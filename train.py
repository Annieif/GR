"""
每日中文 BERT 微调脚本(支持 全参 / LoRA / MoE+Multi-LoRA)
----------------------------------------------------------------
- 默认 base 模型: hfl/chinese-macbert-base  (纯 PyTorch, ~400MB, 中文原生)
- 任务: Masked Language Modeling(掩码语言建模)
- 训练语料: data/corpus.txt + (可选) data/extra_corpora/*.txt
- 输出:  ./output/
  * 全参: model.safetensors + tokenizer
  * LoRA(merge): 完整模型(便于 evaluate.py 直接用)
  * LoRA(no-merge): output/adapter/  (PEFT adapter)
  * MoMo:        output/adapter/  (safetensors + adapter_config.json)

使用:
  python train.py                                    # 全参微调
  python train.py --use_lora --merge_lora            # LoRA 微调并合并
  python train.py --use_momo                         # MoE + Multi-LoRA
  python train.py --use_momo --momo_n_experts 8 --momo_top_k 2
  python train.py --max_steps 100               # 快速试跑(只跑 100 步)
  python train.py --resume_from_checkpoint output/checkpoint-100   # 断点续训
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# 本地 MoMo 模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
from momo_lora import (  # noqa: E402
    add_momo_aux_loss_hook,
    get_momo_param_count,
    inject_momo_lora,
    save_momo_checkpoint,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="每日中文 MLM 微调(支持全参/LoRA/MoMo)")
    p.add_argument("--model_name", default="hfl/chinese-macbert-base",
                   help="HuggingFace 上的 base 模型(纯 PyTorch)")
    p.add_argument("--data_path", default="data/corpus.txt",
                   help="主训练语料,一行一句 UTF-8 文本")
    p.add_argument("--extra_corpus_dir", default="data/extra_corpora",
                   help="额外语料目录(可选),目录下所有 .txt 都会被读入")
    p.add_argument("--output_dir", default="output",
                   help="微调后模型输出目录")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--mlm_prob", type=float, default=0.15)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_steps", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="训练最大步数(默认 -1 = 用 epochs),用于快速试跑或限速")
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="从某个 checkpoint 目录恢复训练(transformers Trainer 原生支持)")
    p.add_argument("--dry_run", action="store_true",
                   help="只做加载数据 + 模型 + 1 步训练就退出,用于快速验证环境")
    p.add_argument("--save_total_limit", type=int, default=1)

    # ---- PEFT LoRA ----
    p.add_argument("--use_lora", action="store_true",
                   help="启用 PEFT LoRA(单 adapter)")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_target", default="query,key,value")
    p.add_argument("--merge_lora", action="store_true",
                   help="训练完成后把 LoRA 合并进 base,保存为完整模型")

    # ---- MoE + Multi-LoRA(MoMo) ----
    p.add_argument("--use_momo", action="store_true",
                   help="启用 MoE + Multi-LoRA(自定义 momo_lora 模块)")
    p.add_argument("--momo_target", default="query,value",
                   help="MoMo 注入的目标 Linear 名称,逗号分隔")
    p.add_argument("--momo_n_experts", type=int, default=4)
    p.add_argument("--momo_top_k", type=int, default=2)
    p.add_argument("--momo_lora_r", type=int, default=8)
    p.add_argument("--momo_lora_alpha", type=int, default=16)
    p.add_argument("--momo_lora_dropout", type=float, default=0.0)
    p.add_argument("--momo_aux_alpha", type=float, default=0.01,
                   help="MoMo 负载均衡辅助 loss 系数")
    return p.parse_args()


# ---------------------------------------------------------------- data
def list_corpus_files(data_path: str, extra_dir: str) -> list[str]:
    """收集主语料 + 额外语料目录下的所有 .txt,统计每文件行数。"""
    files = []
    if data_path and os.path.exists(data_path):
        files.append(data_path)
    if extra_dir and os.path.isdir(extra_dir):
        for fn in sorted(glob.glob(os.path.join(extra_dir, "*.txt"))):
            if fn not in files:
                files.append(fn)
    print(f"[INFO] 加载 {len(files)} 个语料文件:")
    total_lines = 0
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
        n = len(lines)
        # 估算平均字符数
        avg_chars = sum(len(l.strip()) for l in lines) / max(n, 1)
        total_lines += n
        print(f"  - {f}  ({n} lines, ~{avg_chars:.0f} chars/line)")
    print(f"[INFO] 总行数: {total_lines}")
    return files


def load_combined_dataset(files: list):
    """把多个 .txt 合并成一个 HF Dataset(每行一句)。"""
    from datasets import Dataset
    rows = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if len(t) >= 4:
                    rows.append({"text": t})
    if len(rows) < 10:
        sys.exit(f"[ERROR] 合并后语料过少({len(rows)} 条),无法训练")
    print(f"[INFO] 合并语料共 {len(rows)} 条")
    return Dataset.from_list(rows)


def build_features(ds, tokenizer, max_len: int):
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
    cols_to_remove = [c for c in ds.column_names if c != "text"]
    return ds.map(tokenize_fn, batched=True, remove_columns=cols_to_remove,
                  desc="Tokenizing")


# ---------------------------------------------------------------- lora
def maybe_apply_peft_lora(model, args):
    if not args.use_lora:
        return model, False
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        sys.exit("[ERROR] 启用 --use_lora 但未安装 peft")
    target_modules = [m.strip() for m in args.lora_target.split(",") if m.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=["cls"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, True


# ---------------------------------------------------------------- momo
def maybe_apply_momo(model, args):
    if not args.use_momo:
        return model, False
    targets = [m.strip() for m in args.momo_target.split(",") if m.strip()]
    n = inject_momo_lora(
        model,
        target_module_names=targets,
        n_experts=args.momo_n_experts,
        top_k=args.momo_top_k,
        lora_r=args.momo_lora_r,
        lora_alpha=args.momo_lora_alpha,
        lora_dropout=args.momo_lora_dropout,
        aux_loss_alpha=args.momo_aux_alpha,
    )
    print(f"[INFO] MoMo: 替换了 {n} 个 Linear -> MoLoRALinear")
    if n == 0:
        sys.exit("[ERROR] MoMo 注入了 0 层,请检查 --momo_target (默认 query,value)")
    # 在 model 上记一下元信息(便于 save)
    model._momo_n_experts = args.momo_n_experts
    model._momo_top_k = args.momo_top_k
    model._momo_lora_r = args.momo_lora_r
    model._momo_lora_alpha = args.momo_lora_alpha
    model._momo_lora_dropout = args.momo_lora_dropout
    model._momo_aux_loss_alpha = args.momo_aux_alpha
    model._momo_targets = targets
    # 关键:加 hook 让 Trainer 拿到的 loss 自动加上 MoMo 辅助 loss
    add_momo_aux_loss_hook(model)
    return model, True


# ---------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    if args.use_lora and args.use_momo:
        sys.exit("[ERROR] --use_lora 与 --use_momo 互斥,请只选一个")

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] 模型: {args.model_name}")
    print(f"[INFO] 输出: {args.output_dir}")
    print(f"[INFO] Epochs={args.epochs}  bs={args.batch_size}  lr={args.lr}")
    if args.use_momo:
        print(f"[INFO] 模式: MoMo  experts={args.momo_n_experts} "
              f"top_k={args.momo_top_k} r={args.momo_lora_r} "
              f"alpha={args.momo_lora_alpha} targets={args.momo_target}")
    elif args.use_lora:
        print(f"[INFO] 模式: PEFT-LoRA  r={args.lora_r} alpha={args.lora_alpha} "
              f"targets={args.lora_target}")
    else:
        print(f"[INFO] 模式: 全参微调")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # 2. 加载 base
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] base 参数量: {n_params/1e6:.1f}M")

    # 3. 可选 LoRA / MoMo
    model, used_peft_lora = maybe_apply_peft_lora(model, args)
    model, used_momo = maybe_apply_momo(model, args)

    if used_momo:
        stats = get_momo_param_count(model)
        print(f"[INFO] MoMo trainable: {stats['momo_params']/1e6:.3f}M "
              f"({stats['trainable_pct']}%) layers={stats['n_momo_layers']}")

    # 4. 数据
    files = list_corpus_files(args.data_path, args.extra_corpus_dir)
    raw_ds = load_combined_dataset(files)
    tokenized = build_features(raw_ds, tokenizer, args.max_len)

    # 5. 数据整理器
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )

    # 5.5 dry-run:验证一切就绪,跑 1 步就退出
    if args.dry_run:
        print("[DRY-RUN] 验证环境:数据 / 模型 / 1 步训练")
        # 用一个小子集做 1 步训练
        small_ds = tokenized.select(range(min(8, len(tokenized))))
        small_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, "_dryrun"),
            overwrite_output_dir=True,
            num_train_epochs=1,
            max_steps=1,
            per_device_train_batch_size=2,
            learning_rate=args.lr,
            logging_steps=1,
            report_to=[],
            seed=args.seed,
            fp16=torch.cuda.is_available(),
            disable_tqdm=True,
        )
        small_trainer = Trainer(
            model=model,
            args=small_args,
            data_collator=collator,
            train_dataset=small_ds,
            tokenizer=tokenizer,
        )
        small_trainer.train()
        print("[DRY-RUN] 一切就绪 ✅")
        return

    # 6. 训练参数
    fp16_ok = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        logging_steps=args.log_steps,
        report_to=[],
        seed=args.seed,
        fp16=fp16_ok,
        dataloader_num_workers=2,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_seconds = time.time() - t0

    # 7. 保存 — output/ 始终包含「merged 完整 HF 模型」+ tokenizer,便于打包进 release
    #              训练产生的原始 adapter(PEFT-LoRA / MoMo)单独存到 output/adapter/ 子目录
    if used_peft_lora and args.merge_lora:
        print("[INFO] 合并 LoRA 到 base 并保存完整模型")
        merged = model.merge_and_unload()
        merged.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        try:
            model.save_pretrained(os.path.join(args.output_dir, "adapter"))
        except Exception as e:
            print(f"[WARN] 保存 PEFT adapter 备份失败(可忽略): {e}")
    elif used_peft_lora:
        print("[INFO] 额外合并 LoRA 到 base 存为完整模型;adapter 也存到子目录")
        merged = model.merge_and_unload()
        merged.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        try:
            model.save_pretrained(os.path.join(args.output_dir, "adapter"))
        except Exception as e:
            print(f"[WARN] 保存 PEFT adapter 备份失败(可忽略): {e}")
    elif used_momo:
        from momo_lora import merge_momo_into_base
        # 先保存 MoMo adapter(必须在 merge 之前,merge 会原地替换 MoLoRALinear)
        adapter_dir = os.path.join(args.output_dir, "adapter")
        save_momo_checkpoint(model, tokenizer, adapter_dir,
                             base_model_name=args.model_name)
        # 再合并并保存完整模型
        print("[INFO] 合并 MoMo 到 base(均匀近似),存为完整模型")
        merged = merge_momo_into_base(model)
        merged.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    print(f"[INFO] output/ 内已包含 merged 完整模型,可直接 AutoModelForMaskedLM.from_pretrained 加载")

    # 9. 列出 output/ 实际产物,方便排查
    print(f"[INFO] output/ 产物清单:")
    for root, dirs, files in os.walk(args.output_dir):
        # 排除大体积文件
        for f in sorted(files):
            full = os.path.join(root, f)
            try:
                size = os.path.getsize(full)
                rel = os.path.relpath(full, args.output_dir)
                if size > 1024 * 1024:
                    print(f"  {rel}  ({size/1024/1024:.1f} MB)")
                else:
                    print(f"  {rel}  ({size/1024:.1f} KB)")
            except OSError:
                pass

    # 8. 写训练指标
    log_path = Path(args.output_dir) / "training_log.json"
    metrics = {
        "model_name": args.model_name,
        "data_path": args.data_path,
        "extra_corpus_dir": args.extra_corpus_dir,
        "n_corpus_files": len(files),
        "n_samples": len(tokenized),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_len": args.max_len,
        "mode": ("momo" if used_momo else ("peft_lora" if used_peft_lora else "full")),
        "use_lora": used_peft_lora,
        "lora_r": args.lora_r if used_peft_lora else None,
        "use_momo": used_momo,
        "momo_n_experts": args.momo_n_experts if used_momo else None,
        "momo_top_k": args.momo_top_k if used_momo else None,
        "momo_target": args.momo_target if used_momo else None,
        "train_runtime_sec": round(train_seconds, 2),
        "train_loss": train_result.training_loss,
        "metrics": train_result.metrics,
        "history": trainer.state.log_history,
    }
    if used_momo:
        stats = get_momo_param_count(model)
        metrics["momo_stats"] = stats

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 训练完成,耗时 {train_seconds:.1f}s,最终 loss={train_result.training_loss:.4f}")
    print(f"[INFO] 模型与日志已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
