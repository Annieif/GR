"""
每日中文 BERT 微调脚本(支持可选 LoRA)
----------------------------------------
- 默认 base 模型: hfl/chinese-macbert-base  (纯 PyTorch, ~400MB, 中文原生)
- 任务: Masked Language Modeling(掩码语言建模)
- 训练语料: data/corpus.txt  (一行一句,UTF-8 纯中文文本)
- 输出:  ./output/  (含 model + tokenizer + 训练日志)
- 可选:  --use_lora 走 PEFT LoRA 路线,显存/磁盘更省,便于频繁日训

使用:
  python train.py                                  # 全参微调 macbert-base
  python train.py --use_lora                      # LoRA 微调(默认参数)
  python train.py --use_lora --lora_r 16 --epochs 5
"""
from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="每日中文 MLM 微调(支持 LoRA)")
    p.add_argument("--model_name", default="hfl/chinese-macbert-base",
                   help="HuggingFace 上的 base 模型(纯 PyTorch)")
    p.add_argument("--data_path", default="data/corpus.txt",
                   help="训练语料,一行一句 UTF-8 文本")
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
    p.add_argument("--save_total_limit", type=int, default=1)

    # ---- LoRA 参数 ----
    p.add_argument("--use_lora", action="store_true",
                   help="启用 LoRA(PEFT),只训练低秩适配器,节省资源")
    p.add_argument("--lora_r", type=int, default=8, help="LoRA 秩")
    p.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    p.add_argument("--lora_target", default="query,key,value",
                   help="LoRA 注入的目标模块名,逗号分隔")
    p.add_argument("--merge_lora", action="store_true",
                   help="训练完成后将 LoRA 合并进 base,保存为完整模型 "
                        "(evaluate.py 不用感知 LoRA)")
    return p.parse_args()


def load_text_dataset(data_path: str):
    if not os.path.exists(data_path):
        sys.exit(f"[ERROR] 找不到训练语料: {data_path}")
    ds = load_dataset("text", data_files={"train": data_path})["train"]
    ds = ds.filter(lambda x: x["text"] is not None and len(x["text"].strip()) >= 4)
    print(f"[INFO] 加载语料: {len(ds)} 条")
    if len(ds) < 10:
        sys.exit(f"[ERROR] 语料过少({len(ds)} 条),无法训练")
    return ds


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


def maybe_apply_lora(model, args):
    """如果 --use_lora,用 PEFT 包一层 LoRA。"""
    if not args.use_lora:
        return model, False

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        sys.exit("[ERROR] 启用 --use_lora 但未安装 peft,pip install peft")

    target_modules = [m.strip() for m in args.lora_target.split(",") if m.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # MLM 走 FEATURE_EXTRACTION
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=["cls"],  # 同时训练 MLM 头,否则 loss 不会下降
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, True


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] 模型: {args.model_name}")
    print(f"[INFO] 语料: {args.data_path}")
    print(f"[INFO] 输出: {args.output_dir}")
    print(f"[INFO] Epochs={args.epochs}  bs={args.batch_size}  lr={args.lr}")
    print(f"[INFO] LoRA: {'ON (r=%d, alpha=%d)' % (args.lora_r, args.lora_alpha)
                       if args.use_lora else 'OFF (全参微调)'}")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # 2. 加载 base
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] base 参数量: {n_params/1e6:.1f}M")

    # 3. 可选 LoRA
    model, used_lora = maybe_apply_lora(model, args)
    if used_lora:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] LoRA trainable: {trainable/1e6:.3f}M "
              f"({100*trainable/n_params:.4f}%)")

    # 4. 数据
    raw_ds = load_text_dataset(args.data_path)
    tokenized = build_features(raw_ds, tokenizer, args.max_len)

    # 5. 数据整理器
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )

    # 6. 训练参数
    fp16_ok = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
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
    train_result = trainer.train()
    train_seconds = time.time() - t0

    # 7. 保存:LoRA 走两种保存策略
    if used_lora and args.merge_lora:
        print("[INFO] 合并 LoRA 到 base 并保存完整模型")
        merged = model.merge_and_unload()
        merged.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
    elif used_lora:
        # 只保存 LoRA adapter
        print("[INFO] 仅保存 LoRA adapter(注意 evaluate.py 需要先加载 base)")
        model.save_pretrained(os.path.join(args.output_dir, "adapter"))
        tokenizer.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # 8. 写训练指标
    log_path = Path(args.output_dir) / "training_log.json"
    metrics = {
        "model_name": args.model_name,
        "data_path": args.data_path,
        "n_samples": len(tokenized),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_len": args.max_len,
        "use_lora": used_lora,
        "lora_r": args.lora_r if used_lora else None,
        "lora_alpha": args.lora_alpha if used_lora else None,
        "lora_target": args.lora_target if used_lora else None,
        "merge_lora": used_lora and args.merge_lora,
        "train_runtime_sec": round(train_seconds, 2),
        "train_loss": train_result.training_loss,
        "metrics": train_result.metrics,
        "history": trainer.state.log_history,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 训练完成,耗时 {train_seconds:.1f}s,最终 loss={train_result.training_loss:.4f}")
    print(f"[INFO] 模型与日志已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
