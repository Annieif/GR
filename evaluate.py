"""
模型评估脚本
==============
对 base / 微调后的模型计算:
  1. Perplexity (PPL)           — 整体语言建模困惑度
  2. Top-k Masked Token Accuracy — MLM 任务最直接的正确率
  3. Fill-Mask 样例               — 定性对比展示

支持:
  --base_model   : 原版 base 模型(HF 仓库名或本地路径)
  --ft_model     : 微调后模型路径(默认 ./output)
  --eval_data    : 评估语料,默认 data/eval.txt
  --report_path  : 评估报告输出 JSON,默认 output/eval_report.json

使用:
  python evaluate.py                                  # 同时评 base + ft
  python evaluate.py --ft_model output/adapter        # 评 LoRA adapter
  python evaluate.py --base_model hfl/chinese-macbert-base --ft_model output
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


# 一些 [MASK] 填空的演示句,用于定性对比
DEMO_SENTENCES = [
    "北京是中国的[MASK].",
    "今天天气很[MASK].",
    "我喜欢吃[MASK].",
    "人工智能是[MASK]的技术.",
    "李白是唐代著名的[MASK].",
    "开源软件让[MASK]更加平等.",
    "杭州以西湖和[MASK]闻名.",
    "深度学习需要大量的[MASK].",
]

TOP_K = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="中文 BERT 评估")
    p.add_argument("--base_model", default="hfl/chinese-macbert-base",
                   help="base 模型(HF id 或路径)")
    p.add_argument("--ft_model", default="output",
                   help="微调后模型路径,或 LoRA adapter 父目录 "
                        "(若路径下存在 adapter/ 子目录则按 LoRA 加载)")
    p.add_argument("--eval_data", default="data/eval.txt",
                   help="评估语料,一行一句 UTF-8 文本")
    p.add_argument("--report_path", default="output/eval_report.json",
                   help="评估报告 JSON 输出路径")
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--mlm_prob", type=float, default=0.15)
    p.add_argument("--skip_base", action="store_true",
                   help="跳过 base 模型评估(只评微调后)")
    p.add_argument("--max_eval_samples", type=int, default=-1,
                   help="最多评估多少条样本(默认 -1 = 全部),用于快速验证/CI")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_eval_data(path: str):
    if not os.path.exists(path):
        sys.exit(f"[ERROR] 找不到评估语料: {path}")
    ds = load_dataset("text", data_files={"eval": path})["eval"]
    ds = ds.filter(lambda x: x["text"] is not None and len(x["text"].strip()) >= 4)
    print(f"[INFO] 评估语料: {len(ds)} 条")
    if len(ds) < 5:
        sys.exit(f"[ERROR] 评估语料过少({len(ds)} 条),无法评估")
    return ds


def prepare_eval_tokenized(eval_ds, tokenizer, args):
    """对评估语料做一次性 tokenize。后续 PPL/Acc/demos 都共享。"""
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True,
                         padding="max_length", max_length=args.max_len)
    # 按 max_eval_samples 截断
    if args.max_eval_samples > 0 and len(eval_ds) > args.max_eval_samples:
        eval_ds = eval_ds.select(range(args.max_eval_samples))
        print(f"[INFO] 评估样本数截断到 {len(eval_ds)}")
    tokenized = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    print(f"[INFO] tokenize 完成: {len(tokenized)} 样本")
    return tokenized


def load_model_any(path: str, tokenizer):
    """统一加载 base / fine-tuned / LoRA adapter / MoMo adapter 四种形式。

    加载优先级:
      1. path/config.json 存在 → 当作合并后的完整 HF 模型
      2. path/adapter/adapter_config.json 存在 → 按 PEFT-LoRA / MoMo adapter 加载
      3. 其它 → 直接当 HF 仓库路径
    """
    path = Path(path)
    config_file = path / "config.json"
    adapter_dir = path / "adapter"
    adapter_cfg = adapter_dir / "adapter_config.json"

    if config_file.exists():
        # ---- 完整 HF 模型(可能是全参/PEFT-merged/MoMo-merged)----
        print(f"[INFO] 加载完整模型: {path}")
        model = AutoModelForMaskedLM.from_pretrained(str(path))
    elif adapter_cfg.exists():
        # ---- Adapter 模式(PEFT-LoRA 或 MoMo)----
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type", "peft")
        base_name = cfg.get("base_model_name_or_path")
        if not base_name:
            sys.exit("[ERROR] adapter_config.json 缺 base_model_name_or_path")
        if model_type == "momo_lora":
            # MoMo adapter:重新注入 MoLoRALinear 结构,加载权重
            try:
                from momo_lora import (
                    inject_momo_lora, load_momo_checkpoint, merge_momo_into_base,
                )
            except ImportError as e:
                sys.exit(f"[ERROR] 加载 MoMo adapter 需要 momo_lora.py: {e}")
            print(f"[INFO] MoMo 模式:base={base_name}, adapter={adapter_dir}")
            base_model = AutoModelForMaskedLM.from_pretrained(base_name)
            n = inject_momo_lora(
                base_model,
                target_module_names=cfg.get("target_module_names", ["query", "value"]),
                n_experts=cfg.get("n_experts", 4),
                top_k=cfg.get("top_k", 2),
                lora_r=cfg.get("lora_r", 8),
                lora_alpha=cfg.get("lora_alpha", 16),
                lora_dropout=cfg.get("lora_dropout", 0.0),
                aux_loss_alpha=cfg.get("aux_loss_alpha", 0.01),
            )
            if n == 0:
                sys.exit(f"[ERROR] MoMo 注入了 0 层,target={cfg.get('target_module_names')}")
            load_momo_checkpoint(base_model, str(adapter_dir))
            model = merge_momo_into_base(base_model)
        else:
            # PEFT-LoRA adapter
            try:
                from peft import PeftModel
            except ImportError:
                sys.exit("[ERROR] 检测到 LoRA adapter,但未安装 peft")
            print(f"[INFO] LoRA 模式:base={base_name}, adapter={adapter_dir}")
            base_model = AutoModelForMaskedLM.from_pretrained(base_name)
            model = PeftModel.from_pretrained(base_model, str(adapter_dir))
            model = model.merge_and_unload()
    else:
        print(f"[INFO] 加载模型: {path}")
        model = AutoModelForMaskedLM.from_pretrained(str(path))
    model.eval()
    return model


@torch.no_grad()
def compute_perplexity(model, tokenizer, tokenized, args) -> float:
    """整体 MLM 困惑度(tokenized 必须已被 tokenize_fn 处理过)。"""
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(tokenized, batch_size=args.batch_size,
                        collate_fn=collator, num_workers=0)

    total_loss = 0.0
    total_count = 0
    t0 = time.time()
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch)
        labels = batch["labels"]
        n = (labels != -100).sum().item()
        if n == 0:
            continue
        total_loss += out.loss.item() * n
        total_count += n

    avg_loss = total_loss / max(total_count, 1)
    ppl = math.exp(avg_loss)
    print(f"[INFO] PPL 计算:avg_loss={avg_loss:.4f}, ppl={ppl:.3f} "
          f"({time.time()-t0:.1f}s, {total_count} masked tokens)")
    return float(ppl)


@torch.no_grad()
def compute_mask_accuracy(model, tokenizer, tokenized, args,
                          k_list=(1, 5)) -> dict:
    """Top-k masked token accuracy(tokenized 必须已被 tokenize_fn 处理过)。"""
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(tokenized, batch_size=args.batch_size,
                        collate_fn=collator, num_workers=0)

    max_k = max(k_list)
    correct = {k: 0 for k in k_list}
    total = 0
    t0 = time.time()
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        labels = batch["labels"]
        mask = (labels != -100)
        if mask.sum() == 0:
            continue
        out = model(**batch)
        logits = out.logits
        topk = logits.topk(max_k, dim=-1).indices
        labels_exp = labels.unsqueeze(-1).expand_as(topk)
        hit = (topk == labels_exp) & mask.unsqueeze(-1)
        for k in k_list:
            correct[k] += hit[..., :k].any(dim=-1).sum().item()
        total += mask.sum().item()
    acc = {k: (correct[k] / total if total else 0.0) for k in k_list}
    print(f"[INFO] Acc 计算:{acc} ({time.time()-t0:.1f}s, {total} masked tokens)")
    return {f"top{k}_acc": round(v, 4) for k, v in acc.items()}


@torch.no_grad()
def run_fill_mask_demos(model, tokenizer, sentences: List[str],
                        top_k: int = TOP_K) -> List[dict]:
    """对若干 [MASK] 句子做定性 fill-mask。"""
    results = []
    for sent in sentences:
        if tokenizer.mask_token not in sent:
            continue
        try:
            enc = tokenizer(sent, return_tensors="pt").to(model.device)
            out = model(**enc)
            mask_idx = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_idx) == 0:
                continue
            midx = mask_idx[0].item()
            probs = torch.softmax(out.logits[0, midx], dim=-1)
            top_probs, top_ids = probs.topk(top_k)
            preds = []
            for pid, pp in zip(top_ids.tolist(), top_probs.tolist()):
                preds.append({"token": tokenizer.convert_ids_to_tokens(pid),
                              "id": pid, "prob": round(float(pp), 4)})
            results.append({"sentence": sent, "predictions": preds})
        except Exception as e:
            results.append({"sentence": sent, "error": str(e)})
    return results


def evaluate_one(name: str, model_path: str, tokenizer, tokenized, args) -> dict:
    print(f"\n========== 评估 [{name}] ==========")
    t0 = time.time()
    model = load_model_any(model_path, tokenizer).to("cpu")
    # device 选择
    if torch.cuda.is_available():
        model = model.to("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()

    ppl = compute_perplexity(model, tokenizer, tokenized, args)
    acc = compute_mask_accuracy(model, tokenizer, tokenized, args)
    demos = run_fill_mask_demos(model, tokenizer, DEMO_SENTENCES, top_k=TOP_K)

    # 释放显存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": name,
        "model_path": model_path,
        "perplexity": round(ppl, 4),
        **acc,
        "fill_mask_demos": demos,
        "eval_time_sec": round(time.time() - t0, 1),
    }


def make_comparison(base_res: dict | None, ft_res: dict) -> dict:
    """对比 base 与 ft 的关键指标。"""
    if not base_res:
        return {}
    delta = {}
    for k in ("perplexity", "top1_acc", "top5_acc"):
        if k in base_res and k in ft_res:
            base_v = base_res[k]
            ft_v = ft_res[k]
            delta[k] = {
                "base": base_v,
                "ft": ft_v,
                "delta": round(ft_v - base_v, 4),
                # 困惑度越低越好,准确率越高越好
                "better": "ft" if (
                    (k == "perplexity" and ft_v < base_v) or
                    (k != "perplexity" and ft_v > base_v)
                ) else "base",
            }
    return delta


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    eval_ds = load_eval_data(args.eval_data)

    # 用 ft 的 tokenizer(若存在)做基准;否则用 base 的
    if Path(args.ft_model, "tokenizer_config.json").exists() or \
       Path(args.ft_model, "tokenizer.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(args.ft_model, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    results: dict = {
        "base_model": args.base_model,
        "ft_model": args.ft_model,
        "eval_data": args.eval_data,
        "n_eval_samples": len(eval_ds),
        "demo_sentences": DEMO_SENTENCES,
    }

    # 一次性 tokenize,base 和 ft 共享(避免重复 ~200MB 数据的 tokenize)
    tokenized = prepare_eval_tokenized(eval_ds, tokenizer, args)

    base_res = None
    if not args.skip_base:
        base_res = evaluate_one("BASE", args.base_model, tokenizer, tokenized, args)
        results["base"] = base_res

    ft_res = evaluate_one("FINETUNED", args.ft_model, tokenizer, tokenized, args)
    results["finetuned"] = ft_res

    results["comparison"] = make_comparison(base_res, ft_res)

    # 打印简洁总结
    print("\n========== 评估小结 ==========")
    for k in ("perplexity", "top1_acc", "top5_acc"):
        line = f"{k}: "
        if base_res:
            line += f"base={base_res.get(k, 'N/A')}  "
        line += f"ft={ft_res.get(k, 'N/A')}"
        print(line)
    if "comparison" in results and "perplexity" in results["comparison"]:
        p = results["comparison"]["perplexity"]
        print(f"  → PPL 变化: {p['base']} → {p['ft']}  "
              f"(Δ {p['delta']:+.4f}, 优胜方: {p['better']})")

    os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 报告已保存: {args.report_path}")


if __name__ == "__main__":
    main()
