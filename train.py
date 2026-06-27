"""
每日中文 MLM 微调脚本
-----------------------
- 默认 base 模型: hfl/chinese-macbert-base  (纯 PyTorch, ~400MB, 中文原生)
- 任务: Masked Language Modeling(掩码语言建模)
- 训练语料: data/corpus.txt  (一行一句,UTF-8 纯中文文本)
- 输出:  ./output/  (含 model + tokenizer + 训练日志)

使用:
  python train.py
  python train.py --model_name bert-base-chinese --epochs 5
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
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="每日中文 MLM 微调")
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
    return p.parse_args()


def load_text_dataset(data_path: str) -> Dataset:
    """加载一行一句的 UTF-8 纯文本语料,自动过滤空行/超短行。"""
    if not os.path.exists(data_path):
        sys.exit(f"[ERROR] 找不到训练语料: {data_path}")

    ds = load_dataset("text", data_files={"train": data_path})["train"]
    # 过滤空行
    ds = ds.filter(lambda x: x["text"] is not None and len(x["text"].strip()) >= 4)
    print(f"[INFO] 加载语料: {len(ds)} 条")
    if len(ds) < 10:
        sys.exit(f"[ERROR] 语料过少({len(ds)} 条),无法训练")
    return ds


def build_features(ds: Dataset, tokenizer, max_len: int) -> Dataset:
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

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # 2. 模型
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 模型参数量: {n_params/1e6:.1f}M")

    # 3. 数据
    raw_ds = load_text_dataset(args.data_path)
    tokenized = build_features(raw_ds, tokenizer, args.max_len)

    # 4. 数据整理器(随机 mask)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )

    # 5. 训练参数
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
        report_to=[],            # 关闭 wandb / tensorboard
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

    # 6. 保存最终模型
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 7. 写训练指标
    log_path = Path(args.output_dir) / "training_log.json"
    metrics = {
        "model_name": args.model_name,
        "data_path": args.data_path,
        "n_samples": len(tokenized),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_len": args.max_len,
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
