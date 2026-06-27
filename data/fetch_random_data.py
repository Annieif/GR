"""
每日随机拉取一个未用过的中文数据集
===================================

从 data/dataset_pool.json 候选池中随机抽一个未在 data/used_datasets.json
出现过的数据集,下载 → 抽取纯文本 → 写入 data/extra_corpora/<ts>_<id>.txt,
并把这次选择追加到 manifest,写回 used_datasets.json。

若候选全部失败,在 max_attempts 次重试后报错,让 workflow 失败并告警。

使用:
    python data/fetch_random_data.py
    python data/fetch_random_data.py --max_samples 300 --max_attempts 5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_PATH = REPO_ROOT / "data" / "dataset_pool.json"
MANIFEST_PATH = REPO_ROOT / "data" / "used_datasets.json"
EXTRA_DIR = REPO_ROOT / "data" / "extra_corpora"


def load_pool() -> list:
    with open(POOL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"used": [], "history": []}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(m: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


def fetch_one(entry: dict, max_samples: int) -> tuple[int, str]:
    """
    拉取单个数据集,返回 (写入条数, 临时文件路径)。
    """
    from datasets import load_dataset

    ds_id = entry["id"]
    config = entry.get("config")
    split = entry.get("split", "train")
    text_fields = entry.get("text_fields", ["text"])
    limit = min(max_samples, int(entry.get("max_samples", max_samples)))

    print(f"[INFO] load_dataset({ds_id}, config={config}, split={split}) ...")
    try:
        if config:
            ds = load_dataset(ds_id, config, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(ds_id, split=split, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"load_dataset 失败: {e}")

    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_id = ds_id.replace("/", "_")
    out_path = EXTRA_DIR / f"{ts}_{safe_id}.txt"

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            if count >= limit:
                break
            # 拼接 text_fields
            parts = []
            for tf in text_fields:
                v = row.get(tf) if isinstance(row, dict) else None
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    parts.append(s)
            if not parts:
                continue
            text = " ".join(parts)
            if len(text) < 4:
                continue
            f.write(text + "\n")
            count += 1

    if count == 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"数据集 {ds_id} 抽取后 0 条")

    return count, str(out_path.relative_to(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=300,
                    help="每个数据集最多取的样本数")
    ap.add_argument("--max_attempts", type=int, default=4,
                    help="候选全部失败时,最大重试次数")
    ap.add_argument("--allow_repeat_when_exhausted", action="store_true",
                    help="所有候选都跑过一遍时是否允许重复")
    ap.add_argument("--seed", type=int, default=None,
                    help="随机种子(默认用当前时间)")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    else:
        random.seed(datetime.now(timezone.utc).timestamp())

    pool = load_pool()
    manifest = load_manifest()
    used_ids = {x["id"] for x in manifest.get("used", [])}

    available = [d for d in pool if d["id"] not in used_ids]
    if not available:
        if not args.allow_repeat_when_exhausted:
            print("[WARN] 候选池已全部用过,这次允许重复(--allow_repeat_when_exhausted)")
        available = list(pool)

    random.shuffle(available)
    print(f"[INFO] 候选池 {len(pool)} 个,未用过 {len(available)} 个")

    last_err = None
    for i, entry in enumerate(available[: args.max_attempts]):
        try:
            n, rel_path = fetch_one(entry, args.max_samples)
            print(f"[OK] 选 {entry['id']} ({entry.get('description','')}) 写入 {n} 条到 {rel_path}")

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            manifest.setdefault("used", []).append({
                "id": entry["id"],
                "timestamp": ts,
                "n_samples": n,
                "out_path": rel_path,
            })
            manifest.setdefault("history", []).append({
                "id": entry["id"],
                "timestamp": ts,
                "n_samples": n,
                "out_path": rel_path,
                "description": entry.get("description", ""),
            })
            save_manifest(manifest)
            print(f"[INFO] 更新 {MANIFEST_PATH.relative_to(REPO_ROOT)} (已用 {len(manifest['used'])} 个)")
            # 写到 stdout 一行关键信息,方便 workflow 解析
            print(f"::set-output name=dataset_id::{entry['id']}")
            print(f"::set-output name=n_samples::{n}")
            print(f"::set-output name=out_path::{rel_path}")
            return 0
        except Exception as e:
            last_err = e
            print(f"[WARN] 第 {i+1} 次尝试 {entry['id']} 失败: {e}", file=sys.stderr)
            continue

    print(f"[ERROR] 尝试 {args.max_attempts} 次都失败,最后错误: {last_err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
