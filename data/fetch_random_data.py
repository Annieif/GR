"""
每日随机拉取一个未用过的中文数据集(健壮版)
================================================

从 data/dataset_pool.json 候选池中,先做:
  1) HfApi.dataset_info() 校验存在性(快速 HEAD 风格),跳过不存在的
  2) load_dataset() 加载
  3) 按 text_fields 抽取;若 0 条,自动回退到「任意 string 字段」
  4) 写到 data/extra_corpora/<ts>_<id>.txt 并追加 used_datasets.json

如果整个候选池都失败,在 max_attempts 次后报错,让 workflow 失败并告警。

使用:
    python data/fetch_random_data.py
    python data/fetch_random_data.py --max_samples 300 --max_attempts 10
    python data/fetch_random_data.py --validate_only           # 只校验候选池
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


# ---------------------------------------------------------------- helpers
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


def verify_exists(ds_id: str, config: str | None = None,
                  timeout: int = 8) -> tuple[bool, str]:
    """用 HfApi.dataset_info 做 HEAD 风格校验。返回 (是否存在, 错误信息)。"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.dataset_info(ds_id, config_name=config, timeout=timeout)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"


def get_string_fields(ds) -> list[str]:
    """从 dataset.features 中识别 string 类字段(优先 ClassLabel/Value('string'))。"""
    out = []
    for name, feat in ds.features.items():
        s = str(feat).lower()
        # 常见 string 字段表达
        if "string" in s or s.startswith("value"):
            out.append(name)
    return out


def extract_text_from_row(row: dict, preferred: list[str], string_fields: list[str],
                          min_len: int = 4) -> str | None:
    """优先用 preferred 字段拼接;若都没有,回退到「任意 string 字段」
    或「conversations 风格(list[dict{from, value}])」。"""
    # 1) 优先:拼接所有命中的 preferred(都是 string)
    parts = []
    for f in preferred:
        v = row.get(f)
        if isinstance(v, str):
            s = v.strip()
            if s and len(s) >= min_len:
                parts.append(s)
    if parts:
        return " ".join(parts)

    # 2) 回退:任意 string 字段
    for f in string_fields:
        if f in preferred:
            continue
        v = row.get(f)
        if isinstance(v, str):
            s = v.strip()
            if s and len(s) >= min_len:
                return s

    # 3) 回退:list[dict{from, value}] / list[dict{role, content}] 等
    #    (BelleGroup / ShareGPT / 通用 ChatML 风格)
    for f in string_fields:
        v = row.get(f)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            parts = []
            for item in v:
                if not isinstance(item, dict):
                    continue
                txt = (item.get("value") or item.get("text") or
                       item.get("content") or item.get("answer"))
                if isinstance(txt, str):
                    s = txt.strip()
                    if s and len(s) >= min_len:
                        parts.append(s)
            if parts:
                return " ".join(parts)

    return None


# ---------------------------------------------------------------- core
def fetch_one(entry: dict, max_samples: int) -> tuple[int, str]:
    from datasets import load_dataset

    ds_id = entry["id"]
    config = entry.get("config")
    split = entry.get("split", "train")
    preferred = list(entry.get("text_fields", ["text"]))
    per_dataset_cap = int(entry.get("max_samples", max_samples))
    limit = min(max_samples, per_dataset_cap)

    print(f"[INFO] load_dataset({ds_id}, config={config}, split={split}) ...")
    try:
        if config:
            ds = load_dataset(ds_id, config, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(ds_id, split=split, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"load_dataset 失败: {e}")

    cols = list(ds.column_names)
    string_fields = get_string_fields(ds)
    print(f"[INFO] schema={cols}  string_fields={string_fields}")

    # 如果数据超大,先采样再遍历,加速
    if len(ds) > limit * 5:
        idx = random.sample(range(len(ds)), limit * 2)
        ds = ds.select(idx)
        print(f"[INFO] 采样到 {len(ds)} 条 (原 {len(ds)})")

    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_id = ds_id.replace("/", "_")
    out_path = EXTRA_DIR / f"{ts}_{safe_id}.txt"

    count = 0
    fallback_used = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            if count >= limit:
                break
            text = extract_text_from_row(row, preferred, string_fields)
            if text is None:
                continue
            # 若不是从 preferred 字段抽出的,记一笔(用于排查)
            if not any(isinstance(row.get(pf), str) and row.get(pf, "").strip()
                       for pf in preferred):
                fallback_used += 1
            f.write(text + "\n")
            count += 1

    if count == 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"数据集 {ds_id} 抽取后 0 条 (schema={cols})")
    if fallback_used:
        print(f"[INFO] 抽取 {count} 条,其中 {fallback_used} 条用了字段自动探测")

    return count, str(out_path.relative_to(REPO_ROOT))


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=300)
    ap.add_argument("--max_attempts", type=int, default=10,
                    help="对加载/抽取失败的「重试次数」")
    ap.add_argument("--allow_repeat_when_exhausted", action="store_true",
                    help="所有候选都跑过一遍时是否允许重复")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--validate_only", action="store_true",
                    help="只校验候选池,打印每个数据集是否存在,不做实际拉取")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    else:
        random.seed(datetime.now(timezone.utc).timestamp())

    pool = load_pool()
    manifest = load_manifest()
    used_ids = {x["id"] for x in manifest.get("used", [])}

    available_unused = [d for d in pool if d["id"] not in used_ids]
    if not available_unused:
        if not args.allow_repeat_when_exhausted:
            print("[WARN] 候选池已全部用过,这次允许重复(--allow_repeat_when_exhausted)")
        available_unused = list(pool)

    # ---- 0. 预校验:哪些数据集在 Hub 上真实存在 ----
    print(f"[INFO] 候选池 {len(pool)} 个,未用过 {len(available_unused)} 个,开始预校验 ...")
    valid_pool = []
    for entry in available_unused:
        ok, err = verify_exists(entry["id"], entry.get("config"))
        marker = "✓" if ok else "✗"
        msg = "" if ok else f" ({err})"
        print(f"  {marker} {entry['id']}{msg}")
        if ok:
            valid_pool.append(entry)

    if args.validate_only:
        print(f"\n[validate_only] {len(valid_pool)}/{len(available_unused)} 个候选可用")
        return 0 if valid_pool else 1

    if not valid_pool:
        print("[ERROR] 候选池里没有任何数据集在 Hub 上真实存在。请检查 data/dataset_pool.json",
              file=sys.stderr)
        return 1

    # 打乱顺序,逐个尝试
    random.shuffle(valid_pool)
    last_err = None
    attempts = 0
    for entry in valid_pool:
        if attempts >= args.max_attempts:
            print(f"[INFO] 已达 max_attempts={args.max_attempts},停止尝试")
            break
        attempts += 1
        try:
            n, rel_path = fetch_one(entry, args.max_samples)
            print(f"[OK] 选 {entry['id']} ({entry.get('description','')}) "
                  f"写入 {n} 条到 {rel_path}")

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            record = {
                "id": entry["id"],
                "timestamp": ts,
                "n_samples": n,
                "out_path": rel_path,
            }
            manifest.setdefault("used", []).append(record)
            manifest.setdefault("history", []).append({**record,
                "description": entry.get("description", "")})
            save_manifest(manifest)
            print(f"[INFO] 更新 {MANIFEST_PATH.relative_to(REPO_ROOT)} "
                  f"(已用 {len(manifest['used'])} 个)")
            print(f"::set-output name=dataset_id::{entry['id']}")
            print(f"::set-output name=n_samples::{n}")
            print(f"::set-output name=out_path::{rel_path}")
            return 0
        except Exception as e:
            last_err = e
            print(f"[WARN] 第 {attempts}/{args.max_attempts} 次尝试 "
                  f"{entry['id']} 失败: {e}", file=sys.stderr)
            continue

    print(f"[ERROR] 全部 {attempts} 次尝试都失败,最后错误: {last_err}",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
