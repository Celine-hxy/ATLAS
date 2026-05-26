#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算 splits_by_source 中每个 source 与 splits_by_dataset/test 中每个测试集之间的
data-to-data similarity（embedding cos_sim），存为 npy。
- 使用 utils.plot_mapping.normalize_source 做 source 归一化
- 每个 (source, test_dataset) 单独存一个 npy，形状 (n_source_samples, n_test_samples)
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from sentence_transformers import util
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.plot_mapping import normalize_source


def _safe_fname(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def _cache_key(prompt: str) -> str:
    """与 s3_get_embedding 一致：pkl 文件名用 MD5(prompt)。"""
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def batch_load_embeddings_by_prompt(tuples_sha1_prompt, emb_cache_dir: Path, num_workers: int, desc=None):
    """Load embeddings from emb_cache_dir / {md5(prompt)}.pkl. tuples_sha1_prompt = [(sha1, prompt), ...]."""
    if not tuples_sha1_prompt:
        return []
    emb_cache_dir = Path(emb_cache_dir)
    keys = [_cache_key(p) for _, p in tuples_sha1_prompt]
    result = [None] * len(keys)

    def _load_one(i, ckey):
        cache_path = emb_cache_dir / f"{ckey}.pkl"
        if not cache_path.exists():
            return (i, None)
        try:
            with cache_path.open("rb") as f:
                return (i, pickle.load(f))
        except (EOFError, pickle.UnpicklingError):
            return (i, None)

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        futures = {ex.submit(_load_one, i, k): i for i, k in enumerate(keys)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc or "Load pkl", unit=" pkl"):
            i, emb = fut.result()
            result[i] = emb
    return result


def load_sources_to_tuples(splits_by_source_dir: Path) -> dict:
    """normalized_source -> list of (sha1, prompt)."""
    norm_to_tuples = {}
    for json_file in sorted(splits_by_source_dir.glob("*.json")):
        if json_file.stem == "__no_source__":
            continue
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        raw_source = None
        for item in data.values():
            if isinstance(item, dict) and "source" in item:
                raw_source = item["source"]
                break
        norm = normalize_source(raw_source)
        if norm is None:
            continue
        tuples = [(k, v.get("prompt", "")) for k, v in data.items() if isinstance(v, dict) and v.get("prompt")]
        if norm not in norm_to_tuples:
            norm_to_tuples[norm] = []
        norm_to_tuples[norm].extend(tuples)
    return norm_to_tuples


def load_test_datasets_to_tuples(splits_by_dataset_test_dir: Path) -> dict:
    """test_stem -> list of (sha1, prompt)."""
    stem_to_tuples = {}
    for json_file in sorted(splits_by_dataset_test_dir.glob("*.json")):
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        tuples = [(k, v.get("prompt", "")) for k, v in data.items() if isinstance(v, dict) and v.get("prompt")]
        stem_to_tuples[json_file.stem] = tuples
    return stem_to_tuples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--splits_by_source_dir",
        type=str,
        default="$ROOT/ATLAS/results/splits_by_source",
        help="Directory of JSON files per source",
    )
    ap.add_argument(
        "--splits_by_dataset_test_dir",
        type=str,
        default="$ROOT/ATLAS/results/splits_by_dataset/test",
        help="Directory of JSON files per test dataset",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="$HOME/similarity",
        help="Output directory for npy files (one per source x test_dataset)",
    )
    ap.add_argument(
        "--emb_cache_dir",
        type=str,
        default="$HOME/emb_cache",
        help="Directory of precomputed embedding pkl files",
    )
    ap.add_argument("--num_workers", type=int, default=128)
    ap.add_argument("--num_threads", type=int, default=64)
    args = ap.parse_args()

    torch.set_num_threads(args.num_threads)
    splits_by_source_dir = Path(args.splits_by_source_dir)
    splits_by_dataset_test_dir = Path(args.splits_by_dataset_test_dir)
    out_dir = Path(args.out_dir)
    emb_cache_dir = Path(args.emb_cache_dir)

    if not splits_by_source_dir.exists():
        print(f"Error: not found: {splits_by_source_dir}")
        return
    if not splits_by_dataset_test_dir.exists():
        print(f"Error: not found: {splits_by_dataset_test_dir}")
        return
    if not emb_cache_dir.exists():
        print(f"Error: emb_cache not found: {emb_cache_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_to_tuples = load_sources_to_tuples(splits_by_source_dir)
    stem_to_tuples = load_test_datasets_to_tuples(splits_by_dataset_test_dir)
    print(f"Sources (normalized): {len(norm_to_tuples)}, Test datasets: {len(stem_to_tuples)}")

    # Embedding cache 与 s3_get_embedding 一致：pkl 文件名 = MD5(prompt)
    stem_to_tensor = {}
    for test_stem, tuples in tqdm(stem_to_tuples.items(), desc="Load test embeddings"):
        if not tuples:
            continue
        test_embs = batch_load_embeddings_by_prompt(tuples, emb_cache_dir, args.num_workers, desc=test_stem[:24])
        valid = [e for e in test_embs if e is not None]
        if valid:
            stem_to_tensor[test_stem] = torch.stack(valid)
        else:
            print(f"  [WARN] {test_stem}: 0/{len(tuples)} embeddings in cache")

    for norm_source, tuples in tqdm(norm_to_tuples.items(), desc="Source vs test"):
        if not tuples:
            continue
        src_embs = batch_load_embeddings_by_prompt(
            tuples, emb_cache_dir, args.num_workers,
            desc=f"src_{_safe_fname(norm_source)[:24]}",
        )
        valid_src = [e for e in src_embs if e is not None]
        if not valid_src:
            print(f"  [WARN] {norm_source}: 0/{len(tuples)} embeddings in cache, skip")
            continue
        src_tensor = torch.stack(valid_src)

        for test_stem, test_tensor in stem_to_tensor.items():
            sim = util.cos_sim(src_tensor, test_tensor)
            arr = sim.cpu().numpy().astype(np.float32)
            fname = f"{_safe_fname(norm_source)}_{_safe_fname(test_stem)}.npy"
            np.save(out_dir / fname, arr)
        print(f"  {norm_source} -> {len(stem_to_tensor)} npy")

    print(f"Done. Outputs under {out_dir}")


if __name__ == "__main__":
    main()
