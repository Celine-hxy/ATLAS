#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute source-vs-test semantic similarity (cosine) and save per (source, test_dataset) as .npy.
Enhancements:
- Use utils.plot_mapping.normalize_source to normalize sources
- Skip test-vs-test similarity computations (if a normalized source name equals a test stem)
- Save meta mapping for each saved npy:
    - row_sha1s: sha1 list aligned with npy rows (valid src embeddings only)
    - col_sha1s: sha1 list aligned with npy cols (valid test embeddings only)
    - counts / cache hit stats
- Save argmax indices for leakage inspection:
    - top1_src_idx_per_test: for each test column j, argmax row i (np.int32), shape (n_test,)
    - top1_sim_per_test: max similarity per test column, shape (n_test,)
Notes:
- Embedding cache key: MD5(prompt) to match s3_get_embedding.
- Embeddings are coerced to 1D float tensors and L2-normalized explicitly.
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.plot_mapping import normalize_source  # noqa: E402


def _safe_fname(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def _cache_key(prompt: str) -> str:
    """Match s3_get_embedding: pkl filename uses MD5(prompt)."""
    # If your upstream normalizes prompt before hashing, mirror it here.
    prompt = prompt.strip()
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def _to_1d_float_tensor(x) -> Optional[torch.Tensor]:
    """Convert loaded embedding to 1D float tensor. Return None if invalid."""
    if x is None:
        return None
    try:
        if isinstance(x, torch.Tensor):
            t = x
        elif isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        else:
            t = torch.tensor(x)
        t = t.float().view(-1)
        if t.numel() == 0 or not torch.isfinite(t).all():
            return None
        return t
    except Exception:
        return None


def batch_load_embeddings_by_prompt(
    tuples_sha1_prompt: List[Tuple[str, str]],
    emb_cache_dir: Path,
    num_workers: int,
    desc: Optional[str] = None,
) -> List[Optional[torch.Tensor]]:
    """
    Load embeddings from emb_cache_dir / {md5(prompt)}.pkl.
    tuples_sha1_prompt = [(sha1, prompt), ...]
    Returns a list aligned to tuples_sha1_prompt order; each item is a 1D torch.FloatTensor or None.
    """
    if not tuples_sha1_prompt:
        return []
    emb_cache_dir = Path(emb_cache_dir)
    keys = [_cache_key(p) for _, p in tuples_sha1_prompt]
    result: List[Optional[torch.Tensor]] = [None] * len(keys)

    def _load_one(i: int, ckey: str):
        cache_path = emb_cache_dir / f"{ckey}.pkl"
        if not cache_path.exists():
            return (i, None)
        try:
            with cache_path.open("rb") as f:
                obj = pickle.load(f)
            return (i, _to_1d_float_tensor(obj))
        except (EOFError, pickle.UnpicklingError, OSError):
            return (i, None)

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        futures = {ex.submit(_load_one, i, k): i for i, k in enumerate(keys)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc or "Load pkl", unit=" pkl"):
            i, emb = fut.result()
            result[i] = emb
    return result


def load_sources_to_tuples(splits_by_source_dir: Path) -> Dict[str, List[Tuple[str, str]]]:
    """normalized_source -> list of (sha1, prompt)."""
    norm_to_tuples: Dict[str, List[Tuple[str, str]]] = {}
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
        if not tuples:
            continue
        norm_to_tuples.setdefault(norm, []).extend(tuples)
    return norm_to_tuples


def load_test_datasets_to_tuples(splits_by_dataset_test_dir: Path) -> Dict[str, List[Tuple[str, str]]]:
    """test_stem -> list of (sha1, prompt)."""
    stem_to_tuples: Dict[str, List[Tuple[str, str]]] = {}
    for json_file in sorted(splits_by_dataset_test_dir.glob("*.json")):
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        tuples = [(k, v.get("prompt", "")) for k, v in data.items() if isinstance(v, dict) and v.get("prompt")]
        stem_to_tuples[json_file.stem] = tuples
    return stem_to_tuples


def _stack_valid(
    tuples_sha1_prompt: List[Tuple[str, str]],
    embs: List[Optional[torch.Tensor]],
) -> Tuple[Optional[torch.Tensor], List[str], int]:
    """
    Filter None embeddings, stack into (n, d) tensor, and return:
    - tensor or None
    - valid_sha1s aligned with tensor rows
    - total_count
    """
    assert len(tuples_sha1_prompt) == len(embs)
    valid_tensors: List[torch.Tensor] = []
    valid_sha1s: List[str] = []
    for (sha1, _), e in zip(tuples_sha1_prompt, embs):
        if e is None:
            continue
        valid_tensors.append(e)
        valid_sha1s.append(sha1)
    if not valid_tensors:
        return None, [], len(tuples_sha1_prompt)
    # Ensure consistent dim
    d0 = valid_tensors[0].numel()
    valid_tensors = [t for t in valid_tensors if t.numel() == d0]
    valid_sha1s = valid_sha1s[: len(valid_tensors)]
    if not valid_tensors:
        return None, [], len(tuples_sha1_prompt)
    return torch.stack(valid_tensors, dim=0), valid_sha1s, len(tuples_sha1_prompt)


def _save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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
        default="$HOME/similarity_re",
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
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device for similarity (cuda if available).",
    )
    ap.add_argument(
        "--save_top1_only",
        action="store_true",
        help="If set, do NOT save full (n_src,n_test) matrix; only save top1 per test + argmax + meta.",
    )
    args = ap.parse_args()

    torch.set_num_threads(args.num_threads)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

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
    test_stems = sorted(stem_to_tuples.keys())
    test_set = set(test_stems)

    print(f"Sources (normalized): {len(norm_to_tuples)}, Test datasets: {len(stem_to_tuples)}")
    print(f"Device: {device}")

    # ---- Preload test embeddings once (and keep sha1 mapping for columns) ----
    stem_to_tensor: Dict[str, torch.Tensor] = {}
    stem_to_sha1s: Dict[str, List[str]] = {}
    stem_to_total: Dict[str, int] = {}

    for test_stem, tuples in tqdm(stem_to_tuples.items(), desc="Load test embeddings"):
        if not tuples:
            continue
        test_embs = batch_load_embeddings_by_prompt(tuples, emb_cache_dir, args.num_workers, desc=test_stem[:24])
        tensor, sha1s, total = _stack_valid(tuples, test_embs)
        stem_to_total[test_stem] = total
        if tensor is None:
            print(f"  [WARN] {test_stem}: 0/{total} embeddings in cache")
            continue

        # normalize for cosine
        tensor = torch.nn.functional.normalize(tensor, p=2, dim=1).to(device)
        stem_to_tensor[test_stem] = tensor
        stem_to_sha1s[test_stem] = sha1s

    if not stem_to_tensor:
        print("No test embeddings loaded; nothing to do.")
        return

    # ---- Compute per source vs all tests ----
    for norm_source, tuples in tqdm(norm_to_tuples.items(), desc="Source vs test"):
        if not tuples:
            continue

        # Skip test-vs-test: if this normalized source name is also a test dataset stem
        if norm_source in test_set:
            print(f"  [SKIP test-test] {norm_source} is a test dataset stem; skip computing against tests.")
            continue

        src_embs = batch_load_embeddings_by_prompt(
            tuples, emb_cache_dir, args.num_workers, desc=f"src_{_safe_fname(norm_source)[:24]}"
        )
        src_tensor, src_sha1s, src_total = _stack_valid(tuples, src_embs)
        if src_tensor is None:
            print(f"  [WARN] {norm_source}: 0/{src_total} embeddings in cache, skip")
            continue

        src_tensor = torch.nn.functional.normalize(src_tensor, p=2, dim=1).to(device)

        for test_stem, test_tensor in stem_to_tensor.items():
            # (n_src, n_test)
            sim = src_tensor @ test_tensor.t()

            # top1 for each test sample (column)
            top1_sim, top1_idx = torch.max(sim, dim=0)  # shapes: (n_test,), (n_test,)

            # filenames
            base = f"{_safe_fname(norm_source)}_{_safe_fname(test_stem)}"
            npy_path = out_dir / f"{base}.npy"
            meta_path = out_dir / f"{base}.meta.json"
            top1_sim_path = out_dir / f"{base}.top1_sim.npy"
            top1_idx_path = out_dir / f"{base}.top1_src_idx.npy"

            # Save top1 always (small + useful)
            np.save(top1_sim_path, top1_sim.detach().cpu().numpy().astype(np.float32))
            np.save(top1_idx_path, top1_idx.detach().cpu().numpy().astype(np.int32))

            # Save full matrix unless user requests top1 only
            if not args.save_top1_only:
                arr = sim.detach().cpu().numpy().astype(np.float32)
                np.save(npy_path, arr)

            # Save meta mapping (row/col sha1s) + hit stats
            meta = {
                "normalized_source": norm_source,
                "test_stem": test_stem,
                "rows": {
                    "n_total": int(src_total),
                    "n_valid": int(len(src_sha1s)),
                    "sha1s": src_sha1s,
                },
                "cols": {
                    "n_total": int(stem_to_total.get(test_stem, len(stem_to_sha1s.get(test_stem, [])))),
                    "n_valid": int(len(stem_to_sha1s.get(test_stem, []))),
                    "sha1s": stem_to_sha1s.get(test_stem, []),
                },
                "files": {
                    "full_matrix_npy": str(npy_path) if not args.save_top1_only else "",
                    "top1_sim_npy": str(top1_sim_path),
                    "top1_src_idx_npy": str(top1_idx_path),
                },
                "cosine_range_hint": "[-1, 1] after L2-normalization",
            }
            _save_json(meta_path, meta)

        print(f"  {norm_source} -> {len(stem_to_tensor)} outputs")

    print(f"Done. Outputs under {out_dir}")


if __name__ == "__main__":
    main()