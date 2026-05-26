#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预计算 JSON 中所有 prompt 的 embedding，写入 emb_cache_dir；已存在 pkl 则跳过。
"""

import argparse
import hashlib
import pickle
from pathlib import Path

import ijson
import torch
from sentence_transformers import SentenceTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_json_file",
        type=str,
        default="$ROOT/ATLAS/results/stage2_train_test-matched_with_source.json",
        help="Input JSON file path",
    )
    ap.add_argument(
        "--test_json_file",
        type=str,
        default=None,
        help="Optional second JSON file path",
    )
    ap.add_argument(
        "--model_path",
        type=str,
        default="$HOME/model/sentence-transformers/all-MiniLM-L6-v2",
        help="BERT model path for embedding",
    )
    ap.add_argument(
        "--emb_cache_dir",
        type=str,
        default="$HOME/emb_cache",
        help="Directory to save embedding cache (pkl per prompt)",
    )
    ap.add_argument(
        "--num_threads",
        type=int,
        default=64,
        help="PyTorch/tokenizer CPU threads",
    )
    args = ap.parse_args()

    json_path = Path(args.train_json_file)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        return

    test_json_path = Path(args.test_json_file) if args.test_json_file else None
    if test_json_path and not test_json_path.exists():
        print(f"Error: Test JSON file not found: {test_json_path}")
        return

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    print(f"Loading BERT model: {args.model_path}")
    model = SentenceTransformer(args.model_path)

    emb_cache_dir = Path(args.emb_cache_dir)
    emb_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_embedding_cache_key(text):
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get_embedding(text, cache_key=None):
        if cache_key is None:
            cache_key = get_embedding_cache_key(text)
        cache_path = emb_cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                cache_path.unlink(missing_ok=True)
        embedding = model.encode(text, convert_to_tensor=True)
        with cache_path.open("wb") as f:
            pickle.dump(embedding, f)
        return embedding

    seen_texts = set()
    paths = [json_path] + ([test_json_path] if test_json_path else [])
    for path in paths:
        with path.open("rb") as f:
            for prompt_sha1, data in ijson.kvitems(f, ""):
                if not isinstance(data, dict) or "prompt" not in data:
                    continue
                text = data["prompt"]
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    get_embedding(text)
    print(f"Done. Cached {len(seen_texts)} unique prompts to {emb_cache_dir}")


if __name__ == "__main__":
    main()
