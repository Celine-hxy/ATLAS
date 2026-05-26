#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
只对 label_list 中的数据进行 match（embedding 已预计算在 emb_cache_dir）：
1. 按数据量从少到多逐 label 处理，每次只 load 当前 label + 对应 match_source 数据
2. 从 emb_cache_dir 的 pkl 加载 embedding，无 encode
3. 不做合并：不把目标样本加入最相似 prompt 的 occ_list，只输出 sim_match 结果
4. 每个 label 单独写一个 JSON：{sim_matched_output_stem}_{label}.json，每样本键 sim_match_top3 为 list（最多 3 条），每项含 similarity、prompt_sha1、prompt 以及 occ_list[0]（或 source==match_source 的 occ）中全部字段
"""

import argparse
import json
import ijson
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import util
import torch
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        help="Input JSON file path",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="$ROOT/ATLAS/results/stage3_sim_match.json",
        help="Output JSON file path",
    )
    ap.add_argument(
        "--test_output",
        type=str,
        default=None,
        help="Output test JSON file path (optional, only used if test_json_file is provided)",
    )
    ap.add_argument(
        "--sim_matched_output",
        type=str,
        default="$ROOT/ATLAS/results/stage3_sim_match_known_source/sim_match_ids.json",
        help="Output path (prefix) for sim_match; one JSON per label will be written as {prefix_dir}/{prefix_stem}_{label}.json",
    )
    ap.add_argument(
        "--label_list",
        type=str,
        nargs="+",
        default=["numina_amc_aime", "numina_aops_forum", "numina_cn_k12", "numina_olympiads",
            "numina_synthetic_amc", "numina_synthetic_math"],
        help="List of source labels to process (only these are matched)",
    )
    ap.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for matching (only merge if similarity >= threshold)",
    )
    ap.add_argument(
        "--emb_cache_dir",
        type=str,
        default="$HOME/emb_cache",
        help="Directory of precomputed embedding pkl files",
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=128,
        help="Parallel workers for loading pkl from cache (default 64)",
    )
    ap.add_argument(
        "--num_threads",
        type=int,
        default=64,
        help="PyTorch CPU threads for cos_sim (default 64)",
    )
    args = ap.parse_args()

    json_path = Path(args.train_json_file)
    label_list = set(args.label_list) if args.label_list else set()
    
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        return
    
    test_json_path = Path(args.test_json_file) if args.test_json_file else None

    if test_json_path and not test_json_path.exists():
        print(f"Error: Test JSON file not found: {test_json_path}")
        return
    
    emb_cache_dir = Path(args.emb_cache_dir)
    if not emb_cache_dir.exists():
        print(f"Error: emb_cache_dir not found: {emb_cache_dir}")
        return
    
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    
    num_workers = max(1, args.num_workers)
    
    def get_embedding_cache_key(text):
        return hashlib.md5(text.encode("utf-8")).hexdigest()
    
    def _load_one(args_item):
        i, text, cache_key = args_item
        cache_path = emb_cache_dir / f"{cache_key}.pkl"
        if not cache_path.exists():
            return (i, None)
        try:
            with cache_path.open("rb") as f:
                return (i, pickle.load(f))
        except (EOFError, pickle.UnpicklingError):
            return (i, None)

    def load_embedding_from_cache(text, cache_key=None):
        """仅从 pkl 加载 embedding，不存在则返回 None"""
        if cache_key is None:
            cache_key = get_embedding_cache_key(text)
        cache_path = emb_cache_dir / f"{cache_key}.pkl"
        if not cache_path.exists():
            return None
        try:
            with cache_path.open("rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            return None

    def batch_load_embeddings(texts, desc=None):
        """批量从缓存加载 embeddings（多线程），与 texts 同序，缺失为 None"""
        if not texts:
            return []
        keys = [get_embedding_cache_key(t) for t in texts]
        result = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_load_one, (i, t, k)): i for i, (t, k) in enumerate(zip(texts, keys))}
            pbar = tqdm(as_completed(futures), total=len(futures), desc=desc or "Load pkl", unit=" pkl")
            for fut in pbar:
                i, emb = fut.result()
                result[i] = emb
        return result
    
    def _first_source(occ_list):
        for occ in occ_list:
            s = occ.get("source")
            if s is not None:
                return s
        return None

    def _stream_count_by_label(paths, label_list):
        """Stream JSON(s), return label -> count (only first_source in label_list)."""
        counts = {}
        for path in paths:
            if not path or not path.exists():
                continue
            with path.open("rb") as f:
                for prompt_sha1, data in ijson.kvitems(f, ""):
                    if not isinstance(data, dict) or "occ_list" not in data:
                        continue
                    occ_list = data.get("occ_list", [])
                    if not occ_list:
                        continue
                    fs = _first_source(occ_list)
                    if fs and fs in label_list:
                        counts[fs] = counts.get(fs, 0) + 1
        return counts

    def _stream_load_label_and_candidates(paths, label, match_source):
        """Stream JSON(s), collect (1) label_data: [(sha1, data)] where first_source==label,
           (2) candidate_tuples: [(sha1, data, prompt_text)] where any occ.source==match_source."""
        label_data = []
        candidate_tuples = []
        for path in paths:
            if not path or not path.exists():
                continue
            with path.open("rb") as f:
                for prompt_sha1, data in ijson.kvitems(f, ""):
                    if not isinstance(data, dict) or "occ_list" not in data:
                        continue
                    occ_list = data.get("occ_list", [])
                    if not occ_list:
                        continue
                    fs = _first_source(occ_list)
                    prompt_text = data.get("prompt", "")
                    if fs == label:
                        label_data.append((prompt_sha1, data))
                    if match_source and any(occ.get("source") == match_source for occ in occ_list):
                        candidate_tuples.append((prompt_sha1, data, prompt_text))
        return label_data, candidate_tuples

    sim_matched_output_dir = Path(args.sim_matched_output).parent
    sim_matched_output_stem = Path(args.sim_matched_output).stem
    sim_matched_output_dir.mkdir(parents=True, exist_ok=True)
    paths = [json_path]
    if test_json_path:
        paths.append(test_json_path)

    # 步骤1: 流式统计各 label 数量，按数量升序处理
    print("Counting prompts per label (streaming)...")
    label_counts = _stream_count_by_label(paths, label_list)
    labels_sorted = sorted(label_counts.keys(), key=lambda x: label_counts[x])
    print(f"  Labels (ascending count): {[(l, label_counts[l]) for l in labels_sorted]}")

    # 断点：已存在输出文件的 label 跳过
    done_labels = set()
    for label in list(labels_sorted):
        out_path = sim_matched_output_dir / f"{sim_matched_output_stem}_{label}.json"
        if out_path.exists():
            done_labels.add(label)
    if done_labels:
        labels_sorted = [l for l in labels_sorted if l not in done_labels]
        print(f"  Resuming: skip {len(done_labels)} already-done labels")

    total_no_emb_count = 0

    for label in labels_sorted:
        out_path = sim_matched_output_dir / f"{sim_matched_output_stem}_{label}.json"
        if out_path.exists():
            print(f"  [SKIP] {label}: output already exists -> {out_path.name}")
            continue
        match_source = label[len("numina_"):] if label.startswith("numina_") else None
        if not match_source:
            print(f"  [SKIP] {label}: not numina_*")
            total_no_emb_count += label_counts.get(label, 0)
            continue

        # 只 load 当前 label + 对应 match_source 数据
        print(f"  Loading data for label={label} (match_source={match_source})...")
        label_data, candidate_tuples = _stream_load_label_and_candidates(paths, label, match_source)
        if not label_data:
            print(f"  [SKIP] {label}: no prompts")
            continue
        if not candidate_tuples:
            print(f"  [SKIP] {label}: no candidates for '{match_source}'")
            total_no_emb_count += len(label_data)
            continue

        # 去重候选（同一 prompt 可能多条 occ），保留 (sha1, data, text) 唯一 text
        seen_text = set()
        unique_candidates = []
        for sha1, data, text in candidate_tuples:
            if text and text not in seen_text:
                seen_text.add(text)
                unique_candidates.append((sha1, data, text))

        cand_texts = [t[2] for t in unique_candidates]
        cand_embs = batch_load_embeddings(cand_texts, desc=f"cand_{label}")
        text_to_emb = {t: e for t, e in zip(cand_texts, cand_embs) if e is not None}
        if not text_to_emb:
            print(f"  [SKIP] {label}: no candidate embeddings")
            total_no_emb_count += len(label_data)
            continue

        valid_tuples = [(sha1, data, text) for (sha1, data, text) in unique_candidates if text in text_to_emb]
        candidate_embeddings = torch.stack([text_to_emb[t[2]] for t in valid_tuples])
        prompt_sha1s_valid = [t[0] for t in valid_tuples]
        sha1_to_prompt = {t[0]: t[2] for t in valid_tuples}

        def _first_occ_for_source(data, src):
            occ_list = data.get("occ_list", [])
            if not occ_list:
                return {}
            for occ in occ_list:
                if occ.get("source") == src:
                    return dict(occ)
            return dict(occ_list[0])

        sha1_to_occ = {t[0]: _first_occ_for_source(t[1], match_source) for t in valid_tuples}

        query_texts = [d.get("prompt", "") for _, d in label_data]
        query_embs = batch_load_embeddings(query_texts, desc=f"query_{label}")
        query_embedding_by_sha1 = {}
        for i, (sha1, data) in enumerate(label_data):
            if query_embs[i] is not None:
                query_embedding_by_sha1[sha1] = query_embs[i]

        no_emb_count = 0
        sim_matched_data = {}

        k_top = min(3, len(prompt_sha1s_valid))
        for prompt_sha1, data in tqdm(label_data, desc=f"Sim match {label}", unit=" samples"):
            query_embedding = query_embedding_by_sha1.get(prompt_sha1)
            if query_embedding is None:
                no_emb_count += 1
                continue
            similarities = util.cos_sim(query_embedding, candidate_embeddings)[0]
            top_sims, top_idxs = torch.topk(similarities, k=k_top, largest=True)
            sim_match_top3 = []
            for j, i in enumerate(top_idxs.tolist()):
                cand_sha1 = prompt_sha1s_valid[i]
                occ = sha1_to_occ.get(cand_sha1, {})
                item = dict(occ)
                item["similarity"] = round(top_sims[j].item(), 6)
                item["prompt_sha1"] = cand_sha1
                item["prompt"] = sha1_to_prompt.get(cand_sha1, "")
                sim_match_top3.append(item)
            d = data.copy()
            d["sim_match_top3"] = sim_match_top3
            sim_matched_data[prompt_sha1] = d

        total_no_emb_count += no_emb_count

        out_path = sim_matched_output_dir / f"{sim_matched_output_stem}_{label}.json"
        if sim_matched_data:
            with out_path.open("w", encoding="utf-8") as f:
                f.write("{\n")
                first_item = True
                for pk, d in sorted(sim_matched_data.items()):
                    if not first_item:
                        f.write(",\n")
                    f.write(f'  "{pk}": ')
                    json.dump(d, f, ensure_ascii=False, indent=2)
                    first_item = False
                f.write("\n}")
            print(f"  Saved {len(sim_matched_data)} samples (top-3 match) for {label} -> {out_path}")
        else:
            with out_path.open("w", encoding="utf-8") as f:
                f.write("{}")
            print(f"  No query embeddings for {label} -> {out_path} (empty)")

    print(f"\n[DONE]")
    print(f"  Samples without query embedding (skipped): {total_no_emb_count}")
    print(f"  Sim-matched JSONs (top-3 per sample): {sim_matched_output_dir}/{sim_matched_output_stem}_<label>.json")


if __name__ == "__main__":
    main()
