#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 no source（所有 occ 的 source 均为 null）的样本与除 label_list 以外的所有数据进行相似度匹配（embedding 从 emb_cache_dir 的 pkl 加载）：
1. Query：仅 no source 样本
2. 候选池：除 label_list 以外的所有数据
3. 不做合并：只输出 sim_match 结果存为 JSON；格式同 s3_sim_match_known_source，每样本键 sim_match_top3（最多 3 条），
   每项含 similarity、prompt_sha1、prompt 以及 occ_list[0] 中全部字段
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


def _first_source(occ_list):
    for occ in occ_list:
        s = occ.get("source")
        if s is not None:
            return s
    return None


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
        help="Input test JSON file path",
    )
    ap.add_argument(
        "--sim_matched_output",
        type=str,
        default="$ROOT/ATLAS/results/stage3_sim_match_no_match/sim_match_ids.json",
        help="Output JSON path (same format as s3_sim_match_known_source)",
    )
    ap.add_argument(
        "--label_list",
        type=str,
        nargs="+",
        default=["numina_amc_aime", "numina_aops_forum", "numina_cn_k12", "numina_olympiads",
            "numina_synthetic_amc", "numina_synthetic_math"],
        help="Candidate pool = data with first_source NOT in this list; query = no source only",
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
        help="Parallel workers for loading pkl from cache",
    )
    ap.add_argument(
        "--num_threads",
        type=int,
        default=64,
        help="PyTorch CPU threads for cos_sim",
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

    def batch_load_embeddings(texts, desc=None):
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

    # 流式加载：no_source_data = query（所有 occ 的 source 均为 null）；non_label_data = 候选池（first_source 不在 label_list）
    paths = [json_path]
    if test_json_path:
        paths.append(test_json_path)

    no_source_data = []   # query: [(prompt_sha1, data), ...]
    non_label_data = []   # 候选池: [(prompt_sha1, data), ...]
    for path in paths:
        if not path.exists():
            continue
        with path.open("rb") as f:
            for prompt_sha1, data in ijson.kvitems(f, ""):
                if not isinstance(data, dict) or "occ_list" not in data:
                    continue
                occ_list = data.get("occ_list", [])
                if not occ_list:
                    continue
                all_null = all(occ.get("source") is None for occ in occ_list)
                fs = _first_source(occ_list)
                if fs is not None and fs in label_list:
                    continue
                non_label_data.append((prompt_sha1, data))
                if all_null:
                    no_source_data.append((prompt_sha1, data))

    print(f"No-source (query) prompts: {len(no_source_data)}, candidate pool (non-label_list): {len(non_label_data)}")
    if not no_source_data:
        print("No query data to process.")
        return
    if not non_label_data:
        print("No candidate pool.")
        return

    # 候选池：去重 prompt 文本，保留 (sha1, data, text)
    seen_text = set()
    unique_candidates = []
    for sha1, data in non_label_data:
        text = data.get("prompt", "")
        if text and text not in seen_text:
            seen_text.add(text)
            unique_candidates.append((sha1, data, text))

    cand_texts = [t[2] for t in unique_candidates]
    cand_embs = batch_load_embeddings(cand_texts, desc="cand_no_match")
    text_to_emb = {t: e for t, e in zip(cand_texts, cand_embs) if e is not None}
    if not text_to_emb:
        print("No candidate embeddings in cache.")
        return

    valid_tuples = [(sha1, data, text) for (sha1, data, text) in unique_candidates if text in text_to_emb]
    candidate_embeddings = torch.stack([text_to_emb[t[2]] for t in valid_tuples])
    prompt_sha1s_valid = [t[0] for t in valid_tuples]
    sha1_to_prompt = {t[0]: t[2] for t in valid_tuples}
    # occ 用 occ_list[0]（no_match 无 match_source）
    sha1_to_occ = {}
    for sha1, data, _ in valid_tuples:
        occ_list = data.get("occ_list", [])
        sha1_to_occ[sha1] = dict(occ_list[0]) if occ_list else {}

    # 查询 embedding（仅 no source）
    query_texts = [d.get("prompt", "") for _, d in no_source_data]
    query_embs = batch_load_embeddings(query_texts, desc="query_no_match")
    query_embedding_by_sha1 = {}
    for i, (sha1, data) in enumerate(no_source_data):
        if query_embs[i] is not None:
            query_embedding_by_sha1[sha1] = query_embs[i]

    # 建立 sha1 -> 在 candidate 中的索引（用于排除自身）
    sha1_to_cand_idx = {s: i for i, s in enumerate(prompt_sha1s_valid)}

    sim_matched_data = {}
    k_top = min(3, len(prompt_sha1s_valid))
    no_emb_count = 0

    for prompt_sha1, data in tqdm(no_source_data, desc="Sim match no_match", unit=" samples"):
        query_embedding = query_embedding_by_sha1.get(prompt_sha1)
        if query_embedding is None:
            no_emb_count += 1
            continue
        similarities = util.cos_sim(query_embedding, candidate_embeddings)[0]
        # 若自身在候选里，把自身相似度设为 -1 避免选到自己
        self_idx = sha1_to_cand_idx.get(prompt_sha1)
        if self_idx is not None:
            similarities[self_idx] = -1.0
        top_sims, top_idxs = torch.topk(similarities, k=k_top, largest=True)
        sim_match_top3 = []
        for j, i in enumerate(top_idxs.tolist()):
            if similarities[i].item() < 0:
                continue
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

    out_path = Path(args.sim_matched_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        print(f"Saved {len(sim_matched_data)} samples -> {out_path}")
    else:
        with out_path.open("w", encoding="utf-8") as f:
            f.write("{}")
        print(f"No query embeddings -> {out_path} (empty)")

    print(f"Samples without query embedding (skipped): {no_emb_count}")


if __name__ == "__main__":
    main()
