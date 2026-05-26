#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化 s3_sim_match_no_match 输出 JSON 的 similarity 分布。"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default="$ROOT/ATLAS/results/stage3_sim_match_no_match/sim_match_ids.json",
        help="s3_sim_match_no_match 输出的 JSON",
    )
    ap.add_argument("--output", type=str,
                    default="$ROOT/ATLAS/results/stage3_sim_match_no_match_dist.pdf")
    ap.add_argument("--bins", type=int, default=50, help="直方图 bin 数")
    args = ap.parse_args()

    p = Path(args.input)
    if not p.exists():
        raise FileNotFoundError(p)

    with p.open(encoding="utf-8") as f:
        data = json.load(f)

    HF_ID_TO_SOURCE = {
        "RUC-AIBOX/STILL-3-RL-90K": "still3",
        "inclusionAI/AReaL-boba-Data": "areal_boba",
    }

    def _effective_source(item):
        hf_id = item.get("hf_id")
        if hf_id in HF_ID_TO_SOURCE:
            return HF_ID_TO_SOURCE[hf_id]
        return item.get("source")

    def _effective_source_still3_only(item):
        if item.get("hf_id") == "RUC-AIBOX/STILL-3-RL-90K":
            return "still3"
        return item.get("source")

    def _effective_source_areal_only(item):
        if item.get("hf_id") == "inclusionAI/AReaL-boba-Data":
            return "areal_boba"
        return item.get("source")

    # 1) occ_list / sim_match_top3 的 source 按 HF_ID_TO_SOURCE 赋值
    # 2) 计算并写入 prompt 级 source，统计 null
    out_path = Path("$ROOT/ATLAS/results/stage3_sim_match_no_match/sim_match_ids_processed.json")
    n_source_null_all_same = 0
    n_source_null_else = 0
    for v in data.values():
        occ_list = v.get("occ_list") or []
        for occ in occ_list:
            occ["source"] = _effective_source(occ)
        top3 = v.get("sim_match_top3") or []
        for it in top3:
            it["source"] = _effective_source(it)
        occ0 = occ_list[0] if occ_list else {}
        occ0_hf_id = occ0.get("hf_id")
        occ0_date = occ0.get("date")
        # sim_match_top3 按 date 从早到晚排序并写回
        top3_by_date = sorted(top3, key=lambda it: (it.get("date") is None, it.get("date") or ""))
        v["sim_match_top3"] = top3_by_date
        if not top3:
            v["source"] = occ0.get("source")
            v["sim_match_info"] = None
            if v["source"] is None:
                n_source_null_else += 1
            continue
        all_same = all(it.get("hf_id") == occ0_hf_id for it in top3)
        if all_same:
            v["source"] = occ0.get("source")
            v["sim_match_info"] = None
            if v["source"] is None:
                n_source_null_all_same += 1
        else:
            found = None
            for it in top3_by_date:
                if (it.get("similarity") is not None and it["similarity"] >= 0.9
                    and it.get("date") is not None and occ0_date is not None
                    and it["date"] < occ0_date):
                    found = it
                    break
            if found is not None:
                v["source"] = found.get("source")
                v["sim_match_info"] = found
            else:
                v["source"] = occ0.get("source")
                v["sim_match_info"] = None
                if v["source"] is None:
                    n_source_null_else += 1
    print(f"Prompt-level source: all_hf_id_same -> source=null: {n_source_null_all_same}; else branch -> source=null: {n_source_null_else}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {out_path}")

    # 将 sim match 的 source 写回 stage2，存为 stage3（no_match + known_source）
    stage2_path = Path("$ROOT/ATLAS/results/stage2_train_test-matched_with_source.json")
    stage3_path = Path("$ROOT/ATLAS/results/stage3_train_test-sim_matched_with_source.json")
    known_source_dir = Path("$ROOT/ATLAS/results/stage3_sim_match_known_source")
    if stage2_path.exists():
        with stage2_path.open(encoding="utf-8") as f:
            stage2 = json.load(f)
        for sha1, proc in data.items():
            if sha1 in stage2:
                stage2[sha1]["source"] = proc["source"]
                stage2[sha1]["is_sim_match"] = True
                stage2[sha1]["sim_match_info"] = proc.get("sim_match_info")
        n_known = 0
        for jpath in sorted(known_source_dir.glob("sim_match_ids_*.json")):
            label = jpath.stem.replace("sim_match_ids_", "", 1)
            with jpath.open(encoding="utf-8") as f:
                known = json.load(f)
            for sha1, v in known.items():
                if sha1 in stage2:
                    top3 = v.get("sim_match_top3") or []
                    stage2[sha1]["source"] = label
                    stage2[sha1]["is_sim_match"] = True
                    stage2[sha1]["sim_match_info"] = top3[0] if top3 else None
                    n_known += 1
        stage3_path.parent.mkdir(parents=True, exist_ok=True)
        with stage3_path.open("w", encoding="utf-8") as f:
            json.dump(stage2, f, ensure_ascii=False, indent=2)
        print(f"Saved stage3: {stage3_path} (no_match: {len(data)} prompts, known_source: {n_known} prompts)")
    else:
        print(f"Stage2 not found, skip: {stage2_path}")

    THRESHOLDS = [0.70, 0.80, 0.90]
    # 每个阈值: (有至少一条 sim>=T 的样本数, 其中「所有 sim>=T 的项 source 全为 null」的样本数), 以及这些样本的 occ_list[0].hf_id 列表
    above_threshold = {t: [0, 0, []] for t in THRESHOLDS}  # [n_any_above, n_all_null_above, first_hf_ids]

    sims = []
    n_all_top3_source_null = 0
    n_all_top3_source_null_after_still3 = 0
    n_all_top3_source_null_after_areal = 0
    for v in data.values():
        top3 = v.get("sim_match_top3", [])
        occ0 = (v.get("occ_list") or [{}])[0] if v.get("occ_list") else {}
        first_hf_id = occ0.get("hf_id") or "(null)"
        if len(top3) == 3:
            if all(item.get("source") is None for item in top3):
                n_all_top3_source_null += 1
            if all(_effective_source_still3_only(item) is None for item in top3):
                n_all_top3_source_null_after_still3 += 1
            if all(_effective_source_areal_only(item) is None for item in top3):
                n_all_top3_source_null_after_areal += 1
        for t in THRESHOLDS:
            items_above = [it for it in top3 if it.get("similarity") is not None and it["similarity"] >= t]
            if not items_above:
                continue
            above_threshold[t][0] += 1
            if all(_effective_source(it) is None for it in items_above):
                above_threshold[t][1] += 1
                above_threshold[t][2].append(first_hf_id)
        for item in top3:
            s = item.get("similarity")
            if s is not None:
                sims.append(s)

    sims = np.array(sims)
    if sims.size == 0:
        print("No similarity values found.")
        return

    print(f"N = {len(sims)}, min = {sims.min():.4f}, max = {sims.max():.4f}, mean = {sims.mean():.4f}, std = {sims.std():.4f}")
    print(f"Samples with sim_match_top3 all 3 source=null: {n_all_top3_source_null} / {len(data)} ({100*n_all_top3_source_null/len(data):.1f}%)")
    print(f"  (if hf_id=RUC-AIBOX/STILL-3-RL-90K -> source=still3) still all 3 null: {n_all_top3_source_null_after_still3} / {len(data)} ({100*n_all_top3_source_null_after_still3/len(data):.1f}%)")
    print(f"  (if hf_id=inclusionAI/AReaL-boba-Data -> source=areal_boba) still all 3 null: {n_all_top3_source_null_after_areal} / {len(data)} ({100*n_all_top3_source_null_after_areal/len(data):.1f}%)")
    print("By threshold (sim>=T 的项中 all effective_source=null, STILL-3->still3, AReaL-boba-Data->areal_boba):")
    for t in THRESHOLDS:
        n_any, n_all_null, first_hf_ids = above_threshold[t]
        pct = 100 * n_all_null / n_any if n_any else 0
        print(f"  sim>={t}: all_null={n_all_null} / n_above={n_any} ({pct:.1f}%)")
        hf_id_counts = Counter(first_hf_ids)
        for hf_id, cnt in hf_id_counts.most_common(15):
            print(f"      {cnt:5d}  {hf_id}")
    for q in [0.25, 0.5, 0.75]:
        print(f"  p{int(q*100)} = {np.quantile(sims, q):.4f}")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(sims, bins=args.bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Similarity distribution (sim_match_no_match)")
    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
