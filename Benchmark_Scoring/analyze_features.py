"""
深入分析各数据集的learnability分布差异，找出与GT排名最相关的特征组合。
"""
import ijson, collections, math, itertools

DATASET_HF_IDS = {
    "DeepScaleR":        ["agentica-org/DeepScaleR-Preview-Dataset"],
    "DeepMath-103K":     ["zwhe99/DeepMath-103K"],
    "OpenR1-Math-220k":  ["open-r1/OpenR1-Math-220k", "open-r1/Big-Math-RL-Verified-Processed"],
    "DAPO-Math-17k":     ["BytedTsinghua-SIA/DAPO-Math-17k"],
    "Skywork-OR1":       ["Skywork/Skywork-OR1-RL-Data"],
}

# Ground truth from paper
GT_1B = {
    "DeepScaleR": 14.7, "DeepMath-103K": 15.4, "OpenR1-Math-220k": 14.0,
    "DAPO-Math-17k": 15.0, "Skywork-OR1": 15.1,
}
GT_8B = {
    "DeepScaleR": 26.1, "DeepMath-103K": 25.1, "OpenR1-Math-220k": 25.0,
    "DAPO-Math-17k": 29.6, "Skywork-OR1": 25.1,
}

stats = {ds: {
    "total": 0,
    "learn_dist": collections.Counter(),
    "mcq": 0,
    "occ_list": [],
    "cont_list": [],
    "domains": collections.Counter(),
    "sources": collections.Counter(),
    "consistent": 0,
    "inconsistent": 0,
} for ds in DATASET_HF_IDS}

count = 0
with open("/Users/celine/Downloads/stage4_final.json", "rb") as f:
    parser = ijson.kvitems(f, "")
    for key, item in parser:
        count += 1
        occ_list = item.get("occ_list", [])
        entry_datasets = set()
        for occ in occ_list:
            hf_id = occ.get("hf_id", "")
            for ds, hf_ids in DATASET_HF_IDS.items():
                if any(h == hf_id for h in hf_ids):
                    entry_datasets.add(ds)

        learnability = item.get("learnability_score")
        contamination = float(item.get("contamination_penalty") or 0)
        is_mcq = item.get("is_mcq", False)
        domain = item.get("domain", "Other")
        source = item.get("source") or "unknown"
        occ_count = len(occ_list)
        answers = list(set(occ.get("answer", "") for occ in occ_list if occ.get("answer")))
        consistent = len(answers) <= 1

        for ds in entry_datasets:
            s = stats[ds]
            s["total"] += 1
            s["learn_dist"][learnability] += 1
            if is_mcq: s["mcq"] += 1
            s["occ_list"].append(occ_count)
            s["cont_list"].append(contamination)
            s["domains"][domain] += 1
            s["sources"][source] += 1
            if consistent: s["consistent"] += 1
            else: s["inconsistent"] += 1

        if count % 100000 == 0:
            print(f"  {count}...", flush=True)
        if count >= 800000:
            break

print(f"\nProcessed {count} entries\n")

# ── Print detailed feature breakdown per dataset ──────────────────────────────
def to_float(v):
    if v is None: return None
    try: return float(v)
    except: return None

print(f"{'Dataset':<22} {'Total':>8} {'p00':>7} {'p10':>7} {'p11':>7} {'p01':>7} {'pNone':>7} {'MCQ%':>7} {'AvgOcc':>8} {'Cont%':>7} {'Olym%':>7}")
print("-" * 100)

OLYMPIAD_SOURCES = {"olympiads", "aops_forum", "cn_contest"}
features = {}
for ds, s in stats.items():
    n = s["total"]
    if n == 0: print(f"{ds:<22} NO DATA"); continue
    d = s["learn_dist"]
    p00 = d.get(0.0, d.get("0.0", 0)) / n
    p10 = d.get(0.2, d.get("0.2", 0)) / n
    p11 = d.get(0.5, d.get("0.5", 0)) / n
    p01 = d.get(1.0, d.get("1.0", 0)) / n
    pNone = d.get(None, 0) / n
    mcq_pct = s["mcq"] / n
    avg_occ = sum(s["occ_list"]) / n if s["occ_list"] else 1
    cont_pct = sum(1 for v in s["cont_list"] if v > 0) / n
    olym_count = sum(v for k, v in s["sources"].items() if k and any(o in k.lower() for o in OLYMPIAD_SOURCES))
    olym_pct = olym_count / n

    # Fix decimal keys from ijson
    actual_p00 = actual_p10 = actual_p11 = actual_p01 = 0
    for k, v in d.items():
        fk = to_float(k)
        if fk is None: continue
        if abs(fk - 0.0) < 0.01: actual_p00 += v
        elif abs(fk - 0.2) < 0.01: actual_p10 += v
        elif abs(fk - 0.5) < 0.01: actual_p11 += v
        elif abs(fk - 1.0) < 0.01: actual_p01 += v
    actual_p00 /= n; actual_p10 /= n; actual_p11 /= n; actual_p01 /= n

    features[ds] = {
        "p00": actual_p00, "p10": actual_p10, "p11": actual_p11, "p01": actual_p01,
        "pNone": pNone, "mcq": mcq_pct, "avg_occ": avg_occ,
        "cont": cont_pct, "olym": olym_pct, "n": n,
        "domain_diversity": len(s["domains"]),
        "source_diversity": len(s["sources"]),
    }

    print(f"{ds:<22} {n:>8} {actual_p00:>7.3f} {actual_p10:>7.3f} {actual_p11:>7.3f} {actual_p01:>7.3f} {pNone:>7.3f} {mcq_pct:>7.3f} {avg_occ:>8.2f} {cont_pct:>7.3f} {olym_pct:>7.3f}")

# ── Feature correlation analysis ───────────────────────────────────────────────
print("\n\n=== Feature correlation with GT ranking ===")

def pearson(x, y):
    n = len(x)
    if n < 2: return 0
    mx, my = sum(x)/n, sum(y)/n
    num = sum((xi-mx)*(yi-my) for xi,yi in zip(x,y))
    den = math.sqrt(sum((xi-mx)**2 for xi in x) * sum((yi-my)**2 for yi in y))
    return num/(den+1e-12)

feat_names = ["p00", "p10", "p11", "p01", "pNone", "mcq", "avg_occ", "cont", "olym", "n", "domain_diversity"]
ds_list = list(features.keys())

for gt_name, gt_scores in [("1.7B", GT_1B), ("8B", GT_8B)]:
    print(f"\n--- {gt_name} ---")
    gt_vals = [gt_scores.get(ds, 0) for ds in ds_list]
    for fn in feat_names:
        feat_vals = [features[ds][fn] for ds in ds_list]
        r = pearson(feat_vals, gt_vals)
        print(f"  {fn:<20} r={r:+.4f}")

# ── Find best linear combination ───────────────────────────────────────────────
print("\n\n=== Best feature combinations (brute force) ===")
for gt_name, gt_scores in [("1.7B", GT_1B), ("8B", GT_8B)]:
    gt_vals = [gt_scores.get(ds, 0) for ds in ds_list]
    best_r = -999
    best_combo = None
    for combo in itertools.combinations(feat_names, 3):
        for signs in itertools.product([-1, 1], repeat=3):
            feat_vals = [sum(s * features[ds][f] for s, f in zip(signs, combo)) for ds in ds_list]
            r = pearson(feat_vals, gt_vals)
            if r > best_r:
                best_r = r
                best_combo = (combo, signs)
    print(f"\n{gt_name}: best Pearson r={best_r:.4f}")
    print(f"  combo: {[(s,f) for s,f in zip(best_combo[1], best_combo[0])]}")
