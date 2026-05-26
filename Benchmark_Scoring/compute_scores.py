"""
按照最新 3_benchmark_of_dataset.tex 方法计算分数：
- L_SCA = α01·p01 + α11·p11 + α10·p10 + α00·p00  (scale-adaptive α)
- S1a = 0.5·R_con + 0.3·(1-0.5·R_mcq) + 0.2·P_reuse
- S1b = σ(L_SCA) + ε_div
- S1c = 1 - N_leak/N
- S1 = w1a·S1a + w1b·S1b + w1c·S1c
- S2 = λ·ΔMean@4_norm + (1-λ)·L_SCA  (L_SCA raw, clamped)
- S3 = λ·ΔPass@4_norm + (1-λ)·L_SCA  (L_SCA raw, clamped)
- Q  = w1·S1 + w2·S2 + w3·S3
计算 Pearson/Spearman 相关性，输出 LaTeX 表格
"""
import math, json

# ── Ground truth ──────────────────────────────────────────────────────────────
GT = {
    "Qwen3-1.7B": {
        "DeepScaleR": 14.7, "DeepMath-103K": 15.4, "OpenR1-Math-220k": 14.0,
        "DAPO-Math-17k": 15.0, "Skywork-OR1": 15.1,
    },
    "Qwen3-8B": {
        "DeepScaleR": 26.1, "DeepMath-103K": 25.1, "OpenR1-Math-220k": 25.0,
        "DAPO-Math-17k": 29.6, "Skywork-OR1": 25.1,
    },
}

# ── Features from stage4_final.json (pre-computed) ──────────────────────────
FEATURES = {
    "DeepScaleR":       {"n":21488,"p00":0.0488,"p10":0.6384,"p11":0.1957,"p01":0.1392,
                         "r_leak":0.1418,"r_mcq":0.0250,"avg_occ":6.37,"r_con":0.8526,
                         "h_domain":0.6984,"r_olym":0.2949},
    "DeepMath-103K":    {"n":55979,"p00":0.0429,"p10":0.7067,"p11":0.1418,"p01":0.1114,
                         "r_leak":0.0032,"r_mcq":0.0239,"avg_occ":1.36,"r_con":0.9911,
                         "h_domain":0.6472,"r_olym":0.0061},
    "OpenR1-Math-220k": {"n":207984,"p00":0.0543,"p10":0.5206,"p11":0.2953,"p01":0.1476,
                         "r_leak":0.0180,"r_mcq":0.1538,"avg_occ":3.48,"r_con":0.9085,
                         "h_domain":0.5943,"r_olym":0.3340},
    "DAPO-Math-17k":    {"n":9488, "p00":0.0454,"p10":0.7205,"p11":0.0744,"p01":0.1325,
                         "r_leak":0.0466,"r_mcq":0.0053,"avg_occ":3.46,"r_con":0.9595,
                         "h_domain":0.6883,"r_olym":0.3906},
    "Skywork-OR1":      {"n":57714,"p00":0.0486,"p10":0.7045,"p11":0.1378,"p01":0.1229,
                         "r_leak":0.0602,"r_mcq":0.0836,"avg_occ":4.76,"r_con":0.8680,
                         "h_domain":0.6675,"r_olym":0.6942},
}

MATH500_MEAN4 = {
    "base_1b":48.4,"base_8b":61.6,
    "DeepScaleR_1b":58.1,"DeepMath-103K_1b":57.9,"OpenR1-Math-220k_1b":58.9,
    "DAPO-Math-17k_1b":58.5,"Skywork-OR1_1b":58.8,
    "DeepScaleR_8b":75.9,"DeepMath-103K_8b":73.4,"OpenR1-Math-220k_8b":73.1,
    "DAPO-Math-17k_8b":77.7,"Skywork-OR1_8b":73.2,
}
MATH500_PASS4 = {
    "base_1b":66.6,"base_8b":76.4,
    "DeepScaleR_1b":69.4,"DeepMath-103K_1b":68.8,"OpenR1-Math-220k_1b":70.0,
    "DAPO-Math-17k_1b":71.0,"Skywork-OR1_1b":70.8,
    "DeepScaleR_8b":83.2,"DeepMath-103K_8b":80.6,"OpenR1-Math-220k_8b":79.6,
    "DAPO-Math-17k_8b":84.6,"Skywork-OR1_8b":79.2,
}

ds_list = list(FEATURES.keys())

# ── Helpers ───────────────────────────────────────────────────────────────────
def pearson(x, y):
    n = len(x); mx, my = sum(x)/n, sum(y)/n
    num = sum((xi-mx)*(yi-my) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi-mx)**2 for xi in x)*sum((yi-my)**2 for yi in y))
    return num/(den+1e-12)

def spearman(x, y):
    def rank(arr):
        si = sorted(range(len(arr)), key=lambda i: arr[i])
        r = [0]*len(arr)
        for rv, idx in enumerate(si, 1): r[idx] = rv
        return r
    return pearson(rank(x), rank(y))

def alpha(M):
    """Scale-adaptive α coefficients for L_SCA"""
    if M <= 3:
        return {"a10": +1.5, "a01": -0.3, "a11": -0.5, "a00": -1.5}
    else:
        return {"a10": +0.5, "a01": +1.5, "a11": -0.8, "a00": -0.8}

def compute_all(M):
    """Compute S1a, S1b, S1c, S1, S2, S3, Q for all datasets at model scale M"""
    suffix = "1b" if M <= 3 else "8b"
    lam = max(0, min(1, math.log10(M)))       # λ(M)
    a = alpha(M)

    # weights
    if M <= 3:
        w1, w2, w3 = 0.60, 0.25, 0.15
        w1a, w1b, w1c = 0.20, 0.55, 0.25
    else:
        w1, w2, w3 = 0.35, 0.35, 0.30
        w1a, w1b, w1c = 0.20, 0.50, 0.30

    results = {}
    for ds in ds_list:
        f = FEATURES[ds]

        # L_SCA (raw, not clamped yet)
        L = a["a10"]*f["p10"] + a["a01"]*f["p01"] + a["a11"]*f["p11"] + a["a00"]*f["p00"]

        # S1a: verifiability
        P_reuse = max(0.3, min(1.0, 4/max(f["avg_occ"], 0.01)))
        s1a = 0.5*f["r_con"] + 0.3*(1 - 0.5*f["r_mcq"]) + 0.2*P_reuse

        # S1b: learnability  σ(L_SCA) + ε_div
        sig_L = 1/(1 + math.exp(-L*3))
        eps_div = 0.05*(-sum(v*math.log(v+1e-9) for v in
                             [f["p00"],f["p10"],f["p11"],f["p01"]])/math.log(4))
        s1b = sig_L + eps_div

        # S1c: contamination  1 - N_leak/N
        s1c = 1.0 - f["r_leak"]

        # S1
        s1 = w1a*s1a + w1b*s1b + w1c*s1c

        # S2  λ·ΔMean@4_norm + (1-λ)·L_SCA (clamped)
        dm4 = (MATH500_MEAN4[ds+"_"+suffix] - MATH500_MEAN4["base_"+suffix]) / 20.0
        s2 = lam*dm4 + (1-lam)*max(0, min(1, L))

        # S3  λ·ΔPass@4_norm + (1-λ)·L_SCA (clamped)
        dp4 = (MATH500_PASS4[ds+"_"+suffix] - MATH500_PASS4["base_"+suffix]) / 15.0
        s3 = lam*dp4 + (1-lam)*max(0, min(1, L))

        # Q
        Q = w1*s1 + w2*s2 + w3*s3

        results[ds] = {
            "L_SCA": L, "S1a": s1a, "S1b": s1b, "S1c": s1c,
            "S1": s1, "S2": s2, "S3": s3, "Q": Q,
            "dm4": dm4*20, "dp4": dp4*15,  # raw pp values
        }
    return results

# ── Compute for both model sizes ──────────────────────────────────────────────
scores_1b = compute_all(1.7)
scores_8b = compute_all(8.0)

# ── Correlations ──────────────────────────────────────────────────────────────
def correlations(scores, gt):
    metrics = ["S1", "S1a", "S1b", "S1c", "S2", "S3", "Q"]
    gt_vals = [gt[ds] for ds in ds_list]
    print(f"\n{'Metric':<8} {'Pearson':>9} {'Spearman':>10}")
    print("-"*30)
    for m in metrics:
        vals = [scores[ds][m] for ds in ds_list]
        pr = pearson(vals, gt_vals)
        sp = spearman(vals, gt_vals)
        print(f"{m:<8} {pr:>+9.4f} {sp:>+10.4f}")

print("="*50)
print("Qwen3-1.7B  (M=1.7, λ=0.23)")
print("="*50)
correlations(scores_1b, GT["Qwen3-1.7B"])
print("\nRanking by Q:", [ds for ds,_ in sorted(scores_1b.items(), key=lambda x:-x[1]["Q"])])
print("GT ranking:  ", [ds for ds,_ in sorted(GT["Qwen3-1.7B"].items(), key=lambda x:-x[1])])

print("\n" + "="*50)
print("Qwen3-8B  (M=8, λ=0.90)")
print("="*50)
correlations(scores_8b, GT["Qwen3-8B"])
print("\nRanking by Q:", [ds for ds,_ in sorted(scores_8b.items(), key=lambda x:-x[1]["Q"])])
print("GT ranking:  ", [ds for ds,_ in sorted(GT["Qwen3-8B"].items(), key=lambda x:-x[1])])

# ── Print full score table ────────────────────────────────────────────────────
print("\n\n" + "="*90)
print("Full score table")
print("="*90)
for M_name, scores in [("1.7B", scores_1b), ("8B", scores_8b)]:
    print(f"\n[{M_name}]")
    print(f"{'Dataset':<22} {'L_SCA':>7} {'S1a':>6} {'S1b':>6} {'S1c':>6} {'S1':>6} {'S2':>6} {'S3':>6} {'Q':>6}  GT")
    print("-"*85)
    gt = GT["Qwen3-1.7B"] if M_name=="1.7B" else GT["Qwen3-8B"]
    for ds in ds_list:
        sc = scores[ds]
        print(f"{ds:<22} {sc['L_SCA']:>+7.3f} {sc['S1a']:>6.3f} {sc['S1b']:>6.3f} {sc['S1c']:>6.3f} "
              f"{sc['S1']:>6.3f} {sc['S2']:>6.3f} {sc['S3']:>6.3f} {sc['Q']:>6.3f}  {gt[ds]:.1f}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
out = {"1.7B": {ds: {k: round(v,4) for k,v in sc.items()} for ds,sc in scores_1b.items()},
       "8B":   {ds: {k: round(v,4) for k,v in sc.items()} for ds,sc in scores_8b.items()}}
with open("/Users/celine/WorkBuddy/2026-05-08-task-1/scores_final.json","w") as f:
    json.dump(out, f, indent=2)
print("\nSaved scores_final.json")
