#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from math import comb


MCQ_LETTERS = ("A", "B", "C", "D", "E")

# Defaults for this repo layout
DEFAULT_TEST_JSONL = Path("$ROOT/verl/recipe/eval/data/gpqa/test.jsonl")
DEFAULT_GPQA_CSV = Path("$ROOT/verl/recipe/eval/data/gpqa/gpqa_diamond.csv")
DEFAULT_OUTPUTS_ROOT = Path("$ROOT/verl/recipe/eval/results_gpqa/outputs")


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl_atomic(path: Path, rows: List[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
        f.flush()
    tmp.replace(path)


def last_boxed_only_string(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed")
    if "\\boxed " in text:
        # legacy form: \boxed A
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else text[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    if s.startswith("\\boxed "):
        return s[len("\\boxed ") :].strip()
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1].strip()
    return s.strip()


def extract_mcq_letter(response: str) -> Optional[str]:
    """
    Robust MCQ extraction for GPQA-style prompts.
    Priority:
      1) last \\boxed{...} or \\boxed ... (single-letter)
      2) explicit "answer is X" / "choice is X"
      3) last standalone A-E token near end
    """
    boxed = last_boxed_only_string(response)
    if boxed:
        content = remove_boxed(boxed)
        content = re.sub(r"\s+", "", content)
        if content in MCQ_LETTERS:
            return content
        # sometimes boxed like "(A)" or "A."
        m = re.fullmatch(r"[\(\[]?([A-E])[\)\]]?\.?", content)
        if m:
            return m.group(1)

    m = re.search(r"(answer|choice)\s+is\s*\(?\s*([A-E])\s*\)?", response, flags=re.IGNORECASE)
    if m:
        return m.group(2).upper()

    tail = response[-800:] if len(response) > 800 else response
    candidates = re.findall(r"\b([A-E])\b", tail.upper())
    return candidates[-1] if candidates else None


def compute_answers_correctness_mcq(generated_responses: List[str], gold: str) -> List[bool]:
    gold = str(gold).strip().upper()
    out: List[bool] = []
    for r in generated_responses:
        pred = extract_mcq_letter(r)
        out.append(pred == gold)
    return out


def mean_at_n(answers_correctness: List[List[bool]]) -> float:
    total = 0
    correct = 0
    for lst in answers_correctness:
        total += len(lst)
        correct += sum(bool(x) for x in lst)
    return 0.0 if total == 0 else 100.0 * correct / total


def pass_at_k_for_question(is_correct_list: List[bool], k: int) -> float:
    n = len(is_correct_list)
    c = sum(is_correct_list)
    if n == 0:
        return 0.0
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (comb(n - c, k) / comb(n, k))


def pass_at_n_percent(per_question_correctness: List[List[bool]]) -> float:
    """
    Pass@n where n is the number of samples available for each question:
      indicator = any(correct samples)
      metric = mean(indicator) * 100
    """
    if not per_question_correctness:
        return 0.0
    return 100.0 * sum(1.0 for lst in per_question_correctness if any(lst)) / len(per_question_correctness)


def load_subdomain_and_major_by_row(
    gpqa_csv: Path, test_rows: List[dict]
) -> Tuple[List[str], List[str], Dict[str, Tuple[str, str]]]:
    """
    Returns:
      - subdomain_per_test_row (aligned with test_rows order)
      - major_domain_per_test_row (Biology/Physics/Chemistry) (aligned)
      - question_to_(subdomain, major_domain) mapping (best-effort)
    Alignment strategy:
      - If CSV length == test length: assume row-by-row
      - Else: build question->subdomain dict from CSV's "Question" field and map by exact string
    """
    import pandas as pd

    df = pd.read_csv(gpqa_csv)
    # column name in file is "Subdomain" (capital S), but tolerate variants
    sub_col = None
    for c in ("Subdomain", "subdomain", "SUBDOMAIN"):
        if c in df.columns:
            sub_col = c
            break
    if sub_col is None:
        raise KeyError(f"Cannot find subdomain column in CSV. Columns: {list(df.columns)}")

    q_col = None
    for c in ("Question", "question"):
        if c in df.columns:
            q_col = c
            break

    major_col = None
    for c in ("High-level domain", "High-level Domain", "high-level domain", "domain"):
        if c in df.columns:
            major_col = c
            break
    if major_col is None:
        raise KeyError(
            f"Cannot find high-level domain column in CSV. Columns: {list(df.columns)}"
        )

    n_test = len(test_rows)
    if len(df) == n_test:
        subs = [str(x) for x in df[sub_col].tolist()]
        majors_raw = [str(x) for x in df[major_col].tolist()]
        majors = [normalize_high_level_domain(x) for x in majors_raw]
        q2: Dict[str, Tuple[str, str]] = {}
        if q_col:
            for q, s, m in zip(df[q_col].tolist(), subs, majors):
                q2[str(q)] = (s, m)
        return subs, majors, q2

    if not q_col:
        raise ValueError(
            f"CSV rows ({len(df)}) != test rows ({n_test}) and cannot align without a Question column."
        )

    q2: Dict[str, Tuple[str, str]] = {}
    for q, s, m in zip(df[q_col].tolist(), df[sub_col].tolist(), df[major_col].tolist()):
        q2[str(q)] = (str(s), normalize_high_level_domain(str(m)))

    subs: List[str] = []
    majors: List[str] = []
    missing = 0
    for r in test_rows:
        q = str(r.get("question", ""))
        pair = q2.get(q)
        if pair is None:
            missing += 1
            subs.append("UNKNOWN")
            majors.append("UNKNOWN")
        else:
            s, m = pair
            subs.append(s)
            majors.append(m)
    if missing:
        raise ValueError(
            f"Failed to align {missing}/{n_test} questions by exact text. "
            f"CSV len={len(df)}. If you expect row-by-row, please pass a CSV already filtered to test split."
        )
    return subs, majors, q2


def normalize_high_level_domain(x: str) -> str:
    """
    GPQA diamond CSV has a 'High-level domain' column that should already be one of:
      Biology / Chemistry / Physics
    We canonicalize spelling/casing and fail fast on anything else.
    """
    s = (x or "").strip().lower()
    if s == "physics":
        return "Physics"
    if s == "chemistry":
        return "Chemistry"
    if s == "biology":
        return "Biology"
    raise ValueError(f"Unexpected High-level domain value: {x!r}")


def infer_model_name_from_path(pred_jsonl: Path) -> str:
    """
    Return "<model_dir>_<step>" to match naming like:
      GRPO_Qwen3-1.7B-Base_agentica-org_DeepScaleR-Preview-Dataset_16384_temp-1.0_350

    We infer:
      - model_dir: path segment right after ".../outputs/"
      - step: from a "global_step_<N>" segment (if present)
    """
    parts = pred_jsonl.parts
    model_dir = pred_jsonl.stem
    if "outputs" in parts:
        idx = parts.index("outputs")
        if idx + 1 < len(parts):
            model_dir = parts[idx + 1]

    step = None
    for p in parts:
        m = re.fullmatch(r"global_step_(\d+)", p)
        if m:
            step = m.group(1)
            break

    if step and not re.search(rf"_{re.escape(step)}$", model_dir):
        return f"{model_dir}_{step}"
    return model_dir


def fix_one_file(
    pred_jsonl: Path,
    test_jsonl: Path,
    gpqa_csv: Path,
    out_jsonl: Optional[Path] = None,
    model: Optional[str] = None,
) -> Tuple[Dict[str, object], List[str]]:
    pred_rows = read_jsonl(pred_jsonl)
    test_rows = read_jsonl(test_jsonl)
    if len(pred_rows) != len(test_rows):
        raise ValueError(
            f"{pred_jsonl}: pred_jsonl lines={len(pred_rows)} != test_jsonl lines={len(test_rows)}"
        )

    subdomains, majors, _ = load_subdomain_and_major_by_row(gpqa_csv, test_rows)
    uniq_major = sorted(set(majors))

    all_correctness: List[List[bool]] = []
    for i, (pr, tr) in enumerate(zip(pred_rows, test_rows)):
        gold = tr.get("answer")
        gen = pr.get("generated_responses")
        if not isinstance(gen, list):
            raise TypeError(f"{pred_jsonl} line {i+1}: generated_responses is not a list")
        corr = compute_answers_correctness_mcq(gen, str(gold))
        pr["id"] = pr.get("id", tr.get("id", i + 1))
        pr["gold_answer"] = str(gold).strip()
        pr["answers_correctness"] = corr
        pr["is_correct"] = any(corr)
        pr["subdomain"] = subdomains[i]
        pr["major_domain"] = majors[i]
        all_correctness.append(corr)

    out_jsonl = out_jsonl or pred_jsonl.with_name(pred_jsonl.stem + "_fixed.jsonl")
    write_jsonl_atomic(out_jsonl, pred_rows)

    def mean_for_domain(domain: str) -> float:
        per_q = [c for c, m in zip(all_correctness, majors) if (domain == "Overall" or m == domain)]
        return mean_at_n(per_q)

    def passn_for_domain(domain: str) -> float:
        per_q = [c for c, m in zip(all_correctness, majors) if (domain == "Overall" or m == domain)]
        return pass_at_n_percent(per_q)

    model_name = model or infer_model_name_from_path(pred_jsonl)
    row = {
        "Model": model_name,
        "Biology": mean_for_domain("Biology"),
        "Physics": mean_for_domain("Physics"),
        "Chemistry": mean_for_domain("Chemistry"),
        "Overall": mean_for_domain("Overall"),
        "Passn_Biology": passn_for_domain("Biology"),
        "Passn_Physics": passn_for_domain("Physics"),
        "Passn_Chemistry": passn_for_domain("Chemistry"),
        "Passn_Overall": passn_for_domain("Overall"),
        "_pred_jsonl": str(pred_jsonl),
        "_fixed_jsonl": str(out_jsonl),
    }
    return row, uniq_major


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", type=Path, default=None, help="single mode: e.g. test_t0.6_k4.jsonl")
    ap.add_argument("--outputs_root", type=Path, default=None, help="batch mode: scan under this dir (outputs/)")
    ap.add_argument(
        "--test_jsonl",
        type=Path,
        default=DEFAULT_TEST_JSONL,
        help=f"e.g. data/gpqa/test.jsonl (default: {DEFAULT_TEST_JSONL})",
    )
    ap.add_argument(
        "--gpqa_csv",
        type=Path,
        default=DEFAULT_GPQA_CSV,
        help=f"e.g. data/gpqa/gpqa_diamond.csv (default: {DEFAULT_GPQA_CSV})",
    )
    ap.add_argument("--out_jsonl", type=Path, default=None)
    ap.add_argument("--out_xlsx", type=Path, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--k", type=int, default=4, help="k for pass@k (used only if you want it later)")
    args = ap.parse_args()

    if args.outputs_root is None and args.pred_jsonl is None:
        args.outputs_root = DEFAULT_OUTPUTS_ROOT

    test_rows = read_jsonl(args.test_jsonl)
    subdomains, majors, q2 = load_subdomain_and_major_by_row(args.gpqa_csv, test_rows)
    uniq_sub = sorted(set(str(x) for x in q2.values())) if q2 else sorted(set(subdomains))

    uniq_major = sorted(set(majors))
    try:
        import pandas as pd

        # also include a second sheet to show the subdomain taxonomy we observed
        # build subdomain->major mapping from the CSV rows we saw
        sub_map: Dict[str, str] = {}
        for s, m in zip(subdomains, majors):
            sub_map.setdefault(str(s), str(m))
        taxonomy = pd.DataFrame(
            [{"subdomain": s, "major": sub_map.get(s, "UNKNOWN")} for s in sorted(sub_map.keys())],
            columns=["subdomain", "major"],
        )

        # batch mode
        if args.outputs_root:
            root = args.outputs_root
            candidates = sorted(root.glob("**/gpqa/test_t*_k*.jsonl"))
            # avoid re-processing our own fixed files
            candidates = [p for p in candidates if not p.name.endswith("_fixed.jsonl")]
            if not candidates:
                raise ValueError(f"No gpqa/test_t*_k*.jsonl found under {root}")

            rows: List[Dict[str, object]] = []
            for p in candidates:
                fixed = p.with_name(p.stem + "_fixed.jsonl")
                row, _ = fix_one_file(
                    pred_jsonl=p,
                    test_jsonl=args.test_jsonl,
                    gpqa_csv=args.gpqa_csv,
                    out_jsonl=fixed,
                    model=None,
                )
                rows.append(row)

            out_xlsx = args.out_xlsx or (root / "gpqa_subdomain_mean.xlsx")
            df = pd.DataFrame(rows)
            df = df.sort_values("Model")
            with pd.ExcelWriter(out_xlsx) as w:
                df[["Model", "Biology", "Physics", "Chemistry", "Overall"]].to_excel(w, sheet_name="mean@n", index=False)
                df[
                    ["Model", "Passn_Biology", "Passn_Physics", "Passn_Chemistry", "Passn_Overall"]
                ].rename(
                    columns={
                        "Passn_Biology": "Biology",
                        "Passn_Physics": "Physics",
                        "Passn_Chemistry": "Chemistry",
                        "Passn_Overall": "Overall",
                    }
                ).to_excel(w, sheet_name="pass@n", index=False)
                taxonomy.to_excel(w, sheet_name="subdomain_map", index=False)
                df[["_pred_jsonl", "_fixed_jsonl", "Model"]].to_excel(w, sheet_name="files", index=False)

            print("written excel:", str(out_xlsx))
            print("processed files:", len(candidates))
            return

        # single mode
        if not args.pred_jsonl:
            raise ValueError("Need either --pred_jsonl (single) or --outputs_root (batch).")

        out_jsonl = args.out_jsonl or args.pred_jsonl.with_name(args.pred_jsonl.stem + "_fixed.jsonl")
        row, _ = fix_one_file(
            pred_jsonl=args.pred_jsonl,
            test_jsonl=args.test_jsonl,
            gpqa_csv=args.gpqa_csv,
            out_jsonl=out_jsonl,
            model=args.model,
        )
        out_xlsx = args.out_xlsx or out_jsonl.with_suffix(".subdomain_mean.xlsx")
        df = pd.DataFrame([row])
        with pd.ExcelWriter(out_xlsx) as w:
            df[["Model", "Biology", "Physics", "Chemistry", "Overall"]].to_excel(w, sheet_name="mean@n", index=False)
            df[
                ["Model", "Passn_Biology", "Passn_Physics", "Passn_Chemistry", "Passn_Overall"]
            ].rename(
                columns={
                    "Passn_Biology": "Biology",
                    "Passn_Physics": "Physics",
                    "Passn_Chemistry": "Chemistry",
                    "Passn_Overall": "Overall",
                }
            ).to_excel(w, sheet_name="pass@n", index=False)
            taxonomy.to_excel(w, sheet_name="subdomain_map", index=False)
    except Exception as e:
        raise RuntimeError(
            f"Excel export failed ({e}). If pandas/openpyxl is missing, install it or change output format."
        )

    print("written fixed jsonl:", str(out_jsonl))
    print("written excel:", str(out_xlsx))
    print("unique majors:", uniq_major)
    if "UNKNOWN" in uniq_major:
        print("WARNING: some subdomains classified as UNKNOWN; check subdomain_map sheet.")


if __name__ == "__main__":
    main()

