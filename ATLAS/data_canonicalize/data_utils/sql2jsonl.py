#!/usr/bin/env python3
"""Convert ATLAS SQLite to JSONL with keys matching verl/recipe/eval/data format.

Output keys: problem, solution, answer [, unique_id ]
Example: math500/test.jsonl, aime24/test.jsonl
"""
import argparse
import json
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="SQLite to JSONL (verl eval format)")
    parser.add_argument("sqlite_path", type=str, nargs="?", default="$ROOT/ATLASDB/test/math/hle_math_exact_match_no_image_int_answer_random128/test.sqlite")
    parser.add_argument("-o", "--output", type=str, default="$ROOT/verl/recipe/eval/data/hle_math/test.jsonl")
    # parser.add_argument("sqlite_path", type=str, nargs="?", default="$ROOT/ATLASDB/test/math/AMO-Bench/test.sqlite")
    # parser.add_argument("-o", "--output", type=str, default="$ROOT/verl/recipe/eval/data/amo_bench/test.jsonl", help="Output JSONL path (default: <sqlite_stem>.jsonl beside sqlite)")
    parser.add_argument("-t", "--table", type=str, default="data", help="Table name")
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite_path).resolve()
    if not sqlite_path.exists():
        raise FileNotFoundError(sqlite_path)

    out_path = Path(args.output).resolve() if args.output else sqlite_path.with_suffix(".jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    cur = conn.execute(f'SELECT * FROM {args.table} ORDER BY rowid')
    rows = cur.fetchall()
    conn.close()

    # Column mapping: ATLAS canonical -> verl eval keys
    # verl format: problem, solution, answer [, unique_id, subject, level, ... ]
    def row_to_obj(row):
        d = dict(row)
        problem = d.get("prompt") or d.get("problem") or ""
        solution = d.get("solution") or ""
        answer = d.get("answer") or ""
        out = {"problem": problem, "solution": solution, "answer": answer}
        if "DL_row_idx" in d and d["DL_row_idx"] is not None:
            out["unique_id"] = str(d["DL_row_idx"])
        elif "prompt_sha1" in d and d["prompt_sha1"]:
            out["unique_id"] = d["prompt_sha1"]
        else:
            out["unique_id"] = ""
        return out

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            obj = row_to_obj(row)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
