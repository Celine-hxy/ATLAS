#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
from pathlib import Path
from typing import Dict, Any
import pyarrow.parquet as pq

global_extraction_failed_count = 0
def add_extraction_failed_count() -> int:
    global global_extraction_failed_count
    global_extraction_failed_count += 1
    return global_extraction_failed_count

def get_extraction_failed_count() -> int:
    global global_extraction_failed_count
    return global_extraction_failed_count

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

# ========================
# Dataset Split Mapping
# ========================
SPLIT_MAP = {
    "allenai/lila": "train",
    "nvidia/AceReason-Math": "math",
    "inclusionAI/AReaL-boba-Data": "AReaL-boba-106k",
    "Kwai-Klear/KlearReasoner-MathSub-30K": "train_math_30K",
    "agentica-org/DeepScaleR-Preview-Dataset": "deepscaler",
    "meta-math/MetaMathQA": "MetaMathQA-395K",
    "Open-Reasoner-Zero/orz_math_72k_collection_extended": "orz_math_72k_collection_extended",
    "POLARIS-Project/Polaris-Dataset-53K": "polaris-data-53K",
    "open-r1/Big-Math-RL-Verified-Processed": "all",
    # "open-r1/DAPO-Math-17k-Processed": "train",
    "zwhe99/DeepMath-103K": "train",
    "PRIME-RL/Eurus-2-RL-Data": "train",
    "AI-MO/NuminaMath-1.5": "train",
    "AI-MO/NuminaMath-CoT": "train", # "train,test"
    "nvidia/OpenMathReasoning": "cot,tir,genselect",
    "open-r1/OpenR1-Math-220k": "all", # "default,all,extended"
    "knoveleng/open-rs": "train",
    "microsoft/orca-math-word-problems-200k": "train",
    "ElonTusk2001/rstar_ppm": "train",
    "Skywork/Skywork-OR1-RL-Data": "math", # "math,code"
    "RUC-AIBOX/long_form_thought_data_5k": "train",
    "RUC-AIBOX/STILL-3-RL-90K": "train",
    "AI-MO/olympiads-ref-base": "train",
    "BytedTsinghua-SIA/DAPO-Math-17k": "dapo-math-17k",
}


# ========================
# Answer Extraction Utils
# ========================
def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


# ========================
# Common Processing Utils
# ========================
def find_parquet_file(dataset_dir: Path, split: str) -> Path:
    """查找parquet文件"""
    if split.endswith(".parquet"):
        p = dataset_dir / split
        if not p.exists():
            raise FileNotFoundError(f"Parquet file not found: {p}")
        return p
    
    parquet_files = sorted(dataset_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {dataset_dir}")
    
    cand = [p for p in parquet_files if split.lower() in p.name.lower()]
    if len(cand) == 1:
        return cand[0]
    if len(cand) > 1:
        names = "\n  - " + "\n  - ".join([c.name for c in cand])
        raise RuntimeError(f"Multiple parquet files match split='{split}' under {dataset_dir}:{names}")
    
    if len(parquet_files) == 1:
        return parquet_files[0]
    
    names = "\n  - " + "\n  - ".join([p.name for p in parquet_files])
    raise RuntimeError(
        f"Cannot uniquely determine parquet for split='{split}' under {dataset_dir}. "
        f"Candidates:{names}\nTip: pass --split <exact_filename.parquet>"
    )


def pretty_print_sample(before: Dict[str, Any], after: Dict[str, Any], max_len: int = 100):
    """打印处理前后的样本对比，每个字段的value分别限制长度"""
    def clip_dict(d: Dict[str, Any]) -> str:
        result = []
        for key, value in d.items():
            value_str = json.dumps(value, ensure_ascii=False, default=str)
            if len(value_str) > max_len:
                value_str = value_str[:max_len] + "...(truncated)"
            result.append(f'  "{key}": {value_str}')
        return "{\n" + ",\n".join(result) + "\n}"
    
    print("\n================ SAMPLE (BEFORE) ================")
    print(clip_dict(before), "\n")
    print("\n================ SAMPLE (AFTER) =================")
    print(clip_dict(after), "\n")


def prompt_yes_no(msg: str) -> bool:
    """提示用户输入yes/no"""
    while True:
        ans = input(msg).strip().lower()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please input y/n.")