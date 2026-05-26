#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
过滤 stage1_data.json：
1. 移除 nvidia/OpenMathReasoning 相关数据
2. 修改 Kwai-Klear/KlearReasoner-MathSub-30K 的 source 为 null
3. 如果 zwhe99/DeepMath-103K 的 source 是 null，则赋值为 deepmath_stack_exchange
4. 将所有 occ_list 中 answer == "proof" 或 "\\text{Proved}" 的数据的 answer 改为 ""
5. 如果 hf_id == "AI-MO/olympiads-ref-base"，将 source 赋值为 "olympiads-ref"
6. 如果 source == "GSM8k_structured"，改为 "gsm8k"
7. 如果 source == "train-math-deepscaler"，置为 null
8. 如果 source 包含 "train-math-"，去掉 "train-math-" 字样
9. 如果 source 包含 "_structured"，去掉 "_structured" 字样
9. 如果 source 是 "addsub", "simuleq", "singleop", "singleq", "multiarith"，改为 "basic_arithmetic"，原值存入 problem_type
10. 如果 source 以 "amps_", "mathqa_", "NumGLUE_", "deepmind_mathematics_" 开头，提取 problem_type 并修改 source
"""

import argparse
import json
import re
import ijson
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json_file",
        type=str,
        default="$ROOT/ATLAS/results/stage0_org_train_data.json",
        help="Input JSON file path",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="$ROOT/ATLAS/results/stage1_train_filtered.json",
        help="Output JSON file path",
    )
    args = ap.parse_args()

    json_path = Path(args.json_file)
    output_path = Path(args.output)
    
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        return

    target_hf_id = "nvidia/OpenMathReasoning"
    source_modify_hf_id = "Kwai-Klear/KlearReasoner-MathSub-30K"
    dapo_math_hf_id = "BytedTsinghua-SIA/DAPO-Math-17k"
    deepmath_hf_id = "zwhe99/DeepMath-103K"
    deepmath_default_source = "deepmath_stack_exchange"
    olympiads_hf_id = "AI-MO/olympiads-ref-base"
    olympiads_source = "olympiads_ref"
    long_form_thought_hf_id = "RUC-AIBOX/long_form_thought_data_5k"
    long_form_thought_source = "still1"
    proof_answers = ["proof", "\\text{Proved}"]
    gsm8k_structured_source = "GSM8k_structured"
    gsm8k_target_source = "gsm8k"
    dropped_prompts = 0
    dropped_occurrences = 0
    modified_sources = 0
    assigned_deepmath_sources = 0
    assigned_olympiads_sources = 0
    assigned_long_form_thought_sources = 0
    modified_gsm8k_sources = 0
    modified_problem_type_sources = 0
    modified_basic_arithmetic_sources = 0
    nullified_deepscaler_sources = 0
    removed_structured_suffix = 0
    removed_train_math_prefix = 0
    modified_proof_answers = 0
    modified_prove_answers = 0
    nullified_dapo_math_sources = 0
    kept_prompts = 0
    
    # 需要提取 problem_type 的 source 前缀列表
    problem_type_prefixes = ["amps_", "mathqa_", "NumGLUE_", "deepmind_mathematics_"]
    # 需要改为 basic_arithmetic 的 source 值列表
    basic_arithmetic_sources = ["addsub", "simuleq", "singleop", "singleq", "multiarith"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 使用 ijson 流式读取，避免内存溢出
    with json_path.open("rb") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        # 开始写入 JSON 对象
        f_out.write("{\n")
        first_item = True
        
        # 使用 ijson.kvitems 解析 JSON 对象的键值对
        for prompt_sha1, data in ijson.kvitems(f_in, ""):
            if not isinstance(data, dict) or "occ_list" not in data:
                continue
            
            occ_list = data.get("occ_list", [])
            
            # 步骤1: 修改 occ_list 中的 source 和 answer
            for occ in occ_list:
                # 第一部分：所有基于 hf_id 的赋值操作（在 source 检索之前）
                # 修改 Kwai-Klear/KlearReasoner-MathSub-30K 的 source 为 null
                if occ.get("hf_id") == source_modify_hf_id:
                    if "source" in occ and occ["source"] is not None:
                        occ["source"] = None
                        modified_sources += 1
                
                # 修改 BytedTsinghua-SIA/DAPO-Math-17k 的 source "math_dapo" 为 null
                if occ.get("hf_id") == dapo_math_hf_id and occ.get("source") == "math_dapo":
                    occ["source"] = None
                    nullified_dapo_math_sources += 1
                
                # 如果 zwhe99/DeepMath-103K 的 source 是 null，则赋值为 deepmath_stack_exchange
                if occ.get("hf_id") == deepmath_hf_id:
                    if occ.get("source") is None:
                        occ["source"] = deepmath_default_source
                        assigned_deepmath_sources += 1
                
                # 如果 AI-MO/olympiads-ref-base，将 source 赋值为 olympiads-ref
                if occ.get("hf_id") == olympiads_hf_id:
                    occ["source"] = olympiads_source
                    assigned_olympiads_sources += 1
                
                # 如果 hf_id == "RUC-AIBOX/long_form_thought_data_5k"，将 source 赋值为 still1
                if occ.get("hf_id") == long_form_thought_hf_id:
                    occ["source"] = long_form_thought_source
                    assigned_long_form_thought_sources += 1
                
                # 第二部分：获取 source 并进行所有修改操作（集中处理所有 source 修改）
                source = occ.get("source")
                
                # 所有基于 source 的修改操作（按顺序执行）
                # 如果 source 是 "GSM8k_structured"，改为 "gsm8k"
                if source == gsm8k_structured_source:
                    occ["source"] = gsm8k_target_source
                    modified_gsm8k_sources += 1
                    source = gsm8k_target_source
                
                # 如果 source 是 "train-math-deepscaler"，置为 null
                if source == "train-math-deepscaler":
                    occ["source"] = None
                    nullified_deepscaler_sources += 1
                    source = None
                
                # 如果 source 包含 "train-math-"，去掉 "train-math-" 字样
                if source and isinstance(source, str) and "train-math-" in source:
                    occ["source"] = source.replace("train-math-", "")
                    removed_train_math_prefix += 1
                    source = occ["source"]
                
                # 如果 source 包含 "_structured"，去掉 "_structured" 字样
                if source and isinstance(source, str) and "_structured" in source:
                    occ["source"] = source.replace("_structured", "")
                    removed_structured_suffix += 1
                    source = occ["source"]
                
                # 如果 source 是 basic_arithmetic 相关值，改为 "basic_arithmetic" 并保存原值到 problem_type
                if source in basic_arithmetic_sources:
                    occ["source"] = "basic_arithmetic"
                    occ["problem_type"] = source
                    modified_basic_arithmetic_sources += 1
                elif source and isinstance(source, str):
                    # 如果 source 以特定前缀开头，提取 problem_type 并修改 source
                    for prefix in problem_type_prefixes:
                        if source.startswith(prefix):
                            problem_type = source[len(prefix):]
                            base_source = prefix.rstrip("_")
                            occ["source"] = base_source
                            occ["problem_type"] = problem_type
                            modified_problem_type_sources += 1
                            break
                
                # 第三部分：处理 answer（在所有 source 修改之后）
                # 如果 answer 是 "proof" 或 "\\text{Proved}"，改为 ""
                answer = occ.get("answer")
                if answer and isinstance(answer, str):
                    if answer in proof_answers:
                        occ["answer"] = ""
                        modified_proof_answers += 1
                    else:
                        # 全字匹配 "prove" 或 "proved"（不区分大小写）
                        # 使用正则表达式进行全字匹配
                        if re.search(r'\bprove\b', answer, re.IGNORECASE) or re.search(r'\bproved\b', answer, re.IGNORECASE):
                            occ["answer"] = ""
                            modified_prove_answers += 1
            
            # 使用修改后的 occ_list
            filtered_occ_list = occ_list
            
            # 步骤3: 处理 nvidia/OpenMathReasoning
            has_target = any(occ.get("hf_id") == target_hf_id for occ in filtered_occ_list)
            
            if len(filtered_occ_list) == 1 and has_target:
                # 情况1: 只有1条且是目标，整个扔掉
                dropped_prompts += 1
                continue
            
            if len(filtered_occ_list) > 1 and has_target:
                # 情况2: 多条且有目标，只过滤掉目标项
                original_len = len(filtered_occ_list)
                filtered_occ_list = [
                    occ for occ in filtered_occ_list 
                    if occ.get("hf_id") != target_hf_id
                ]
                dropped_occurrences += original_len - len(filtered_occ_list)
            
            # 如果过滤后 occ_list 为空，整个 prompt 扔掉
            if len(filtered_occ_list) == 0:
                dropped_prompts += 1
                continue
            
            data["occ_list"] = filtered_occ_list
            
            # 写入保留的数据
            if not first_item:
                f_out.write(",\n")
            f_out.write(f'  "{prompt_sha1}": ')
            json.dump(data, f_out, ensure_ascii=False, indent=2)
            first_item = False
            kept_prompts += 1
        
        f_out.write("\n}")

    print(f"[DONE]")
    print(f"  Dropped prompts (entire): {dropped_prompts}")
    print(f"  Dropped occurrences: {dropped_occurrences}")
    print("  Modified proof answers (proof or \\text{Proved} -> empty):", modified_proof_answers)
    print(f"  Modified prove answers (contains 'prove'/'proved' word -> empty): {modified_prove_answers}")
    print(f"  Modified sources (Kwai-Klear -> null): {modified_sources}")
    print(f"  Modified GSM8k sources (GSM8k_structured -> gsm8k): {modified_gsm8k_sources}")
    print(f"  Nullified deepscaler sources (train-math-deepscaler -> null): {nullified_deepscaler_sources}")
    print(f"  Removed train-math- prefix: {removed_train_math_prefix}")
    print(f"  Removed _structured suffix: {removed_structured_suffix}")
    print(f"  Modified basic_arithmetic sources (addsub/simuleq/singleop/singleq/multiarith -> basic_arithmetic, problem_type=original): {modified_basic_arithmetic_sources}")
    print(f"  Modified problem_type sources (amps_*/mathqa_*/NumGLUE_*/deepmind_mathematics_* -> base, problem_type=*): {modified_problem_type_sources}")
    print(f"  Assigned DeepMath sources (null -> deepmath_stack_exchange): {assigned_deepmath_sources}")
    print(f"  Assigned Olympiads sources: {assigned_olympiads_sources}")
    print(f"  Assigned Long Form Thought sources (RUC-AIBOX/long_form_thought_data_5k -> still1): {assigned_long_form_thought_sources}")
    print(f"  Nullified DAPO Math sources (BytedTsinghua-SIA/DAPO-Math-17k, math_dapo -> null): {nullified_dapo_math_sources}")
    print(f"  Kept prompts: {kept_prompts}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()