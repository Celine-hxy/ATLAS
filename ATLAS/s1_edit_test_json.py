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
        default="$ROOT/ATLAS/results/stage0_org_test_data.json",
        help="Input JSON file path",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="$ROOT/ATLAS/results/stage1_test_filtered.json",
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
    deepmath_hf_id = "zwhe99/DeepMath-103K"
    deepmath_default_source = "deepmath_stack_exchange"
    olympiads_hf_id = "AI-MO/olympiads-ref-base"
    olympiads_source = "olympiads-ref"
    harp_hf_id = "GITHUB/HARP"
    harp_source = "harp"
    math_hf_id = "HuggingFaceH4/MATH-500"
    math_source = "math"
    omni_hf_id = "KbsdJames/Omni-MATH"
    omni_problem_type = "omni_math"
    olympiadbench_hf_id = "knoveleng/OlympiadBench"
    olympiadbench_source = "olympiad_bench"
    imo_answerbench_hf_id = "Hwilner/imo-answerbench"
    imo_answerbench_source = "imo_answerbench"
    long_form_thought_hf_id = "RUC-AIBOX/long_form_thought_data_5k"
    long_form_thought_source = "still1"
    math_odyssey_hf_id = "MathOdyssey/MathOdyssey"
    math_odyssey_source = "math_odyssey"
    beyond_aime_hf_id = "ByteDance-Seed/BeyondAIME"
    beyond_aime_source = "beyond_aime"
    hle_math_hf_id = "hbXNov/hle_math_exact_match_no_image_int_answer_random128"
    hle_math_source = "hle_math"
    minerva_math_hf_id = "math-ai/minervamath"
    minerva_math_source = "minerva_math"
    aime24_hf_id = "math-ai/aime24"
    aime24_source = "aime24"
    aime25_hf_id = "math-ai/aime25"
    aime25_source = "aime25"
    amc23_hf_id = "math-ai/amc23"
    amc23_source = "amc23"
    amo_bench_hf_id = "hf-imo-colab/AMO-Bench"
    amo_bench_source = "amo_bench"
    proof_answers = ["proof", "\\text{Proved}"]
    gsm8k_structured_source = "GSM8k_structured"
    gsm8k_target_source = "gsm8k"
    dropped_prompts = 0
    dropped_occurrences = 0
    modified_sources = 0
    assigned_deepmath_sources = 0
    assigned_olympiads_sources = 0
    assigned_harp_sources = 0
    assigned_math_sources = 0
    assigned_olympiadbench_sources = 0
    assigned_imo_answerbench_sources = 0
    assigned_long_form_thought_sources = 0
    assigned_math_odyssey_sources = 0
    assigned_beyond_aime_sources = 0
    assigned_hle_math_sources = 0
    assigned_minerva_math_sources = 0
    assigned_aime24_sources = 0
    assigned_aime25_sources = 0
    assigned_amc23_sources = 0
    assigned_amo_bench_sources = 0
    modified_omni_sources = 0
    modified_gsm8k_sources = 0
    modified_problem_type_sources = 0
    modified_basic_arithmetic_sources = 0
    nullified_deepscaler_sources = 0
    removed_structured_suffix = 0
    removed_train_math_prefix = 0
    modified_proof_answers = 0
    modified_prove_answers = 0
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
                # 修改 Kwai-Klear/KlearReasoner-MathSub-30K 的 source 为 null
                if occ.get("hf_id") == source_modify_hf_id:
                    if "source" in occ and occ["source"] is not None:
                        occ["source"] = None
                        modified_sources += 1
                
                # 如果 zwhe99/DeepMath-103K 的 source 是 null，则赋值为 deepmath_stack_exchange
                if occ.get("hf_id") == deepmath_hf_id:
                    if occ.get("source") is None:
                        occ["source"] = deepmath_default_source
                        assigned_deepmath_sources += 1
                
                # 如果 AI-MO/olympiads-ref-base，将 source 赋值为 olympiads-ref
                if occ.get("hf_id") == olympiads_hf_id:
                    occ["source"] = olympiads_source
                    assigned_olympiads_sources += 1
                
                # 如果 hf_id == "GITHUB/HARP"，将 source 赋值为 harp
                if occ.get("hf_id") == harp_hf_id:
                    occ["source"] = harp_source
                    assigned_harp_sources += 1
                
                # 如果 hf_id == "HuggingFaceH4/MATH-500"，将 source 赋值为 math
                if occ.get("hf_id") == math_hf_id:
                    occ["source"] = math_source
                    assigned_math_sources += 1
                
                # 如果 hf_id == "knoveleng/OlympiadBench"，将 source 赋值为 olympiad_bench
                if occ.get("hf_id") == olympiadbench_hf_id:
                    occ["source"] = olympiadbench_source
                    assigned_olympiadbench_sources += 1
                
                # 如果 hf_id == "Hwilner/imo-answerbench"，将 source 赋值为 imo_answerbench
                if occ.get("hf_id") == imo_answerbench_hf_id:
                    occ["source"] = imo_answerbench_source
                    assigned_imo_answerbench_sources += 1
                
                # 如果 hf_id == "RUC-AIBOX/long_form_thought_data_5k"，将 source 赋值为 still1
                if occ.get("hf_id") == long_form_thought_hf_id:
                    occ["source"] = long_form_thought_source
                    assigned_long_form_thought_sources += 1
                
                # 如果 hf_id == "MathOdyssey/MathOdyssey"，将 source 赋值为 math_odyssey
                if occ.get("hf_id") == math_odyssey_hf_id:
                    occ["source"] = math_odyssey_source
                    assigned_math_odyssey_sources += 1
                
                # 如果 hf_id == "ByteDance-Seed/BeyondAIME"，将 source 赋值为 beyond_aime
                if occ.get("hf_id") == beyond_aime_hf_id:
                    occ["source"] = beyond_aime_source
                    assigned_beyond_aime_sources += 1
                
                # 如果 hf_id == "hbXNov/hle_math_exact_match_no_image_int_answer_random128"，将 source 赋值为 hle_math
                if occ.get("hf_id") == hle_math_hf_id:
                    occ["source"] = hle_math_source
                    assigned_hle_math_sources += 1
                
                # 如果 hf_id == "math-ai/minervamath"，将 source 赋值为 minerva_math
                if occ.get("hf_id") == minerva_math_hf_id:
                    occ["source"] = minerva_math_source
                    assigned_minerva_math_sources += 1
                
                # 如果 hf_id == "math-ai/aime24"，将 source 赋值为 aime24
                if occ.get("hf_id") == aime24_hf_id:
                    occ["source"] = aime24_source
                    assigned_aime24_sources += 1
                
                # 如果 hf_id == "math-ai/aime25"，将 source 赋值为 aime25
                if occ.get("hf_id") == aime25_hf_id:
                    occ["source"] = aime25_source
                    assigned_aime25_sources += 1
                
                # 如果 hf_id == "math-ai/amc23"，将 source 赋值为 amc23
                if occ.get("hf_id") == amc23_hf_id:
                    occ["source"] = amc23_source
                    assigned_amc23_sources += 1
                
                # 如果 hf_id == "hf-imo-colab/AMO-Bench"，将 source 赋值为 amo_bench
                if occ.get("hf_id") == amo_bench_hf_id:
                    occ["source"] = amo_bench_source
                    assigned_amo_bench_sources += 1
                
                # 如果 hf_id == "KbsdJames/Omni-MATH"，保留原 source 值到 problem_type，source 改为 omni
                if occ.get("hf_id") == omni_hf_id:
                    original_source = occ.get("source")
                    if original_source is not None:
                        occ["problem_type"] = original_source
                    occ["source"] = omni_problem_type
                    modified_omni_sources += 1
                
                # 如果 source 是 "GSM8k_structured"，改为 "gsm8k"
                if occ.get("source") == gsm8k_structured_source:
                    occ["source"] = gsm8k_target_source
                    modified_gsm8k_sources += 1
                
                # 获取 source 进行处理
                source = occ.get("source")
                
                # 如果 source 是 "train-math-deepscaler"，置为 null
                if source == "train-math-deepscaler":
                    occ["source"] = None
                    nullified_deepscaler_sources += 1
                    source = None  # 更新 source 变量，避免后续处理
                
                # 如果 source 包含 "train-math-"，去掉 "train-math-" 字样
                if source and isinstance(source, str) and "train-math-" in source:
                    occ["source"] = source.replace("train-math-", "")
                    removed_train_math_prefix += 1
                    source = occ["source"]  # 更新 source 变量，用于后续处理
                
                # 如果 source 包含 "_structured"，去掉 "_structured" 字样
                if source and isinstance(source, str) and "_structured" in source:
                    occ["source"] = source.replace("_structured", "")
                    removed_structured_suffix += 1
                    source = occ["source"]  # 更新 source 变量，用于后续处理
                
                # 如果 source 是 basic_arithmetic 相关值，改为 "basic_arithmetic" 并保存原值到 problem_type
                if source in basic_arithmetic_sources:
                    occ["source"] = "basic_arithmetic"
                    occ["problem_type"] = source
                    modified_basic_arithmetic_sources += 1
                elif source and isinstance(source, str):
                    # 如果 source 以特定前缀开头，提取 problem_type 并修改 source
                    for prefix in problem_type_prefixes:
                        if source.startswith(prefix):
                            problem_type = source[len(prefix):]  # 提取前缀之后的内容
                            base_source = prefix.rstrip("_")  # 去掉末尾的 "_"
                            occ["source"] = base_source
                            occ["problem_type"] = problem_type
                            modified_problem_type_sources += 1
                            break
                
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
    print(f"  Assigned HARP sources (GITHUB/HARP -> harp): {assigned_harp_sources}")
    print(f"  Assigned MATH sources (HuggingFaceH4/MATH-500 -> math): {assigned_math_sources}")
    print(f"  Assigned OlympiadBench sources (knoveleng/OlympiadBench -> olympiad_bench): {assigned_olympiadbench_sources}")
    print(f"  Assigned IMO AnswerBench sources (Hwilner/imo-answerbench -> imo_answerbench): {assigned_imo_answerbench_sources}")
    print(f"  Assigned Long Form Thought sources (RUC-AIBOX/long_form_thought_data_5k -> still1): {assigned_long_form_thought_sources}")
    print(f"  Assigned MathOdyssey sources (MathOdyssey/MathOdyssey -> math_odyssey): {assigned_math_odyssey_sources}")
    print(f"  Assigned BeyondAIME sources (ByteDance-Seed/BeyondAIME -> beyond_aime): {assigned_beyond_aime_sources}")
    print(f"  Assigned HLE Math sources (hbXNov/hle_math_exact_match_no_image_int_answer_random128 -> hle_math): {assigned_hle_math_sources}")
    print(f"  Assigned Minerva Math sources (math-ai/minervamath -> minerva_math): {assigned_minerva_math_sources}")
    print(f"  Assigned AIME24 sources (math-ai/aime24 -> aime24): {assigned_aime24_sources}")
    print(f"  Assigned AIME25 sources (math-ai/aime25 -> aime25): {assigned_aime25_sources}")
    print(f"  Assigned AMC23 sources (math-ai/amc23 -> amc23): {assigned_amc23_sources}")
    print(f"  Assigned AMO-Bench sources (hf-imo-colab/AMO-Bench -> amo_bench): {assigned_amo_bench_sources}")
    print(f"  Modified Omni-MATH sources (KbsdJames/Omni-MATH: source -> problem_type=omni): {modified_omni_sources}")
    print(f"  Kept prompts: {kept_prompts}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()