#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""

"""

import argparse
import json
import re
import ijson
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_json_file",
        type=str,
        default="$ROOT/ATLAS/results/stage1_train_filtered.json",
        help="Input JSON file path",
    )
    ap.add_argument(
        "--test_json_file",
        type=str,
        default="$ROOT/ATLAS/results/stage1_test_filtered.json",
        help="Input JSON file path",
    )
    ap.add_argument(
        "--matched_json_file",
        type=str,
        default="$ROOT/ATLAS/results/summary_stage2_test_match",
        help="Output JSON file path",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="$ROOT/ATLAS/results/stage2_train_test-matched.json",
        help="Output JSON file path",
    )
    args = ap.parse_args()

    train_json_path = Path(args.train_json_file)
    test_json_path = Path(args.test_json_file)
    output_path = Path(args.output)
    
    if not train_json_path.exists():
        print(f"Error: Train JSON file not found: {train_json_path}")
        return
    if not test_json_path.exists():
        print(f"Error: Test JSON file not found: {test_json_path}")
        return
    
    # 步骤1: 从 test_json_file 构建 prompt_sha1 -> source 的映射
    print("Building source lookup from test_json_file...")
    test_source_lookup = {}
    with test_json_path.open("rb") as f:
        for prompt_sha1, data in ijson.kvitems(f, ""):
            if not isinstance(data, dict) or "occ_list" not in data:
                continue
            
            occ_list = data.get("occ_list", [])
            # 按照 occ_list 顺序查找第一个 source != null 的
            found_source = None
            for occ in occ_list:
                source = occ.get("source")
                if source is not None:
                    found_source = source
                    break
            
            if found_source is not None:
                test_source_lookup[prompt_sha1] = found_source
    
    print(f"Found {len(test_source_lookup)} prompts with source in test_json_file")
    
    # 步骤2: 读取 train_json_file，找到 no source 的数据并匹配
    print("Processing train_json_file...")
    matched_count = 0
    no_source_count = 0
    matched_data = {}  # 保存匹配成功的数据
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with train_json_path.open("rb") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        f_out.write("{\n")
        first_item = True
        
        for prompt_sha1, data in ijson.kvitems(f_in, ""):
            if not isinstance(data, dict) or "occ_list" not in data:
                continue
            
            occ_list = data.get("occ_list", [])
            if len(occ_list) == 0:
                continue
            
            # 检查是否为 no source 数据
            found_source = None
            for occ in occ_list:
                source = occ.get("source")
                if source is not None:
                    found_source = source
                    break
            
            if found_source is None:
                # 这是 no source 数据
                no_source_count += 1
                
                # 尝试在 test_json_file 中匹配
                if prompt_sha1 in test_source_lookup:
                    matched_source = test_source_lookup[prompt_sha1]
                    matched_count += 1
                    
                    # 将 source 写入到第一个 occ 的 source 中
                    if len(occ_list) > 0:
                        occ_list[0]["source"] = matched_source
                    
                    # 添加 is_test=True 和 match_test=True
                    data["is_test"] = True
                    data["exact_match_test"] = True
                    
                    # 保存匹配成功的数据（深拷贝）
                    matched_data[prompt_sha1] = json.loads(json.dumps(data))
            
            # 写入数据
            if not first_item:
                f_out.write(",\n")
            first_item = False
            
            f_out.write(f'  "{prompt_sha1}": ')
            json.dump(data, f_out, ensure_ascii=False, indent=2)
        
        f_out.write("\n}")
    
    # 保存匹配成功的数据
    matched_output_path = Path(args.matched_json_file)
    matched_output_path.parent.mkdir(parents=True, exist_ok=True)
    matched_output_path = matched_output_path / "matched.json"
    if matched_data:
        with matched_output_path.open("w", encoding="utf-8") as f:
            json.dump(matched_data, f, indent=2, ensure_ascii=False)
        print(f"Matched data saved to: {matched_output_path}")
    
    print(f"\nSummary:")
    print(f"  No source data in train_json_file: {no_source_count}")
    print(f"  Successfully matched: {matched_count}")
    print(f"  Output saved to: {output_path}")


if __name__ == "__main__":
    main()