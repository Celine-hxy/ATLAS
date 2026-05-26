#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset: Open-Reasoner-Zero/orz_math_72k_collection_extended
Nested fields:
  - prompt: ["0"]["value"]
  - answer: ["1"]["ground_truth"]["value"]
  - source: N/A
example = {
  "0": {"from": "human", "value": "14. Two circles $C_{1}$ and $C_{2}$ with centers $A$ and $B$ are externa...(truncated),
  "1": {"from": "assistant", "ground_truth": {"value": "48"}},
  "DataLineage_uid": "Open-Reasoner-Zero/orz_math_72k_collection_extended-orz_math_72k_collection_extended-0"
}
"""

from typing import Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data_utils.base_processor import DatasetProcessor


class OrzMath72KCollectionExtendedProcessor(DatasetProcessor):
    """Open-Reasoner-Zero/orz_math_72k_collection_extended 数据集处理器"""
    
    def get_default_hf_id(self) -> str:
        return "Open-Reasoner-Zero/orz_math_72k_collection_extended"
    
    @classmethod
    def get_default_args(cls):
        return {
            "prompt_field": "0",
            "answer_key": "1",
            "source_field": None,
            "print_max_len": 1000,
        }
    
    def process_prompt(self, text: Any) -> Any:
        """处理prompt字段，从嵌套结构 ["0"]["value"] 中提取"""
        # Extract from ["0"]["value"] structure
        if isinstance(text, dict) and "0" in text:
            nested = text["0"]
            if isinstance(nested, dict) and "value" in nested:
                return nested["value"]
        # If already extracted, just return the value
        if isinstance(text, dict) and "value" in text:
            return text["value"]
        return text
    
    def process_answer(self, val: Any) -> Any:
        """处理answer字段，从嵌套结构 ["1"]["ground_truth"]["value"] 中提取"""
        # Extract from ["1"]["ground_truth"]["value"] structure
        if isinstance(val, dict) and "1" in val:
            nested = val["1"]
            if isinstance(nested, dict) and "ground_truth" in nested:
                gt = nested["ground_truth"]
                if isinstance(gt, dict) and "value" in gt:
                    return gt["value"]
        # If already extracted to ground_truth level
        if isinstance(val, dict) and "ground_truth" in val:
            gt = val["ground_truth"]
            if isinstance(gt, dict) and "value" in gt:
                return gt["value"]
        return val
    
    def process_solution(self, val: Any) -> Any:
        """处理solution字段"""
        return val


if __name__ == "__main__":
    processor = OrzMath72KCollectionExtendedProcessor()
    processor.main()
