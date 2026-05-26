#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset: Skywork/Skywork-OR1-RL-Data
Nested fields:
  - prompt: ["prompt"][0]["content"]
  - answer: ["reward_model"]["ground_truth"]
  - source: data_source
"""

from typing import Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data_utils.base_processor import DatasetProcessor


class SkyworkOR1RLDataProcessor(DatasetProcessor):
    """Skywork/Skywork-OR1-RL-Data 数据集处理器"""
    
    def get_default_hf_id(self) -> str:
        return "Skywork/Skywork-OR1-RL-Data"
    
    @classmethod
    def get_default_args(cls):
        return {
            "prompt_field": "prompt",
            "answer_key": "reward_model",
            "source_field": "data_source",
            "print_max_len": 2000,
        }
    
    # def process_prompt(self, text: Any) -> Any:
    #     """处理prompt字段，从嵌套结构 ["prompt"][0]["content"] 中提取"""
    #     # Handle nested field extraction ["prompt"][0]["content"] for `math` split
    #     if isinstance(text, list) and len(text) > 0 and isinstance(text[0], dict):
    #         return text[0].get("content", text)
    
    # def process_answer(self, val: Any) -> Any:
    #     """处理answer字段，从嵌套结构 ["reward_model"]["ground_truth"] 中提取"""
    #     # Handle nested field extraction ["reward_model"]["ground_truth"] for `math` split
    #     if isinstance(val, dict) and "ground_truth" in val:
    #         answer_val = val["ground_truth"]
    #         if isinstance(answer_val, str) and answer_val.startswith("[\"") and answer_val.endswith("\"]"):
    #             # Remove the surrounding brackets and parse the inner value
    #             import json
    #             try:
    #                 # Handles typical cases like '["20"]'
    #                 parsed = json.loads(answer_val)
    #                 if isinstance(parsed, list) and len(parsed) == 1:
    #                     return parsed[0]
    #                 return parsed
    #             except Exception:
    #                 # Fallback: just strip [] and quotes
    #                 return answer_val.strip("[\"]").strip('"\'')
    #         return answer_val
    #     return val
    
    def process_prompt(self, text: Any) -> Any:
        """处理prompt字段，从嵌套结构 ["prompt"][0]["content"] 中提取"""
        # Handle nested field extraction ["prompt"][0]["content"] for `math` split
        if isinstance(text, list) and len(text) > 0 and isinstance(text[0], dict):
            processed = text[0].get("content", text)
            processed = processed.replace("You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n", "")
        if len(processed) == 0:
            return processed
        else:
            return processed
    
    def process_answer(self, val: Any) -> Any:
        """处理answer字段，从嵌套结构 ["reward_model"]["ground_truth"] 中提取"""
        # Handle nested field extraction ["reward_model"]["ground_truth"] for `math` split
        if isinstance(val, dict) and "ground_truth" in val:
            answer_val = val["ground_truth"]
            return answer_val
        return val
    
    def process_source(self, val: Any) -> Any:
        return val
    
    def process_solution(self, val: Any) -> Any:
        """处理solution字段"""
        return val


if __name__ == "__main__":
    processor = SkyworkOR1RLDataProcessor()
    processor.main()
