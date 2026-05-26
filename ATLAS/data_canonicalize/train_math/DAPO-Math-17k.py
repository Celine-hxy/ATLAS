#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset: BytedTsinghua-SIA/DAPO-Math-17k
Nested fields:
  - prompt: ["prompt"][1]["content"]
  - answer: ["reward_model"]["ground_truth"]
  - source: data_source
Prompt_NeedExtract: Y
"""

from typing import Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data_utils.base_processor import DatasetProcessor


class DAPOMath17KProcessor(DatasetProcessor):
    """BytedTsinghua-SIA/DAPO-Math-17k 数据集处理器"""
    
    def get_default_hf_id(self) -> str:
        return "BytedTsinghua-SIA/DAPO-Math-17k"
    
    @classmethod
    def get_default_args(cls):
        return {
            "prompt_field": "prompt",
            "answer_key": "reward_model",
            "source_field": "data_source",
            "print_max_len": 1000,
        }
    
    def process_prompt(self, text: Any) -> Any:
        """处理prompt字段，从嵌套结构 ["prompt"][1]["content"] 中提取并清理文本"""
        # Handle nested field extraction ["prompt"][1]["content"]
        if isinstance(text, list) and len(text) == 1 and isinstance(text[0], dict):
            text = text[0].get("content", text)
        
        text = text.replace("Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n", "")
        text = text.replace("\n\nRemember to put your answer on its own line after \"Answer:\".", "")
        text = text.strip('\n')
        return text
    
    def process_answer(self, val: Any) -> Any:
        """处理answer字段，从嵌套结构 ["reward_model"]["ground_truth"] 中提取"""
        # Handle nested field extraction ["reward_model"]["ground_truth"]
        if isinstance(val, dict) and "ground_truth" in val:
            return val["ground_truth"]
        return val
    
    def process_solution(self, val: Any) -> Any:
        """处理solution字段"""
        return val


if __name__ == "__main__":
    processor = DAPOMath17KProcessor()
    processor.main()
