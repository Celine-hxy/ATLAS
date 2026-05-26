#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset: PRIME-RL/Eurus-2-RL-Data
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


class Eurus2RLDataProcessor(DatasetProcessor):
    """PRIME-RL/Eurus-2-RL-Data 数据集处理器"""
    
    def get_default_hf_id(self) -> str:
        return "PRIME-RL/Eurus-2-RL-Data"
    
    @classmethod
    def get_default_args(cls):
        return {
            "prompt_field": "prompt",
            "answer_key": "reward_model",
            "source_field": "data_source",
            "filter_column": "data_source",  # 过滤处理后的 source 列
            "filter_values": ["codeforces", "taco", "apps", "code_contests", "codecontests"],  # 丢弃这些 source 值的行
            "output_split": "math",
        }
    
    def process_prompt(self, text: Any) -> Any:
        """处理prompt字段，从嵌套结构 ["prompt"][1]["content"] 中提取并清理文本"""
        # Handle nested field extraction ["prompt"][1]["content"]
        if isinstance(text, list) and len(text) > 1 and isinstance(text[1], dict):
            text = text[1].get("content", text)
        
        # Clean text
        if isinstance(text, str):
            text = text.replace("\n\nPresent the answer in LaTex format: \\boxed{Your answer}", "")
            text = text.replace("Present the answer in LaTex format: \\boxed{Your answer}", "")
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
    processor = Eurus2RLDataProcessor()
    processor.main()
