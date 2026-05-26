#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset: Kwai-Klear/KlearReasoner-MathSub-30K
Nested fields:
  - prompt: ["prompt"][0]["content"]
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


class KlearReasonerMathSub30KProcessor(DatasetProcessor):
    """Kwai-Klear/KlearReasoner-MathSub-30K 数据集处理器"""
    
    def get_default_hf_id(self) -> str:
        return "Kwai-Klear/KlearReasoner-MathSub-30K"
    
    @classmethod
    def get_default_args(cls):
        return {
            "prompt_field": "prompt",
            "answer_key": "reward_model",
            "source_field": "data_source",
        }
    
    def process_prompt(self, text: Any) -> Any:
        """处理prompt字段，从嵌套结构 ["prompt"][0]["content"] 中提取"""
        # Handle nested field extraction ["prompt"][0]["content"]
        if isinstance(text, list) and len(text) > 0 and isinstance(text[0], dict):
            return text[0].get("content", text)
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
    processor = KlearReasonerMathSub30KProcessor()
    processor.main()
