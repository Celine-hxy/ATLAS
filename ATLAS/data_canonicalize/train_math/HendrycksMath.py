#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset: meta-math/MetaMathQA
Fields:
  - prompt: original_question 或 query (两种配置)
  - answer: response
  - source: type
Prompt_NeedExtract: Y
Note: 该数据集有两行配置，支持两种prompt字段，通过命令行参数选择
"""

from typing import Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data_utils.base_processor import DatasetProcessor
from data_utils.tools import extract_solution


class HendrycksMathProcessor(DatasetProcessor):
    """meta-math/MetaMathQA 数据集处理器"""
    
    def get_default_hf_id(self) -> str:
        return "EleutherAI/hendrycks_math"


    
    @classmethod
    def get_default_args(cls):
        return {
            "prompt_field": "problem",
            "answer_key": "solution",
            "source_field": None,
            "print_max_len": 1000,
        }
    
    def process_prompt(self, text: Any) -> Any:
        """处理prompt字段"""
        return text
    
    def process_answer(self, val: Any) -> Any:
        """处理answer字段"""
        processed = extract_solution(val)
        return processed
    
    def process_solution(self, val: Any) -> Any:
        """处理solution字段"""
        return val


if __name__ == "__main__":
    processor = HendrycksMathProcessor()
    processor.main()
