#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset: RUC-AIBOX/long_form_thought_data_5k
Fields:
  - prompt: question
  - answer: combined_text
  - source: domain
Answer_NeedExtract: Y
"""

from typing import Any
from data_utils import extract_solution
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data_utils.base_processor import DatasetProcessor


class LongFormThoughtData5KProcessor(DatasetProcessor):
    """RUC-AIBOX/long_form_thought_data_5k 数据集处理器"""
    
    def get_default_hf_id(self) -> str:
        return "RUC-AIBOX/long_form_thought_data_5k"
    
    @classmethod
    def get_default_args(cls):
        return {
            "prompt_field": "question",
            "answer_key": "combined_text",
            "source_field": "domain",
        }
    
    def process_prompt(self, text: Any) -> Any:
        """处理prompt字段"""
        return text
    
    def process_answer(self, val: Any) -> Any:
        """处理answer字段，从combined_text中提取答案"""
        try:
            return extract_solution(val)
        except:
            return ""
    
    def process_solution(self, val: Any) -> Any:
        """处理solution字段"""
        return val


if __name__ == "__main__":
    processor = LongFormThoughtData5KProcessor()
    processor.main()
