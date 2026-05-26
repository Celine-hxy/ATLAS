#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data_utils.base_processor import DatasetProcessor
from data_utils.tools import extract_solution

class GSM8KProcessor(DatasetProcessor):
    
    def get_default_hf_id(self) -> str:
        return "openai/gsm8k"
    
    @classmethod
    def get_default_args(cls):
        return {
            "prompt_field": "question",
            "answer_key": "answer",
            "source_field": None,
            "print_max_len": 2000,
            "keep_org_answer": True,
        }
    
    def process_prompt(self, text: Any) -> Any:
        """处理prompt字段"""
        return text
    
    def process_answer(self, val: Any) -> Any:
        """处理answer字段"""
        # val = extract_solution(val)
        val = val.split("\n#### ")[-1]
        return val
    
    def process_solution(self, val: Any) -> Any:
        """处理solution字段"""
        return val


if __name__ == "__main__":
    processor = GSM8KProcessor()
    processor.main()
