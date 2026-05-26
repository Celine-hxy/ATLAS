#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset: AI-MO/NuminaMath-CoT
Fields:
  - prompt: problem
  - answer: solution
  - source: source
Answer_NeedExtract: Y
"""

from typing import Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data_utils.base_processor import DatasetProcessor
from data_utils.tools import extract_solution
from data_utils.tools import add_extraction_failed_count, get_extraction_failed_count

class NuminaMathCoTProcessor(DatasetProcessor):
    """AI-MO/NuminaMath-CoT 数据集处理器"""
    
    def get_default_hf_id(self) -> str:
        return "AI-MO/NuminaMath-CoT"
    
    @classmethod
    def get_default_args(cls):
        return {
            # "database_root": "$HOME/ATLAS/test/math",
            "prompt_field": "problem",
            "answer_key": "solution",
            "source_field": "source",
            "print_max_len": 2000,
        }
    
    def process_prompt(self, text: Any) -> Any:
        """处理prompt字段"""
        return text
    
    def process_answer(self, val: Any) -> Any:
        """处理answer字段"""
        try:
            extracted = extract_solution(val)
            return extracted
        except:
            add_extraction_failed_count()
            print(f"[WARNING] Failed to extract solution from {val}")
            # input()
            return val
        
    
    def process_solution(self, val: Any) -> Any:
        """处理solution字段"""
        return val


if __name__ == "__main__":
    processor = NuminaMathCoTProcessor()
    processor.main()
    print(f"[INFO] Extraction failed count: {get_extraction_failed_count()}")
