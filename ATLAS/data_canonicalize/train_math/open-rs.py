#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data_utils.base_processor import DatasetProcessor


class OpenRSProcessor(DatasetProcessor):
    """knoveleng/open-rs 数据集处理器"""
    
    def get_default_hf_id(self) -> str:
        return "knoveleng/open-rs"
    
    @classmethod
    def get_default_args(cls):
        return {
            "prompt_field": "problem",
            "answer_key": None,
            "source_field": None,
            "print_max_len": 1000,
        }
    
    def process_prompt(self, text: Any) -> Any:
        """处理prompt字段"""
        return text
    
    def process_answer(self, val: Any) -> Any:
        """处理answer字段"""
        return val
    
    def process_solution(self, val: Any) -> Any:
        """处理solution字段"""
        return val


if __name__ == "__main__":
    processor = OpenRSProcessor()
    processor.main()
