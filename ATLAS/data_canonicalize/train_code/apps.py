#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# newfacade/LeetCodeDataset
# BAAI/TACO
# codeparrot/apps
# deepmind/code_contests
# MatrixStudio/Codeforces-Python-Submissions
# TIGER-Lab/AceCode-87K

from typing import Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data_utils.base_processor import DatasetProcessor


class AppsProcessor(DatasetProcessor):
    """codeparrot/apps 数据集处理器"""
    
    def get_default_hf_id(self) -> str:
        return "codeparrot/apps"
    
    @classmethod
    def get_default_args(cls):
        return {
            "database_root": "$HOME/ATLAS/train/code",
            "prompt_field": "question",
            "answer_key": None,
            "source_field": None,
            "print_max_len": 2000,
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
    processor = AppsProcessor()
    processor.main()
