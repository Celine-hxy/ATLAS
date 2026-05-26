#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data_utils.base_processor import DatasetProcessor
from data_utils.tools import extract_solution

class AReaLBobaDataProcessor(DatasetProcessor):
    """inclusionAI/AReaL-boba-Data 数据集处理器"""
    
    def get_default_hf_id(self) -> str:
        return "inclusionAI/AReaL-boba-Data"
    
    @classmethod
    def get_default_args(cls):
        return {
            "prompt_field": "prompt",
            "answer_key": "solutions",
            "source_field": None,
            "print_max_len": 2000,
        }
    
    def process_prompt(self, text: Any) -> Any:
        """处理prompt字段，清理文本"""
        if text is None:
            return None
        if not isinstance(text, str):
            return text

        s = text
        s = s.replace("<｜User｜>", "")
        # s = s.replace("\n<｜tool▁sep｜>", "")
        # s = s.replace("<｜tool▁sep｜>", "")
        s = s.replace("\nPlease reason step by step, and put your final answer within \\boxed{}.", "")
        s = s.replace("\nPlease reason step by step, and put your final answer within \\boxed{{}}.", "")
        s = s.replace("<｜Assistant｜><think>\n", "")
        s = s.replace("<｜Assistant｜>", "")
        s = s.replace("<think>\n", "")

        # if "Please reason step by step, and put your final answer within" in s:
        #     parts = s.split("Please reason step by step, and put your final answer within")
        #     if len(parts) > 0:
        #         s = "".join(parts[:-1]).strip("\n")

        s = s.strip("\n")
        return s
    
    def process_answer(self, val: Any) -> Any:
        """处理answer字段，从列表中提取solution"""
        try:
            return extract_solution(val[0])
        except Exception:
            return str(val)
    
    def process_solution(self, val: Any) -> Any:
        """处理solution字段"""
        return val


if __name__ == "__main__":
    processor = AReaLBobaDataProcessor()
    processor.main()
