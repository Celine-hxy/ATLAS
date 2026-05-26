#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data_utils.base_processor import DatasetProcessor
from data_utils.tools import extract_solution

class AMOBenchProcessor(DatasetProcessor):
    
    def get_default_hf_id(self) -> str:
        return "hf-imo-colab/AMO-Bench"
        # return "meituan-longcat/AMO-Bench"
    
    @classmethod
    def get_default_args(cls):
        return {
            "database_root": "$HOME/ATLAS/test/math",
            "prompt_field": "prompt",
            "answer_key": "answer",
            "source_field": None,
            "print_max_len": 2000,
        }
    
    def process_prompt(self, text: Any) -> Any:
        """处理prompt字段"""
        text = text.replace("\nAfter solving the above problem, please output your final answer in the following format:\n### The final answer is: $\\boxed{<your answer>}$\nExample:\n### The final answer is: $\\boxed{123}$\nThe final answer should be given as precisely as possible (using LaTeX symbols such as \\sqrt, \\frac, \\pi, etc.). If the final answer involves a decimal approximation, it must be accurate to at least four decimal places.", "")
        text = text.replace("\nAfter solving the above problem, please output your final answer in the following format:\n### The final answer is: $\\boxed{<your answer>}$\nExample:\n### The final answer is: $\\boxed{2n+1}$\nYour final answer must be a formula containing n.\nDo not include any restrictions on the value of n in your final answer.\nThe final answer should be given as precisely as possible (using LaTeX symbols such as \\sqrt, \\frac, \\pi, etc.). If the final answer involves a decimal approximation, it must be accurate to at least four decimal places.", "")
        return text
    
    def process_answer(self, val: Any) -> Any:
        """处理answer字段"""
        try:
            val = extract_solution(val)
        except:
            # Some entries may not be valid answers and do not contain \\boxed{}, so extraction might fail.
            # print(val)
            # input()
            return ""
        return val
    
    def process_solution(self, val: Any) -> Any:
        """处理solution字段"""
        return val


if __name__ == "__main__":
    processor = AMOBenchProcessor()
    processor.main()
