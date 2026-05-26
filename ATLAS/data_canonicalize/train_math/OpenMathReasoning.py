#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data_utils.base_processor import DatasetProcessor


class OpenMathReasoningProcessor(DatasetProcessor):
    """nvidia/OpenMathReasoning 数据集处理器"""
    
    def get_default_hf_id(self) -> str:
        return "nvidia/OpenMathReasoning"
    
    @classmethod
    def get_default_args(cls):
        return {
            "prompt_field": "problem",
            "answer_key": "expected_answer",
            "source_field": "problem_source",
            "print_max_len": 5000,
        }
    
    def process_prompt(self, text: Any) -> Any:
        """处理prompt字段"""
        text = text.split("\n\nYOUR TASK\n\nProblem: ")[-1]
        # text = text.replace("You will be given a challenging math problem followed by 2 solutions. Your task is to systematically analyze these solutions to identify the most mathematically sound approach.\n\nInput Format:\nProblem: A complex mathematical word problem at advanced high school or college level\nSolutions: Detailed solutions indexed 0-1, each concluding with an answer in \\boxed{} notation\n\nYOUR TASK\n\nProblem:", "")
        return text
    
    def process_answer(self, val: Any) -> Any:
        """处理answer字段"""
        return val
    
    def process_solution(self, val: Any) -> Any:
        """处理solution字段"""
        return val


if __name__ == "__main__":
    processor = OpenMathReasoningProcessor()
    processor.main()
