#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset: RUC-AIBOX/STILL-3-RL-90K
Nested fields:
  - prompt: ["prompt"][0]["content"]
  - answer: ["reward_model"]["ground_truth"]
  - source: N/A
Prompt_NeedExtract: Y
"""

from typing import Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data_utils.base_processor import DatasetProcessor


class STILL3RL90KProcessor(DatasetProcessor):
    """RUC-AIBOX/STILL-3-RL-90K 数据集处理器"""
    
    def get_default_hf_id(self) -> str:
        return "RUC-AIBOX/STILL-3-RL-90K"
    
    @classmethod
    def get_default_args(cls):
        return {
            "prompt_field": "prompt",
            "answer_key": "reward_model",
            "source_field": None,
        }
    
    def process_prompt(self, text: Any) -> Any:
        """处理prompt字段，从嵌套结构 ["prompt"][0]["content"] 中提取并清理文本"""
        # Handle nested field extraction ["prompt"][0]["content"]
        if isinstance(text, list) and len(text) > 0 and isinstance(text[0], dict):
            text = text[0].get("content", text)
        
        # Clean text
        if isinstance(text, str):
            text = text.replace("A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first engages in an internal reasoning process, akin to a stream of consciousness, before providing the User with the answer. The reasoning process and answer are enclosed within `<think></think>` and `<answer></answer>` tags, respectively. For example:\n\n```\n<think>\nreasoning process here\n</think>\n<answer>\nanswer here\n</answer>\n```\n\nThe reasoning process includs detailed considerations such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of current steps, refining any errors, and revisiting previous steps. During this process, the Assistant uses casual, genuine phrases such as: \"Hmm\", \"Wait\", \"Alternatively\", \"double check\", \"I wonder...\", \"But\", \"rethink\", etc., to make the reasoning process coherent, clear, and logically sound, effectively simulating human cognitive processes.\n\nThe Assistant shows the reasoning process within `<think></think>` tags, and ONLY return the FINAL ANSWER within `<answer></answer>` tags. For example: `<answer> \\frac{1}{2} </answer>`.\n\nUser: ", "")
            text = text.replace("\nAssistant:\n<think>\n", "")
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
    processor = STILL3RL90KProcessor()
    processor.main()





