from abc import ABC, abstractmethod
from typing import Any, Dict
from core.llm_engine import LLMEngine
from core.base_reward import BaseRewardEngine
from tasks.base_task import BaseTask

class BaseAlgorithm(ABC):
    def __init__(
        self, 
        llm_engine: LLMEngine, 
        rm_engine: BaseRewardEngine, 
        task: BaseTask, 
        config: Any
    ):
        self.llm_engine = llm_engine
        self.rm_engine = rm_engine
        self.task = task
        self.config = config
        self._sampling_counter = 0

    def _get_rm_kwargs(self) -> Dict[str, Any]:
        action = self.task.action_strategy
        return {
            "rm_instruction": action.get("rm_instruction", None) or self.task.dataset_config.get("rm_instruction", None),
            "response_mode": action.get("rm_response_mode", None),
            "prompt_suffix_to_strip": action.get("rm_prompt_suffix_to_strip", None),
            "think_end_token": action.get("think_end_token", "</think>"),
        }

    def _next_sampling_seed(self):
        base_seed = getattr(self.config, "seed", None)
        if base_seed is None:
            return None
        seed = int(base_seed) + self._sampling_counter
        self._sampling_counter += 1
        return seed

    @abstractmethod
    def run(self, problem_data: dict) -> dict:
        """
        Executes the search algorithm on a single problem.
        Returns a dictionary containing the final answer, score, and logs.
        """
        pass
