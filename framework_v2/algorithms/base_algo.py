from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
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
            "secondary_rm_instruction": action.get("secondary_rm_instruction", None),
            "primary_rm_weight": action.get("primary_rm_weight", 1.0),
            "secondary_rm_weight": action.get("secondary_rm_weight", 0.0),
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

    def _encode_text(self, text: str) -> list[int]:
        tokenizer = getattr(self.llm_engine, "tokenizer", None)
        if tokenizer is None:
            return list(range(len((text or "").split())))
        try:
            return tokenizer.encode(text or "", add_special_tokens=False)
        except TypeError:
            return tokenizer.encode(text or "")

    def _token_len(self, text: str) -> int:
        return len(self._encode_text(text))

    def _strict_total_output_tokens(self) -> int | None:
        value = getattr(self.config, "strict_total_output_tokens", None)
        if value is None:
            return None
        value = int(value)
        return value if value > 0 else None

    def _state_response_tokens(self, state) -> int:
        return self._token_len(state.get_full_response())

    def _max_generation_tokens_for_state(self, state, per_call_max_tokens: int | None = None) -> int:
        per_call_limit = int(
            per_call_max_tokens
            if per_call_max_tokens is not None
            else getattr(self.config, "max_tokens", 2048)
        )
        total_limit = self._strict_total_output_tokens()
        if total_limit is None:
            return per_call_limit
        remaining = total_limit - self._state_response_tokens(state)
        return max(0, min(per_call_limit, remaining))

    def _state_within_prefix_window(self, state) -> bool:
        total_limit = self._strict_total_output_tokens()
        if total_limit is None:
            return True

        prefix_tokens = self._state_response_tokens(state)
        min_ratio = getattr(self.config, "strict_prefix_min_ratio", None)
        max_ratio = getattr(self.config, "strict_prefix_max_ratio", None)

        if min_ratio is not None and prefix_tokens < total_limit * float(min_ratio):
            return False
        if max_ratio is not None and prefix_tokens > total_limit * float(max_ratio):
            return False
        return True

    def _root_prefix_normalization_overrides(self) -> Dict[str, Any] | None:
        total_limit = self._strict_total_output_tokens()
        min_ratio = getattr(self.config, "strict_prefix_min_ratio", None)
        max_ratio = getattr(self.config, "strict_prefix_max_ratio", None)
        if total_limit is None or min_ratio is None or max_ratio is None:
            return None
        if self.task.action_strategy.get("chunking_method") != "if_answer_only":
            return None
        return {
            "prefix_normalization_test_mode": True,
            "prefix_normalization_min_ratio": float(min_ratio),
            "prefix_normalization_max_ratio": float(max_ratio),
        }

    def _root_prefix_postselect_disabled(self) -> bool:
        return bool(getattr(self.config, "disable_root_prefix_postselect", False))

    @abstractmethod
    def run(self, problem_data: dict) -> dict:
        """
        Executes the search algorithm on a single problem.
        Returns a dictionary containing the final answer, score, and logs.
        """
        pass
