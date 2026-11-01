from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .data_structures import State


class BaseRewardEngine(ABC):
    @abstractmethod
    def score_states_batch(self, states: List[State], **kwargs) -> List[float]:
        pass

    def set_problem(self, problem_data: Dict[str, Any]) -> None:
        return None

    def inspect_text(self, text: str) -> Dict[str, Any]:
        return {}
