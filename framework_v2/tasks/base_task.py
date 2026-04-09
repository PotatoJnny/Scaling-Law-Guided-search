from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from core.data_structures import Action
from .action_parsers import ACTION_PARSERS
from .parsing_utils import sanitize_response_text

class BaseTask(ABC):
    def __init__(self, dataset_config: dict, action_strategy: dict, tokenizer=None):
        self.dataset_config = dataset_config
        self.action_strategy = action_strategy # Defines how to split LM output into actions/states
        self.tokenizer = tokenizer # Optional, only needed if using token_count strategy

    @abstractmethod
    def get_prompt(self, problem_data: dict) -> str:
        """Tasks must define how to format the dataset into a prompt."""
        pass

    @abstractmethod
    def extract_answer(self, text: str) -> str:
        """Tasks must define how to extract the final ground truth."""
        pass

    def build_result_metrics(
        self,
        problem_data: dict,
        search_result: Dict[str, Any],
        reward_source: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Task-specific result metrics added to results.json.
        Override this for tasks that need richer summaries than exact-match metrics.
        """
        return {}

    def serialize_search_artifacts(self, search_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Task-owned serialization of search traces for results.json.
        Override when storing full answer texts is unnecessary or too heavy.
        """
        return {
            "all_scores_and_answers": search_result.get("all_scores_and_answers", []),
        }

    def get_true_answer(self, problem_data: dict) -> Optional[str]:
        return None

    def build_core_result_fields(
        self,
        problem_data: dict,
        search_result: Dict[str, Any],
        evaluator: Any,
    ) -> Dict[str, Any]:
        """
        Task-owned core evaluation fields. By default, tasks without exact-match labels
        do not report pass@k-style metrics.
        """
        return {
            "true_answer": self.get_true_answer(problem_data),
            "pass_at_1": None,
            "pass_at_all": None,
            "majority_vote": None,
            "all_answers": [],
        }

    def prepare_dataset(self, dataset, cache_root: str):
        return dataset, {}

    def prepare_problem_data(
        self,
        problem_data: dict,
        runtime_context: Optional[Dict[str, Any]] = None,
        reward_source: Optional[Any] = None,
    ) -> Dict[str, Any]:
        return dict(problem_data)

    def sanitize_response_text(self, text: str) -> str:
        """Remove obvious generation artifacts before parsing."""
        return sanitize_response_text(text)

    def parse_response_to_actions(self, full_response: str) -> List[Action]:
        """
        Dispatch response parsing to the registered action parser.
        This keeps task code generic while allowing action strategies to evolve independently.
        """
        method = self.action_strategy.get("chunking_method")
        parser = ACTION_PARSERS.get(method)
        if parser is None:
            raise ValueError(f"Unknown chunking method: {method}")
        return parser.parse(self, full_response)
