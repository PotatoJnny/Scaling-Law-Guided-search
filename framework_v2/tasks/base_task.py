from abc import ABC, abstractmethod
from core.data_structures import Action

class BaseTask(ABC):
    """Abstract base class that all tasks (Math, Code, etc.) must follow."""
    
    @abstractmethod
    def get_prompt(self, problem_data: dict) -> str:
        """Formats the dataset row into a prompt string."""
        pass

    @abstractmethod
    def extract_answer(self, raw_response: str) -> str:
        """Extracts the final answer from the model's raw text."""
        pass

    @abstractmethod
    def evaluate_correctness(self, model_answer: str, ground_truth: str) -> float:
        """Returns 1.0 for correct, 0.0 for incorrect."""
        pass

    @abstractmethod
    def parse_response_to_actions(self, full_response: str) -> List[Action]:
        """
        Takes a full generated string from the LLM and chunks it into 
        individual Actions (states) based on custom task logic.
        """
        pass