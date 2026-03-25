from abc import ABC, abstractmethod
from typing import Any
from core.llm_engine import LLMEngine
from core.rm_engine import RMEngine
from tasks.base_task import BaseTask

class BaseAlgorithm(ABC):
    def __init__(
        self, 
        llm_engine: LLMEngine, 
        rm_engine: RMEngine, 
        task: BaseTask, 
        config: Any
    ):
        self.llm_engine = llm_engine
        self.rm_engine = rm_engine
        self.task = task
        self.config = config

    @abstractmethod
    def run(self, problem_data: dict) -> dict:
        """
        Executes the search algorithm on a single problem.
        Returns a dictionary containing the final answer, score, and logs.
        """
        pass