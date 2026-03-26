import re
from abc import ABC, abstractmethod
from typing import List, Optional
from core.data_structures import Action

class BaseTask(ABC):
    def __init__(self, dataset_config: dict, action_strategy: dict, tokenizer=None):
        self.dataset_config = dataset_config
        self.action_strategy = action_strategy
        self.tokenizer = tokenizer # Optional, only needed if using token_count strategy

    @abstractmethod
    def get_prompt(self, problem_data: dict) -> str:
        """Tasks must define how to format the dataset into a prompt."""
        pass

    @abstractmethod
    def extract_answer(self, text: str) -> str:
        """Tasks must define how to extract the final ground truth."""
        pass

    def parse_response_to_actions(self, full_response: str) -> List[Action]:
        """
        Universal parsing logic based on the chosen Action Strategy.
        Works for Math, Code, Logic, etc.
        """
        method = self.action_strategy.get("chunking_method")
        actions = []

        if method == "delimiter":
            delimiter = self.action_strategy["delimiter"]
            chunks = [s for s in full_response.split(delimiter) if s.strip()]
            for i, chunk in enumerate(chunks):
                is_last = (i == len(chunks) - 1)
                text = chunk + delimiter if not is_last else chunk
                actions.append(Action(step_text=text, is_final=is_last))

        elif method == "regex":
            pattern = self.action_strategy["regex_pattern"]
            chunks = [s for s in re.split(pattern, full_response) if s.strip()]
            for i, chunk in enumerate(chunks):
                is_last = (i == len(chunks) - 1)
                actions.append(Action(step_text=chunk, is_final=is_last))

        elif method == "token_count":
            if not self.tokenizer:
                raise ValueError("Tokenizer must be provided to BaseTask for token_count strategy.")
            tokens = self.tokenizer.encode(full_response)
            chunk_size = self.action_strategy["token_count"]
            
            for i in range(0, len(tokens), chunk_size):
                chunk_tokens = tokens[i : i + chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                is_last = (i + chunk_size >= len(tokens))
                actions.append(Action(step_text=chunk_text, is_final=is_last))

        else:
            raise ValueError(f"Unknown chunking method: {method}")

        return actions