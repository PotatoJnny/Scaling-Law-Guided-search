# core/data_structures.py

from dataclasses import dataclass
from typing import List

@dataclass
class Action:
    """Represents one reasoning step (a generated piece of text)."""
    step_text: str
    is_final: bool = False

@dataclass
class State:
    """
    Represents the current state of the generation process, including the
    initial prompt and all subsequent actions taken.
    """
    prompt: str
    steps: List[Action]
    is_complete: bool = False

    def get_full_text(self) -> str:
        """
        Concatenates the prompt and all previous steps to form the full context
        for the LLM.
        """
        full_text = self.prompt
        for action in self.steps:
            full_text += action.step_text
        return full_text

    def get_full_response(self) -> str:
        """
        Concatenates all steps to form the full response generated so far.
        """
        full_response = ""
        for action in self.steps:
            full_response += action.step_text
        return full_response

    def append_step(self, action: Action):
        """
        Appends a new action to the steps and updates the completion status.
        """
        self.steps.append(action)
        if action.is_final:
            self.is_complete = True

    def print_state(self):
        """
        Prints the current state in a readable format.
        """
        for i, action in enumerate(self.steps):
           print(self.steps[i].step_text + '\n')
