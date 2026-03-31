import re
from .base_task import BaseTask

_FENCED_BLOCK = re.compile(r'```(?:python)?\s*\n(.*?)```', re.DOTALL)

CODE_PROMPT_TEMPLATE = (
    "You are given a Python program that is correct but slow. "
    "Your task is to optimize it to run faster while keeping the exact same input/output behavior.\n\n"
    "Original program:\n"
    "```python\n{original_code}\n```\n\n"
    "{injection}"
    "Provide the optimized Python program in a fenced ```python``` block."
)

CODE_PLAN_INJECTION = (
    "First write a comment block starting with `# PLAN:` explaining your optimization strategy "
    "(e.g., algorithmic improvement, data structure change, vectorization). "
    "Then provide the optimized code.\n\n"
)


class CodeTask(BaseTask):
    """
    Task for code optimization using the PIE dataset.
    Score comes from CodeExecutor (actual execution), not a reward model.
    """

    def get_prompt(self, problem_data: dict) -> str:
        original_code = problem_data["original_code"]
        injection = self.action_strategy.get("prompt_injection", CODE_PLAN_INJECTION)
        return CODE_PROMPT_TEMPLATE.format(
            original_code=original_code,
            injection=injection,
        )

    def extract_answer(self, text: str) -> str:
        """Extract the last fenced code block, or the raw text if it looks like code."""
        matches = _FENCED_BLOCK.findall(text)
        if matches:
            return matches[-1].strip()
        return self.sanitize_response_text(text)
