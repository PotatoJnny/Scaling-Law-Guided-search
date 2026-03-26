import re
from .base_task import BaseTask

class MathTask(BaseTask):

    def get_prompt(self, problem_data: dict) -> str:
        base = self.dataset_config["base_prompt"].format(
            question=problem_data[self.dataset_config["question_column"]]
        )
        injection = self.action_strategy.get("prompt_injection", "")
        return base.replace("Problem:", f"{injection}\nProblem:")

    def extract_answer(self, text: str) -> str:
        delimiter = self.dataset_config.get("primary_delimiter", "")
        if not delimiter or not text: return ""

        if delimiter == "\\boxed{":
            match = re.search(r'\\boxed{([^}]*)}', text)
            if match: return match.group(1).strip()
            
        elif delimiter in text:
            return text.split(delimiter)[-1].strip()

        fallback = self.dataset_config.get("fallback_regex")
        if fallback:
            matches = re.findall(fallback, text)
            if matches: return matches[-1].strip()

        return ""
