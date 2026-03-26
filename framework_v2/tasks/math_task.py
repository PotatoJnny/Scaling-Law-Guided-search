import re
from .base_task import BaseTask

class MathTask(BaseTask):

    def get_prompt(self, problem_data: dict) -> str:
        base = self.dataset_config["base_prompt"].format(
            question=problem_data[self.dataset_config["question_column"]]
        )
        injection = self.action_strategy.get("prompt_injection", "")
        if injection:
            return base.replace("Problem:", f"{injection}\nProblem:")
        return base

    def extract_answer(self, text: str) -> str:
        delimiter = self.dataset_config.get("primary_delimiter", "")
        if not delimiter or not text: return ""

        if delimiter == "\\boxed{":
            match = re.search(r'\\boxed{([^}]*)}', text)
            if match: return match.group(1).strip()
            
        elif delimiter in text:
            part = text.split(delimiter)[-1].strip()
            match = re.search(r'-?[\d,]+\.?\d*', part)
            return match.group(0).replace(',', '') if match else part

        fallback = self.dataset_config.get("fallback_regex")
        if fallback:
            matches = re.findall(fallback, text)
            if matches: return matches[-1].strip()

        return ""
