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

    def _extract_boxed_content(self, text: str) -> str:
        """Extract content of last \\boxed{...}, correctly handling nested braces."""
        marker = r'\boxed{'
        last_start = -1
        idx = 0
        while True:
            pos = text.find(marker, idx)
            if pos == -1:
                break
            last_start = pos
            idx = pos + 1
        if last_start == -1:
            return ""
        start = last_start + len(marker)
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == '{':
                depth += 1
            elif text[pos] == '}':
                depth -= 1
            pos += 1
        return text[start:pos - 1].strip() if depth == 0 else ""

    def extract_answer(self, text: str) -> str:
        delimiter = self.dataset_config.get("primary_delimiter", "")
        if not delimiter or not text:
            return ""

        text = self.sanitize_response_text(text)
        if not text:
            return ""

        boxed_content = self._extract_boxed_content(text)
        if boxed_content:
            # For LaTeX answer types (e.g. MATH-500), keep the full expression intact
            if self.dataset_config.get("answer_type") == "latex":
                return boxed_content
            return self._extract_last_number(boxed_content) or boxed_content

        marker_patterns = [
            r"####\s*([^\n]*)",
            r"final answer(?: is|:)?\s*([^\n.]*)",
            r"the answer(?: is|:)?\s*([^\n.]*)",
            r"answer(?: is|:)?\s*([^\n.]*)",
        ]
        for pattern in marker_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                candidate = matches[-1].strip()
                extracted = self._extract_last_number(candidate)
                if extracted:
                    return extracted
                if candidate and not self._looks_like_step_heading(candidate):
                    return candidate

        if delimiter in text:
            parts = [part.strip() for part in text.split(delimiter) if part.strip()]
            for part in reversed(parts):
                if self._looks_like_step_heading(part):
                    continue
                extracted = self._extract_last_number(part)
                if extracted:
                    return extracted
                if part:
                    return part

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines[-5:]):
            if self._looks_like_step_heading(line):
                continue
            extracted = self._extract_last_number(line)
            if extracted:
                return extracted

        fallback = self.dataset_config.get("fallback_regex")
        if fallback:
            matches = re.findall(fallback, text)
            if matches:
                return matches[-1].strip()

        return self._extract_last_number(text) or ""

    def _extract_last_number(self, text: str) -> str:
        if not text:
            return ""
        matches = re.findall(r'-?\d[\d,]*(?:\.\d+)?', text)
        if not matches:
            return ""
        return matches[-1].replace(',', '')

    def _looks_like_step_heading(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return True
        return bool(re.match(r'^(?:#+\s*)?(?:step|##\s*step)\b[\s:.-]*$', text, flags=re.IGNORECASE))
