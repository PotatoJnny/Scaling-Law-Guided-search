from .base_task import BaseTask


class LanguageTask(BaseTask):
    """
    Task class for open-ended language generation (instruction following, summarization, etc.).
    There is no ground-truth answer — quality is measured by RM score and/or LLM judge score.
    """

    def get_prompt(self, problem_data: dict) -> str:
        question_col = self.dataset_config["question_column"]
        instruction = problem_data[question_col]
        injection = self.action_strategy.get("prompt_injection", "")
        if injection:
            return f"{injection}\n{instruction}"
        return instruction

    def extract_answer(self, text: str) -> str:
        """Return the full response text. No extraction needed for language tasks."""
        return self.sanitize_response_text(text)

    def serialize_search_artifacts(self, search_result: dict) -> dict:
        return {
            "all_scores_and_answers": [
                {"score": x["score"]}
                for x in search_result.get("all_scores_and_answers", [])
            ]
        }
