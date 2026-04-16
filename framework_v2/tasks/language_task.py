from .base_task import BaseTask


class LanguageTask(BaseTask):
    """
    Task class for open-ended language generation (instruction following, summarization, etc.).
    There is no ground-truth answer — quality is measured by RM score and/or LLM judge score.
    """

    def get_prompt(self, problem_data: dict) -> str:
        precomputed_prompt = problem_data.get("prompt_text")
        if precomputed_prompt:
            return precomputed_prompt
        question_col = self.dataset_config["question_column"]
        instruction = problem_data[question_col]
        injection = self.action_strategy.get("prompt_injection", "")
        if injection:
            return f"{injection}\n{instruction}"
        return instruction

    def prepare_problem_data(
        self,
        problem_data: dict,
        runtime_context=None,
        reward_source=None,
    ) -> dict:
        row = dict(problem_data)
        runtime_context = runtime_context or {}
        llm = runtime_context.get("llm_engine")
        max_new_tokens = runtime_context.get("generation_max_new_tokens", 1024)
        if llm is None:
            return row

        prompt = row.get("instruction") or ""
        messages = row.get("messages")
        if messages:
            prompt = llm.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        elif prompt:
            prompt = llm.apply_chat_template(str(prompt))

        row["prompt_text"] = llm.truncate_prompt_to_fit(
            prompt,
            max_output_tokens=max_new_tokens,
        )
        return row

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
