from .base_task import BaseTask
from .if_state_utils import clean_answer_only_response, split_open_ended_blocks


_COMPACT_STATE_METADATA_KEYS = {
    "num_blocks",
    "block_lengths",
    "num_spans",
    "span_lengths",
    "state_mode",
    "chunking_path",
    "segmentation_path",
    "root_prefix_postselect_applied",
    "root_prefix_postselect_mode",
    "root_prefix_postselect_ratio",
    "root_prefix_postselect_num_tokens",
    "root_prefix_postselect_total_tokens",
    "root_prefix_postselect_source_actions",
    "root_prefix_postselect_min_ratio",
    "root_prefix_postselect_max_ratio",
    "segmentation_path_final",
    "join_delimiter",
    "num_paragraphs_before_fallback",
    "largest_block_fraction",
    "single_block_after_paragraph_split",
    "blocks_per_action",
    "target_depth",
    "contains_think",
    "changed_by_cleaning",
}

_TEST_ONLY_STATE_METADATA_KEYS = {
    "block_text_preview",
    "span_preview",
    "local_first_block_split",
    "local_first_block_split_test_mode",
    "prefix_normalization_applied",
    "fixed_prefix_ratio",
    "fixed_prefix_num_tokens",
    "fixed_prefix_num_blocks",
    "search_blocks_num_blocks",
    "repair_mode",
    "boundary_block_split_applied",
    "largest_block_fraction_before",
    "largest_block_fraction_after",
    "root_prefix_postselect_applied",
    "root_prefix_postselect_mode",
    "root_prefix_postselect_ratio",
    "root_prefix_postselect_num_tokens",
    "root_prefix_postselect_total_tokens",
    "root_prefix_postselect_source_actions",
    "root_prefix_postselect_min_ratio",
    "root_prefix_postselect_max_ratio",
}


class LanguageTask(BaseTask):
    """
    Task class for open-ended language generation (instruction following, summarization, etc.).
    There is no ground-truth answer — quality is measured by RM score and/or LLM judge score.
    """

    def __init__(self, dataset_config: dict, action_strategy: dict, tokenizer=None):
        super().__init__(dataset_config, action_strategy, tokenizer=tokenizer)

    def _is_open_ended_if(self) -> bool:
        return self.dataset_config.get("if_state_family", "open_ended_if") == "open_ended_if"

    def _structured_answer_test_enabled(self) -> bool:
        return self._is_open_ended_if() and bool(
            self.action_strategy.get("structured_answer_test_mode", False)
        )

    def _legacy_prompt_path_enabled(self) -> bool:
        return self._is_open_ended_if() and bool(
            self.action_strategy.get("legacy_prompt_path", False)
        )

    def _local_first_block_split_test_enabled(self) -> bool:
        return self._is_open_ended_if() and bool(
            self.action_strategy.get("local_first_block_split_test_mode", False)
        )

    def _prefix_normalization_test_enabled(self) -> bool:
        return self._is_open_ended_if() and bool(
            self.action_strategy.get("prefix_normalization_test_mode", False)
        )

    def _prefix_normalization_ratio_bounds(self) -> tuple[float, float]:
        return (
            float(self.action_strategy.get("prefix_normalization_min_ratio", 0.1)),
            float(self.action_strategy.get("prefix_normalization_max_ratio", 0.5)),
        )

    def _active_prompt_injection(self) -> str:
        base_injection = self.action_strategy.get("prompt_injection", "").strip()
        return base_injection

    def _apply_prompt_injection(self, row: dict) -> tuple[str, list | None, str]:
        injection = self._active_prompt_injection()
        prompt = row.get("instruction") or ""
        messages = row.get("messages")
        if not injection:
            return prompt, messages, ""

        if messages:
            updated_messages = [dict(msg) for msg in messages]
            for idx in range(len(updated_messages) - 1, -1, -1):
                if updated_messages[idx].get("role") == "user":
                    content = str(updated_messages[idx].get("content", "")).strip()
                    updated_messages[idx]["content"] = (
                        f"{injection}\n{content}" if content else injection
                    )
                    return prompt, updated_messages, injection
            return f"{injection}\n{prompt}".strip(), updated_messages, injection

        return f"{injection}\n{prompt}".strip(), messages, injection

    def get_prompt(self, problem_data: dict) -> str:
        precomputed_prompt = problem_data.get("prompt_text")
        if precomputed_prompt:
            return precomputed_prompt
        question_col = self.dataset_config["question_column"]
        instruction = problem_data[question_col]
        injection = self._active_prompt_injection()
        if injection:
            return f"{injection}\n{instruction}"
        return instruction

    def split_open_ended_response(self, full_response: str, root_level: bool = True):
        prefix_min_ratio, prefix_max_ratio = self._prefix_normalization_ratio_bounds()
        split = split_open_ended_blocks(
            full_response,
            tokenizer=self.tokenizer,
            max_blocks=4,
            local_first_block_split_test_mode=self._local_first_block_split_test_enabled(),
            prefix_normalization_test_mode=self._prefix_normalization_test_enabled(),
            prefix_min_ratio=prefix_min_ratio,
            prefix_max_ratio=prefix_max_ratio,
            root_level=root_level,
        )
        return split

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
        output_mode = runtime_context.get("output_mode", "answer_only_no_think")
        if llm is None:
            return row

        prompt, messages, injection = self._apply_prompt_injection(row)
        if self._legacy_prompt_path_enabled():
            # Pre-2026-04-16 prompt path: use the normalized instruction text
            # directly, without chat-template rendering.
            # Runtime safety guard: if the raw legacy prompt exceeds the model's
            # available input budget, truncate just enough to fit. This is a
            # no-op for prompts already within the budget.
            # Exception: for Qwen3 with answer_only_no_think, apply the chat
            # template solely to suppress thinking tokens (<think> chains).
            if output_mode == "answer_only_no_think" and "qwen3" in getattr(llm, "model_name", "").lower():
                rendered = llm.apply_chat_template(str(prompt), output_mode=output_mode)
                row["prompt_text"] = llm.truncate_prompt_to_fit(rendered, max_output_tokens=max_new_tokens)
            else:
                row["prompt_text"] = llm.truncate_prompt_to_fit(str(prompt), max_output_tokens=max_new_tokens)
            row["raw_prompt_text"] = prompt
            row["output_mode"] = output_mode
            row["search_state_family"] = self.dataset_config.get("if_state_family", "open_ended_if")
            row["structured_answer_test_mode"] = self._structured_answer_test_enabled()
            row["local_first_block_split_test_mode"] = self._local_first_block_split_test_enabled()
            row["prefix_normalization_test_mode"] = self._prefix_normalization_test_enabled()
            row["legacy_prompt_path"] = True
            row["injection_applied"] = bool(injection)
            row["prompt_injection_text"] = injection or ""
            self.output_mode = output_mode
            return row

        if messages:
            rendered_prompt = llm.render_messages(messages, output_mode=output_mode)
        elif prompt:
            rendered_prompt = llm.apply_chat_template(str(prompt), output_mode=output_mode)
        else:
            rendered_prompt = ""

        row["prompt_text"] = llm.truncate_prompt_to_fit(
            rendered_prompt,
            max_output_tokens=max_new_tokens,
        )
        row["raw_prompt_text"] = rendered_prompt
        row["output_mode"] = output_mode
        row["search_state_family"] = self.dataset_config.get("if_state_family", "open_ended_if")
        row["structured_answer_test_mode"] = self._structured_answer_test_enabled()
        row["local_first_block_split_test_mode"] = self._local_first_block_split_test_enabled()
        row["prefix_normalization_test_mode"] = self._prefix_normalization_test_enabled()
        row["legacy_prompt_path"] = False
        row["injection_applied"] = bool(injection)
        row["prompt_injection_text"] = injection or ""
        self.output_mode = output_mode
        return row

    def clean_model_output(self, raw_response: str, problem_data=None):
        output_mode = getattr(self, "output_mode", None)
        if problem_data:
            output_mode = problem_data.get("output_mode", output_mode)
        if output_mode == "answer_only_no_think":
            return clean_answer_only_response(raw_response)
        return super().clean_model_output(raw_response, problem_data=problem_data)

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

    def serialize_state_metadata(self, state_metadata: dict) -> dict:
        metadata = dict(state_metadata or {})
        keep = set(_COMPACT_STATE_METADATA_KEYS)
        if (
            self._structured_answer_test_enabled()
            or self._local_first_block_split_test_enabled()
            or self._prefix_normalization_test_enabled()
        ):
            keep.update(_TEST_ONLY_STATE_METADATA_KEYS)
        return {key: value for key, value in metadata.items() if key in keep}
