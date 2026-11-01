import re
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from core.data_structures import Action, State
from .action_parsers import ACTION_PARSERS
from .parsing_utils import sanitize_response_text


ROOT_STATE_METADATA_KEYS = {
    "num_blocks",
    "block_lengths",
    "block_text_preview",
    "state_mode",
    "chunking_path",
    "segmentation_path_final",
    "join_delimiter",
    "num_paragraphs_before_fallback",
    "largest_block_fraction",
    "single_block_after_paragraph_split",
    "blocks_per_action",
    "target_depth",
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


class BaseTask(ABC):
    def __init__(self, dataset_config: dict, action_strategy: dict, tokenizer=None):
        self.dataset_config = dataset_config
        self.action_strategy = action_strategy # Defines how to split LM output into actions/states
        self.tokenizer = tokenizer # Optional, only needed if using token_count strategy

    @contextmanager
    def _temporary_action_strategy_overrides(self, overrides: Optional[dict] = None):
        if not overrides:
            yield
            return
        original = self.action_strategy
        self.action_strategy = {
            **self.action_strategy,
            **overrides,
        }
        try:
            yield
        finally:
            self.action_strategy = original

    @abstractmethod
    def get_prompt(self, problem_data: dict) -> str:
        """Tasks must define how to format the dataset into a prompt."""
        pass

    @abstractmethod
    def extract_answer(self, text: str) -> str:
        """Tasks must define how to extract the final ground truth."""
        pass

    def build_result_metrics(
        self,
        problem_data: dict,
        search_result: Dict[str, Any],
        reward_source: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Task-specific result metrics added to results.json.
        Override this for tasks that need richer summaries than exact-match metrics.
        """
        return {}

    def serialize_search_artifacts(self, search_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Task-owned serialization of search traces for results.json.
        Override when storing full answer texts is unnecessary or too heavy.
        """
        return {
            "all_scores_and_answers": search_result.get("all_scores_and_answers", []),
        }

    def serialize_state_metadata(self, state_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Task-owned filtering of per-state parsing metadata before it is written to
        results.json. Tasks can override this to keep the default reports compact.
        """
        return dict(state_metadata or {})

    def get_true_answer(self, problem_data: dict) -> Optional[str]:
        return None

    def build_core_result_fields(
        self,
        problem_data: dict,
        search_result: Dict[str, Any],
        evaluator: Any,
    ) -> Dict[str, Any]:
        """
        Task-owned core evaluation fields. By default, tasks without exact-match labels
        do not report pass@k-style metrics.
        """
        return {
            "true_answer": self.get_true_answer(problem_data),
            "pass_at_1": None,
            "pass_at_all": None,
            "majority_vote": None,
            "all_answers": [],
        }

    def prepare_dataset(self, dataset, cache_root: str):
        return dataset, {}

    def prepare_problem_data(
        self,
        problem_data: dict,
        runtime_context: Optional[Dict[str, Any]] = None,
        reward_source: Optional[Any] = None,
    ) -> Dict[str, Any]:
        return dict(problem_data)

    def sanitize_response_text(self, text: str) -> str:
        """Remove obvious generation artifacts before parsing."""
        return sanitize_response_text(text)

    def clean_model_output(
        self,
        raw_response: str,
        problem_data: Optional[dict] = None,
    ) -> Dict[str, Any]:
        clean_response = self.sanitize_response_text(raw_response)
        return {
            "raw_response": raw_response,
            "clean_response": clean_response,
            "contains_think": False,
            "changed_by_cleaning": clean_response != (raw_response or "").strip(),
        }

    def _encode_text(self, text: str) -> List[int]:
        if self.tokenizer is None:
            return list(range(len((text or "").split())))
        try:
            return self.tokenizer.encode(text or "", add_special_tokens=False)
        except TypeError:
            return self.tokenizer.encode(text or "")

    def _decode_tokens(self, token_ids: List[int]) -> str:
        if self.tokenizer is None:
            return " ".join(str(tok) for tok in token_ids)
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        except TypeError:
            return self.tokenizer.decode(token_ids)

    def _token_len(self, text: str) -> int:
        return len(self._encode_text(text))

    @staticmethod
    def _interval_gap(value: float, low: float, high: float) -> float:
        if low <= value <= high:
            return 0.0
        return min(abs(value - low), abs(value - high))

    def _best_prefix_split(
        self,
        text: str,
        *,
        total_tokens: int,
        prefix_tokens_before: int,
        prefix_min_ratio: float,
        prefix_max_ratio: float,
    ) -> Optional[Dict[str, Any]]:
        if not text or total_tokens <= 0:
            return None

        best: Optional[Dict[str, Any]] = None
        seen_splits = set()
        boundary_patterns = [
            r"(?<=[.!?])\s+",
            r"\n\s*\n+",
            r"(?<=;)\s+",
            r"(?<=:)\s+",
            r"(?<=,)\s+",
        ]

        def consider(prefix_text: str, remainder_text: str, mode: str) -> None:
            nonlocal best
            prefix_text = prefix_text.rstrip()
            remainder_text = remainder_text.lstrip()
            if not prefix_text or not remainder_text:
                return
            prefix_tokens = prefix_tokens_before + self._token_len(prefix_text)
            ratio = prefix_tokens / total_tokens
            candidate = {
                "prefix_text": prefix_text,
                "remainder_text": remainder_text,
                "prefix_tokens": prefix_tokens,
                "ratio": ratio,
                "mode": mode,
                "exact": prefix_min_ratio <= ratio <= prefix_max_ratio,
                "gap": self._interval_gap(ratio, prefix_min_ratio, prefix_max_ratio),
            }
            if best is None:
                best = candidate
                return
            if candidate["exact"] and not best["exact"]:
                best = candidate
                return
            if candidate["exact"] == best["exact"]:
                if candidate["gap"] < best["gap"] - 1e-9:
                    best = candidate
                    return
                if abs(candidate["gap"] - best["gap"]) <= 1e-9 and candidate["prefix_tokens"] < best["prefix_tokens"]:
                    best = candidate

        for pattern in boundary_patterns:
            for match in re.finditer(pattern, text):
                key = (match.start(), match.end())
                if key in seen_splits:
                    continue
                seen_splits.add(key)
                consider(text[: match.start()], text[match.end() :], "split_text_boundary")

        token_ids = self._encode_text(text)
        if len(token_ids) >= 2:
            for split_idx in range(1, len(token_ids)):
                prefix_text = self._decode_tokens(token_ids[:split_idx])
                remainder_text = self._decode_tokens(token_ids[split_idx:])
                consider(prefix_text, remainder_text, "split_token_boundary")

        return best

    def _select_postselected_root_prefix(
        self,
        action_texts: List[str],
        *,
        total_tokens: int,
        prefix_min_ratio: float,
        prefix_max_ratio: float,
    ) -> Optional[Dict[str, Any]]:
        if not action_texts or total_tokens <= 0:
            return None

        best: Optional[Dict[str, Any]] = None

        def consider(prefix_text: str, mode: str, source_actions: int) -> Optional[Dict[str, Any]]:
            nonlocal best
            if not prefix_text:
                return None
            prefix_tokens = self._token_len(prefix_text)
            ratio = prefix_tokens / total_tokens
            candidate = {
                "prefix_text": prefix_text,
                "prefix_tokens": prefix_tokens,
                "ratio": ratio,
                "mode": mode,
                "source_actions": source_actions,
                "exact": prefix_min_ratio <= ratio <= prefix_max_ratio,
                "gap": self._interval_gap(ratio, prefix_min_ratio, prefix_max_ratio),
            }
            if best is None:
                best = candidate
            elif candidate["exact"] and not best["exact"]:
                best = candidate
            elif candidate["exact"] == best["exact"]:
                if candidate["gap"] < best["gap"] - 1e-9:
                    best = candidate
                elif abs(candidate["gap"] - best["gap"]) <= 1e-9 and candidate["prefix_tokens"] < best["prefix_tokens"]:
                    best = candidate
            return candidate

        prefix_text = action_texts[0]
        prefix_tokens = self._token_len(prefix_text)
        first_candidate = consider(prefix_text, "original_first_action", 1)
        if first_candidate and first_candidate["exact"]:
            return first_candidate

        if first_candidate and first_candidate["ratio"] > prefix_max_ratio:
            split = self._best_prefix_split(
                action_texts[0],
                total_tokens=total_tokens,
                prefix_tokens_before=0,
                prefix_min_ratio=prefix_min_ratio,
                prefix_max_ratio=prefix_max_ratio,
            )
            if split:
                candidate = consider(split["prefix_text"], split["mode"], 1)
                if candidate and candidate["exact"]:
                    return candidate
            return best

        for idx, next_text in enumerate(action_texts[1:], start=1):
            next_tokens = self._token_len(next_text)
            if (prefix_tokens + next_tokens) / total_tokens > prefix_max_ratio:
                split = self._best_prefix_split(
                    next_text,
                    total_tokens=total_tokens,
                    prefix_tokens_before=prefix_tokens,
                    prefix_min_ratio=prefix_min_ratio,
                    prefix_max_ratio=prefix_max_ratio,
                )
                if split:
                    candidate = consider(
                        prefix_text + split["prefix_text"],
                        split["mode"],
                        idx + 1,
                    )
                    if candidate and candidate["exact"]:
                        return candidate
                break

            prefix_text += next_text
            prefix_tokens += next_tokens
            candidate = consider(prefix_text, "grow_prefix_whole_actions", idx + 1)
            if candidate and candidate["exact"]:
                return candidate

        return best

    def build_postselected_root_branch_state(
        self,
        base_state: State,
        candidate_state: State,
        *,
        prefix_min_ratio: float,
        prefix_max_ratio: float,
    ) -> State:
        action_texts = [action.step_text for action in candidate_state.steps if action.step_text]
        full_response = candidate_state.get_full_response()
        total_tokens = self._token_len(full_response)
        selection = self._select_postselected_root_prefix(
            action_texts,
            total_tokens=total_tokens,
            prefix_min_ratio=prefix_min_ratio,
            prefix_max_ratio=prefix_max_ratio,
        )
        if not selection:
            return candidate_state

        metadata = dict(candidate_state.state_metadata)
        metadata.update(
            {
                "root_prefix_postselect_applied": selection["mode"] != "original_first_action",
                "root_prefix_postselect_mode": selection["mode"],
                "root_prefix_postselect_ratio": selection["ratio"],
                "root_prefix_postselect_num_tokens": selection["prefix_tokens"],
                "root_prefix_postselect_total_tokens": total_tokens,
                "root_prefix_postselect_source_actions": selection["source_actions"],
                "root_prefix_postselect_min_ratio": float(prefix_min_ratio),
                "root_prefix_postselect_max_ratio": float(prefix_max_ratio),
            }
        )

        if candidate_state.steps and selection["mode"] == "original_first_action":
            candidate_state.state_metadata = metadata
            return candidate_state

        branch_state = base_state.get_truncated_copy(len(base_state.steps))
        is_full_response = selection["prefix_text"] == full_response
        branch_state.append_step(
            Action(
                step_text=selection["prefix_text"],
                is_final=is_full_response and candidate_state.is_complete,
            )
        )
        branch_state.raw_response = candidate_state.raw_response or full_response
        branch_state.clean_response = selection["prefix_text"]
        metadata.update(
            {
                "reconstructed_response": branch_state.get_full_response(),
                "reconstructed_response_includes_prefix": False,
            }
        )
        branch_state.state_metadata = metadata
        return branch_state

    def parse_response_to_actions_with_metadata(
        self,
        full_response: str,
        base_state: Optional[State] = None,
        action_strategy_overrides: Optional[dict] = None,
    ):
        with self._temporary_action_strategy_overrides(action_strategy_overrides):
            method = self.action_strategy.get("chunking_method")
            parser = ACTION_PARSERS.get(method)
            if parser is None:
                raise ValueError(f"Unknown chunking method: {method}")
            return parser.parse_with_metadata(self, full_response, base_state=base_state)

    def parse_response_to_actions(
        self,
        full_response: str,
        base_state: Optional[State] = None,
        action_strategy_overrides: Optional[dict] = None,
    ) -> List[Action]:
        """
        Dispatch response parsing to the registered action parser.
        This keeps task code generic while allowing action strategies to evolve independently.
        """
        actions, _ = self.parse_response_to_actions_with_metadata(
            full_response,
            base_state=base_state,
            action_strategy_overrides=action_strategy_overrides,
        )
        return actions

    def build_state_from_raw_response(
        self,
        base_state: State,
        raw_response: str,
        problem_data: Optional[dict] = None,
        action_strategy_overrides: Optional[dict] = None,
    ) -> State:
        clean_info = self.clean_model_output(raw_response, problem_data=problem_data)
        state = base_state.get_truncated_copy(len(base_state.steps))
        actions, parse_metadata = self.parse_response_to_actions_with_metadata(
            clean_info["clean_response"],
            base_state=base_state,
            action_strategy_overrides=action_strategy_overrides,
        )
        if not actions:
            actions = [Action(step_text=clean_info["clean_response"], is_final=True)]
        for action in actions:
            state.append_step(action)
        state.raw_response = clean_info["raw_response"]
        segmented_response = "".join(action.step_text for action in actions)
        state.clean_response = clean_info["clean_response"]
        parse_metadata = self._merge_root_state_metadata(
            inherited_metadata=base_state.state_metadata,
            new_metadata=parse_metadata,
            has_prefix=bool(base_state.steps),
        )
        state.state_metadata = {
            **parse_metadata,
            "contains_think": clean_info.get("contains_think", False),
            "changed_by_cleaning": clean_info.get("changed_by_cleaning", False),
            "clean_response_changed_by_segmentation": segmented_response != clean_info["clean_response"],
            "segmented_response": segmented_response,
            "reconstructed_response": state.get_full_response(),
            "reconstructed_response_includes_prefix": state.get_full_response() != clean_info["clean_response"],
        }
        return state

    @staticmethod
    def _merge_root_state_metadata(
        inherited_metadata: Dict[str, Any],
        new_metadata: Dict[str, Any],
        has_prefix: bool,
    ) -> Dict[str, Any]:
        if not has_prefix:
            return dict(new_metadata)

        merged = dict(new_metadata)
        for key in ROOT_STATE_METADATA_KEYS:
            if key in inherited_metadata:
                merged[key] = inherited_metadata[key]
        return merged
