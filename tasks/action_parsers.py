import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from core.data_structures import Action
from .if_state_utils import (
    split_exact_if_spans,
    split_open_ended_blocks,
    split_paper_default_state,
)
from .parsing_utils import group_chunks, sanitize_response_text, split_with_structure_fallback


class ActionParser(ABC):
    @abstractmethod
    def parse_with_metadata(
        self,
        task,
        full_response: str,
        base_state: Optional[object] = None,
    ) -> Tuple[List[Action], Dict]:
        pass

    def parse(self, task, full_response: str, base_state: Optional[object] = None) -> List[Action]:
        actions, _ = self.parse_with_metadata(task, full_response, base_state=base_state)
        return actions


class DelimiterParser(ActionParser):
    def parse_with_metadata(
        self,
        task,
        full_response: str,
        base_state: Optional[object] = None,
    ) -> Tuple[List[Action], Dict]:
        delimiter = task.action_strategy["delimiter"]
        cleaned = sanitize_response_text(full_response)
        chunks = [s for s in cleaned.split(delimiter) if s.strip()]
        if len(chunks) <= 1:
            chunks = split_with_structure_fallback(cleaned)
        chunks = group_chunks(
            chunks,
            cleaned,
            delimiter,
            int(task.action_strategy.get("min_chunks_per_action", 1)),
        )

        actions = []
        for i, chunk in enumerate(chunks):
            is_last = i == len(chunks) - 1
            if len(chunks) > 1 and delimiter in cleaned:
                text = chunk + delimiter if not is_last else chunk
            else:
                text = chunk + "\n\n" if not is_last else chunk
            actions.append(Action(step_text=text, is_final=is_last))
        return actions, {}


class RegexParser(ActionParser):
    def parse_with_metadata(
        self,
        task,
        full_response: str,
        base_state: Optional[object] = None,
    ) -> Tuple[List[Action], Dict]:
        pattern = task.action_strategy["regex_pattern"]
        cleaned = sanitize_response_text(full_response)
        chunks = [s for s in re.split(pattern, cleaned) if s.strip()]
        if len(chunks) <= 1:
            chunks = split_with_structure_fallback(cleaned)
        regex_delimiter = task.action_strategy.get("join_delimiter", "\n")
        chunks = group_chunks(
            chunks,
            cleaned,
            regex_delimiter,
            int(task.action_strategy.get("min_chunks_per_action", 1)),
        )
        actions = [Action(step_text=chunk, is_final=(i == len(chunks) - 1)) for i, chunk in enumerate(chunks)]
        return actions, {}


class TokenCountParser(ActionParser):
    def parse_with_metadata(
        self,
        task,
        full_response: str,
        base_state: Optional[object] = None,
    ) -> Tuple[List[Action], Dict]:
        if not task.tokenizer:
            raise ValueError("Tokenizer must be provided to BaseTask for token_count strategy.")

        cleaned = sanitize_response_text(full_response)
        tokens = task.tokenizer.encode(cleaned)
        chunk_size = task.action_strategy["token_count"]
        actions = []

        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i : i + chunk_size]
            try:
                chunk_text = task.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            except TypeError:
                chunk_text = task.tokenizer.decode(chunk_tokens)
                chunk_text = sanitize_response_text(chunk_text)
            is_last = i + chunk_size >= len(tokens)
            actions.append(Action(step_text=chunk_text, is_final=is_last))

        return actions, {}


class ThinkBlockParser(ActionParser):
    def parse_with_metadata(
        self,
        task,
        full_response: str,
        base_state: Optional[object] = None,
    ) -> Tuple[List[Action], Dict]:
        think_end = task.action_strategy.get("think_end_token", "</think>")
        cleaned = sanitize_response_text(full_response)
        if think_end in cleaned:
            end_idx = cleaned.find(think_end) + len(think_end)
            thinking_part = cleaned[:end_idx]
            answer_part = cleaned[end_idx:].strip()
            actions = [Action(step_text=thinking_part, is_final=(not answer_part))]
            if answer_part:
                actions.append(Action(step_text=answer_part, is_final=True))
            return actions, {}

        # vllm strips the stop token from output, so think_end is absent even when
        # generation correctly stopped at </think>. Re-append it so that downstream
        # code can detect that the probe phase is done.
        return [Action(step_text=cleaned + think_end, is_final=False)], {}


class CodeBlockParser(ActionParser):
    _CODE_LINE_PATTERN = re.compile(
        r"^(?:from\s+\S+\s+import\s+\S+|import\s+\S+|def\s+\w+|class\s+\w+|if\s+__name__\s*==|"
        r"for\s+.+:|while\s+.+:|with\s+.+:|try:|A\s*,|[A-Za-z_][A-Za-z0-9_,\s]*=\s*|"
        r"print\s*\(|sys\.stdout|sys\.stdin)"
    )

    def parse_with_metadata(
        self,
        task,
        full_response: str,
        base_state: Optional[object] = None,
    ) -> Tuple[List[Action], Dict]:
        cleaned = sanitize_response_text(full_response)
        if not cleaned:
            return [], {}

        lines = cleaned.splitlines(keepends=True)
        code_start = None

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("```"):
                code_start = idx
                break
            if self._CODE_LINE_PATTERN.match(stripped):
                code_start = idx
                break

        if code_start is not None and code_start > 0:
            plan_part = "".join(lines[:code_start]).rstrip()
            code_part = "".join(lines[code_start:]).lstrip()
            if plan_part:
                return [
                    Action(step_text=plan_part + "\n\n", is_final=False),
                    Action(step_text=code_part, is_final=True),
                ], {}
            return [Action(step_text=code_part, is_final=True)], {}

        return [Action(step_text=cleaned, is_final=True)], {}


class IfAnswerOnlyParser(ActionParser):
    def parse_with_metadata(
        self,
        task,
        full_response: str,
        base_state: Optional[object] = None,
    ) -> Tuple[List[Action], Dict]:
        family = task.dataset_config.get("if_state_family", "open_ended_if")
        tokenizer = task.tokenizer
        if family == "exact_if":
            split = split_exact_if_spans(full_response, tokenizer=tokenizer, max_blocks=6)
            chunks = split["chunks"]
            metadata = split["metadata"]
        else:
            root_level = base_state is None or not getattr(base_state, "steps", [])
            if task.action_strategy.get("paper_default_state", False):
                split = split_paper_default_state(
                    full_response,
                    tokenizer=tokenizer,
                    paragraph_blocks=int(
                        task.action_strategy.get("paper_state_paragraph_blocks", 4)
                    ),
                    fallback_tokens=int(
                        task.action_strategy.get("paper_state_fallback_tokens", 64)
                    ),
                    root_level=root_level,
                )
            elif hasattr(task, "split_open_ended_response"):
                split = task.split_open_ended_response(full_response, root_level=root_level)
            else:
                split = split_open_ended_blocks(
                    full_response,
                    tokenizer=tokenizer,
                    max_blocks=4,
                    root_level=root_level,
                )
            chunks = split["chunks"]
            metadata = split["metadata"]

        if not chunks:
            chunks = [sanitize_response_text(full_response)]

        blocks_per_action = max(
            1,
            int(metadata.get("blocks_per_action", task.action_strategy.get("blocks_per_action", 1))),
        )
        if blocks_per_action > 1 and len(chunks) > 1:
            join_delimiter = metadata.get("join_delimiter", "")
            grouped_chunks = []
            for idx in range(0, len(chunks), blocks_per_action):
                grouped_chunks.append(join_delimiter.join(chunks[idx:idx + blocks_per_action]).strip())
            chunks = grouped_chunks

        join_delimiter = metadata.get("join_delimiter", "")
        force_first_action_nonfinal = bool(metadata.get("force_first_action_nonfinal", False))
        metadata = {
            **metadata,
            "blocks_per_action": blocks_per_action,
            "num_action_chunks": len(chunks),
        }
        actions = [
            Action(
                step_text=(chunk + join_delimiter) if idx < len(chunks) - 1 else chunk,
                is_final=(
                    idx == len(chunks) - 1
                    and not (idx == 0 and force_first_action_nonfinal)
                ),
            )
            for idx, chunk in enumerate(chunks)
        ]
        return actions, metadata


ACTION_PARSERS: Dict[str, ActionParser] = {
    "delimiter": DelimiterParser(),
    "regex": RegexParser(),
    "token_count": TokenCountParser(),
    "think_block": ThinkBlockParser(),
    "code_block": CodeBlockParser(),
    "if_answer_only": IfAnswerOnlyParser(),
}
