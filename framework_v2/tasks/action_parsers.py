import re
from abc import ABC, abstractmethod
from typing import Dict, List

from core.data_structures import Action
from .parsing_utils import group_chunks, sanitize_response_text, split_with_structure_fallback


class ActionParser(ABC):
    @abstractmethod
    def parse(self, task, full_response: str) -> List[Action]:
        pass


class DelimiterParser(ActionParser):
    def parse(self, task, full_response: str) -> List[Action]:
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
        return actions


class RegexParser(ActionParser):
    def parse(self, task, full_response: str) -> List[Action]:
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
        return [Action(step_text=chunk, is_final=(i == len(chunks) - 1)) for i, chunk in enumerate(chunks)]


class TokenCountParser(ActionParser):
    def parse(self, task, full_response: str) -> List[Action]:
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

        return actions


class ThinkBlockParser(ActionParser):
    def parse(self, task, full_response: str) -> List[Action]:
        think_end = task.action_strategy.get("think_end_token", "</think>")
        cleaned = sanitize_response_text(full_response)
        if think_end in cleaned:
            end_idx = cleaned.find(think_end) + len(think_end)
            thinking_part = cleaned[:end_idx]
            answer_part = cleaned[end_idx:].strip()
            actions = [Action(step_text=thinking_part, is_final=(not answer_part))]
            if answer_part:
                actions.append(Action(step_text=answer_part, is_final=True))
            return actions

        # vllm strips the stop token from output, so think_end is absent even when
        # generation correctly stopped at </think>. Re-append it so that downstream
        # code can detect that the probe phase is done.
        return [Action(step_text=cleaned + think_end, is_final=False)]


class CodeBlockParser(ActionParser):
    _CODE_LINE_PATTERN = re.compile(
        r"^(?:from\s+\S+\s+import\s+\S+|import\s+\S+|def\s+\w+|class\s+\w+|if\s+__name__\s*==|"
        r"for\s+.+:|while\s+.+:|with\s+.+:|try:|A\s*,|[A-Za-z_][A-Za-z0-9_,\s]*=\s*|"
        r"print\s*\(|sys\.stdout|sys\.stdin)"
    )

    def parse(self, task, full_response: str) -> List[Action]:
        cleaned = sanitize_response_text(full_response)
        if not cleaned:
            return []

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
                ]
            return [Action(step_text=code_part, is_final=True)]

        return [Action(step_text=cleaned, is_final=True)]


ACTION_PARSERS: Dict[str, ActionParser] = {
    "delimiter": DelimiterParser(),
    "regex": RegexParser(),
    "token_count": TokenCountParser(),
    "think_block": ThinkBlockParser(),
    "code_block": CodeBlockParser(),
}
