import re
from abc import ABC, abstractmethod
from typing import List, Optional
from core.data_structures import Action

class BaseTask(ABC):
    def __init__(self, dataset_config: dict, action_strategy: dict, tokenizer=None):
        self.dataset_config = dataset_config
        self.action_strategy = action_strategy
        self.tokenizer = tokenizer # Optional, only needed if using token_count strategy

    @abstractmethod
    def get_prompt(self, problem_data: dict) -> str:
        """Tasks must define how to format the dataset into a prompt."""
        pass

    @abstractmethod
    def extract_answer(self, text: str) -> str:
        """Tasks must define how to extract the final ground truth."""
        pass

    def sanitize_response_text(self, text: str) -> str:
        """Remove obvious generation artifacts before parsing."""
        if not text:
            return ""
        text = re.sub(r"<\|[^>]+?\|>", "", text)
        text = text.replace("\r\n", "\n")
        return text.strip()

    def _split_with_structure_fallback(self, full_response: str) -> List[str]:
        """
        Fallback splitter for cases where the requested delimiter was not obeyed.
        This keeps SLG usable even when the LM ignores formatting instructions.
        """
        text = self.sanitize_response_text(full_response)
        if not text:
            return []

        patterns = [
            r"(?=^\s*#{1,6}\s*Step\b)",
            r"(?=^\s*#{1,6}\s*\d+[\.\):])",
            r"(?=^\s*Step\s+\d+[\s:.-])",
            r"(?=^\s*\d+[\.\)]\s+)",
            r"(?=^\s*(Final answer|Answer|Therefore)\b)",
            r"(?=####)",
        ]

        for pattern in patterns:
            chunks = [s.strip() for s in re.split(pattern, text, flags=re.MULTILINE) if s.strip()]
            if len(chunks) > 1:
                return chunks

        paragraphs = [s.strip() for s in re.split(r"\n{1,}", text) if s.strip()]
        return paragraphs if len(paragraphs) > 1 else [text]

    def _group_chunks(self, chunks: List[str], cleaned: str, delimiter: str) -> List[str]:
        """
        Optionally merge multiple low-level chunks into one action.
        This lets SLG branch after 2-3 logical steps instead of the first boilerplate step.
        """
        min_chunks = int(self.action_strategy.get("min_chunks_per_action", 1))
        if min_chunks <= 1 or len(chunks) <= 1:
            return chunks

        grouped = []
        used_delimiter = delimiter if delimiter in cleaned else "\n\n"
        for start in range(0, len(chunks), min_chunks):
            grouped.append(used_delimiter.join(part.strip() for part in chunks[start:start + min_chunks] if part.strip()))
        return grouped

    def parse_response_to_actions(self, full_response: str) -> List[Action]:
        """
        Universal parsing logic based on the chosen Action Strategy.
        Works for Math, Code, Logic, etc.
        """
        method = self.action_strategy.get("chunking_method")
        actions = []

        if method == "delimiter":
            delimiter = self.action_strategy["delimiter"]
            cleaned = self.sanitize_response_text(full_response)
            chunks = [s for s in cleaned.split(delimiter) if s.strip()]
            if len(chunks) <= 1:
                chunks = self._split_with_structure_fallback(cleaned)
            chunks = self._group_chunks(chunks, cleaned, delimiter)
            for i, chunk in enumerate(chunks):
                is_last = (i == len(chunks) - 1)
                if len(chunks) > 1 and delimiter in cleaned:
                    text = chunk + delimiter if not is_last else chunk
                else:
                    text = chunk + "\n\n" if not is_last else chunk
                actions.append(Action(step_text=text, is_final=is_last))

        elif method == "regex":
            pattern = self.action_strategy["regex_pattern"]
            cleaned = self.sanitize_response_text(full_response)
            chunks = [s for s in re.split(pattern, cleaned) if s.strip()]
            if len(chunks) <= 1:
                chunks = self._split_with_structure_fallback(cleaned)
            regex_delimiter = self.action_strategy.get("join_delimiter", "\n")
            chunks = self._group_chunks(chunks, cleaned, regex_delimiter)
            for i, chunk in enumerate(chunks):
                is_last = (i == len(chunks) - 1)
                actions.append(Action(step_text=chunk, is_final=is_last))

        elif method == "token_count":
            if not self.tokenizer:
                raise ValueError("Tokenizer must be provided to BaseTask for token_count strategy.")
            cleaned = self.sanitize_response_text(full_response)
            tokens = self.tokenizer.encode(cleaned)
            chunk_size = self.action_strategy["token_count"]
            
            for i in range(0, len(tokens), chunk_size):
                chunk_tokens = tokens[i : i + chunk_size]
                try:
                    chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                except TypeError:
                    chunk_text = self.tokenizer.decode(chunk_tokens)
                    chunk_text = self.sanitize_response_text(chunk_text)
                is_last = (i + chunk_size >= len(tokens))
                actions.append(Action(step_text=chunk_text, is_final=is_last))

        elif method == "think_block":
            think_end = self.action_strategy.get("think_end_token", "</think>")
            cleaned = self.sanitize_response_text(full_response)
            if think_end in cleaned:
                end_idx = cleaned.find(think_end) + len(think_end)
                thinking_part = cleaned[:end_idx]
                answer_part = cleaned[end_idx:].strip()
                actions = [Action(step_text=thinking_part, is_final=(not answer_part))]
                if answer_part:
                    actions.append(Action(step_text=answer_part, is_final=True))
            else:
                # vllm strips the stop token from output, so think_end is absent even when
                # generation correctly stopped at </think>. Re-append it so that downstream
                # code (e.g. _get_stop_seqs_for_state) can detect the probe phase is done.
                actions = [Action(step_text=cleaned + think_end, is_final=False)]

        elif method == "code_block":
            cleaned = self.sanitize_response_text(full_response)
            if not cleaned:
                return []

            lines = cleaned.splitlines(keepends=True)
            code_start = None

            for idx, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                # Prefer an explicit fenced code block if present.
                if stripped.startswith("```"):
                    code_start = idx
                    break
                # PIE programs are usually script-style code, not def/class-based.
                # Treat the first obvious Python line as the start of executable code.
                if re.match(
                    r"^(?:from\s+\S+\s+import\s+\S+|import\s+\S+|def\s+\w+|class\s+\w+|if\s+__name__\s*==|"
                    r"for\s+.+:|while\s+.+:|with\s+.+:|try:|A\s*,|[A-Za-z_][A-Za-z0-9_,\s]*=\s*|"
                    r"print\s*\(|sys\.stdout|sys\.stdin)",
                    stripped,
                ):
                    code_start = idx
                    break

            if code_start is not None and code_start > 0:
                plan_part = "".join(lines[:code_start]).rstrip()
                code_part = "".join(lines[code_start:]).lstrip()
                if plan_part:
                    actions = [
                        Action(step_text=plan_part + "\n\n", is_final=False),
                        Action(step_text=code_part, is_final=True),
                    ]
                else:
                    actions = [Action(step_text=code_part, is_final=True)]
            else:
                actions = [Action(step_text=cleaned, is_final=True)]

        else:
            raise ValueError(f"Unknown chunking method: {method}")

        return actions
