import re
from typing import Dict, List, Tuple


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_FINE_PUNCT_RE = re.compile(
    r"(?:\n+|(?<=;)\s+|(?<=[.!?])\s+|(?<=,)\s+|(?<=:)\s+(?=(?:[-*•]|\d+[\.)])))"
)


def clean_answer_only_response(raw_response: str) -> Dict[str, object]:
    text = raw_response or ""
    sanitized = text.replace("\r\n", "\n")
    contains_think = "<think>" in sanitized.lower() or "</think>" in sanitized.lower()

    without_blocks = _THINK_BLOCK_RE.sub("", sanitized)
    clean = without_blocks.replace("<think>", "").replace("</think>", "")
    clean = re.sub(r"<\|[^>]+?\|>", "", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()

    if not clean:
        fallback = sanitized
        if "</think>" in fallback:
            fallback = fallback.rsplit("</think>", 1)[-1]
        clean = re.sub(r"<\|[^>]+?\|>", "", fallback).replace("<think>", "").replace("</think>", "").strip()

    return {
        "raw_response": text,
        "clean_response": clean,
        "contains_think": contains_think,
        "changed_by_cleaning": clean != sanitized.strip(),
    }


def count_paragraphs(text: str) -> int:
    clean = (text or "").strip()
    if not clean:
        return 0
    return len([chunk for chunk in re.split(r"\n\s*\n", clean) if chunk.strip()])


def split_paper_default_state(
    text: str,
    tokenizer,
    *,
    paragraph_blocks: int = 4,
    fallback_tokens: int = 64,
    root_level: bool = True,
) -> Dict[str, object]:
    """Build the intermediate state definition reported in the paper."""
    clean = (text or "").strip()
    if not root_level:
        return split_open_ended_blocks(clean, tokenizer=tokenizer, max_blocks=4, root_level=False)

    paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n", clean) if chunk.strip()]
    if len(paragraphs) >= paragraph_blocks:
        prefix = "\n\n".join(paragraphs[:paragraph_blocks])
        suffix = "\n\n".join(paragraphs[paragraph_blocks:])
        chunks = [prefix] + ([suffix] if suffix else [])
        path = f"paper_first_{paragraph_blocks}_paragraphs"
        join_delimiter = "\n\n"
        prefix_token_count = _token_len(prefix, tokenizer)
    else:
        token_ids = _encode(clean, tokenizer)
        prefix_ids = token_ids[:fallback_tokens]
        prefix = _decode(prefix_ids, tokenizer)
        prefix_token_count = len(prefix_ids)
        suffix = (
            clean[len(prefix):]
            if clean.startswith(prefix)
            else _decode(token_ids[fallback_tokens:], tokenizer)
        )
        chunks = [prefix] + ([suffix] if suffix else [])
        path = f"paper_fallback_first_{fallback_tokens}_tokens"
        join_delimiter = ""

    chunks = [chunk for chunk in chunks if chunk] or [clean]
    return {
        "chunks": chunks,
        "metadata": {
            "num_blocks": len(chunks),
            "block_lengths": [_token_len(chunk, tokenizer) for chunk in chunks],
            "block_text_preview": [_preview(chunk) for chunk in chunks],
            "state_mode": "paper_default",
            "chunking_path": path,
            "segmentation_path_final": path,
            "join_delimiter": join_delimiter,
            "num_paragraphs_before_fallback": len(paragraphs),
            "single_block_after_paragraph_split": len(paragraphs) < paragraph_blocks,
            "blocks_per_action": 1,
            "target_depth": 1,
            "paper_default_state": True,
            "paper_state_paragraph_blocks": paragraph_blocks,
            "paper_state_fallback_tokens": fallback_tokens,
            "paper_state_prefix_tokens": prefix_token_count,
            "force_first_action_nonfinal": True,
        },
    }


def split_open_ended_blocks(
    text: str,
    tokenizer,
    max_blocks: int = 4,
    local_first_block_split_test_mode: bool = False,
    prefix_normalization_test_mode: bool = False,
    prefix_min_ratio: float = 0.1,
    prefix_max_ratio: float = 0.5,
    root_level: bool = True,
) -> Dict[str, object]:
    clean = (text or "").strip()
    local_first_block_split = False
    normalization_metadata = _default_prefix_normalization_metadata()
    if not clean:
        blocks = [""]
        path = "token_window"
        num_paragraphs = 0
        paragraph_blocks = [""]
    else:
        paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n", clean) if chunk.strip()]
        num_paragraphs = len(paragraphs)
        paragraph_blocks = _pack_balanced(paragraphs, max_blocks, joiner="\n\n") if paragraphs else []
        paragraph_diag = _block_diagnostics(paragraph_blocks or [clean], tokenizer)
        if len(paragraphs) >= 2:
            blocks = paragraph_blocks
            path = "paragraph"
        else:
            sentence_groups = _build_sentence_groups(
                clean,
                tokenizer,
                min_tokens=48,
                max_tokens=80,
            )
            if len(sentence_groups) >= 2:
                blocks = _pack_balanced(
                    sentence_groups,
                    max_blocks,
                    joiner=" ",
                )
                path = "sentence_group"
            else:
                blocks = _token_windows(clean, tokenizer, window_size=64)
                blocks = _pack_balanced(blocks, max_blocks, joiner=" ")
                path = "token_window"

        if local_first_block_split_test_mode and path == "paragraph":
            repaired_blocks = _repair_first_block_if_needed(
                blocks=blocks,
                tokenizer=tokenizer,
                max_blocks=max_blocks,
            )
            if repaired_blocks is not None:
                blocks = repaired_blocks
                path = "paragraph_first_block_split"
                local_first_block_split = True

        if prefix_normalization_test_mode and root_level:
            normalized = _normalize_root_prefix(
                blocks,
                tokenizer,
                prefix_min_ratio=prefix_min_ratio,
                prefix_max_ratio=prefix_max_ratio,
            )
            blocks = normalized["chunks"]
            normalization_metadata = normalized["metadata"]
            if normalized["metadata"].get("prefix_normalization_applied"):
                path = f"{path}_prefix_normalized"

    if not blocks:
        blocks = [clean]

    final_diag = _block_diagnostics(blocks, tokenizer)
    blocks_per_action = 1

    state_mode = "single_block" if len(blocks) < 2 else "chunked"
    return {
        "chunks": blocks,
        "metadata": {
            "num_blocks": len(blocks),
            "block_lengths": final_diag["lengths"],
            "block_text_preview": [_preview(block) for block in blocks],
            "state_mode": state_mode,
            "chunking_path": path,
            "segmentation_path_final": path,
            "join_delimiter": "\n\n",
            "num_paragraphs_before_fallback": num_paragraphs,
            "largest_block_fraction": final_diag["largest_block_fraction"],
            "single_block_after_paragraph_split": len(paragraph_blocks) < 2,
            "blocks_per_action": blocks_per_action,
            "target_depth": None,
            "local_first_block_split": local_first_block_split,
            "local_first_block_split_test_mode": local_first_block_split_test_mode,
            **normalization_metadata,
        },
    }


def split_exact_if_spans(text: str, tokenizer, max_blocks: int = 6) -> Dict[str, object]:
    clean = (text or "").strip()
    if not clean:
        spans = [""]
        path = "token_window"
    else:
        punct_units = [chunk.strip() for chunk in re.split(_FINE_PUNCT_RE, clean) if chunk and chunk.strip()]
        punct_units = [chunk for chunk in punct_units if chunk]
        if len(punct_units) >= 2:
            spans = _pack_balanced(punct_units, max_blocks, joiner=" ")
            path = "fine_punct"
        else:
            spans = _token_windows(clean, tokenizer, window_size=16)
            spans = _pack_balanced(spans, max_blocks, joiner=" ")
            path = "token_window"

    if not spans:
        spans = [clean]

    state_mode = "single_span" if len(spans) < 2 else "chunked"
    return {
        "chunks": spans,
        "metadata": {
            "num_spans": len(spans),
            "span_lengths": [_token_len(span, tokenizer) for span in spans],
            "span_preview": [_preview(span) for span in spans],
            "state_mode": state_mode,
            "segmentation_path": path,
            "join_delimiter": " ",
        },
    }


def _build_sentence_groups(text: str, tokenizer, min_tokens: int, max_tokens: int) -> List[str]:
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return sentences

    groups: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = _token_len(sentence, tokenizer)
        if current and current_tokens >= min_tokens and current_tokens + sentence_tokens > max_tokens:
            groups.append(" ".join(current).strip())
            current = [sentence]
            current_tokens = sentence_tokens
            continue
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        groups.append(" ".join(current).strip())
    return [group for group in groups if group]


def _split_sentences(text: str) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []
    pieces = [piece.strip() for piece in re.split(_SENTENCE_BOUNDARY_RE, stripped) if piece.strip()]
    return pieces if len(pieces) > 1 else [stripped]


def _token_windows(text: str, tokenizer, window_size: int) -> List[str]:
    token_ids = _encode(text, tokenizer)
    if not token_ids:
        return [text.strip()] if text.strip() else [""]
    windows = []
    for start in range(0, len(token_ids), window_size):
        chunk_ids = token_ids[start:start + window_size]
        chunk_text = _decode(chunk_ids, tokenizer).strip()
        if chunk_text:
            windows.append(chunk_text)
    return windows or [text.strip()]


def _pack_balanced(chunks: List[str], max_blocks: int, joiner: str) -> List[str]:
    clean_chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    if len(clean_chunks) <= max_blocks:
        return clean_chunks

    packed = []
    total = len(clean_chunks)
    start = 0
    for remaining_groups in range(max_blocks, 0, -1):
        remaining_items = total - start
        take = max(1, remaining_items // remaining_groups)
        if remaining_items % remaining_groups:
            take += 1
        group = clean_chunks[start:start + take]
        packed.append(joiner.join(group).strip())
        start += take
    return [chunk for chunk in packed if chunk]


def _repair_first_block_if_needed(
    blocks: List[str],
    tokenizer,
    max_blocks: int,
) -> List[str] | None:
    diagnostics = _block_diagnostics(blocks, tokenizer)
    if not _has_bad_shape(blocks, diagnostics):
        return None

    if len(blocks) >= max_blocks:
        return None

    head = blocks[0].strip()
    tail = [block.strip() for block in blocks[1:] if block.strip()]
    available_head_slots = max_blocks - len(tail)
    if available_head_slots < 2 or not head:
        return None

    head_chunks = _build_sentence_groups(head, tokenizer, min_tokens=24, max_tokens=72)
    if len(head_chunks) < 2:
        window_size = max(24, (_token_len(head, tokenizer) + available_head_slots - 1) // available_head_slots)
        head_chunks = _token_windows(head, tokenizer, window_size=window_size)

    repaired_head = _pack_balanced(head_chunks, available_head_slots, joiner=" ")
    if len(repaired_head) < 2:
        return None

    candidate = repaired_head + tail
    candidate_diag = _block_diagnostics(candidate, tokenizer)
    if candidate_diag["largest_block_fraction"] >= diagnostics["largest_block_fraction"]:
        return None
    if candidate_diag["first_block_fraction"] >= diagnostics["first_block_fraction"]:
        return None
    return candidate


def _normalize_root_prefix(
    blocks: List[str],
    tokenizer,
    prefix_min_ratio: float = 0.1,
    prefix_max_ratio: float = 0.5,
) -> Dict[str, object]:
    clean_blocks = [block.strip() for block in blocks if block and block.strip()]
    if not clean_blocks:
        return {
            "chunks": clean_blocks,
            "metadata": _default_prefix_normalization_metadata(),
        }

    diagnostics_before = _block_diagnostics(clean_blocks, tokenizer)
    lengths = diagnostics_before["lengths"]
    total_tokens = diagnostics_before["total_tokens"]
    if total_tokens <= 0:
        metadata = _default_prefix_normalization_metadata()
        metadata["largest_block_fraction_before"] = diagnostics_before["largest_block_fraction"]
        metadata["largest_block_fraction_after"] = diagnostics_before["largest_block_fraction"]
        metadata["search_blocks_num_blocks"] = len(clean_blocks)
        return {"chunks": clean_blocks, "metadata": metadata}

    first_ratio = lengths[0] / total_tokens
    final_chunks: List[str]
    fixed_prefix = ""
    fixed_prefix_ratio = None
    fixed_prefix_num_tokens = None
    fixed_prefix_num_blocks = 0
    search_blocks_num_blocks = len(clean_blocks)
    repair_mode = "none"
    prefix_normalization_applied = False
    boundary_block_split_applied = False

    if first_ratio > prefix_max_ratio:
        split_result = _split_block_for_prefix_target(
            clean_blocks[0],
            tokenizer=tokenizer,
            total_tokens=total_tokens,
            prefix_tokens_before=0,
            prefix_min_ratio=prefix_min_ratio,
            prefix_max_ratio=prefix_max_ratio,
        )
        if split_result is not None:
            prefix_part, remainder_part, prefix_tokens = split_result
            fixed_prefix = prefix_part
            fixed_prefix_num_tokens = prefix_tokens
            fixed_prefix_ratio = prefix_tokens / total_tokens
            fixed_prefix_num_blocks = 1
            search_blocks = ([remainder_part] if remainder_part else []) + clean_blocks[1:]
            final_chunks = [fixed_prefix] + search_blocks
            search_blocks_num_blocks = len(search_blocks)
            repair_mode = "split_large_first_block"
            prefix_normalization_applied = True
            boundary_block_split_applied = True
        else:
            final_chunks = clean_blocks
    else:
        prefix_parts = [clean_blocks[0]]
        prefix_tokens = lengths[0]
        idx = 1

        if prefix_min_ratio <= first_ratio <= prefix_max_ratio:
            fixed_prefix = clean_blocks[0]
            fixed_prefix_num_tokens = prefix_tokens
            fixed_prefix_ratio = first_ratio
            fixed_prefix_num_blocks = 1
            search_blocks = clean_blocks[1:]
            final_chunks = [fixed_prefix] + search_blocks
            search_blocks_num_blocks = len(search_blocks)
        else:
            split_applied = False
            while idx < len(clean_blocks) and (prefix_tokens / total_tokens) < prefix_min_ratio:
                next_block = clean_blocks[idx]
                next_len = _token_len(next_block, tokenizer)
                if (prefix_tokens + next_len) / total_tokens > prefix_max_ratio:
                    split_result = _split_block_for_prefix_target(
                        next_block,
                        tokenizer=tokenizer,
                        total_tokens=total_tokens,
                        prefix_tokens_before=prefix_tokens,
                        prefix_min_ratio=prefix_min_ratio,
                        prefix_max_ratio=prefix_max_ratio,
                    )
                    if split_result is not None:
                        prefix_part, remainder_part, updated_prefix_tokens = split_result
                        prefix_parts.append(prefix_part)
                        fixed_prefix = "\n\n".join(part for part in prefix_parts if part).strip()
                        fixed_prefix_num_tokens = updated_prefix_tokens
                        fixed_prefix_ratio = updated_prefix_tokens / total_tokens
                        fixed_prefix_num_blocks = idx + 1
                        search_blocks = ([remainder_part] if remainder_part else []) + clean_blocks[idx + 1:]
                        final_chunks = [fixed_prefix] + search_blocks
                        search_blocks_num_blocks = len(search_blocks)
                        repair_mode = "split_boundary_block"
                        prefix_normalization_applied = True
                        boundary_block_split_applied = True
                        split_applied = True
                    break

                prefix_parts.append(next_block)
                prefix_tokens += next_len
                idx += 1

            if not split_applied:
                fixed_prefix = "\n\n".join(part for part in prefix_parts if part).strip()
                fixed_prefix_num_tokens = prefix_tokens
                fixed_prefix_ratio = prefix_tokens / total_tokens
                fixed_prefix_num_blocks = len(prefix_parts)
                search_blocks = clean_blocks[len(prefix_parts):]
                final_chunks = [fixed_prefix] + search_blocks
                search_blocks_num_blocks = len(search_blocks)
                if len(prefix_parts) > 1 and prefix_min_ratio <= fixed_prefix_ratio <= prefix_max_ratio:
                    repair_mode = "grow_prefix"
                    prefix_normalization_applied = True

    diagnostics_after = _block_diagnostics(final_chunks, tokenizer)
    metadata = {
        "prefix_normalization_applied": prefix_normalization_applied,
        "fixed_prefix_ratio": fixed_prefix_ratio,
        "fixed_prefix_num_tokens": fixed_prefix_num_tokens,
        "fixed_prefix_num_blocks": fixed_prefix_num_blocks,
        "search_blocks_num_blocks": search_blocks_num_blocks,
        "repair_mode": repair_mode,
        "boundary_block_split_applied": boundary_block_split_applied,
        "largest_block_fraction_before": diagnostics_before["largest_block_fraction"],
        "largest_block_fraction_after": diagnostics_after["largest_block_fraction"],
    }
    return {
        "chunks": final_chunks,
        "metadata": metadata,
    }


def _has_bad_shape(blocks: List[str], diagnostics: Dict[str, float]) -> bool:
    if len(blocks) <= 2:
        return True
    if diagnostics["largest_block_fraction"] > 0.60:
        return True
    if diagnostics["first_block_fraction"] > 0.50 and diagnostics["rest_mean_fraction"] < 0.18:
        return True
    return False


def _split_block_for_prefix_target(
    block_text: str,
    tokenizer,
    total_tokens: int,
    prefix_tokens_before: int,
    prefix_min_ratio: float = 0.1,
    prefix_max_ratio: float = 0.5,
) -> Tuple[str, str, int] | None:
    units = _split_sentences(block_text)
    if len(units) < 2:
        units = _split_fine_units(block_text)
    if len(units) < 2:
        units = _token_windows(block_text, tokenizer, window_size=32)

    if len(units) < 2:
        return None

    prefix_parts: List[str] = []
    for idx, unit in enumerate(units[:-1]):
        prefix_parts.append(unit)
        prefix_text = " ".join(prefix_parts).strip()
        prefix_tokens = prefix_tokens_before + _token_len(prefix_text, tokenizer)
        ratio = prefix_tokens / total_tokens if total_tokens else 0.0
        if prefix_min_ratio <= ratio <= prefix_max_ratio:
            remainder_text = " ".join(units[idx + 1:]).strip()
            return prefix_text, remainder_text, prefix_tokens

    return None


def _split_fine_units(text: str) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []
    pieces = [piece.strip() for piece in re.split(_FINE_PUNCT_RE, stripped) if piece and piece.strip()]
    return pieces if len(pieces) > 1 else [stripped]


def _block_diagnostics(blocks: List[str], tokenizer) -> Dict[str, float]:
    lengths = [_token_len(block, tokenizer) for block in blocks]
    total = sum(lengths)
    if not lengths or total <= 0:
        return {
            "lengths": lengths,
            "total_tokens": total,
            "largest_block_fraction": 1.0 if lengths else 0.0,
            "first_block_fraction": 1.0 if lengths else 0.0,
            "rest_mean_fraction": 0.0,
            "coefficient_of_variation": 0.0,
        }

    mean_len = total / len(lengths)
    variance = sum((length - mean_len) ** 2 for length in lengths) / len(lengths)
    std_len = variance ** 0.5
    rest = lengths[1:]
    return {
        "lengths": lengths,
        "total_tokens": total,
        "largest_block_fraction": max(lengths) / total,
        "first_block_fraction": lengths[0] / total,
        "rest_mean_fraction": ((sum(rest) / len(rest)) / total) if rest else 0.0,
        "coefficient_of_variation": (std_len / mean_len) if mean_len else 0.0,
    }


def _default_prefix_normalization_metadata() -> Dict[str, object]:
    return {
        "prefix_normalization_applied": False,
        "fixed_prefix_ratio": None,
        "fixed_prefix_num_tokens": None,
        "fixed_prefix_num_blocks": 0,
        "search_blocks_num_blocks": 0,
        "repair_mode": "none",
        "boundary_block_split_applied": False,
        "largest_block_fraction_before": None,
        "largest_block_fraction_after": None,
    }


def _encode(text: str, tokenizer) -> List[int]:
    if tokenizer is None:
        return list(range(len(text.split())))
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def _decode(token_ids: List[int], tokenizer) -> str:
    if tokenizer is None:
        return " ".join(str(tok) for tok in token_ids)
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(token_ids)


def _token_len(text: str, tokenizer) -> int:
    return len(_encode(text, tokenizer))


def _preview(text: str, limit: int = 120) -> str:
    single_line = re.sub(r"\s+", " ", (text or "").strip())
    return single_line[:limit]
