import re
from typing import List


def sanitize_response_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<\|[^>]+?\|>", "", text)
    text = text.replace("\r\n", "\n")
    return text.strip()


def split_with_structure_fallback(full_response: str) -> List[str]:
    text = sanitize_response_text(full_response)
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


def group_chunks(chunks: List[str], cleaned: str, delimiter: str, min_chunks_per_action: int) -> List[str]:
    if min_chunks_per_action <= 1 or len(chunks) <= 1:
        return chunks

    grouped = []
    used_delimiter = delimiter if delimiter in cleaned else "\n\n"
    for start in range(0, len(chunks), min_chunks_per_action):
        grouped.append(used_delimiter.join(part.strip() for part in chunks[start:start + min_chunks_per_action] if part.strip()))
    return grouped
