import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, List


ARENA_HARD_QUESTION_URL = (
    "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/"
    "data/arena-hard-v2.0/question.jsonl"
)
IFBENCH_TEST_URL = (
    "https://raw.githubusercontent.com/allenai/IFBench/main/data/IFBench_test.jsonl"
)


def _cache_dir(cache_root: str) -> Path:
    path = Path(cache_root) / "instruction_following_phase1"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_jsonl(cache_root: str, name: str, url: str) -> Path:
    path = _cache_dir(cache_root) / name
    if path.exists():
        return path

    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    path.write_bytes(data)
    return path


def _normalize_messages(messages: List[Dict[str, object]]) -> List[Dict[str, str]]:
    cleaned = []
    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def _messages_to_instruction(messages: List[Dict[str, object]]) -> str:
    parts = []
    for msg in messages:
        role = str(msg.get("role", "user")).strip().capitalize()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts).strip()


def load_instruction_following_dataset(dataset_name: str, cache_root: str) -> List[dict]:
    if dataset_name == "alpaca_eval":
        return _load_alpaca_eval_rows()
    if dataset_name == "arena_hard":
        return _load_arena_hard_rows(cache_root)
    if dataset_name == "wildbench_v2":
        return _load_wildbench_v2_rows()
    if dataset_name == "ifbench":
        return _load_ifbench_rows(cache_root)
    raise ValueError(f"Unsupported instruction-following dataset: {dataset_name}")


def _load_alpaca_eval_rows() -> List[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        repo_type="dataset",
        filename="alpaca_eval.json",
    )
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    out = []
    for source_index, row in enumerate(rows):
        instruction = (row.get("instruction") or "").strip()
        if not instruction:
            continue
        out.append(
            {
                "problem_id": f"alpaca_eval_{source_index}",
                "source_index": source_index,
                "instruction": instruction,
                "messages": [{"role": "user", "content": instruction}],
                "dataset": row.get("dataset", "alpaca_eval"),
            }
        )
    return out


def _load_arena_hard_rows(cache_root: str) -> List[dict]:
    path = _cache_jsonl(cache_root, "arena_hard_v2_question.jsonl", ARENA_HARD_QUESTION_URL)

    out = []
    with open(path, "r", encoding="utf-8") as f:
        for source_index, line in enumerate(f):
            row = json.loads(line)
            prompt = (row.get("prompt") or "").strip()
            uid = str(row.get("uid", source_index))
            if not prompt:
                continue
            out.append(
                {
                    "problem_id": f"arena_hard_{uid}",
                    "source_index": source_index,
                    "instruction": prompt,
                    "messages": [{"role": "user", "content": prompt}],
                    "uid": uid,
                    "category": row.get("category"),
                    "subcategory": row.get("subcategory"),
                    "arena_prompt": prompt,
                }
            )
    return out


def _load_wildbench_v2_rows() -> List[dict]:
    from datasets import load_dataset

    ds = load_dataset("allenai/WildBench", "v2", split="test")
    out = []
    for source_index, row in enumerate(ds):
        messages = _normalize_messages(row.get("conversation_input", []))
        instruction = _messages_to_instruction(messages)
        if not instruction:
            continue
        session_id = row.get("session_id", row.get("id", source_index))
        out.append(
            {
                "problem_id": f"wildbench_v2_{session_id}",
                "source_index": source_index,
                "instruction": instruction,
                "messages": messages,
                "session_id": session_id,
                "conversation_input": row.get("conversation_input", []),
                "primary_tag": row.get("primary_tag"),
                "checklist": row.get("checklist", []),
            }
        )
    return out


def _load_ifbench_rows(cache_root: str) -> List[dict]:
    path = _cache_jsonl(cache_root, "ifbench_test.jsonl", IFBENCH_TEST_URL)

    out = []
    with open(path, "r", encoding="utf-8") as f:
        for source_index, line in enumerate(f):
            row = json.loads(line)
            prompt = (row.get("prompt") or "").strip()
            key = str(row.get("key", source_index))
            if not prompt:
                continue
            out.append(
                {
                    "problem_id": f"ifbench_{key}",
                    "source_index": source_index,
                    "instruction": prompt,
                    "messages": [{"role": "user", "content": prompt}],
                    "key": key,
                    "instruction_id_list": row.get("instruction_id_list", []),
                    "kwargs": row.get("kwargs", []),
                }
            )
    return out

