import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tasks.instruction_following_datasets import load_instruction_following_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "results" / "phase1" / "subsets"
DEFAULT_CACHE_ROOT = ROOT / "data" / "local_cache"
DEFAULT_DATASETS = ["alpaca_eval", "arena_hard", "wildbench_v2"]
WILDBENCH_ARROW_CACHE = Path(
    os.environ.get("SLG_WILDBENCH_ARROW", DEFAULT_CACHE_ROOT / "wild_bench-test.arrow")
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    return parser.parse_args()


def _stable_rank(dataset_name: str, problem_id: str, seed: int) -> int:
    payload = f"{dataset_name}::{problem_id}::{seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _build_subset(dataset_name: str, seed: int, subset_size: int, cache_root: Path) -> dict:
    if dataset_name == "wildbench_v2" and WILDBENCH_ARROW_CACHE.exists():
        rows = _load_wildbench_from_arrow_cache()
    else:
        rows = load_instruction_following_dataset(dataset_name, str(cache_root))
    ranked = []
    for dataset_index, row in enumerate(rows):
        problem_id = str(row.get("problem_id", f"{dataset_name}_{dataset_index}"))
        ranked.append(
            {
                "dataset_index": dataset_index,
                "problem_id": problem_id,
                "source_index": row.get("source_index", dataset_index),
                "rank": _stable_rank(dataset_name, problem_id, seed),
            }
        )
    ranked.sort(key=lambda item: (item["rank"], item["dataset_index"], item["problem_id"]))
    chosen = ranked[: min(subset_size, len(ranked))]
    chosen.sort(key=lambda item: item["dataset_index"])
    subset_name = f"{dataset_name}_frozen{subset_size}_seed{seed}"
    return {
        "dataset": dataset_name,
        "subset_name": subset_name,
        "seed": seed,
        "subset_size": len(chosen),
        "dataset_size": len(rows),
        "problem_indices": [item["dataset_index"] for item in chosen],
        "problem_ids": [item["problem_id"] for item in chosen],
        "source_indices": [item["source_index"] for item in chosen],
    }


def _load_wildbench_from_arrow_cache() -> list[dict]:
    import pyarrow.ipc as ipc

    with WILDBENCH_ARROW_CACHE.open("rb") as f:
        table = ipc.open_stream(f).read_all()
    rows = table.to_pylist()

    def _normalize_messages(messages):
        cleaned = []
        for msg in messages or []:
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", "")).strip()
            if role not in {"system", "user", "assistant"} or not content:
                continue
            cleaned.append({"role": role, "content": content})
        return cleaned

    def _messages_to_instruction(messages):
        parts = []
        for msg in messages:
            role = str(msg.get("role", "user")).strip().capitalize()
            content = str(msg.get("content", "")).strip()
            if content:
                parts.append(f"{role}: {content}")
        return "\n\n".join(parts).strip()

    out = []
    for source_index, row in enumerate(rows):
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


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    for dataset_name in args.datasets:
        payload = _build_subset(dataset_name, args.seed, args.subset_size, args.cache_root)
        output_path = args.output_dir / f"{payload['subset_name']}.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
