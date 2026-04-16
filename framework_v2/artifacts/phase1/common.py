import csv
import json
import fcntl
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

import yaml


ROOT = Path(__file__).resolve().parents[2]
PHASE1_RESULTS_ROOT = ROOT / "data" / "results" / "phase1"
PHASE1_MANIFEST_PATH = PHASE1_RESULTS_ROOT / "phase1_experiment_manifest.jsonl"
PHASE1_SUMMARY_DIR = PHASE1_RESULTS_ROOT / "summaries"
PHASE1_ASSETS_DIR = PHASE1_RESULTS_ROOT / "assets"
PHASE1_DEFAULTS_PATH = ROOT / "configs" / "phase1_generation_defaults.yaml"
PHASE1_SEEDS_PATH = ROOT / "configs" / "phase1_seeds.json"


LM_SPECS = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "short": "llama31_8b",
        "pretty": "Llama-3.1-8B-Instruct",
    },
    "Qwen/Qwen3-8B": {
        "short": "qwen3_8b",
        "pretty": "Qwen3-8B",
    },
    "google/gemma-3-12b-it": {
        "short": "gemma3_12b",
        "pretty": "gemma-3-12b-it",
    },
}

RM_SPECS = {
    "Skywork/Skywork-Reward-V2-Llama-3.1-8B": {
        "short": "skywork",
        "pretty": "Skywork-Reward-V2-Llama-3.1-8B",
    },
    "weqweasdas/RM-Mistral-7B": {
        "short": "rm_mistral",
        "pretty": "RM-Mistral-7B",
    },
}

DATASET_SUBPHASE = {
    "alpaca_eval": "main_scaling",
    "arena_hard": "main_scaling",
    "wildbench_v2": "main_scaling",
    "ifbench": "ifbench",
}


def load_generation_defaults() -> dict:
    with open(PHASE1_DEFAULTS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_seed_config() -> dict:
    with open(PHASE1_SEEDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def lm_short_name(lm_name: str) -> str:
    return LM_SPECS[lm_name]["short"]


def rm_short_name(rm_name: str) -> str:
    return RM_SPECS[rm_name]["short"]


def output_dir_for_job(subphase: str, dataset: str, lm_name: str, rm_name: str, method: str, budget: int) -> Path:
    if subphase == "ifbench":
        base = PHASE1_RESULTS_ROOT / "ifbench"
    else:
        base = PHASE1_RESULTS_ROOT / subphase
    return base / f"{dataset}__{lm_short_name(lm_name)}__{rm_short_name(rm_name)}__{method}__N{budget}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_manifest_status(manifest_path: Path, expected_output_dir: str, status: str) -> None:
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    ensure_parent(lock_path)
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        rows = read_jsonl(manifest_path)
        changed = False
        for row in rows:
            if row.get("expected_output_dir") == expected_output_dir:
                row["status"] = status
                changed = True
        if changed:
            tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
            write_jsonl(tmp_path, rows)
            tmp_path.replace(manifest_path)


def git_metadata() -> Dict[str, str]:
    commit = _run_git(["rev-parse", "HEAD"])
    describe = _run_git(["describe", "--always", "--dirty"])
    return {
        "git_commit_hash": commit,
        "code_version_tag": describe,
    }


def _run_git(args: List[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
