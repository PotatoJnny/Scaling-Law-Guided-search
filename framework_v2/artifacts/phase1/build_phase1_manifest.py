import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from artifacts.phase1.common import PHASE1_MANIFEST_PATH, PHASE1_RESULTS_ROOT, output_dir_for_job, write_jsonl


MAIN_DATASETS = ["alpaca_eval", "arena_hard", "wildbench_v2"]
MAIN_LMS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
]
ALL_LMS = MAIN_LMS + ["google/gemma-3-12b-it"]
DEFAULT_RM = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
SECONDARY_RM = "weqweasdas/RM-Mistral-7B"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PHASE1_MANIFEST_PATH)
    return parser.parse_args()


def _job_rows():
    rows = []
    main_methods = ["best_of_n", "bbon", "mean_search", "slg"]

    # Priority 1: Phase 1A
    for dataset in MAIN_DATASETS:
        for lm in MAIN_LMS:
            for budget in [64, 128, 256, 512, 1024, 2048]:
                for method in main_methods:
                    rows.append(_row("phase1", "main_scaling", dataset, lm, DEFAULT_RM, method, budget, 1))
                rows.append(_row("phase1", "main_scaling", dataset, lm, DEFAULT_RM, "darwin", budget, 5))

    # Priority 2: Phase 1D
    for budget in [256, 512, 1024]:
        for method in main_methods:
            rows.append(_row("phase1", "ifbench", "ifbench", "Qwen/Qwen3-8B", DEFAULT_RM, method, budget, 2))
        rows.append(_row("phase1", "ifbench", "ifbench", "Qwen/Qwen3-8B", DEFAULT_RM, "darwin", budget, 5))

    # Priority 3: Phase 1B
    for dataset in MAIN_DATASETS:
        for lm in ALL_LMS:
            for budget in [256, 512, 1024, 2048]:
                for method in main_methods:
                    rows.append(_row("phase1", "cross_lm", dataset, lm, DEFAULT_RM, method, budget, 3))

    # Priority 4: Phase 1C
    for dataset in MAIN_DATASETS:
        for rm in [DEFAULT_RM, SECONDARY_RM]:
            for budget in [256, 512, 1024, 2048]:
                for method in main_methods:
                    rows.append(_row("phase1", "cross_rm", dataset, "Qwen/Qwen3-8B", rm, method, budget, 4))

    return rows


def _row(phase, subphase, dataset, lm, rm, method, budget, priority):
    out_dir = output_dir_for_job(subphase, dataset, lm, rm, method, budget)
    return {
        "phase": phase,
        "subphase": subphase,
        "dataset": dataset,
        "lm": lm,
        "rm": rm,
        "method": method,
        "budget": budget,
        "priority": priority,
        "expected_output_dir": str(out_dir),
        "status": "planned",
    }


def main():
    args = parse_args()
    PHASE1_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, _job_rows())
    print(f"Wrote manifest to {args.output}")


if __name__ == "__main__":
    main()
