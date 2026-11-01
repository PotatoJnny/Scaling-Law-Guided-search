import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.search.common import PHASE1_MANIFEST_PATH, PHASE1_RESULTS_ROOT, lm_short_name, read_jsonl, rm_short_name, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = PHASE1_RESULTS_ROOT / "summaries"
SUBSET_DIR = PHASE1_RESULTS_ROOT / "subsets"
SUBSET_OUTPUT_ROOT = PHASE1_RESULTS_ROOT / "main_scaling_subsets"
DATASET_SIZES = {"alpaca_eval": 805, "arena_hard": 750, "wildbench_v2": 1024}
P1_BUDGETS = [64, 128, 256, 512, 1024]
QWEN = "Qwen/Qwen3-8B"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
SKYWORK = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
P1_SHARD_OVERRIDES = {
    ("alpaca_eval", 512): 14,
    ("alpaca_eval", 1024): 7,
    ("arena_hard", 512): 13,
}
EMPIRICAL_PROMPT_SECONDS = {
    "qwen3_8b": {
        "alpaca_eval": {64: 114.7, 128: 185.8, 256: 927.2, 512: 1924.0, 1024: 3739.0},
        "arena_hard": {64: 134.3, 128: 212.3, 256: 412.0, 512: 2122.6, 1024: 2122.6},
        "wildbench_v2": {64: 295.7, 128: 458.8, 256: 854.6, 512: 1627.9, 1024: 2875.4},
    },
    "llama31_8b": {
        "alpaca_eval": {64: 135.0, 128: 328.5, 256: 318.3, 512: 679.9, 1024: 1374.9},
        "arena_hard": {64: 128.3, 128: 264.0, 256: 520.2, 512: 1100.6, 1024: 1981.0},
        "wildbench_v2": {64: 192.0, 128: 347.3, 256: 564.4, 512: 1181.6, 1024: 2804.2},
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, default=PHASE1_MANIFEST_PATH)
    parser.add_argument("--subset-dir", type=Path, default=SUBSET_DIR)
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    return parser.parse_args()


def _subset_output_dir(subset_name: str, lm: str, rm: str, method: str, budget: int) -> str:
    return str(
        SUBSET_OUTPUT_ROOT
        / f"{subset_name}__{lm_short_name(lm)}__{rm_short_name(rm)}__{method}__N{budget}"
    )


def _find_base_row(rows: list[dict], *, dataset: str, lm: str, rm: str, method: str, budget: int) -> dict:
    for row in rows:
        if (
            row.get("subphase") == "main_scaling"
            and row.get("dataset") == dataset
            and row.get("lm") == lm
            and row.get("rm") == rm
            and row.get("method") == method
            and int(row.get("budget")) == budget
            and row.get("config_label") == "cfgA_baseline"
        ):
            return dict(row)
    raise KeyError((dataset, lm, rm, method, budget))


def _load_subset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_completed(output_dir: Path) -> int:
    results_path = output_dir / "results.json"
    if not results_path.exists():
        return 0
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    return len(payload.get("experiments", []))


def _runtime_from_outputs(output_dir: Path) -> float | None:
    results_path = output_dir / "results.json"
    if not results_path.exists():
        return None
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload.get("summary", {}).get("average_time")


def _estimated_prompt_seconds(row: dict) -> float:
    output_dir = Path(row["expected_output_dir"])
    observed = _runtime_from_outputs(output_dir)
    if observed:
        return float(observed)
    family = lm_short_name(row["lm"])
    return float(EMPIRICAL_PROMPT_SECONDS[family][row["dataset"]][int(row["budget"])])


def _manifest_summary(rows: list[dict], *, is_full: bool) -> dict:
    unresolved = 0
    remaining_prompts = 0
    remaining_gpu_hours = 0.0
    for row in rows:
        output_dir = Path(row["expected_output_dir"])
        completed = _count_completed(output_dir)
        if is_full:
            target = DATASET_SIZES[row["dataset"]]
        else:
            target = len(row.get("problem_indices") or [])
        if completed < target:
            unresolved += 1
        remaining = max(target - completed, 0)
        remaining_prompts += remaining
        remaining_gpu_hours += remaining * _estimated_prompt_seconds(row) / 3600.0
    return {
        "logical_runs": len(rows),
        "unresolved_runs": unresolved,
        "remaining_prompts": remaining_prompts,
        "remaining_gpu_hours": remaining_gpu_hours,
    }


def _write_p1_summary(path: Path, summary: dict):
    hours = summary["remaining_gpu_hours"]
    lines = [
        "# Priority 1 Manifest Summary",
        "",
        f"- logical runs: {summary['logical_runs']}",
        f"- unresolved runs: {summary['unresolved_runs']}",
        f"- datasets: alpaca_eval, arena_hard, wildbench_v2",
        f"- LMs: Qwen/Qwen3-8B",
        f"- methods: best_of_n, slg",
        f"- budgets: 64, 128, 256, 512, 1024",
        f"- remaining prompts: {summary['remaining_prompts']}",
        f"- estimated remaining GPU-hours: {hours:.1f}",
        f"- estimated walltime at 16 L40S: {hours / 16 / 24:.1f} days",
        f"- estimated walltime at 24 L40S: {hours / 24 / 24:.1f} days",
        f"- estimated walltime at 32 L40S: {hours / 32 / 24:.1f} days",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_p2_summary(path: Path, subset_paths: list[Path], summary: dict):
    lines = [
        "# Priority 2 Manifest Summary",
        "",
        f"- logical runs: {summary['logical_runs']}",
        f"- unresolved runs: {summary['unresolved_runs']}",
        f"- frozen100 files: {', '.join(str(p) for p in subset_paths)}",
        f"- LMs: Qwen/Qwen3-8B",
        f"- methods: bbon, mean_search",
        f"- budgets: 64, 128, 256, 512, 1024",
        f"- remaining prompts: {summary['remaining_prompts']}",
        f"- estimated remaining GPU-hours: {summary['remaining_gpu_hours']:.1f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_p3_summary(path: Path, full_summary: dict, reduced_summary: dict):
    lines = [
        "# Priority 3 Manifest Summary",
        "",
        f"- logical runs: {full_summary['logical_runs']}",
        f"- unresolved runs: {full_summary['unresolved_runs']}",
        f"- LMs: meta-llama/Llama-3.1-8B-Instruct",
        f"- methods: best_of_n, slg",
        f"- full-budget remaining prompts: {full_summary['remaining_prompts']}",
        f"- full-budget estimated GPU-hours: {full_summary['remaining_gpu_hours']:.1f}",
        f"- reduced-budget remaining prompts: {reduced_summary['remaining_prompts']}",
        f"- reduced-budget estimated GPU-hours: {reduced_summary['remaining_gpu_hours']:.1f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_p4_plan(path: Path):
    lines = [
        "# Priority 4 Small Ablation Draft",
        "",
        "- status: draft only; not submitted",
        "- datasets: alpaca_eval_frozen100_seed20260422, optional arena_hard_frozen100_seed20260422",
        "- LM: Qwen/Qwen3-8B",
        "- RM ablation: Skywork main, optional second RM only if cleanly supported",
        "- budget: N=512",
        "- methods: SLG, optional BoN for paired comparison",
        "- planned ablations:",
        "  - RM ablation",
        "  - state definition ablation",
        "  - m,K ablation with fixed total budget N",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    subset_paths = [args.subset_dir / f"{dataset}_frozen100_seed20260422.json" for dataset in DATASET_SIZES]
    subsets = {payload["dataset"]: payload for payload in [_load_subset(path) for path in subset_paths]}
    base_rows = read_jsonl(args.base_manifest)

    p1_rows = []
    for dataset in DATASET_SIZES:
        for method in ["best_of_n", "slg"]:
            for budget in P1_BUDGETS:
                row = _find_base_row(base_rows, dataset=dataset, lm=QWEN, rm=SKYWORK, method=method, budget=budget)
                row["priority"] = 1
                row["phase1_priority_label"] = "P1_qwen_main"
                override = P1_SHARD_OVERRIDES.get((dataset, budget))
                if override:
                    row["shard_size_override"] = override
                p1_rows.append(row)

    p2_rows = []
    for dataset, subset in subsets.items():
        for method in ["bbon", "mean_search"]:
            for budget in P1_BUDGETS:
                row = _find_base_row(base_rows, dataset=dataset, lm=QWEN, rm=SKYWORK, method=method, budget=budget)
                row["priority"] = 2
                row["phase1_priority_label"] = "P2_qwen_baselines_frozen100"
                row["subset_label"] = subset["subset_name"]
                row["subset_seed"] = subset["seed"]
                row["problem_indices"] = subset["problem_indices"]
                row["dataset_size_override"] = len(subset["problem_indices"])
                row["expected_output_dir"] = _subset_output_dir(subset["subset_name"], QWEN, SKYWORK, method, budget)
                row["status"] = "planned"
                p2_rows.append(row)

    p3_rows = []
    p3_rows_reduced = []
    for dataset, subset in subsets.items():
        for method in ["best_of_n", "slg"]:
            for budget in P1_BUDGETS:
                row = _find_base_row(base_rows, dataset=dataset, lm=LLAMA, rm=SKYWORK, method=method, budget=budget)
                row["priority"] = 3
                row["phase1_priority_label"] = "P3_llama_cross_model_frozen100"
                row["subset_label"] = subset["subset_name"]
                row["subset_seed"] = subset["seed"]
                row["problem_indices"] = subset["problem_indices"]
                row["dataset_size_override"] = len(subset["problem_indices"])
                row["expected_output_dir"] = _subset_output_dir(subset["subset_name"], LLAMA, SKYWORK, method, budget)
                row["status"] = "planned"
                p3_rows.append(row)
                if budget in {128, 512, 1024}:
                    p3_rows_reduced.append(dict(row))

    p4_rows = []
    for dataset in ["alpaca_eval", "arena_hard"]:
        subset = subsets[dataset]
        for variant_name, notes in [
            ("rm_ablation", "Compare Skywork main reward with optional second RM on frozen100 at N=512."),
            ("state_definition_ablation", "Compare current state definition against alternative state segmentation on frozen100 at N=512."),
            ("mk_ablation", "Vary m and K while keeping total budget N fixed at 512."),
        ]:
            p4_rows.append(
                {
                    "phase": "phase1",
                    "subphase": "main_scaling",
                    "dataset": dataset,
                    "subset_label": subset["subset_name"],
                    "problem_indices": subset["problem_indices"],
                    "dataset_size_override": len(subset["problem_indices"]),
                    "lm": QWEN,
                    "rm": SKYWORK,
                    "method": "slg",
                    "budget": 512,
                    "config_label": "cfgA_baseline",
                    "phase1_priority_label": "P4_small_ablation_draft",
                    "draft_only": True,
                    "ablation_family": variant_name,
                    "notes": notes,
                    "expected_output_dir": _subset_output_dir(subset["subset_name"], QWEN, SKYWORK, f"slg_{variant_name}", 512),
                    "status": "draft",
                }
            )

    p1_path = PHASE1_RESULTS_ROOT / "phase1_cfgA_P1_qwen_bon_slg_full_N64_1024.jsonl"
    p2_path = PHASE1_RESULTS_ROOT / "phase1_cfgA_P2_qwen_bbon_meansearch_frozen100_N64_1024.jsonl"
    p3_path = PHASE1_RESULTS_ROOT / "phase1_cfgA_P3_llama_bon_slg_frozen100_N64_1024.jsonl"
    p3_reduced_path = PHASE1_RESULTS_ROOT / "phase1_cfgA_P3_llama_bon_slg_frozen100_N128_512_1024_reduced.jsonl"
    p4_path = PHASE1_RESULTS_ROOT / "phase1_cfgA_P4_small_ablation_draft.jsonl"

    write_jsonl(p1_path, p1_rows)
    write_jsonl(p2_path, p2_rows)
    write_jsonl(p3_path, p3_rows)
    write_jsonl(p3_reduced_path, p3_rows_reduced)
    write_jsonl(p4_path, p4_rows)

    _write_p1_summary(args.summary_dir / "P1_qwen_bon_slg_manifest_summary.md", _manifest_summary(p1_rows, is_full=True))
    _write_p2_summary(
        args.summary_dir / "P2_qwen_baselines_frozen100_manifest_summary.md",
        subset_paths,
        _manifest_summary(p2_rows, is_full=False),
    )
    _write_p3_summary(
        args.summary_dir / "P3_llama_frozen100_manifest_summary.md",
        _manifest_summary(p3_rows, is_full=False),
        _manifest_summary(p3_rows_reduced, is_full=False),
    )
    _write_p4_plan(args.summary_dir / "P4_small_ablation_plan.md")

    print(f"Wrote {p1_path}")
    print(f"Wrote {p2_path}")
    print(f"Wrote {p3_path}")
    print(f"Wrote {p3_reduced_path}")
    print(f"Wrote {p4_path}")


if __name__ == "__main__":
    main()
