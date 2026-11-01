import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.search.common import PHASE1_RESULTS_ROOT, PHASE1_SUMMARY_DIR, read_jsonl, write_csv


ROOT = Path(__file__).resolve().parents[2]
SUBSET_DIR = PHASE1_RESULTS_ROOT / "subsets"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1-manifest", type=Path, required=True)
    parser.add_argument("--p2-manifest", type=Path, required=True)
    parser.add_argument("--p3-manifest", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, default=PHASE1_SUMMARY_DIR)
    parser.add_argument("--subset-dir", type=Path, default=SUBSET_DIR)
    return parser.parse_args()


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _experiments_by_id(output_dir: Path) -> dict:
    payload = _load_json(output_dir / "results.json")
    if payload is None:
        return {}
    rows = {}
    for experiment in payload.get("experiments", []):
        key = experiment.get("problem_id")
        if key is None:
            key = experiment.get("source_index")
        if key is None:
            key = experiment.get("experiment_index")
        rows[key] = experiment
    return rows


def _summary_row(row: dict, *, label: str) -> dict:
    output_dir = Path(row["expected_output_dir"])
    payload = _load_json(output_dir / "results.json") or {}
    final = _load_json(output_dir / "final_summary.json") or {}
    experiments = payload.get("experiments", [])
    target_prompts = row.get("dataset_size_override") or len(row.get("problem_indices") or []) or payload.get("total_experiments")
    return {
        "priority": label,
        "dataset": row.get("subset_label") or row["dataset"],
        "base_dataset": row["dataset"],
        "LM": row["lm"],
        "RM": row["rm"],
        "method": row["method"],
        "N": int(row["budget"]),
        "completed_prompts": len(experiments),
        "target_prompts": target_prompts,
        "avg_best_score": payload.get("summary", {}).get("average_score"),
        "runtime_per_prompt": payload.get("summary", {}).get("average_time"),
        "manifest_status": row.get("status"),
        "run_status": "final" if final else ("partial" if payload else "missing"),
        "output_dir": str(output_dir),
    }


def _pairwise(method_to_exp: dict, methods: list[str], meta: dict) -> list[dict]:
    rows = []
    for method_a, method_b in methods:
        exp_a = method_to_exp.get(method_a, {})
        exp_b = method_to_exp.get(method_b, {})
        common = sorted(set(exp_a).intersection(exp_b), key=str)
        if not common:
            continue
        diffs = []
        wins = ties = losses = 0
        for key in common:
            score_a = exp_a[key].get("best_score")
            score_b = exp_b[key].get("best_score")
            if score_a is None or score_b is None:
                continue
            diff = float(score_a) - float(score_b)
            diffs.append(diff)
            if diff > 0:
                wins += 1
            elif diff < 0:
                losses += 1
            else:
                ties += 1
        if not diffs:
            continue
        rows.append(
            {
                **meta,
                "method_a": method_a,
                "method_b": method_b,
                "num_common_prompts": len(diffs),
                "mean_diff_a_minus_b": sum(diffs) / len(diffs),
                "wins_a": wins,
                "ties": ties,
                "losses_a": losses,
                "win_rate_a": wins / len(diffs),
            }
        )
    return rows


def _build_p1_outputs(manifest_rows: list[dict], summary_dir: Path):
    result_rows = [_summary_row(row, label="P1") for row in manifest_rows]
    write_csv(
        summary_dir / "P1_qwen_bon_slg_results.csv",
        result_rows,
        list(result_rows[0].keys()) if result_rows else ["priority"],
    )

    grouped = defaultdict(dict)
    for row in manifest_rows:
        grouped[(row["dataset"], row["lm"], row["rm"], int(row["budget"]))][row["method"]] = row

    pairwise_rows = []
    for (dataset, lm, rm, budget), method_rows in grouped.items():
        if "best_of_n" not in method_rows or "slg" not in method_rows:
            continue
        bon_dir = Path(method_rows["best_of_n"]["expected_output_dir"])
        slg_dir = Path(method_rows["slg"]["expected_output_dir"])
        exp_bon = _experiments_by_id(bon_dir)
        exp_slg = _experiments_by_id(slg_dir)
        common = sorted(set(exp_bon).intersection(exp_slg), key=str)
        if not common:
            continue
        diffs = []
        wins = ties = losses = 0
        for key in common:
            diff = float(exp_slg[key]["best_score"]) - float(exp_bon[key]["best_score"])
            diffs.append(diff)
            if diff > 0:
                wins += 1
            elif diff < 0:
                losses += 1
            else:
                ties += 1
        pairwise_rows.append(
            {
                "dataset": dataset,
                "LM": lm,
                "RM": rm,
                "N": budget,
                "num_common_prompts": len(diffs),
                "mean_BoN": sum(float(exp_bon[key]["best_score"]) for key in common) / len(common),
                "mean_SLG": sum(float(exp_slg[key]["best_score"]) for key in common) / len(common),
                "mean_diff_SLG_minus_BoN": sum(diffs) / len(diffs),
                "wins_SLG": wins,
                "ties": ties,
                "losses_SLG": losses,
                "win_rate_SLG": wins / len(diffs),
                "runtime_per_prompt_BoN": (_load_json(bon_dir / "results.json") or {}).get("summary", {}).get("average_time"),
                "runtime_per_prompt_SLG": (_load_json(slg_dir / "results.json") or {}).get("summary", {}).get("average_time"),
            }
        )
    write_csv(
        summary_dir / "P1_qwen_bon_slg_pairwise.csv",
        pairwise_rows,
        list(pairwise_rows[0].keys()) if pairwise_rows else ["dataset"],
    )

    completed = sum(1 for row in result_rows if row["run_status"] == "final")
    partial = sum(1 for row in result_rows if row["run_status"] == "partial")
    missing = sum(1 for row in result_rows if row["run_status"] == "missing")
    lines = [
        "# Priority 1 Progress",
        "",
        f"- logical runs: {len(result_rows)}",
        f"- final runs: {completed}",
        f"- partial runs: {partial}",
        f"- missing runs: {missing}",
    ]
    if pairwise_rows:
        lines.append("")
        lines.append("## Pairwise Snapshot")
        for row in pairwise_rows:
            lines.append(
                f"- {row['dataset']} | {row['LM']} | N={row['N']}: "
                f"common={row['num_common_prompts']}, diff={row['mean_diff_SLG_minus_BoN']:.4f}, "
                f"wins/ties/losses={row['wins_SLG']}/{row['ties']}/{row['losses_SLG']}"
            )
    (summary_dir / "P1_qwen_bon_slg_progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_subset_problem_ids(subset_dir: Path) -> dict[str, set[str]]:
    mapping = {}
    for path in subset_dir.glob("*_frozen100_seed20260422.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mapping[payload["dataset"]] = set(payload["problem_ids"])
        mapping[payload["subset_name"]] = set(payload["problem_ids"])
    return mapping


def _build_p2_outputs(p1_rows: list[dict], p2_rows: list[dict], summary_dir: Path, subset_dir: Path):
    result_rows = [_summary_row(row, label="P2") for row in p2_rows]
    write_csv(
        summary_dir / "P2_qwen_baselines_frozen100_results.csv",
        result_rows,
        list(result_rows[0].keys()) if result_rows else ["priority"],
    )

    subset_problem_ids = _load_subset_problem_ids(subset_dir)
    p1_lookup = {}
    for row in p1_rows:
        p1_lookup[(row["dataset"], int(row["budget"]), row["method"])] = row

    grouped = defaultdict(dict)
    for row in p2_rows:
        grouped[(row["dataset"], row["subset_label"], int(row["budget"]))][row["method"]] = row

    pairwise_rows = []
    for (dataset, subset_label, budget), method_rows in grouped.items():
        subset_ids = subset_problem_ids.get(subset_label) or subset_problem_ids.get(dataset) or set()
        method_to_exp = {}
        for method in ["best_of_n", "slg"]:
            p1_row = p1_lookup.get((dataset, budget, method))
            if p1_row:
                full_exp = _experiments_by_id(Path(p1_row["expected_output_dir"]))
                method_to_exp[method] = {k: v for k, v in full_exp.items() if str(k) in subset_ids}
        for method in ["bbon", "mean_search"]:
            if method in method_rows:
                method_to_exp[method] = _experiments_by_id(Path(method_rows[method]["expected_output_dir"]))
        pairwise_rows.extend(
            _pairwise(
                method_to_exp,
                [
                    ("slg", "bbon"),
                    ("slg", "mean_search"),
                    ("bbon", "best_of_n"),
                    ("mean_search", "best_of_n"),
                    ("bbon", "mean_search"),
                ],
                {
                    "dataset": subset_label,
                    "base_dataset": dataset,
                    "LM": "Qwen/Qwen3-8B",
                    "RM": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                    "N": budget,
                },
            )
        )
    write_csv(
        summary_dir / "P2_qwen_baselines_frozen100_pairwise.csv",
        pairwise_rows,
        list(pairwise_rows[0].keys()) if pairwise_rows else ["dataset"],
    )


def _build_p3_outputs(p3_rows: list[dict], summary_dir: Path):
    result_rows = [_summary_row(row, label="P3") for row in p3_rows]
    write_csv(
        summary_dir / "P3_llama_frozen100_results.csv",
        result_rows,
        list(result_rows[0].keys()) if result_rows else ["priority"],
    )

    grouped = defaultdict(dict)
    for row in p3_rows:
        grouped[(row["dataset"], row["subset_label"], int(row["budget"]))][row["method"]] = row
    pairwise_rows = []
    for (dataset, subset_label, budget), method_rows in grouped.items():
        if "best_of_n" not in method_rows or "slg" not in method_rows:
            continue
        pairwise_rows.extend(
            _pairwise(
                {
                    "best_of_n": _experiments_by_id(Path(method_rows["best_of_n"]["expected_output_dir"])),
                    "slg": _experiments_by_id(Path(method_rows["slg"]["expected_output_dir"])),
                },
                [("slg", "best_of_n")],
                {
                    "dataset": subset_label,
                    "base_dataset": dataset,
                    "LM": "meta-llama/Llama-3.1-8B-Instruct",
                    "RM": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                    "N": budget,
                },
            )
        )
    write_csv(
        summary_dir / "P3_llama_frozen100_pairwise.csv",
        pairwise_rows,
        list(pairwise_rows[0].keys()) if pairwise_rows else ["dataset"],
    )


def main():
    args = parse_args()
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    p1_rows = read_jsonl(args.p1_manifest)
    p2_rows = read_jsonl(args.p2_manifest)
    p3_rows = read_jsonl(args.p3_manifest)
    _build_p1_outputs(p1_rows, args.summary_dir)
    _build_p2_outputs(p1_rows, p2_rows, args.summary_dir, args.subset_dir)
    _build_p3_outputs(p3_rows, args.summary_dir)
    print("Wrote priority reports")


if __name__ == "__main__":
    main()
