import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from artifacts.phase1.common import PHASE1_ASSETS_DIR, PHASE1_RESULTS_ROOT, PHASE1_SUMMARY_DIR, ensure_parent, write_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=PHASE1_RESULTS_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = _collect_final_summaries(args.results_root)
    PHASE1_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    PHASE1_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = PHASE1_SUMMARY_DIR / "phase1_merged_summary.json"
    csv_path = PHASE1_SUMMARY_DIR / "phase1_merged_summary.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        write_csv(csv_path, rows, fieldnames)

    _write_main_table(rows, PHASE1_ASSETS_DIR / "tables" / "table1_main_fixed_budget_summary.md")
    _write_ifbench_table(rows, PHASE1_ASSETS_DIR / "tables" / "table4_ifbench_external_validation.md")
    _write_robustness_table(rows, "cross_lm", PHASE1_ASSETS_DIR / "tables" / "table2_cross_lm_robustness.md")
    _write_robustness_table(rows, "cross_rm", PHASE1_ASSETS_DIR / "tables" / "table3_cross_rm_robustness.md")

    _maybe_make_scaling_plot(rows, "alpaca_eval", PHASE1_ASSETS_DIR / "figures" / "figure1_alpaca_eval_scaling.png")
    _maybe_make_scaling_plot(rows, "arena_hard", PHASE1_ASSETS_DIR / "figures" / "figure2_arena_hard_scaling.png")
    _maybe_make_scaling_plot(rows, "wildbench_v2", PHASE1_ASSETS_DIR / "figures" / "figure3_wildbench_scaling.png")
    print(f"Wrote merged summaries to {json_path} and {csv_path}")


def _collect_final_summaries(results_root: Path):
    rows = []
    for path in sorted(results_root.glob("**/final_summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            row = json.load(f)
        row["final_summary_path"] = str(path)
        rows.append(row)
    return rows


def _maybe_make_scaling_plot(rows, dataset, output_path: Path):
    if plt is None:
        return

    filtered = [
        row for row in rows
        if row.get("dataset") == dataset and row.get("subphase") == "main_scaling" and row.get("official_score") is not None
    ]
    if not filtered:
        return

    ensure_parent(output_path)
    plt.figure(figsize=(6, 4))
    groups = {}
    for row in filtered:
        key = (row["lm"], row["method"])
        groups.setdefault(key, []).append((int(row["budget"]), float(row["official_score"])))
    for (lm, method), pts in sorted(groups.items()):
        pts = sorted(pts)
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        plt.plot(xs, ys, marker="o", label=f"{lm.split('/')[-1]} / {method}")
    plt.xscale("log", base=2)
    plt.xlabel("Budget N")
    plt.ylabel("Official score")
    plt.title(dataset)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _write_main_table(rows, output_path: Path):
    relevant = [row for row in rows if row.get("subphase") == "main_scaling"]
    if not relevant:
        return
    lookup = {(r["dataset"], r["lm"], r["method"], int(r["budget"])): _primary_score(r) for r in relevant}
    lines = [
        "| Dataset × LM | BoN@256 | BBON@256 | Mean@256 | SLG@256 | BoN@512 | BBON@512 | Mean@512 | SLG@512 | BoN@1024 | BBON@1024 | Mean@1024 | SLG@1024 | BoN@2048 | BBON@2048 | Mean@2048 | SLG@2048 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ["alpaca_eval", "arena_hard", "wildbench_v2"]:
        for lm in ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3-8B"]:
            label = f"`{dataset}` × `{lm.split('/')[-1]}`"
            vals = [
                lookup.get((dataset, lm, "best_of_n", 256)),
                lookup.get((dataset, lm, "bbon", 256)),
                lookup.get((dataset, lm, "mean_search", 256)),
                lookup.get((dataset, lm, "slg", 256)),
                lookup.get((dataset, lm, "best_of_n", 512)),
                lookup.get((dataset, lm, "bbon", 512)),
                lookup.get((dataset, lm, "mean_search", 512)),
                lookup.get((dataset, lm, "slg", 512)),
                lookup.get((dataset, lm, "best_of_n", 1024)),
                lookup.get((dataset, lm, "bbon", 1024)),
                lookup.get((dataset, lm, "mean_search", 1024)),
                lookup.get((dataset, lm, "slg", 1024)),
                lookup.get((dataset, lm, "best_of_n", 2048)),
                lookup.get((dataset, lm, "bbon", 2048)),
                lookup.get((dataset, lm, "mean_search", 2048)),
                lookup.get((dataset, lm, "slg", 2048)),
            ]
            lines.append("| " + label + " | " + " | ".join(_fmt(v) for v in vals) + " |")
    ensure_parent(output_path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ifbench_table(rows, output_path: Path):
    relevant = [row for row in rows if row.get("subphase") == "ifbench"]
    if not relevant:
        return
    lines = [
        "| Method | LM | Budget | Official IFBench score |",
        "|---|---|---:|---:|",
    ]
    for row in sorted(relevant, key=lambda r: (r["lm"], r["method"], int(r["budget"]))):
        lines.append(
            f"| `{row['method']}` | `{row['lm'].split('/')[-1]}` | {int(row['budget'])} | {_fmt(_primary_score(row))} |"
        )
    ensure_parent(output_path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_robustness_table(rows, subphase: str, output_path: Path):
    relevant = [row for row in rows if row.get("subphase") == subphase]
    if not relevant:
        return
    lines = [
        "| Dataset | LM | RM | Method | Budget | Official score |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in sorted(relevant, key=lambda r: (r["dataset"], r["lm"], r["rm"], r["method"], int(r["budget"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['dataset']}`",
                    f"`{row['lm'].split('/')[-1]}`",
                    f"`{row['rm'].split('/')[-1]}`",
                    f"`{row['method']}`",
                    str(int(row["budget"])),
                    _fmt(_primary_score(row)),
                ]
            )
            + " |"
        )
    ensure_parent(output_path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _primary_score(row):
    if row.get("official_score") is not None:
        return row.get("official_score")
    return row.get("interim_metric_value")


if __name__ == "__main__":
    main()
