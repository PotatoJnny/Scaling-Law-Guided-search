import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from artifacts.phase1.common import PHASE1_MANIFEST_PATH, ROOT, read_jsonl, update_manifest_status


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PHASE1_MANIFEST_PATH)
    parser.add_argument("--selection", choices=["smoke", "priority1", "priority2", "priority3", "priority4", "all"], default="smoke")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-official-eval", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = _select_rows(read_jsonl(args.manifest), args.selection)
    if args.limit > 0:
        rows = rows[: args.limit]

    for row in rows:
        out_dir = row["expected_output_dir"]
        num_problems = _num_problems_for(args.selection, row)
        cmd = (
            "source /etc/profile; "
            "module load StdEnv/2023 python/3.12 arrow/23.0.1; "
            "source /project/6101845/limuheng/transformers-env/bin/activate; "
            f"cd {ROOT}; export PYTHONUNBUFFERED=1; "
            "python artifacts/phase1/run_phase1_job.py "
            f"--manifest-path {args.manifest} "
            f"--expected-output-dir '{out_dir}' "
            f"{'--num-problems ' + str(num_problems) if num_problems else ''} "
            f"{'--strict-official-eval' if args.run_official_eval else '--skip-official-eval'}"
        )
        job_name = f"p1_{row['dataset']}_{row['method']}_N{row['budget']}"
        time_limit = "02:00:00" if args.selection == "smoke" else _time_limit(row)
        mem = "64G" if "gemma" not in row["lm"] else "80G"
        result = subprocess.run(
            [
                "sbatch",
                "--parsable",
                "--account=aip-wmou",
                "--partition=gpubase_h100_b1,gpubase_h100_b2,gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5",
                "--gres=gpu:h100:1",
                "--cpus-per-task=4",
                f"--mem={mem}",
                f"--time={time_limit}",
                f"--job-name={job_name}",
                f"--output={ROOT / 'archives' / 'logs' / (job_name + '_%j.out')}",
                f"--error={ROOT / 'archives' / 'logs' / (job_name + '_%j.err')}",
                "--wrap",
                f"bash -lc {cmd!r}",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"submitted {job_name}: {result.stdout.strip()}")
        update_manifest_status(args.manifest, out_dir, "running")


def _select_rows(rows, selection):
    active_methods = {"best_of_n", "bbon", "mean_search", "slg"}
    if selection == "all":
        return [row for row in rows if row["method"] != "darwin"]
    if selection == "priority1":
        return [row for row in rows if row["priority"] == 1 and row["method"] != "darwin"]
    if selection == "priority2":
        return [row for row in rows if row["priority"] == 2 and row["method"] != "darwin"]
    if selection == "priority3":
        return [row for row in rows if row["priority"] == 3]
    if selection == "priority4":
        return [row for row in rows if row["priority"] == 4]
    return [
        row for row in rows
        if (
            (row["dataset"] in {"alpaca_eval", "arena_hard", "wildbench_v2"} and row["method"] in active_methods and row["budget"] in {64})
            or (row["dataset"] == "ifbench" and row["method"] in active_methods and row["budget"] in {256})
        )
    ]


def _num_problems_for(selection, row):
    if selection != "smoke":
        return None
    return 2


def _time_limit(row):
    budget = int(row["budget"])
    if row["dataset"] == "wildbench_v2" or budget >= 1024:
        return "12:00:00"
    if budget >= 512:
        return "08:00:00"
    return "04:00:00"


if __name__ == "__main__":
    main()
