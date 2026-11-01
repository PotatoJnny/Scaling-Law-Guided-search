import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from artifacts.phase1.common import ROOT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--expected-output-dir", type=str, required=True)
    parser.add_argument("--skip-official-eval", action="store_true")
    parser.add_argument("--strict-official-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        raise RuntimeError("SLURM_ARRAY_TASK_ID is not set")

    shard_index = int(task_id)
    output_dir = Path(args.expected_output_dir)
    plan_path = output_dir / ".shards" / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Shard plan not found: {plan_path}")

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    shard_specs = plan.get("shards", [])
    if shard_index < 0 or shard_index >= len(shard_specs):
        raise IndexError(f"Shard index {shard_index} out of range for {plan_path}")

    shard_spec = shard_specs[shard_index]
    cmd = [
        "python",
        "artifacts/phase1/run_phase1_job.py",
        "--manifest-path",
        str(args.manifest_path),
        "--expected-output-dir",
        args.expected_output_dir,
        "--start-index",
        str(int(shard_spec["start_index"])),
    ]
    if shard_spec.get("num_problems") is not None:
        cmd.extend(["--num-problems", str(int(shard_spec["num_problems"]))])
    if args.skip_official_eval:
        cmd.append("--skip-official-eval")
    if args.strict_official_eval:
        cmd.append("--strict-official-eval")
    if args.overwrite:
        cmd.append("--overwrite")

    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
