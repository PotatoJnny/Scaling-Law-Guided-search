import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from artifacts.phase1.common import ROOT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-output-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--num-problems", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--race-scope-dir", type=Path, default=None)
    parser.add_argument("--skip-official-eval", action="store_true")
    parser.add_argument("--strict-official-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--peer-metadata-timeout-sec", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.race_scope_dir or args.expected_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    race_dir = output_dir / ".race"
    race_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = race_dir / "winner.lock"
    winner_meta_path = race_dir / "winner.json"
    peers_meta_path = race_dir / "peers.json"

    current_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    current_partition = os.environ.get("SLURM_JOB_PARTITION", "unknown")
    hostname = socket.gethostname()

    if _try_claim_lock(lock_dir):
        winner_meta = {
            "winner_job_id": current_job_id,
            "partition": current_partition,
            "hostname": hostname,
            "claimed_at": int(time.time()),
        }
        winner_meta_path.write_text(json.dumps(winner_meta, indent=2), encoding="utf-8")
        print(
            f"[race] claimed winner lock for {output_dir.name}: "
            f"job_id={current_job_id} partition={current_partition} host={hostname}"
        )
        _cancel_peer_jobs(
            peers_meta_path=peers_meta_path,
            current_job_id=current_job_id,
            timeout_sec=args.peer_metadata_timeout_sec,
        )
        _run_payload(args)
        return

    winner_meta = _read_json(winner_meta_path)
    print(
        f"[race] loser detected for {output_dir.name}: "
        f"current_job_id={current_job_id}, winner={winner_meta.get('winner_job_id') if winner_meta else 'unknown'}"
    )


def _try_claim_lock(lock_dir: Path) -> bool:
    try:
        lock_dir.mkdir()
        return True
    except FileExistsError:
        return False


def _cancel_peer_jobs(peers_meta_path: Path, current_job_id: str, timeout_sec: int):
    deadline = time.time() + max(0, timeout_sec)
    peers_meta = None
    while time.time() <= deadline:
        peers_meta = _read_json(peers_meta_path)
        if peers_meta:
            break
        time.sleep(0.5)

    if not peers_meta:
        print("[race] peer metadata not found before timeout; continuing without sibling cancellation")
        return

    all_job_ids = [str(job_id) for job_id in peers_meta.get("job_ids", []) if str(job_id)]
    peer_job_ids = [job_id for job_id in all_job_ids if job_id != str(current_job_id)]
    if not peer_job_ids:
        print("[race] no sibling jobs to cancel")
        return

    for peer_job_id in peer_job_ids:
        try:
            subprocess.run(
                ["scancel", str(peer_job_id)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            print(f"[race] requested cancellation for sibling job {peer_job_id}")
        except Exception as exc:
            print(f"[race] failed to cancel sibling job {peer_job_id}: {exc}")


def _run_payload(args):
    cmd = [
        "python",
        "artifacts/phase1/run_phase1_job.py",
        "--manifest-path",
        str(args.manifest_path),
        "--expected-output-dir",
        str(args.expected_output_dir),
    ]
    if args.num_problems is not None:
        cmd.extend(["--num-problems", str(args.num_problems)])
    if args.start_index:
        cmd.extend(["--start-index", str(args.start_index)])
    if args.skip_official_eval:
        cmd.append("--skip-official-eval")
    if args.strict_official_eval:
        cmd.append("--strict-official-eval")
    if args.overwrite:
        cmd.append("--overwrite")

    print(f"[race] launching payload: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


if __name__ == "__main__":
    main()
