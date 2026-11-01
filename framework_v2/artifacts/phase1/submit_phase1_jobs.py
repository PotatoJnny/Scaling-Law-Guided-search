import argparse
import getpass
import os
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from artifacts.phase1.common import (
    PHASE1_MANIFEST_PATH,
    ROOT,
    lm_short_name,
    read_jsonl,
    rm_short_name,
    update_manifest_status,
)


DEFAULT_H100_PARTITION = os.environ.get("SLURM_PARTITION", "")
DEFAULT_L40S_PARTITION = os.environ.get("SLG_MIRROR_PARTITION", "")
DEFAULT_VENV_PATH = os.environ.get("VIRTUAL_ENV", str(ROOT / ".venv"))
DEFAULT_MODULE_LOADS = os.environ.get("SLG_MODULE_LOADS", "")
INSTRUCTION_DATASET_SIZES = {
    "alpaca_eval": 805,
    "arena_hard": 750,
    "wildbench_v2": 1024,
    "ifbench": 299,
}
L40_SAFE_SHARD_SIZES = {
    "alpaca_eval": {64: 64, 128: 48, 256: 30, 512: 18, 1024: 10, 2048: 5},
    "arena_hard": {64: 72, 128: 54, 256: 36, 512: 20, 1024: 12, 2048: 6},
    "wildbench_v2": {64: 72, 128: 48, 256: 24, 512: 12, 1024: 6, 2048: 3},
    "ifbench": {256: 64, 512: 40, 1024: 24, 2048: 12},
}
EMPIRICAL_PROMPT_SECONDS = {
    "llama31_8b": {
        "alpaca_eval": {64: 135.0, 128: 328.5, 256: 318.3, 512: 679.9, 1024: 1374.9, 2048: 2870.5},
        "arena_hard": {64: 128.3, 128: 264.0, 256: 520.2, 512: 1100.6, 1024: 1981.0, 2048: 3340.4},
        "wildbench_v2": {64: 192.0, 128: 347.3, 256: 564.4, 512: 1181.6, 1024: 2804.2, 2048: 4706.3},
    },
    "qwen3_8b": {
        "alpaca_eval": {64: 114.7, 128: 185.8, 256: 927.2, 512: 1924.0, 1024: 3739.0, 2048: 2656.2},
        "arena_hard": {64: 134.3, 128: 212.3, 256: 412.0, 512: 2122.6},
        "wildbench_v2": {64: 295.7, 128: 458.8, 256: 854.6, 512: 1627.9, 1024: 2875.4, 2048: 4830.0},
    },
}
SBATCH_RETRY_PATTERNS = (
    "temporarily unable to accept job",
    "socket timed out on send/recv operation",
    "connection reset by peer",
    "unable to contact slurm controller",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PHASE1_MANIFEST_PATH)
    parser.add_argument("--selection", choices=["smoke", "priority1", "priority2", "priority3", "priority4", "all"], default="smoke")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-official-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only-unresolved", action="store_true")
    parser.add_argument(
        "--account", type=str, default=os.environ.get("SLURM_ACCOUNT"),
        required=not os.environ.get("SLURM_ACCOUNT"),
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=DEFAULT_H100_PARTITION,
        required=not DEFAULT_H100_PARTITION,
    )
    parser.add_argument("--gres", type=str, default="gpu:h100:1")
    parser.add_argument("--mirror-partition", type=str, default=DEFAULT_L40S_PARTITION)
    parser.add_argument("--mirror-gres", type=str, default="gpu:l40s:1")
    parser.add_argument("--dual-submit", action="store_true")
    parser.add_argument("--array-submit", action="store_true")
    parser.add_argument(
        "--array-max-concurrent",
        type=int,
        default=8,
        help="Maximum concurrently runnable tasks per submitted job array; <=0 disables the cap.",
    )
    parser.add_argument("--job-name-suffix", type=str, default="")
    parser.add_argument("--skip-active-shards", action="store_true")
    parser.add_argument("--max-queued-jobs", type=int, default=0)
    parser.add_argument("--queue-poll-seconds", type=int, default=90)
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--venv-path", type=str, default=DEFAULT_VENV_PATH)
    parser.add_argument("--module-loads", type=str, default=DEFAULT_MODULE_LOADS)
    parser.add_argument("--runtime-cache-dir", type=str, default=None)
    parser.add_argument(
        "--shard-strategy",
        choices=["none", "l40_safe"],
        default="l40_safe",
    )
    args = parser.parse_args()
    if args.dual_submit and not args.mirror_partition:
        parser.error("--dual-submit requires --mirror-partition or SLG_MIRROR_PARTITION")
    return args


def main():
    args = parse_args()
    if args.dual_submit and args.array_submit:
        raise ValueError("--array-submit currently supports single-partition submissions only")

    rows = _select_rows(read_jsonl(args.manifest), args.selection)
    if args.only_unresolved:
        rows = [row for row in rows if not _is_resolved(row)]
    if args.limit > 0:
        rows = rows[: args.limit]

    if args.skip_active_shards:
        queued_single_jobs, queued_array_tasks, queued_single_targets, queued_array_targets = _queued_jobs()
    else:
        queued_single_jobs, queued_array_tasks, queued_single_targets, queued_array_targets = set(), set(), set(), set()

    failures: list[tuple[str, str]] = []
    for row in rows:
        out_dir = row["expected_output_dir"]
        base_job_name = _base_job_name(row)
        try:
            mem = "64G" if "gemma" not in row["lm"] else "80G"
            shard_specs = _shard_specs_for(row=row, selection=args.selection, shard_strategy=args.shard_strategy)
            _write_shard_plan(output_dir=Path(out_dir), shard_specs=shard_specs)

            for shard_index, shard_spec in enumerate(shard_specs):
                if _is_shard_complete(
                    output_dir=Path(out_dir),
                    start_index=int(shard_spec.get("start_index", 0)),
                    num_problems=shard_spec.get("num_problems"),
                ):
                    continue
                shard_suffix = _shard_suffix(shard_index, total_shards=len(shard_specs))
                shard_job_name = f"{base_job_name}{shard_suffix}"
                shard_target = _single_target_key(
                    out_dir=out_dir,
                    start_index=shard_spec.get("start_index", 0),
                    num_problems=shard_spec.get("num_problems"),
                    partition=args.partition,
                )
                shard_array_target = _array_target_key(
                    out_dir=out_dir,
                    shard_index=shard_index,
                    partition=args.partition,
                )
                time_limit = _time_limit(
                    row,
                    selection=args.selection,
                    shard_num_problems=shard_spec.get("num_problems"),
                )
                race_scope_dir = _race_scope_dir(Path(out_dir), shard_spec)

                if args.dual_submit:
                    primary_job_name = f"{shard_job_name}_h100"
                    mirror_job_name = f"{shard_job_name}_l40s"
                    primary_target = _single_target_key(
                        out_dir=out_dir,
                        start_index=shard_spec.get("start_index", 0),
                        num_problems=shard_spec.get("num_problems"),
                        partition=args.partition,
                    )
                    mirror_target = _single_target_key(
                        out_dir=out_dir,
                        start_index=shard_spec.get("start_index", 0),
                        num_problems=shard_spec.get("num_problems"),
                        partition=args.mirror_partition,
                    )
                    primary_job_id = None
                    mirror_job_id = None

                    if primary_job_name not in queued_single_jobs and primary_target not in queued_single_targets:
                        _wait_for_submission_slot(
                            max_queued_jobs=args.max_queued_jobs,
                            poll_seconds=args.queue_poll_seconds,
                        )
                        primary_job_id = _submit_job(
                            manifest=args.manifest,
                            out_dir=out_dir,
                            num_problems=shard_spec.get("num_problems"),
                            start_index=shard_spec.get("start_index", 0),
                            run_official_eval=args.run_official_eval,
                            overwrite=args.overwrite,
                            account=args.account,
                            partition=args.partition,
                            gres=args.gres,
                            cpus_per_task=args.cpus_per_task,
                            mem=mem,
                            time_limit=time_limit,
                            job_name=primary_job_name,
                            use_race_wrapper=True,
                            race_scope_dir=race_scope_dir,
                            venv_path=args.venv_path,
                            module_loads=args.module_loads,
                            runtime_cache_dir=args.runtime_cache_dir,
                        )
                        queued_single_jobs.add(primary_job_name)
                        queued_single_targets.add(primary_target)

                    if mirror_job_name not in queued_single_jobs and mirror_target not in queued_single_targets:
                        _wait_for_submission_slot(
                            max_queued_jobs=args.max_queued_jobs,
                            poll_seconds=args.queue_poll_seconds,
                        )
                        mirror_job_id = _submit_job(
                            manifest=args.manifest,
                            out_dir=out_dir,
                            num_problems=shard_spec.get("num_problems"),
                            start_index=shard_spec.get("start_index", 0),
                            run_official_eval=args.run_official_eval,
                            overwrite=args.overwrite,
                            account=args.account,
                            partition=args.mirror_partition,
                            gres=args.mirror_gres,
                            cpus_per_task=args.cpus_per_task,
                            mem=mem,
                            time_limit=time_limit,
                            job_name=mirror_job_name,
                            use_race_wrapper=True,
                            race_scope_dir=race_scope_dir,
                            venv_path=args.venv_path,
                            module_loads=args.module_loads,
                            runtime_cache_dir=args.runtime_cache_dir,
                        )
                        queued_single_jobs.add(mirror_job_name)
                        queued_single_targets.add(mirror_target)

                    if primary_job_id or mirror_job_id:
                        _write_race_metadata(
                            output_dir=race_scope_dir,
                            job_ids=[job_id for job_id in [primary_job_id, mirror_job_id] if job_id],
                            partitions=[
                                partition
                                for partition, job_id in [
                                    (args.partition, primary_job_id),
                                    (args.mirror_partition, mirror_job_id),
                                ]
                                if job_id
                            ],
                        )

                    if primary_job_id or mirror_job_id:
                        print(
                            f"submitted {shard_job_name}: "
                            f"h100={primary_job_id or 'queued'}, l40s={mirror_job_id or 'queued'}, "
                            f"start={shard_spec.get('start_index', 0)}, "
                            f"count={shard_spec.get('num_problems')}, "
                            f"time={time_limit}"
                        )
                    else:
                        print(
                            f"skipped {shard_job_name}: already queued on both partitions "
                            f"(start={shard_spec.get('start_index', 0)}, count={shard_spec.get('num_problems')})"
                        )
                else:
                    if args.array_submit and len(shard_specs) > 1:
                        break
                    job_name = f"{shard_job_name}{args.job_name_suffix}"
                    if (
                        job_name in queued_single_jobs
                        or shard_target in queued_single_targets
                        or shard_array_target in queued_array_targets
                    ):
                        print(
                            f"skipped {job_name}: already queued "
                            f"(start={shard_spec.get('start_index', 0)}, count={shard_spec.get('num_problems')})"
                        )
                        continue
                    _wait_for_submission_slot(
                        max_queued_jobs=args.max_queued_jobs,
                        poll_seconds=args.queue_poll_seconds,
                    )
                    job_id = _submit_job(
                        manifest=args.manifest,
                        out_dir=out_dir,
                        num_problems=shard_spec.get("num_problems"),
                        start_index=shard_spec.get("start_index", 0),
                        run_official_eval=args.run_official_eval,
                        overwrite=args.overwrite,
                        account=args.account,
                        partition=args.partition,
                        gres=args.gres,
                        cpus_per_task=args.cpus_per_task,
                        mem=mem,
                        time_limit=time_limit,
                        job_name=job_name,
                        use_race_wrapper=False,
                        race_scope_dir=None,
                        venv_path=args.venv_path,
                        module_loads=args.module_loads,
                        runtime_cache_dir=args.runtime_cache_dir,
                    )
                    queued_single_jobs.add(job_name)
                    queued_single_targets.add(shard_target)
                    print(
                        f"submitted {job_name}: {job_id} "
                        f"(start={shard_spec.get('start_index', 0)}, count={shard_spec.get('num_problems')}, time={time_limit})"
                    )

            if not args.dual_submit and args.array_submit and len(shard_specs) > 1:
                array_job_name = f"{base_job_name}{args.job_name_suffix}"
                missing_indices = []
                for shard_index, shard_spec in enumerate(shard_specs):
                    if _is_shard_complete(
                        output_dir=Path(out_dir),
                        start_index=int(shard_spec.get("start_index", 0)),
                        num_problems=shard_spec.get("num_problems"),
                    ):
                        continue
                    shard_name = f"{base_job_name}{_shard_suffix(shard_index, len(shard_specs))}{args.job_name_suffix}"
                    shard_target = _single_target_key(
                        out_dir=out_dir,
                        start_index=shard_spec.get("start_index", 0),
                        num_problems=shard_spec.get("num_problems"),
                        partition=args.partition,
                    )
                    array_target = _array_target_key(
                        out_dir=out_dir,
                        shard_index=shard_index,
                        partition=args.partition,
                    )
                    if (
                        shard_name in queued_single_jobs
                        or (array_job_name, shard_index) in queued_array_tasks
                        or shard_target in queued_single_targets
                        or array_target in queued_array_targets
                    ):
                        continue
                    missing_indices.append(shard_index)

                if not missing_indices:
                    print(f"skipped {array_job_name}: all shard tasks already queued")
                else:
                    _wait_for_submission_slot(
                        max_queued_jobs=args.max_queued_jobs,
                        poll_seconds=args.queue_poll_seconds,
                    )
                    max_time_limit = max(
                        _time_limit(
                            row,
                            selection=args.selection,
                            shard_num_problems=shard_specs[shard_index].get("num_problems"),
                        )
                        for shard_index in missing_indices
                    )
                    array_spec = _array_index_spec(missing_indices)
                    job_id = _submit_job_array(
                        manifest=args.manifest,
                        out_dir=out_dir,
                        run_official_eval=args.run_official_eval,
                        overwrite=args.overwrite,
                        account=args.account,
                        partition=args.partition,
                        gres=args.gres,
                        cpus_per_task=args.cpus_per_task,
                        mem=mem,
                        time_limit=max_time_limit,
                        job_name=array_job_name,
                        array_spec=array_spec,
                        array_max_concurrent=args.array_max_concurrent,
                        venv_path=args.venv_path,
                        module_loads=args.module_loads,
                        runtime_cache_dir=args.runtime_cache_dir,
                    )
                    queued_array_tasks.update((array_job_name, shard_index) for shard_index in missing_indices)
                    queued_array_targets.update(
                        _array_target_key(out_dir=out_dir, shard_index=shard_index, partition=args.partition)
                        for shard_index in missing_indices
                    )
                    print(
                        f"submitted {array_job_name}: {job_id} "
                        f"(array={array_spec}, shards={len(missing_indices)}, max_time={max_time_limit})"
                    )
            update_manifest_status(args.manifest, out_dir, "running")
        except Exception as exc:
            failures.append((base_job_name, str(exc)))
            print(f"failed {base_job_name}: {exc}", flush=True)

    if failures:
        print(f"submission completed with {len(failures)} failures", flush=True)
        for job_name, message in failures:
            print(f"failure {job_name}: {message}", flush=True)


def _submit_job(
    *,
    manifest: Path,
    out_dir: str,
    num_problems: int | None,
    start_index: int,
    run_official_eval: bool,
    overwrite: bool,
    account: str,
    partition: str,
    gres: str,
    cpus_per_task: int,
    mem: str,
    time_limit: str,
    job_name: str,
    use_race_wrapper: bool,
    race_scope_dir: Path | None,
    venv_path: str,
    module_loads: str,
    runtime_cache_dir: str | None,
):
    cache_parts = []
    if runtime_cache_dir:
        cache_parts.extend(
            [
                f"mkdir -p '{runtime_cache_dir}/flashinfer' '{runtime_cache_dir}/triton';",
                f"export FLASHINFER_WORKSPACE_BASE='{runtime_cache_dir}/flashinfer';",
                f"export TRITON_CACHE_DIR='{runtime_cache_dir}/triton';",
            ]
        )
    cmd_parts = [
        "source /etc/profile;",
        f"module load {module_loads};" if module_loads else "",
        f"source {venv_path}/bin/activate;",
        f"cd {ROOT}; export PYTHONUNBUFFERED=1; export PYTHONNOUSERSITE=1;",
        *cache_parts,
        f"python {'artifacts/phase1/race_run_job.py' if use_race_wrapper else 'artifacts/phase1/run_phase1_job.py'}",
        f"--manifest-path {manifest}",
        f"--expected-output-dir '{out_dir}'",
    ]
    if num_problems is not None:
        cmd_parts.append(f"--num-problems {num_problems}")
    if start_index:
        cmd_parts.append(f"--start-index {start_index}")
    if use_race_wrapper and race_scope_dir is not None:
        cmd_parts.append(f"--race-scope-dir '{race_scope_dir}'")
    cmd_parts.append("--strict-official-eval" if run_official_eval else "--skip-official-eval")
    if overwrite:
        cmd_parts.append("--overwrite")
    cmd = " ".join(cmd_parts)

    return _run_sbatch(
        [
            "sbatch",
            "--parsable",
            f"--account={account}",
            f"--partition={partition}",
            f"--gres={gres}",
            f"--cpus-per-task={cpus_per_task}",
            f"--mem={mem}",
            f"--time={time_limit}",
            f"--job-name={job_name}",
            f"--output={ROOT / 'archives' / 'logs' / (job_name + '_%j.out')}",
            f"--error={ROOT / 'archives' / 'logs' / (job_name + '_%j.err')}",
            "--wrap",
            f"bash -lc {cmd!r}",
        ],
        job_name=job_name,
    )


def _submit_job_array(
    *,
    manifest: Path,
    out_dir: str,
    run_official_eval: bool,
    overwrite: bool,
    account: str,
    partition: str,
    gres: str,
    cpus_per_task: int,
    mem: str,
    time_limit: str,
    job_name: str,
    array_spec: str,
    array_max_concurrent: int,
    venv_path: str,
    module_loads: str,
    runtime_cache_dir: str | None,
):
    cache_parts = []
    if runtime_cache_dir:
        cache_parts.extend(
            [
                f"mkdir -p '{runtime_cache_dir}/flashinfer' '{runtime_cache_dir}/triton';",
                f"export FLASHINFER_WORKSPACE_BASE='{runtime_cache_dir}/flashinfer';",
                f"export TRITON_CACHE_DIR='{runtime_cache_dir}/triton';",
            ]
        )
    cmd_parts = [
        "source /etc/profile;",
        f"module load {module_loads};" if module_loads else "",
        f"source {venv_path}/bin/activate;",
        f"cd {ROOT}; export PYTHONUNBUFFERED=1; export PYTHONNOUSERSITE=1;",
        *cache_parts,
        "python artifacts/phase1/run_phase1_shard_from_array.py",
        f"--manifest-path {manifest}",
        f"--expected-output-dir '{out_dir}'",
    ]
    cmd_parts.append("--strict-official-eval" if run_official_eval else "--skip-official-eval")
    if overwrite:
        cmd_parts.append("--overwrite")
    cmd = " ".join(cmd_parts)

    sbatch_array_spec = array_spec
    if array_max_concurrent > 0:
        sbatch_array_spec = f"{array_spec}%{array_max_concurrent}"

    return _run_sbatch(
        [
            "sbatch",
            "--parsable",
            f"--account={account}",
            f"--partition={partition}",
            f"--gres={gres}",
            f"--cpus-per-task={cpus_per_task}",
            f"--mem={mem}",
            f"--time={time_limit}",
            f"--job-name={job_name}",
            f"--array={sbatch_array_spec}",
            f"--output={ROOT / 'archives' / 'logs' / (job_name + '_%A_%a.out')}",
            f"--error={ROOT / 'archives' / 'logs' / (job_name + '_%A_%a.err')}",
            "--wrap",
            f"bash -lc {cmd!r}",
        ],
        job_name=job_name,
    )


def _write_race_metadata(output_dir: Path, job_ids: list[str], partitions: list[str]):
    race_dir = output_dir / ".race"
    race_dir.mkdir(parents=True, exist_ok=True)
    peers_path = race_dir / "peers.json"
    peers_path.write_text(
        json.dumps(
            {
                "job_ids": job_ids,
                "partitions": partitions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_sbatch(
    cmd: list[str],
    *,
    job_name: str,
    max_attempts: int = 8,
    timeout_seconds: float = 180.0,
) -> str:
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = f"sbatch timed out after {timeout_seconds:.0f}s"
            if attempt < max_attempts:
                sleep_seconds = min(60.0, 2.0 * attempt)
                print(
                    f"retrying sbatch for {job_name} after timeout "
                    f"(attempt {attempt}/{max_attempts}, sleep={sleep_seconds:.0f}s)",
                    flush=True,
                )
                time.sleep(sleep_seconds)
                continue
            raise RuntimeError(f"sbatch failed for {job_name}: {last_error}") from exc

        if result.returncode == 0:
            return result.stdout.strip()

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        combined = "\n".join(part for part in [stdout, stderr] if part).lower()
        last_error = "\n".join(part for part in [stdout, stderr] if part)
        retryable = any(pattern in combined for pattern in SBATCH_RETRY_PATTERNS)
        if retryable and attempt < max_attempts:
            sleep_seconds = min(60.0, 2.0 * attempt)
            print(
                f"retrying sbatch for {job_name} after transient error "
                f"(attempt {attempt}/{max_attempts}, sleep={sleep_seconds:.0f}s): {stderr or stdout}",
                flush=True,
            )
            time.sleep(sleep_seconds)
            continue
        break

    raise RuntimeError(f"sbatch failed for {job_name}: {last_error or 'unknown error'}")


def _write_shard_plan(output_dir: Path, shard_specs: list[dict]) -> None:
    shard_root = output_dir / ".shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    (shard_root / "plan.json").write_text(
        json.dumps({"shards": shard_specs}, indent=2),
        encoding="utf-8",
    )


def _race_scope_dir(output_dir: Path, shard_spec: dict) -> Path:
    return output_dir / ".race_shards" / _shard_key(
        start_index=int(shard_spec["start_index"]),
        num_problems=shard_spec.get("num_problems"),
    )


def _shard_key(start_index: int, num_problems: int | None) -> str:
    count_label = "all" if num_problems is None else str(num_problems)
    return f"start_{start_index:05d}_count_{count_label}"


def _shard_suffix(shard_index: int, total_shards: int) -> str:
    if total_shards <= 1:
        return ""
    return f"_s{shard_index:03d}"


def _base_job_name(row: dict) -> str:
    variant = _job_variant_token(row)
    return (
        f"p1_{row['dataset']}_{lm_short_name(row['lm'])}_{rm_short_name(row['rm'])}_"
        f"{row['method']}_N{row['budget']}_{variant}"
    )


def _job_variant_token(row: dict) -> str:
    scope = "|".join(
        [
            str(row.get("phase", "")),
            str(row.get("subphase", "")),
            str(row.get("config_label", "")),
            str(row.get("rm", "")),
            str(row.get("action_strategy_override", "")),
            str(row.get("output_mode_override", "")),
            str(row.get("structured_answer_test_mode", "")),
            str(row.get("local_first_block_split_test_mode", "")),
            str(row.get("prefix_normalization_test_mode", "")),
            str(row.get("expected_output_dir", "")),
        ]
    )
    readable = _slug_token(str(row.get("config_label") or row.get("subphase") or "default"), max_len=16)
    digest = hashlib.sha1(scope.encode("utf-8")).hexdigest()[:8]
    return f"{readable}_{digest}"


def _slug_token(value: str, *, max_len: int) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not token:
        token = "default"
    return token[:max_len].rstrip("_") or "default"


def _is_resolved(row):
    return _is_parent_output_complete(Path(row["expected_output_dir"]))


def _is_parent_output_complete(output_dir: Path) -> bool:
    final_summary_path = output_dir / "final_summary.json"
    results_path = output_dir / "results.json"
    if not final_summary_path.exists() or not results_path.exists():
        return False

    plan_path = output_dir / ".shards" / "plan.json"
    if not plan_path.exists():
        return True

    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    shard_specs = plan.get("shards", [])
    if len(shard_specs) <= 1:
        return True
    expected_total = 0
    for shard_spec in shard_specs:
        num_problems = shard_spec.get("num_problems")
        if num_problems is None:
            return False
        expected_total += int(num_problems)
        if not _is_shard_complete(
            output_dir=output_dir,
            start_index=int(shard_spec["start_index"]),
            num_problems=num_problems,
        ):
            return False

    try:
        merged_payload = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return len(merged_payload.get("experiments", [])) == expected_total


def _is_shard_complete(*, output_dir: Path, start_index: int, num_problems: int | None) -> bool:
    if num_problems is None:
        return _is_parent_output_complete(output_dir)
    shard_results_path = _shard_output_dir(
        output_dir=output_dir,
        start_index=start_index,
        num_problems=num_problems,
    ) / "results.json"
    return _validate_shard_results(
        results_path=shard_results_path,
        start_index=start_index,
        num_problems=num_problems,
    )


def _shard_output_dir(*, output_dir: Path, start_index: int, num_problems: int | None) -> Path:
    if start_index <= 0 and not num_problems:
        return output_dir
    count_label = "all" if num_problems is None else str(num_problems)
    return output_dir / "shards" / f"start_{start_index:05d}_count_{count_label}"


def _validate_shard_results(*, results_path: Path, start_index: int, num_problems: int | None) -> bool:
    if not results_path.exists():
        return False
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    experiments = payload.get("experiments", [])
    if num_problems is None:
        return False

    expected_indices = list(range(start_index, start_index + int(num_problems)))
    actual_indices = sorted(
        row.get("source_index", row.get("experiment_index", 0) - 1)
        for row in experiments
    )
    return len(experiments) == int(num_problems) and actual_indices == expected_indices


def _select_rows(rows, selection):
    active_methods = {"best_of_n", "bbon", "mean_search", "slg"}
    if selection == "all":
        return [row for row in rows if row["method"] != "darwin"]
    if selection == "priority1":
        # We intentionally skip the N=2048 tier for the current priority1 rollout
        # to keep the schedule tractable and avoid re-queueing the heaviest shard set.
        return [
            row
            for row in rows
            if row["priority"] == 1 and row["method"] != "darwin" and int(row["budget"]) != 2048
        ]
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


def _shard_specs_for(*, row: dict, selection: str, shard_strategy: str) -> list[dict]:
    num_problems = _num_problems_for(selection, row)
    if num_problems is not None:
        return [{"start_index": 0, "num_problems": num_problems}]

    if shard_strategy == "none":
        return [{"start_index": 0, "num_problems": None}]

    dataset = row["dataset"]
    problem_indices = row.get("problem_indices") or []
    if row.get("dataset_size_override") is not None:
        total = int(row["dataset_size_override"])
    elif problem_indices:
        total = len(problem_indices)
    else:
        total = INSTRUCTION_DATASET_SIZES.get(dataset)
    if total is None:
        return [{"start_index": 0, "num_problems": None}]

    budget = int(row["budget"])
    shard_size = row.get("shard_size_override") or _lookup_shard_size(dataset, budget)
    if shard_size is not None:
        shard_size = int(shard_size)
    if shard_size is None or shard_size >= total:
        return [{"start_index": 0, "num_problems": None}]

    return [
        {
            "start_index": start_index,
            "num_problems": min(shard_size, total - start_index),
        }
        for start_index in range(0, total, shard_size)
    ]


def _lookup_shard_size(dataset: str, budget: int) -> int | None:
    size_map = L40_SAFE_SHARD_SIZES.get(dataset)
    if not size_map:
        return None
    if budget in size_map:
        return size_map[budget]

    eligible_budgets = sorted(size_map)
    for candidate in eligible_budgets:
        if budget <= candidate:
            return size_map[candidate]
    return size_map[eligible_budgets[-1]]


def _lm_time_family(lm_name: str | None) -> str | None:
    if not lm_name:
        return None
    lm_name = lm_name.lower()
    if "qwen3-8b" in lm_name:
        return "qwen3_8b"
    if "llama-3.1-8b" in lm_name or "llama31_8b" in lm_name:
        return "llama31_8b"
    return None


def _empirical_prompt_seconds(row: dict) -> float | None:
    family = _lm_time_family(row.get("lm"))
    if family is None:
        return None
    dataset_map = EMPIRICAL_PROMPT_SECONDS.get(family, {}).get(row["dataset"], {})
    budget = int(row["budget"])
    if budget in dataset_map:
        return dataset_map[budget]

    smaller = [candidate for candidate in dataset_map if candidate < budget]
    larger = [candidate for candidate in dataset_map if candidate > budget]
    if smaller and larger:
        low = max(smaller)
        high = min(larger)
        low_value = dataset_map[low]
        high_value = dataset_map[high]
        span = high - low
        weight = (budget - low) / span
        return low_value + (high_value - low_value) * weight
    if smaller:
        return dataset_map[max(smaller)]
    if larger:
        return dataset_map[min(larger)]
    return None


def _bucket_time_limit(hours: float) -> str:
    if hours <= 2.95:
        return "02:59:00"
    if hours <= 5.95:
        return "05:59:00"
    if hours <= 8.95:
        return "08:59:00"
    return "11:59:00"


def _single_target_key(*, out_dir: str, start_index: int, num_problems: int | None, partition: str) -> tuple[str, int, int | None, str]:
    return (out_dir, int(start_index), None if num_problems is None else int(num_problems), partition)


def _array_target_key(*, out_dir: str, shard_index: int, partition: str) -> tuple[str, int, str]:
    return (out_dir, int(shard_index), partition)


def _queued_jobs() -> tuple[
    set[str],
    set[tuple[str, int]],
    set[tuple[str, int, int | None, str]],
    set[tuple[str, int, str]],
]:
    result = subprocess.run(
        ["squeue", "-r", "-u", getpass.getuser(), "-h", "-o", "%i|%.200j|%K"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    single_jobs: set[str] = set()
    array_tasks: set[tuple[str, int]] = set()
    single_targets: set[tuple[str, int, int | None, str]] = set()
    array_targets: set[tuple[str, int, str]] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        job_id, job_name, task_id = line.split("|", 2)
        job_name = job_name.strip()
        task_id = task_id.strip()
        if task_id == "N/A":
            single_jobs.add(job_name)
            continue
        try:
            array_tasks.add((job_name, int(task_id)))
        except ValueError:
            continue
    return single_jobs, array_tasks, single_targets, array_targets


def _queued_submit_lines(job_ids: list[str]) -> dict[str, str]:
    if not job_ids:
        return {}

    submit_lines: dict[str, str] = {}
    batch_size = 200
    for start in range(0, len(job_ids), batch_size):
        batch = job_ids[start : start + batch_size]
        result = subprocess.run(
            [
                "sacct",
                "-X",
                "-P",
                "-n",
                "--format=JobIDRaw,State,SubmitLine",
                "-j",
                ",".join(batch),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) != 3:
                continue
            job_id_raw, state, submit_line = parts
            if state not in {"PENDING", "RUNNING"} or not submit_line:
                continue
            submit_lines[job_id_raw.strip()] = submit_line.strip()
    return submit_lines


def _extract_submit_arg(submit_line: str, flag: str) -> str | None:
    quoted = re.search(rf"{re.escape(flag)}\s+'([^']+)'", submit_line)
    if quoted:
        return quoted.group(1)
    plain = re.search(rf"{re.escape(flag)}\s+(\S+)", submit_line)
    if plain:
        return plain.group(1)
    return None


def _extract_submit_partition(submit_line: str) -> str | None:
    match = re.search(r"--partition=([^\s]+)", submit_line)
    if not match:
        return None
    return match.group(1)


def _queued_job_count() -> int:
    result = subprocess.run(
        ["squeue", "-u", getpass.getuser(), "-h", "-o", "%j"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    # Keep the queue cap scoped to Phase 1 jobs only. Other experiment lines
    # under the same account should not block this submitter from backfilling
    # the intended p1_* queue budget.
    return sum(
        1
        for line in result.stdout.splitlines()
        if line.strip().startswith("p1_")
    )


def _wait_for_submission_slot(*, max_queued_jobs: int, poll_seconds: int) -> None:
    if max_queued_jobs <= 0:
        return

    while True:
        queued_jobs = _queued_job_count()
        if queued_jobs < max_queued_jobs:
            return
        print(
            f"queued jobs {queued_jobs} >= limit {max_queued_jobs}; "
            f"sleeping {poll_seconds}s before next submission",
            flush=True,
        )
        time.sleep(max(1, poll_seconds))


def _queued_job_names() -> set[str]:
    single_jobs, _ = _queued_jobs()
    return single_jobs


def _array_index_spec(indices: list[int]) -> str:
    if not indices:
        raise ValueError("indices must not be empty")

    ranges: list[str] = []
    start = prev = indices[0]
    for index in indices[1:]:
        if index == prev + 1:
            prev = index
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = index
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def _time_limit(row, selection=None, shard_num_problems: int | None = None):
    if selection == "smoke":
        if row["dataset"] == "ifbench":
            return "02:00:00"
        return "01:30:00"

    budget = int(row["budget"])
    dataset = row["dataset"]
    shard_size = shard_num_problems or _lookup_shard_size(dataset, budget)

    empirical_seconds = _empirical_prompt_seconds(row)
    if empirical_seconds is not None and shard_size is not None:
        empirical_hours = empirical_seconds * shard_size / 3600.0
        # Add a moderate safety margin and fixed startup overhead, then snap to the
        # cluster's favorable queue buckets (<3h, <6h, <9h, <12h).
        return _bucket_time_limit(empirical_hours * 1.12 + 0.2)

    if dataset == "ifbench":
        if budget <= 256:
            base_hours = 5.0
        elif budget <= 512:
            base_hours = 7.0
        else:
            base_hours = 9.0
    elif budget <= 256:
        base_hours = 4.0
    elif budget <= 512:
        base_hours = 6.0
    elif budget <= 1024:
        base_hours = 8.0
    else:
        base_hours = 9.0

    if dataset != "ifbench" and budget <= 128:
        base_hours = 3.0

    if dataset == "wildbench_v2" and budget >= 512:
        base_hours += 1.0

    reference_shard_size = _lookup_shard_size(dataset, budget)
    if shard_num_problems is not None and reference_shard_size and shard_num_problems < reference_shard_size:
        scaled_hours = base_hours * (shard_num_problems / reference_shard_size)
        # Tail shards still need non-trivial fixed overhead for model load and eval.
        base_hours = max(2.0, scaled_hours + 0.5)

    return _bucket_time_limit(base_hours)


if __name__ == "__main__":
    main()
