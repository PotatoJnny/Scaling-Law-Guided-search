import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.search.benchmark_eval import evaluate_results
from experiments.search.common import (
    PHASE1_DEFAULTS_PATH,
    PHASE1_SEEDS_PATH,
    ROOT,
    git_metadata,
    lm_short_name,
    load_generation_defaults,
    load_seed_config,
    read_jsonl,
    update_manifest_status,
)
from tasks.dataset_configs import DATASET_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--expected-output-dir", type=str, required=True)
    parser.add_argument("--num-problems", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-official-eval", action="store_true")
    parser.add_argument("--strict-official-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    entry = _find_entry(args.manifest_path, args.expected_output_dir)
    output_dir = Path(entry["expected_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir = _resolve_run_output_dir(
        output_dir=output_dir,
        start_index=args.start_index,
        num_problems=args.num_problems,
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)

    update_manifest_status(args.manifest_path, args.expected_output_dir, "running")
    try:
        parent_runtime_config = _build_runtime_config(entry, output_dir)
        parent_runtime_config_path = output_dir / "phase1_runtime_config.json"
        with open(parent_runtime_config_path, "w", encoding="utf-8") as f:
            json.dump(parent_runtime_config, f, ensure_ascii=False, indent=2)

        runtime_config = _build_runtime_config(entry, run_output_dir)
        runtime_config_path = run_output_dir / "phase1_runtime_config.json"
        with open(runtime_config_path, "w", encoding="utf-8") as f:
            json.dump(runtime_config, f, ensure_ascii=False, indent=2)

        if args.prepare_only:
            print(f"Prepared config at {runtime_config_path}")
            update_manifest_status(args.manifest_path, args.expected_output_dir, "planned")
            return

        results_path = run_output_dir / "results.json"
        if args.overwrite or not results_path.exists():
            _run_vllm_runtime_preflight(runtime_config)
            cmd = ["python", "main.py", "--config", str(runtime_config_path)]
            if args.num_problems is not None:
                cmd.extend(["--num_problems", str(args.num_problems)])
            if args.start_index:
                cmd.extend(["--start_index", str(args.start_index)])
            subprocess.run(cmd, cwd=ROOT, check=True)

        if _is_sharded_run(output_dir, args.start_index, args.num_problems):
            shard_ok, shard_message = _validate_shard_results(
                results_path=results_path,
                start_index=args.start_index,
                num_problems=args.num_problems,
            )
            if not shard_ok:
                raise RuntimeError(f"Incomplete shard output: {shard_message}")
            merged = _maybe_merge_shards(
                entry=entry,
                output_dir=output_dir,
                runtime_config_path=parent_runtime_config_path,
                skip_official_eval=args.skip_official_eval,
                strict_official_eval=args.strict_official_eval,
            )
            if merged:
                update_manifest_status(args.manifest_path, args.expected_output_dir, "completed")
                print(f"Wrote merged final summary to {output_dir / 'final_summary.json'}")
            else:
                print(
                    "Shard finished successfully; waiting for remaining shards before "
                    "writing parent results/final summary."
                )
            return

        final_summary = _build_final_summary(entry, output_dir, parent_runtime_config_path)
        _export_best_answers(output_dir / "results.json", output_dir)
        if not args.skip_official_eval:
            official_eval = evaluate_results(
                results_json=output_dir / "results.json",
                dataset=entry["dataset"],
                output_dir=output_dir,
                model_pretty_name=lm_short_name(entry["lm"]),
                strict=args.strict_official_eval,
            )
            final_summary.update(official_eval)
        else:
            final_summary.update(
                {
                    "official_metric_name": None,
                    "official_score": None,
                    "official_eval_status": "deferred",
                    "official_eval_note": "Official benchmark-native evaluation deferred until API/judge access is available.",
                }
            )

        final_summary = _apply_headline_metric(final_summary)
        with open(output_dir / "final_summary.json", "w", encoding="utf-8") as f:
            json.dump(final_summary, f, ensure_ascii=False, indent=2)

        update_manifest_status(args.manifest_path, args.expected_output_dir, "completed")
        print(f"Wrote final summary to {output_dir / 'final_summary.json'}")
    except Exception:
        update_manifest_status(args.manifest_path, args.expected_output_dir, "failed")
        raise


def _find_entry(manifest_path: Path, expected_output_dir: str) -> dict:
    for row in read_jsonl(manifest_path):
        if row.get("expected_output_dir") == expected_output_dir:
            return row
    raise ValueError(f"No manifest entry found for {expected_output_dir}")


def _run_vllm_runtime_preflight(runtime_config: dict) -> None:
    hardware = runtime_config.get("hardware", {})
    if hardware.get("backend", "vllm") != "vllm":
        return

    check_script = ROOT / "experiments" / "search" / "check_vllm_runtime.py"
    subprocess.run(
        ["python", str(check_script)],
        cwd=ROOT,
        check=True,
    )


def _build_runtime_config(entry: dict, output_dir: Path) -> dict:
    defaults = load_generation_defaults()
    seeds = load_seed_config()

    lm_name = entry["lm"]
    method = entry["method"]
    budget = int(entry["budget"])
    lm_short = lm_short_name(lm_name)
    dataset_cfg = DATASET_CONFIGS[entry["dataset"]]
    seed_value = entry.get("seed_override", seeds.get("default_seed", 42))
    max_new_tokens = entry.get(
        "max_new_tokens_override",
        dataset_cfg.get("generation_max_new_tokens", defaults["generation"]["max_new_tokens"]),
    )
    output_mode = entry.get("output_mode_override", defaults["generation"].get("output_mode", "answer_only_no_think"))
    action_strategy = entry.get("action_strategy_override", defaults["action_strategy"])

    hardware_profile = defaults["hardware_profiles"].get(lm_short, defaults["hardware_profiles"]["default"])
    sampling_key = "sampled"
    sampling = defaults["sampling"][sampling_key]

    max_model_len = hardware_profile.get(
        "max_model_len",
        defaults["generation"]["max_model_len"],
    )
    max_num_batched_tokens = max(
        hardware_profile["max_num_batched_tokens"],
        max_model_len,
    )

    config = {
        "experiment_name": output_dir.name,
        "algorithm": _algorithm_name(method),
        "output_mode": output_mode,
        "seed": seed_value,
        "output_dir": str(output_dir),
        "hardware": {
            "lm_name": lm_name,
            "rm_name": entry["rm"],
            "lm_max_batch_size": hardware_profile.get(
                "lm_max_batch_size",
                defaults["generation"]["lm_max_batch_size"],
            ),
            "lm_max_prompt_batch_size": hardware_profile.get(
                "lm_max_prompt_batch_size",
                defaults["generation"].get("lm_max_prompt_batch_size"),
            ),
            "lm_max_samples_per_call": hardware_profile.get(
                "lm_max_samples_per_call",
                defaults["generation"].get("lm_max_samples_per_call"),
            ),
            "rm_max_batch_size": hardware_profile.get(
                "rm_max_batch_size",
                defaults["generation"]["rm_max_batch_size"],
            ),
            "gpu_memory_utilization": hardware_profile["gpu_memory_utilization"],
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "enforce_eager": hardware_profile.get(
                "enforce_eager",
                defaults["generation"]["enforce_eager"],
            ),
            "backend": hardware_profile.get("backend", "vllm"),
            "attention_backend": hardware_profile.get("attention_backend"),
            "enable_chunked_prefill": hardware_profile.get("enable_chunked_prefill"),
            "enable_prefix_caching": hardware_profile.get("enable_prefix_caching"),
        },
        "task_setup": {
            "dataset_name": entry["dataset"],
            "dataset_config": None,
            "action_strategy": action_strategy,
            "dataset_config_overrides": entry.get("dataset_config_overrides") or {},
            "action_strategy_overrides": entry.get("action_strategy_overrides") or {},
            "output_mode": output_mode,
            "problem_indices": entry.get("problem_indices"),
            "legacy_prompt_path": entry.get("legacy_prompt_path", False),
            "structured_answer_test_mode": entry.get("structured_answer_test_mode", False),
            "local_first_block_split_test_mode": entry.get(
                "local_first_block_split_test_mode",
                entry.get("structured_answer_test_mode", False),
            ),
            "prefix_normalization_test_mode": entry.get(
                "prefix_normalization_test_mode",
                False,
            ),
        },
        "phase1_metadata": {
            "manifest_entry": entry,
            "generation_defaults_path": str(PHASE1_DEFAULTS_PATH),
            "seed_config_path": str(PHASE1_SEEDS_PATH),
        },
    }

    hardware_overrides = entry.get("hardware_overrides") or {}
    if hardware_overrides:
        config["hardware"].update(hardware_overrides)

    if method == "best_of_n":
        config["bon_params"] = {
            "N": budget,
            "max_tokens": max_new_tokens,
            "verbose": False,
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "seed": seed_value,
        }
        bon_overrides = entry.get("bon_param_overrides") or {}
        if bon_overrides:
            config["bon_params"].update(bon_overrides)
    elif method == "slg":
        config["slg_params"] = _shared_tree_search_params(
            defaults,
            sampling,
            budget,
            max_new_tokens=max_new_tokens,
            seed_value=seed_value,
        )
        slg_overrides = entry.get("slg_param_overrides") or {}
        if slg_overrides:
            config["slg_params"].update(slg_overrides)
    elif method == "mean_search":
        config["slg_params"] = _shared_tree_search_params(
            defaults,
            sampling,
            budget,
            max_new_tokens=max_new_tokens,
            seed_value=seed_value,
        )
        slg_overrides = entry.get("slg_param_overrides") or {}
        if slg_overrides:
            config["slg_params"].update(slg_overrides)
    elif method == "bbon":
        config["bbon_params"] = _shared_tree_search_params(
            defaults,
            sampling,
            budget,
            max_new_tokens=max_new_tokens,
            seed_value=seed_value,
        )
        bbon_overrides = entry.get("bbon_param_overrides") or {}
        if bbon_overrides:
            config["bbon_params"].update(bbon_overrides)
    else:
        config["bon_params"] = {
            "N": budget,
            "max_tokens": max_new_tokens,
            "verbose": False,
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "seed": seeds.get("default_seed", 42),
        }

    if method in {"slg", "mean_search"} and entry.get("slg_auto_param_strategy"):
        config["slg_params"]["auto_param_strategy"] = entry["slg_auto_param_strategy"]
    if method == "bbon" and entry.get("slg_auto_param_strategy"):
        config["bbon_params"]["auto_param_strategy"] = entry["slg_auto_param_strategy"]

    return config


def _shared_tree_search_params(
    defaults: dict,
    sampling: dict,
    budget: int,
    max_new_tokens: int,
    seed_value: int,
) -> dict:
    slg_defaults = defaults["search"]["slg"]
    params = {
        "N": budget,
        "K": -1,
        "m": -1,
        "max_depth": slg_defaults["max_depth"],
        "max_tokens": max_new_tokens,
        "verbose": False,
        "auto_param_strategy": slg_defaults["auto_param_strategy"],
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "seed": seed_value,
    }
    for optional_key in ["min_m", "default_K", "max_explore_fraction"]:
        if optional_key in slg_defaults:
            params[optional_key] = slg_defaults[optional_key]
    return params


def _algorithm_name(method: str) -> str:
    if method == "best_of_n":
        return "bon"
    if method == "bbon":
        return "bbon"
    if method == "mean_search":
        return "mean_search"
    if method == "slg":
        return "slg"
    return method


def _resolve_run_output_dir(output_dir: Path, start_index: int, num_problems: int | None) -> Path:
    if not _is_sharded_run(output_dir, start_index, num_problems):
        return output_dir
    count_label = "all" if num_problems is None else str(num_problems)
    return output_dir / "shards" / f"start_{start_index:05d}_count_{count_label}"


def _is_sharded_run(output_dir: Path, start_index: int, num_problems: int | None) -> bool:
    if start_index > 0:
        return True
    if not num_problems:
        return False
    plan_path = _shard_plan_path(output_dir)
    if not plan_path.exists():
        return False
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return len(plan.get("shards", [])) > 1


def _shard_plan_path(output_dir: Path) -> Path:
    return output_dir / ".shards" / "plan.json"


def _maybe_merge_shards(
    *,
    entry: dict,
    output_dir: Path,
    runtime_config_path: Path,
    skip_official_eval: bool,
    strict_official_eval: bool,
) -> bool:
    plan_path = _shard_plan_path(output_dir)
    if not plan_path.exists():
        return False

    aggregate_lock = output_dir / ".shards" / "aggregate.lock"
    aggregate_lock.parent.mkdir(parents=True, exist_ok=True)
    with open(aggregate_lock, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        shard_specs = plan.get("shards", [])
        shard_payloads = []
        missing = []
        incomplete = []

        for shard_spec in shard_specs:
            start_index = int(shard_spec["start_index"])
            num_problems = shard_spec.get("num_problems")
            shard_dir = _resolve_run_output_dir(
                output_dir=output_dir,
                start_index=start_index,
                num_problems=num_problems,
            )
            shard_results_path = shard_dir / "results.json"
            if not shard_results_path.exists():
                missing.append(str(shard_results_path))
                continue
            shard_ok, shard_message = _validate_shard_results(
                results_path=shard_results_path,
                start_index=start_index,
                num_problems=num_problems,
            )
            if not shard_ok:
                incomplete.append(shard_message)
                continue
            with open(shard_results_path, "r", encoding="utf-8") as f:
                shard_payloads.append(json.load(f))

        if missing:
            print(f"Shard merge pending; {len(missing)} shard result files still missing.")
            return False
        if incomplete:
            print(
                "Shard merge pending; incomplete shard outputs detected:\n"
                + "\n".join(f"  - {message}" for message in incomplete[:10])
            )
            return False

        merged = _merge_shard_payloads(shard_payloads, output_dir)
        expected_total = sum(int(spec.get("num_problems") or 0) for spec in shard_specs)
        actual_total = len(merged.get("experiments", []))
        if expected_total and actual_total != expected_total:
            raise RuntimeError(
                f"Merged shard payload count mismatch for {output_dir}: "
                f"expected {expected_total}, got {actual_total}"
            )
        with open(output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        final_summary = _build_final_summary(entry, output_dir, runtime_config_path)
        final_summary["sharded_execution"] = True
        final_summary["shard_count"] = len(shard_specs)
        final_summary["shard_plan_path"] = str(plan_path)
        _export_best_answers(output_dir / "results.json", output_dir)
        if not skip_official_eval:
            official_eval = evaluate_results(
                results_json=output_dir / "results.json",
                dataset=entry["dataset"],
                output_dir=output_dir,
                model_pretty_name=lm_short_name(entry["lm"]),
                strict=strict_official_eval,
            )
            final_summary.update(official_eval)
        else:
            final_summary.update(
                {
                    "official_metric_name": None,
                    "official_score": None,
                    "official_eval_status": "deferred",
                    "official_eval_note": "Official benchmark-native evaluation deferred until API/judge access is available.",
                }
            )

        final_summary = _apply_headline_metric(final_summary)
        with open(output_dir / "final_summary.json", "w", encoding="utf-8") as f:
            json.dump(final_summary, f, ensure_ascii=False, indent=2)
    return True


def _merge_shard_payloads(shard_payloads: list[dict], output_dir: Path) -> dict:
    ordered_payloads = sorted(
        shard_payloads,
        key=lambda payload: min(
            (
                row.get("source_index", row.get("experiment_index", 0) - 1)
                for row in payload.get("experiments", [])
            ),
            default=10**9,
        ),
    )
    merged_experiments = []
    for payload in ordered_payloads:
        merged_experiments.extend(payload.get("experiments", []))

    merged_experiments.sort(
        key=lambda row: (
            row.get("source_index", row.get("experiment_index", 0) - 1),
            row.get("experiment_index", 0),
        )
    )
    for experiment_index, row in enumerate(merged_experiments, start=1):
        row["experiment_index"] = experiment_index

    merged = json.loads(json.dumps(ordered_payloads[0]))
    merged["experiment_name"] = output_dir.name
    merged_config = merged.get("experiment_config", {})
    merged_config["experiment_name"] = output_dir.name
    merged_config["output_dir"] = str(output_dir)
    merged["experiment_config"] = merged_config
    merged["experiments"] = merged_experiments
    merged["total_experiments"] = len(merged_experiments)
    merged["summary"] = _recompute_summary(merged_experiments)
    return merged


def _validate_shard_results(
    *,
    results_path: Path,
    start_index: int,
    num_problems: int | None,
) -> tuple[bool, str]:
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        return False, f"{results_path}: failed to read results.json ({exc})"

    experiments = payload.get("experiments", [])
    if num_problems is None:
        if not experiments:
            return False, f"{results_path}: no experiments recorded"
        return True, f"{results_path}: complete"

    expected_indices = list(range(start_index, start_index + int(num_problems)))
    actual_indices = sorted(
        row.get("source_index", row.get("experiment_index", 0) - 1)
        for row in experiments
    )

    if len(experiments) != int(num_problems):
        return (
            False,
            f"{results_path}: expected {num_problems} experiments, found {len(experiments)}",
        )
    if actual_indices != expected_indices:
        missing = sorted(set(expected_indices) - set(actual_indices))
        extra = sorted(set(actual_indices) - set(expected_indices))
        detail_bits = []
        if missing:
            detail_bits.append(f"missing={missing[:8]}")
        if extra:
            detail_bits.append(f"extra={extra[:8]}")
        detail = ", ".join(detail_bits) if detail_bits else "index mismatch"
        return False, f"{results_path}: {detail}"
    return True, f"{results_path}: complete"


def _recompute_summary(experiments: list[dict]) -> dict:
    if not experiments:
        return {
            "total_experiments": 0,
            "average_time": 0.0,
            "average_score": 0.0,
            "total_generated_responses": 0,
        }

    search_times = [float(row.get("search_time", 0.0) or 0.0) for row in experiments]
    scores = [float(row.get("best_score", 0.0) or 0.0) for row in experiments]
    total_rollouts = [int(row.get("total_rollouts", 0) or 0) for row in experiments]
    return {
        "total_experiments": len(experiments),
        "average_time": sum(search_times) / len(search_times),
        "average_score": sum(scores) / len(scores),
        "total_generated_responses": sum(total_rollouts),
    }


def _build_final_summary(entry: dict, output_dir: Path, runtime_config_path: Path) -> dict:
    with open(output_dir / "results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    experiments = data.get("experiments", [])
    metadata = git_metadata()
    return {
        "phase": entry["phase"],
        "subphase": entry["subphase"],
        "dataset": entry["dataset"],
        "lm": entry["lm"],
        "rm": entry["rm"],
        "method": entry["method"],
        "budget": entry["budget"],
        "output_mode": data.get("experiment_config", {}).get("output_mode"),
        "generation_config_path": str(runtime_config_path),
        "results_json_path": str(output_dir / "results.json"),
        "best_answers_path": str(output_dir / "best_answers.jsonl"),
        "final_headline_metric_name": "average_best_rm_score",
        "final_headline_score": summary.get("average_score"),
        "final_headline_source": "interim_reward_model",
        "interim_metric_name": "average_best_rm_score",
        "interim_metric_value": summary.get("average_score"),
        "total_prompt_count": len(experiments),
        "total_generated_responses": summary.get("total_generated_responses"),
        "total_tokens": None,
        **metadata,
    }


def _apply_headline_metric(summary: dict) -> dict:
    official_score = summary.get("official_score")
    official_metric_name = summary.get("official_metric_name")
    official_status = summary.get("official_eval_status")
    if official_metric_name and official_score is not None and official_status == "completed":
        summary["final_headline_metric_name"] = official_metric_name
        summary["final_headline_score"] = official_score
        summary["final_headline_source"] = "official_benchmark"
    return summary


def _export_best_answers(results_path: Path, output_dir: Path) -> None:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    best_answers_path = output_dir / "best_answers.jsonl"
    with open(best_answers_path, "w", encoding="utf-8") as f:
        for row in data.get("experiments", []):
            export_row = {
                "problem_id": row.get("problem_id"),
                "source_index": row.get("source_index"),
                "instruction": row.get("instruction"),
                "prompt": row.get("prompt"),
                "raw_prompt": row.get("raw_prompt"),
                "output_mode": row.get("output_mode"),
                "best_response": row.get("best_response"),
                "raw_response": row.get("raw_response"),
                "clean_response": row.get("clean_response"),
                "best_score": row.get("best_score"),
                "search_time": row.get("search_time"),
                "total_rollouts": row.get("total_rollouts"),
                "messages": row.get("messages"),
                "uid": row.get("uid"),
                "session_id": row.get("session_id"),
                "conversation_input": row.get("conversation_input"),
                "primary_tag": row.get("primary_tag"),
                "checklist": row.get("checklist"),
                "key": row.get("key"),
                "instruction_id_list": row.get("instruction_id_list"),
                "kwargs": row.get("kwargs"),
                "num_blocks": row.get("num_blocks"),
                "block_lengths": row.get("block_lengths"),
                "block_text_preview": row.get("block_text_preview"),
                "num_spans": row.get("num_spans"),
                "span_lengths": row.get("span_lengths"),
                "span_preview": row.get("span_preview"),
                "state_mode": row.get("state_mode"),
                "chunking_path": row.get("chunking_path"),
                "segmentation_path": row.get("segmentation_path"),
                "contains_think": row.get("contains_think"),
                "changed_by_cleaning": row.get("changed_by_cleaning"),
            }
            f.write(json.dumps(export_row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
