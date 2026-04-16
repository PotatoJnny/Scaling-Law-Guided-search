import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from artifacts.phase1.benchmark_eval import evaluate_results
from artifacts.phase1.common import (
    PHASE1_DEFAULTS_PATH,
    PHASE1_SEEDS_PATH,
    ROOT,
    git_metadata,
    lm_short_name,
    load_generation_defaults,
    load_seed_config,
    read_jsonl,
    rm_short_name,
    update_manifest_status,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--expected-output-dir", type=str, required=True)
    parser.add_argument("--num-problems", type=int, default=None)
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

    update_manifest_status(args.manifest_path, args.expected_output_dir, "running")
    try:
        runtime_config = _build_runtime_config(entry, output_dir)
        runtime_config_path = output_dir / "phase1_runtime_config.json"
        with open(runtime_config_path, "w", encoding="utf-8") as f:
            json.dump(runtime_config, f, ensure_ascii=False, indent=2)

        if args.prepare_only:
            print(f"Prepared config at {runtime_config_path}")
            update_manifest_status(args.manifest_path, args.expected_output_dir, "planned")
            return

        results_path = output_dir / "results.json"
        if args.overwrite or not results_path.exists():
            cmd = ["python", "main.py", "--config", str(runtime_config_path)]
            if args.num_problems is not None:
                cmd.extend(["--num_problems", str(args.num_problems)])
            subprocess.run(cmd, cwd=ROOT, check=True)

        final_summary = _build_final_summary(entry, output_dir, runtime_config_path)
        _export_best_answers(output_dir / "results.json", output_dir)
        if not args.skip_official_eval:
            official_eval = evaluate_results(
                results_json=results_path,
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


def _build_runtime_config(entry: dict, output_dir: Path) -> dict:
    defaults = load_generation_defaults()
    seeds = load_seed_config()

    lm_name = entry["lm"]
    rm_name = entry["rm"]
    method = entry["method"]
    budget = int(entry["budget"])
    lm_short = lm_short_name(lm_name)

    hardware_profile = defaults["hardware_profiles"].get(lm_short, defaults["hardware_profiles"]["default"])
    sampling_key = "sampled"
    sampling = defaults["sampling"][sampling_key]

    config = {
        "experiment_name": output_dir.name,
        "algorithm": _algorithm_name(method),
        "seed": seeds.get("default_seed", 42),
        "output_dir": str(output_dir),
        "hardware": {
            "lm_name": lm_name,
            "rm_name": rm_name,
            "lm_max_batch_size": defaults["generation"]["lm_max_batch_size"],
            "rm_max_batch_size": defaults["generation"]["rm_max_batch_size"],
            "gpu_memory_utilization": hardware_profile["gpu_memory_utilization"],
            "max_model_len": defaults["generation"]["max_model_len"],
            "max_num_batched_tokens": hardware_profile["max_num_batched_tokens"],
            "enforce_eager": defaults["generation"]["enforce_eager"],
        },
        "task_setup": {
            "dataset_name": entry["dataset"],
            "dataset_config": None,
            "action_strategy": defaults["action_strategy"],
        },
        "phase1_metadata": {
            "manifest_entry": entry,
            "generation_defaults_path": str(PHASE1_DEFAULTS_PATH),
            "seed_config_path": str(PHASE1_SEEDS_PATH),
        },
    }

    if method == "best_of_n":
        config["bon_params"] = {
            "N": budget,
            "max_tokens": defaults["generation"]["max_new_tokens"],
            "verbose": False,
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "seed": seeds.get("default_seed", 42),
        }
    elif method == "slg":
        config["slg_params"] = {
            "N": budget,
            "K": -1,
            "m": -1,
            "max_depth": defaults["search"]["slg"]["max_depth"],
            "max_tokens": defaults["generation"]["max_new_tokens"],
            "verbose": False,
            "auto_param_strategy": defaults["search"]["slg"]["auto_param_strategy"],
            "min_m": defaults["search"]["slg"]["min_m"],
            "default_K": defaults["search"]["slg"]["default_K"],
            "max_explore_fraction": defaults["search"]["slg"]["max_explore_fraction"],
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "seed": seeds.get("default_seed", 42),
        }
    elif method == "mean_search":
        config["slg_params"] = {
            "N": budget,
            "K": -1,
            "m": -1,
            "max_depth": defaults["search"]["slg"]["max_depth"],
            "max_tokens": defaults["generation"]["max_new_tokens"],
            "verbose": False,
            "auto_param_strategy": defaults["search"]["slg"]["auto_param_strategy"],
            "min_m": defaults["search"]["slg"]["min_m"],
            "default_K": defaults["search"]["slg"]["default_K"],
            "max_explore_fraction": defaults["search"]["slg"]["max_explore_fraction"],
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "seed": seeds.get("default_seed", 42),
        }
    elif method == "bbon":
        config["bbon_params"] = {
            "N": budget,
            "K": -1,
            "m": -1,
            "max_depth": defaults["search"]["slg"]["max_depth"],
            "max_tokens": defaults["generation"]["max_new_tokens"],
            "verbose": False,
            "auto_param_strategy": defaults["search"]["slg"]["auto_param_strategy"],
            "min_m": defaults["search"]["slg"]["min_m"],
            "default_K": defaults["search"]["slg"]["default_K"],
            "max_explore_fraction": defaults["search"]["slg"]["max_explore_fraction"],
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "seed": seeds.get("default_seed", 42),
        }
    else:
        config["bon_params"] = {
            "N": budget,
            "max_tokens": defaults["generation"]["max_new_tokens"],
            "verbose": False,
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "seed": seeds.get("default_seed", 42),
        }

    return config


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
        "generation_config_path": str(runtime_config_path),
        "results_json_path": str(output_dir / "results.json"),
        "best_answers_path": str(output_dir / "best_answers.jsonl"),
        "final_headline_score": None,
        "interim_metric_name": "average_best_rm_score",
        "interim_metric_value": summary.get("average_score"),
        "total_prompt_count": len(experiments),
        "total_generated_responses": summary.get("total_generated_responses"),
        "total_tokens": None,
        **metadata,
    }


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
                "best_response": row.get("best_response"),
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
            }
            f.write(json.dumps(export_row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
