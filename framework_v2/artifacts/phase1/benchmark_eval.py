import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List

from artifacts.phase1.common import ROOT, ensure_parent


ARENA_REPO = "https://github.com/lmarena/arena-hard-auto.git"
WILDBENCH_REPO = "https://github.com/allenai/WildBench.git"
IFBENCH_REPO = "https://github.com/allenai/IFBench.git"


def evaluate_results(results_json: Path, dataset: str, output_dir: Path, model_pretty_name: str, strict: bool = True) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if dataset == "alpaca_eval":
        return _eval_alpaca_eval(results_json, output_dir, strict=strict)
    if dataset == "arena_hard":
        return _eval_arena_hard(results_json, output_dir, model_pretty_name, strict=strict)
    if dataset == "wildbench_v2":
        return _eval_wildbench(results_json, output_dir, model_pretty_name, strict=strict)
    if dataset == "ifbench":
        return _eval_ifbench(results_json, output_dir, model_pretty_name, strict=strict)
    raise ValueError(f"Unsupported dataset for evaluation: {dataset}")


def _load_results(results_json: Path) -> dict:
    with open(results_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _eval_alpaca_eval(results_json: Path, output_dir: Path, strict: bool) -> Dict[str, object]:
    data = _load_results(results_json)
    formatted = [
        {
            "instruction": row["instruction"],
            "output": row["best_response"],
        }
        for row in data["experiments"]
    ]

    inputs_path = output_dir / "alpaca_eval_model_outputs.json"
    ensure_parent(inputs_path)
    with open(inputs_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    raw_path = output_dir / "alpaca_eval_raw_output.txt"
    metrics = {
        "official_metric_name": "LC-WR",
        "official_score": None,
        "official_eval_status": "not_run",
        "formatted_input_path": str(inputs_path),
        "raw_official_output_path": str(raw_path),
    }

    cmd = ["alpaca_eval", "--model_outputs", str(inputs_path)]
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        raw_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
        metrics["official_eval_status"] = "completed"
        metrics["official_score"] = _extract_first_float(completed.stdout)
    except Exception as exc:
        raw_path.write_text(str(exc), encoding="utf-8")
        metrics["official_eval_status"] = "failed"
        if strict:
            raise
    return metrics


def _eval_arena_hard(results_json: Path, output_dir: Path, model_pretty_name: str, strict: bool) -> Dict[str, object]:
    workdir = _clone_workdir(ARENA_REPO, output_dir / "official_eval" / "arena_hard_auto")
    data = _load_results(results_json)

    answer_dir = workdir / "data" / "arena-hard-v2.0" / "model_answer"
    answer_dir.mkdir(parents=True, exist_ok=True)
    answer_path = answer_dir / f"{model_pretty_name}.jsonl"
    with open(answer_path, "w", encoding="utf-8") as f:
        for row in data["experiments"]:
            answer = {
                "uid": row["uid"],
                "ans_id": f"{row['uid']}-{model_pretty_name}",
                "model": model_pretty_name,
                "messages": [
                    {"role": "user", "content": row["instruction"]},
                    {"role": "assistant", "content": {"answer": row["best_response"]}},
                ],
                "tstamp": 0,
                "metadata": {"token_len": row.get("best_response_num_tokens", 0)},
            }
            f.write(json.dumps(answer, ensure_ascii=False) + "\n")

    config_path = workdir / "config" / "phase1_arena_eval.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    "bench_name: arena-hard-v2.0",
                    "judge_model: gpt-4.1",
                    "reference: null",
                    "temperature: 0.0",
                    "max_tokens: 16000",
                    "prompt_template: |-",
                    "  {$QUESTION}",
                    "regex_patterns:",
                    "  - \"A>>B|A>B|A=B|B>A|B>>A|A<<B|B<<A|B<A\"",
                    "model_list:",
                    f"  - {model_pretty_name}",
                ]
            )
        )

    endpoint_path = workdir / "config" / "phase1_api_config.yaml"
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    endpoint_path.write_text(
        "\n".join(
            [
                "gpt-4.1:",
                "  api_type: openai",
                "  parallel: 4",
                "  endpoints:",
                "    - api_base: https://api.openai.com/v1",
                f"      api_key: {openai_key}",
            ]
        ),
        encoding="utf-8",
    )

    raw_path = output_dir / "official_eval" / "arena_hard_raw_output.txt"
    metrics = {
        "official_metric_name": "Arena-Hard score",
        "official_score": None,
        "official_eval_status": "not_run",
        "formatted_input_path": str(answer_path),
        "raw_official_output_path": str(raw_path),
    }
    try:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for Arena-Hard official judgment.")
        judgment = subprocess.run(
            ["python", "gen_judgment.py", "--setting-file", str(config_path), "--endpoint-file", str(endpoint_path)],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        )
        show = subprocess.run(
            ["python", "show_result.py", "--judge-names", "gpt-4.1"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        )
        raw_path.write_text(judgment.stdout + "\n" + judgment.stderr + "\n" + show.stdout + "\n" + show.stderr, encoding="utf-8")
        metrics["official_eval_status"] = "completed"
        metrics["official_score"] = _extract_first_float(show.stdout)
    except Exception as exc:
        raw_path.write_text(str(exc), encoding="utf-8")
        metrics["official_eval_status"] = "failed"
        if strict:
            raise
    return metrics


def _eval_wildbench(results_json: Path, output_dir: Path, model_pretty_name: str, strict: bool) -> Dict[str, object]:
    workdir = _clone_workdir(WILDBENCH_REPO, output_dir / "official_eval" / "WildBench")
    data = _load_results(results_json)

    local_result_dir = workdir / "result_dirs" / "wild_bench_v2"
    local_result_dir.mkdir(parents=True, exist_ok=True)
    local_result_path = local_result_dir / f"{model_pretty_name}.json"
    formatted = []
    for row in data["experiments"]:
        formatted.append(
            {
                "session_id": row["session_id"],
                "chat_history": [msg["content"] for msg in row["conversation_input"]],
                "model_input": row["prompt"],
                "output": [row["best_response"]],
                "generator": model_pretty_name,
                "configs": {
                    "temperature": data["experiment_config"].get("bon_params", {}).get(
                        "temperature",
                        data["experiment_config"].get("bbon_params", {}).get(
                            "temperature",
                            data["experiment_config"].get("slg_params", {}).get("temperature", 0.7),
                        ),
                    ),
                    "top_p": data["experiment_config"].get("bon_params", {}).get(
                        "top_p",
                        data["experiment_config"].get("bbon_params", {}).get(
                            "top_p",
                            data["experiment_config"].get("slg_params", {}).get("top_p", 0.95),
                        ),
                    ),
                },
                "dataset": "wild_bench",
                "primary_tag": row.get("primary_tag"),
            }
        )
    with open(local_result_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    raw_path = output_dir / "official_eval" / "wildbench_raw_output.txt"
    metrics = {
        "official_metric_name": "WB-Score",
        "official_score": None,
        "official_eval_status": "not_run",
        "formatted_input_path": str(local_result_path),
        "raw_official_output_path": str(raw_path),
    }

    try:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for WildBench official scoring.")
        prep = subprocess.run(
            ["bash", "evaluation/run_score_eval_batch.sh", model_pretty_name],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        )
        raw_path.write_text(prep.stdout + "\n" + prep.stderr, encoding="utf-8")
        metrics["official_eval_status"] = "prepared_batch_eval"
    except Exception as exc:
        raw_path.write_text(str(exc), encoding="utf-8")
        metrics["official_eval_status"] = "failed"
        if strict:
            raise
    return metrics


def _eval_ifbench(results_json: Path, output_dir: Path, model_pretty_name: str, strict: bool) -> Dict[str, object]:
    workdir = _clone_workdir(IFBENCH_REPO, output_dir / "official_eval" / "IFBench")
    data = _load_results(results_json)

    response_path = output_dir / "official_eval" / f"{model_pretty_name}-responses.jsonl"
    with open(response_path, "w", encoding="utf-8") as f:
        for row in data["experiments"]:
            f.write(json.dumps({"prompt": row["instruction"], "response": row["best_response"]}, ensure_ascii=False) + "\n")

    raw_path = output_dir / "official_eval" / "ifbench_raw_output.txt"
    metrics = {
        "official_metric_name": "IFBench loose accuracy",
        "official_score": None,
        "official_eval_status": "not_run",
        "formatted_input_path": str(response_path),
        "raw_official_output_path": str(raw_path),
    }
    try:
        completed = subprocess.run(
            [
                "python",
                "run_eval.py",
                f"--input_data={workdir / 'data' / 'IFBench_test.jsonl'}",
                f"--input_response_data={response_path}",
                f"--output_dir={output_dir / 'official_eval' / 'ifbench_outputs'}",
            ],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        )
        raw_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
        metrics["official_eval_status"] = "completed"
        metrics["official_score"] = _extract_last_float(completed.stdout)
    except Exception as exc:
        raw_path.write_text(str(exc), encoding="utf-8")
        metrics["official_eval_status"] = "failed"
        if strict:
            raise
    return metrics


def _clone_workdir(repo_url: str, workdir: Path) -> Path:
    if workdir.exists():
        return workdir
    workdir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(workdir)], check=True, cwd=ROOT)
    return workdir


def _extract_first_float(text: str):
    import re

    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def _extract_last_float(text: str):
    import re

    matches = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    return float(matches[-1]) if matches else None
