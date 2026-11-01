import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from artifacts.phase1.common import ROOT, ensure_parent


ARENA_REPO = "https://github.com/lmarena/arena-hard-auto.git"
WILDBENCH_REPO = "https://github.com/allenai/WildBench.git"
IFBENCH_REPO = "https://github.com/allenai/IFBench.git"
WILDBENCH_EVAL_MODEL = "gpt-4o-2024-05-13"
WILDBENCH_SCORE_DIR = Path("eval_results") / "v2.0625" / "score.v2" / f"eval={WILDBENCH_EVAL_MODEL}"


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


def _base_metrics(
    metric_name: str,
    raw_path: Path,
    formatted_input_path: Optional[Path] = None,
    provider: Optional[str] = None,
) -> Dict[str, object]:
    metrics = {
        "official_metric_name": metric_name,
        "official_score": None,
        "official_eval_status": "not_run",
        "formatted_input_path": str(formatted_input_path) if formatted_input_path else None,
        "raw_official_output_path": str(raw_path),
        "official_eval_provider": provider,
        "official_eval_note": None,
        "official_eval_missing_requirements": [],
    }
    return metrics


def _blocked_metrics(
    metrics: Dict[str, object],
    raw_path: Path,
    message: str,
    missing_requirements: List[str],
    strict: bool,
) -> Dict[str, object]:
    raw_path.write_text(message, encoding="utf-8")
    metrics["official_eval_status"] = "blocked_missing_requirements"
    metrics["official_eval_note"] = message
    metrics["official_eval_missing_requirements"] = missing_requirements
    if strict:
        raise RuntimeError(message)
    return metrics


def _failed_metrics(metrics: Dict[str, object], raw_path: Path, message: str, strict: bool) -> Dict[str, object]:
    raw_path.write_text(message, encoding="utf-8")
    metrics["official_eval_status"] = "failed"
    metrics["official_eval_note"] = message
    if strict:
        raise RuntimeError(message)
    return metrics


def _completed_metrics(metrics: Dict[str, object], score: Optional[float], note: Optional[str] = None, **extra) -> Dict[str, object]:
    metrics["official_eval_status"] = "completed"
    metrics["official_score"] = score
    if note:
        metrics["official_eval_note"] = note
    metrics.update(extra)
    return metrics


def _require_command(command: str) -> Optional[str]:
    return shutil.which(command)


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
    metrics = _base_metrics("LC-WR", raw_path, formatted_input_path=inputs_path, provider="alpaca_eval_cli")

    if _require_command("alpaca_eval") is None:
        return _blocked_metrics(
            metrics,
            raw_path,
            "alpaca_eval CLI is required for AlpacaEval official scoring.",
            missing_requirements=["alpaca_eval"],
            strict=strict,
        )

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
        score = _extract_first_float(completed.stdout)
        return _completed_metrics(metrics, score)
    except Exception as exc:
        return _failed_metrics(metrics, raw_path, str(exc), strict=strict)


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
    metrics = _base_metrics("Arena-Hard score", raw_path, formatted_input_path=answer_path, provider="openai_judge")
    metrics["official_eval_judge_model"] = "gpt-4.1"

    if not os.environ.get("OPENAI_API_KEY"):
        return _blocked_metrics(
            metrics,
            raw_path,
            "OPENAI_API_KEY is required for Arena-Hard official judgment.",
            missing_requirements=["OPENAI_API_KEY"],
            strict=strict,
        )
    try:
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
        score = _extract_first_float_in_line_containing(show.stdout, model_pretty_name)
        if score is None:
            score = _extract_first_float(show.stdout)
        return _completed_metrics(metrics, score)
    except Exception as exc:
        return _failed_metrics(metrics, raw_path, str(exc), strict=strict)


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
    metrics = _base_metrics("WB-Score", raw_path, formatted_input_path=local_result_path, provider="openai_batch")
    metrics["official_eval_judge_model"] = WILDBENCH_EVAL_MODEL
    score_submit_path = workdir / WILDBENCH_SCORE_DIR / f"{model_pretty_name}.batch-submit.jsonl"
    score_json_path = workdir / WILDBENCH_SCORE_DIR / f"{model_pretty_name}.json"
    score_batch_results_path = workdir / WILDBENCH_SCORE_DIR / f"{model_pretty_name}.batch_results.jsonl"
    metrics["official_eval_batch_submit_path"] = str(score_submit_path)
    metrics["official_eval_batch_results_path"] = str(score_batch_results_path)
    metrics["official_eval_result_json_path"] = str(score_json_path)

    recovered_score = _read_wildbench_score(score_json_path)
    if recovered_score is not None:
        wb_score, raw_mean, item_count = recovered_score
        raw_path.write_text(
            (
                f"Recovered existing WildBench score from {score_json_path}\n"
                f"raw_mean_score={raw_mean}\n"
                f"wb_score={wb_score}\n"
                f"item_count={item_count}\n"
            ),
            encoding="utf-8",
        )
        return _completed_metrics(
            metrics,
            wb_score,
            note="Recovered from existing downloaded WildBench batch results.",
            official_score_raw_mean=raw_mean,
            official_score_count=item_count,
            official_score_formula="WB-Score = (mean_judge_score - 5) * 2",
        )

    try:
        if not os.environ.get("OPENAI_API_KEY"):
            return _blocked_metrics(
                metrics,
                raw_path,
                "OPENAI_API_KEY is required for WildBench official scoring and batch polling.",
                missing_requirements=["OPENAI_API_KEY"],
                strict=strict,
            )
        prep = subprocess.run(
            ["bash", "evaluation/run_score_eval_batch.sh", model_pretty_name],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        )
        poll = subprocess.run(
            ["python", "src/openai_batch_eval/check_batch_status_with_model_name.py", model_pretty_name],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        )
        raw_path.write_text(prep.stdout + "\n" + prep.stderr + "\n" + poll.stdout + "\n" + poll.stderr, encoding="utf-8")

        recovered_score = _read_wildbench_score(score_json_path)
        if recovered_score is not None:
            wb_score, raw_mean, item_count = recovered_score
            return _completed_metrics(
                metrics,
                wb_score,
                note="WildBench batch results were available and converted to a final score.",
                official_score_raw_mean=raw_mean,
                official_score_count=item_count,
                official_score_formula="WB-Score = (mean_judge_score - 5) * 2",
            )

        metrics["official_eval_status"] = "submitted_batch_eval"
        metrics["official_eval_note"] = (
            "WildBench OpenAI batch has been submitted or is still pending. "
            "Re-run official evaluation after the batch completes to materialize WB-Score."
        )
    except Exception as exc:
        return _failed_metrics(metrics, raw_path, str(exc), strict=strict)
    return metrics


def _eval_ifbench(results_json: Path, output_dir: Path, model_pretty_name: str, strict: bool) -> Dict[str, object]:
    workdir = _clone_workdir(IFBENCH_REPO, output_dir / "official_eval" / "IFBench")
    data = _load_results(results_json)

    response_path = output_dir / "official_eval" / f"{model_pretty_name}-responses.jsonl"
    with open(response_path, "w", encoding="utf-8") as f:
        for row in data["experiments"]:
            f.write(json.dumps({"prompt": row["instruction"], "response": row["best_response"]}, ensure_ascii=False) + "\n")

    raw_path = output_dir / "official_eval" / "ifbench_raw_output.txt"
    metrics = _base_metrics("IFBench loose accuracy", raw_path, formatted_input_path=response_path, provider="local_repo_eval")
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
        score = _extract_ifbench_loose_prompt_accuracy(completed.stdout)
        return _completed_metrics(metrics, score)
    except Exception as exc:
        return _failed_metrics(metrics, raw_path, str(exc), strict=strict)


def _clone_workdir(repo_url: str, workdir: Path) -> Path:
    if workdir.exists():
        return workdir
    workdir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(workdir)], check=True, cwd=ROOT)
    return workdir


def _extract_first_float(text: str):
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def _extract_last_float(text: str):
    matches = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    return float(matches[-1]) if matches else None


def _extract_first_float_in_line_containing(text: str, needle: str) -> Optional[float]:
    for line in text.splitlines():
        if needle not in line:
            continue
        value = _extract_first_float(line)
        if value is not None:
            return value
    return None


def _extract_ifbench_loose_prompt_accuracy(text: str) -> Optional[float]:
    match = re.search(
        r"eval_results_loose.*?prompt-level:\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.DOTALL,
    )
    if match:
        return float(match.group(1))

    match = re.search(r"Accuracy:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        return float(match.group(1))
    return None


def _read_wildbench_score(score_json_path: Path) -> Optional[Tuple[float, float, int]]:
    if not score_json_path.exists():
        return None

    with open(score_json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    scores = []
    for row in rows:
        score = row.get("score")
        if score is None:
            score = (row.get("parsed_result") or {}).get("score")
        try:
            if score is not None:
                scores.append(float(score))
        except (TypeError, ValueError):
            continue

    if not scores:
        return None

    raw_mean = sum(scores) / len(scores)
    wb_score = (raw_mean - 5.0) * 2.0
    return wb_score, raw_mean, len(scores)
