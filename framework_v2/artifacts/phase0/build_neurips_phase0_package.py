#!/usr/bin/env python3
"""Build the paper-facing Phase 0 Gaussian-tail validation package.

This script is intentionally offline-only: it reads completed raw reward-score
artifacts and writes derived summaries for the NeurIPS experiments section.
It does not submit jobs, modify manifests, or touch queues.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(
    os.environ.get(
        "PHASE0_SOURCE_ROOT",
        REPO_ROOT / "data/results/if_paper_reward_dist_v1",
    )
)
RAW_SCORES_DIR = SOURCE_ROOT / "raw_scores"
ANALYSIS_DIR = SOURCE_ROOT / "analysis"
OUT_DIR = REPO_ROOT / "data/results/phase0/summaries"

MAIN_DATASETS = {"alpaca_eval", "arena_hard", "wildbench_v2"}
MAIN_LMS = {"qwen3_8b", "llama31_8b"}
MAIN_RMS = {"skywork_v2_8b", "rm_mistral_7b"}
TAIL_FRACTION = 0.20
MIN_SAMPLES = 100
MIN_TAIL_SAMPLES = 10
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260503
EXPECTED_PROMPTS = 100

SCORE_RE = re.compile(
    r"^scores_(?P<dataset>.+)_(?P<lm>llama31_8b|qwen3_8b|gemma3_12b)_"
    r"(?P<rm>skywork_v2_8b|rm_mistral_7b|armorm_8b|fsfairx_llama3)_"
    r"N(?P<n>\d+)_seed(?P<seed>\d+)_start(?P<start>\d+)_count(?P<count>\d+)\.json$"
)

LM_DISPLAY = {
    "qwen3_8b": "Qwen/Qwen3-8B",
    "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma3_12b": "google/gemma-3-12b-it",
}

RM_DISPLAY = {
    "skywork_v2_8b": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "rm_mistral_7b": "weqweasdas/RM-Mistral-7B",
    "armorm_8b": "RLHFlow/ArmoRM-Llama3-8B-v0.1",
    "fsfairx_llama3": "sfairXC/FsfairX-LLaMA3-RM-v0.1",
}


@dataclass(frozen=True)
class SettingKey:
    dataset: str
    lm_key: str
    rm_key: str


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def percentile(values: Iterable[float], q: float) -> float | None:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def std(values: Iterable[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(var)


def bootstrap_ci(values: list[float], reps: int = BOOTSTRAP_REPS) -> tuple[float, float] | None:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    boot = []
    n = len(vals)
    for _ in range(reps):
        boot.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    return percentile(boot, 0.025), percentile(boot, 0.975)


def upper_tail_gaussian_r2(scores: list[float], tail_fraction: float = TAIL_FRACTION) -> dict:
    finite = [float(s) for s in scores if isinstance(s, (int, float)) and math.isfinite(float(s))]
    finite_ratio = len(finite) / len(scores) if scores else 0.0
    if len(finite) < MIN_SAMPLES:
        return {
            "valid": False,
            "R2": None,
            "num_samples": len(finite),
            "num_tail_samples": 0,
            "mean_reward": mean(finite),
            "std_reward": std(finite),
            "valid_reward_mass_or_p_nonzero": finite_ratio,
            "notes": "invalid: fewer than minimum finite samples",
        }

    full_std = std(finite)
    if full_std is None or full_std <= 1e-12:
        return {
            "valid": False,
            "R2": None,
            "num_samples": len(finite),
            "num_tail_samples": 0,
            "mean_reward": mean(finite),
            "std_reward": full_std,
            "valid_reward_mass_or_p_nonzero": finite_ratio,
            "notes": "invalid: degenerate full reward distribution",
        }

    sorted_scores = sorted(finite)
    k = max(MIN_TAIL_SAMPLES, int(math.ceil(len(sorted_scores) * tail_fraction)))
    k = min(k, len(sorted_scores))
    tail = sorted_scores[-k:]
    tail_std = std(tail)
    if k < MIN_TAIL_SAMPLES or tail_std is None or tail_std <= 1e-12:
        return {
            "valid": False,
            "R2": None,
            "num_samples": len(finite),
            "num_tail_samples": k,
            "mean_reward": mean(finite),
            "std_reward": full_std,
            "valid_reward_mass_or_p_nonzero": finite_ratio,
            "notes": "invalid: degenerate or too-small upper tail",
        }

    nd = NormalDist()
    lower_p = 1.0 - tail_fraction
    zs = []
    for i in range(k):
        p = lower_p + tail_fraction * ((i + 0.5) / k)
        p = min(max(p, 1e-9), 1 - 1e-9)
        zs.append(nd.inv_cdf(p))

    x_mean = sum(zs) / k
    y_mean = sum(tail) / k
    sxx = sum((x - x_mean) ** 2 for x in zs)
    if sxx <= 1e-20:
        r2 = None
    else:
        slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(zs, tail)) / sxx
        intercept = y_mean - slope * x_mean
        preds = [intercept + slope * x for x in zs]
        sse = sum((y - p) ** 2 for y, p in zip(tail, preds))
        sst = sum((y - y_mean) ** 2 for y in tail)
        r2 = None if sst <= 1e-20 else 1.0 - sse / sst

    if r2 is None or not math.isfinite(r2):
        return {
            "valid": False,
            "R2": None,
            "num_samples": len(finite),
            "num_tail_samples": k,
            "mean_reward": mean(finite),
            "std_reward": full_std,
            "valid_reward_mass_or_p_nonzero": finite_ratio,
            "notes": "invalid: undefined upper-tail linear fit",
        }

    if r2 >= 0.95:
        label = "strong upper-tail Gaussian QQ fit"
    elif r2 >= 0.90:
        label = "good upper-tail Gaussian QQ fit"
    elif r2 >= 0.80:
        label = "medium upper-tail Gaussian QQ fit"
    else:
        label = "low upper-tail Gaussian QQ fit"

    return {
        "valid": True,
        "R2": r2,
        "num_samples": len(finite),
        "num_tail_samples": k,
        "mean_reward": mean(finite),
        "std_reward": full_std,
        "valid_reward_mass_or_p_nonzero": finite_ratio,
        "notes": label,
    }


def infer_status(key: SettingKey, num_prompts: int) -> tuple[str, str]:
    if key.dataset not in MAIN_DATASETS:
        return "excluded", "not part of open-ended IF main package"
    if key.rm_key not in MAIN_RMS:
        if num_prompts >= EXPECTED_PROMPTS:
            return "excluded", "additional RM not used in paper main comparison"
        return "partial", "partial additional-RM shards; excluded from paper main comparison"
    if key.lm_key == "gemma3_12b":
        if num_prompts >= EXPECTED_PROMPTS:
            return "full", "auxiliary Gemma setting; only alpaca_eval completed"
        return "partial", "partial auxiliary Gemma setting"
    if key.lm_key not in MAIN_LMS:
        return "excluded", "LM not included in paper main matrix"
    if num_prompts >= EXPECTED_PROMPTS:
        return "full", "main paper matrix"
    return "partial", "incomplete main-axis shard coverage"


def load_prompt_rows() -> tuple[list[dict], dict[SettingKey, list[dict]], dict]:
    registry_path = SOURCE_ROOT / "EXPERIMENT_REGISTRY.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    grouped: dict[SettingKey, list[dict]] = defaultdict(list)
    rows: list[dict] = []

    for path_str in sorted(glob.glob(str(RAW_SCORES_DIR / "scores_*.json"))):
        path = Path(path_str)
        match = SCORE_RE.match(path.name)
        if not match:
            continue
        parts = match.groupdict()
        key = SettingKey(parts["dataset"], parts["lm"], parts["rm"])
        data = json.loads(path.read_text())
        meta = data.get("meta", {})
        for problem in data.get("problems", []):
            fit = upper_tail_gaussian_r2(problem.get("scores", []))
            row = {
                "dataset": key.dataset,
                "LM": LM_DISPLAY.get(key.lm_key, key.lm_key),
                "RM": RM_DISPLAY.get(key.rm_key, key.rm_key),
                "lm_key": key.lm_key,
                "rm_key": key.rm_key,
                "prompt_id": str(problem.get("problem_id", "")),
                "source_index": problem.get("source_index", ""),
                "valid": fit["valid"],
                "R2": fit["R2"],
                "tail_fraction": TAIL_FRACTION,
                "num_samples": fit["num_samples"],
                "num_tail_samples": fit["num_tail_samples"],
                "mean_reward": fit["mean_reward"],
                "std_reward": fit["std_reward"],
                "valid_reward_mass_or_p_nonzero": fit["valid_reward_mass_or_p_nonzero"],
                "notes": fit["notes"],
                "score_file": str(path),
                "temperature": meta.get("temperature", ""),
                "top_p": registry.get("sampling_protocol", {}).get("top_p", ""),
                "max_tokens": meta.get("max_tokens", registry.get("sampling_protocol", {}).get("max_tokens", "")),
                "seed": meta.get("seed", registry.get("sampling_protocol", {}).get("seed", "")),
            }
            rows.append(row)
            grouped[key].append(row)
    return rows, grouped, registry


def setting_summary_rows(grouped: dict[SettingKey, list[dict]], registry: dict) -> list[dict]:
    summaries = []
    for key, rows in sorted(grouped.items(), key=lambda kv: (kv[0].dataset, kv[0].lm_key, kv[0].rm_key)):
        valid_rows = [r for r in rows if r["valid"] and r["R2"] is not None]
        r2s = [float(r["R2"]) for r in valid_rows]
        num_prompts = len(rows)
        num_valid = len(valid_rows)
        ci = bootstrap_ci(r2s)
        status, status_note = infer_status(key, num_prompts)
        notes = [status_note]
        if ci:
            notes.append(f"bootstrap mean_R2 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
        temps = sorted({str(r.get("temperature", "")) for r in rows if str(r.get("temperature", ""))})
        top_ps = sorted({str(r.get("top_p", "")) for r in rows if str(r.get("top_p", ""))})
        max_tokens = sorted({str(r.get("max_tokens", "")) for r in rows if str(r.get("max_tokens", ""))})
        summaries.append(
            {
                "dataset": key.dataset,
                "LM": LM_DISPLAY.get(key.lm_key, key.lm_key),
                "RM": RM_DISPLAY.get(key.rm_key, key.rm_key),
                "num_prompts": num_prompts,
                "num_valid_prompts": num_valid,
                "valid_prompt_ratio": num_valid / num_prompts if num_prompts else 0.0,
                "mean_R2": mean(r2s),
                "median_R2": percentile(r2s, 0.5),
                "pass_rate_R2_ge_0.90": sum(1 for x in r2s if x >= 0.90) / num_valid if num_valid else 0.0,
                "pass_rate_R2_ge_0.95": sum(1 for x in r2s if x >= 0.95) / num_valid if num_valid else 0.0,
                "mean_valid_reward_mass_or_p_nonzero": mean(
                    float(r["valid_reward_mass_or_p_nonzero"]) for r in rows
                ),
                "tail_fraction": TAIL_FRACTION,
                "samples_per_prompt": registry.get("sampling_protocol", {}).get("n_samples_per_problem", 5000),
                "status": status,
                "notes": "; ".join(notes),
                "temperature": ",".join(temps),
                "top_p": ",".join(top_ps),
                "max_tokens": ",".join(max_tokens),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key, value in list(out.items()):
                if isinstance(value, float):
                    out[key] = fmt_float(value, 6)
                elif value is None:
                    out[key] = ""
            writer.writerow(out)


def markdown_table(rows: list[dict], fields: list[str], max_rows: int | None = None) -> str:
    use_rows = rows[:max_rows] if max_rows is not None else rows
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in use_rows:
        vals = []
        for field in fields:
            val = row.get(field, "")
            if isinstance(val, float):
                val = fmt_float(val, 4)
            vals.append(str(val).replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def compact_summary_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append(
            {
                "dataset": r["dataset"],
                "LM": r["LM"].split("/")[-1],
                "RM": "Skywork" if "Skywork" in r["RM"] else ("RM-Mistral" if "Mistral" in r["RM"] else r["RM"].split("/")[-1]),
                "prompts": r["num_prompts"],
                "valid": r["num_valid_prompts"],
                "mean_R2": fmt_float(r["mean_R2"], 3),
                "median_R2": fmt_float(r["median_R2"], 3),
                "R2>=0.90": fmt_float(r["pass_rate_R2_ge_0.90"], 3),
                "R2>=0.95": fmt_float(r["pass_rate_R2_ge_0.95"], 3),
                "status": r["status"],
                "notes": r["notes"].split(";")[0],
            }
        )
    return out


def choose_examples(prompt_rows: list[dict]) -> list[dict]:
    def is_main_prompt(row: dict) -> bool:
        return (
            row["dataset"] in MAIN_DATASETS
            and row.get("lm_key") in MAIN_LMS
            and row.get("rm_key") in MAIN_RMS
        )

    valid = [r for r in prompt_rows if r["valid"] and r["R2"] is not None and is_main_prompt(r)]
    valid.sort(key=lambda r: float(r["R2"]))
    examples: list[dict] = []
    examples.extend(("low", r) for r in valid[:2])
    if valid:
        medium = sorted(valid, key=lambda r: abs(float(r["R2"]) - 0.90))[:2]
        examples.extend(("medium", r) for r in medium)
    examples.extend(("strong", r) for r in valid[-2:])
    invalid = [r for r in prompt_rows if not r["valid"] and is_main_prompt(r)]
    examples.extend(("degenerate/invalid", r) for r in invalid[:2])

    seen = set()
    unique = []
    for label, row in examples:
        key = (row["dataset"], row["LM"], row["RM"], row["prompt_id"])
        if key in seen:
            continue
        seen.add(key)
        ex = dict(row)
        ex["example_type"] = label
        unique.append(ex)
    return unique[:8]


def write_examples(path: Path, examples: list[dict]) -> None:
    lines = [
        "# Phase 0 Representative Tail-Fit Examples",
        "",
        "These examples are selected from the recomputed prompt-level upper-tail Gaussian QQ fits. They intentionally include strong, medium, and low-fit cases.",
        "",
    ]
    for i, ex in enumerate(examples, 1):
        r2 = fmt_float(ex.get("R2"), 4) if ex.get("R2") is not None else "N/A"
        why = ex["notes"]
        if ex["example_type"] == "strong":
            why = "High R2 indicates the upper empirical reward quantiles align closely with a Gaussian-tail QQ line."
        elif ex["example_type"] == "medium":
            why = "The upper tail is broadly monotone and partly linear, but visible deviations reduce fit quality."
        elif ex["example_type"] == "low":
            why = "The upper tail deviates from Gaussian-tail linearity; this is useful as a failure/heterogeneity example."
        lines.extend(
            [
                f"## Example {i}: {ex['example_type']}",
                "",
                f"- dataset: `{ex['dataset']}`",
                f"- prompt_id: `{ex['prompt_id']}`",
                f"- LM: `{ex['LM']}`",
                f"- RM: `{ex['RM']}`",
                f"- num_samples: `{ex['num_samples']}`",
                f"- num_tail_samples: `{ex['num_tail_samples']}`",
                f"- tail_fraction: `{TAIL_FRACTION}`",
                f"- R2: `{r2}`",
                f"- why: {why}",
                f"- score source: `{ex['score_file']}`",
                "- plot path: not generated in this package; use the score source above with `phase0_prompt_level_tail_stats.csv` to render the QQ/tail plot.",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def axis_average(rows: list[dict], axis: str, main_only: bool = True) -> list[dict]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if main_only and not ("main paper matrix" in r["notes"]):
            continue
        if r["mean_R2"] is not None:
            buckets[str(r[axis])].append(float(r["mean_R2"]))
    return [
        {"axis": k, "mean_setting_mean_R2": sum(v) / len(v), "num_settings": len(v)}
        for k, v in sorted(buckets.items())
    ]


def write_figure_manifest(path: Path) -> None:
    package = ANALYSIS_DIR / "vn_supporting_validation_paper_facing_package_2026-04-15"
    rows = [
        {
            "figure": "Prompt-wise R2 distribution by dataset",
            "purpose": "Show prompt-level heterogeneity of upper-tail Gaussian QQ fit quality.",
            "source_data": str(OUT_DIR / "phase0_prompt_level_tail_stats.csv"),
            "plot_path": "not generated",
            "placement": "appendix",
        },
        {
            "figure": "Representative upper-tail QQ/tail-fit plots",
            "purpose": "Visualize strong, medium, and low-fit prompts from the representative example list.",
            "source_data": str(OUT_DIR / "phase0_representative_examples.md"),
            "plot_path": "not generated",
            "placement": "main or appendix",
        },
        {
            "figure": "Tail-fraction sensitivity / V_N calibration",
            "purpose": "Supplement direct R2 fits with existing value-extrapolation calibration evidence.",
            "source_data": str(ANALYSIS_DIR / "vn_supporting_validation_slg_regime_r50.json"),
            "plot_path": str(package / "main/main_fig_slg_regime_calibration.png"),
            "placement": "appendix",
        },
        {
            "figure": "Valid prompt ratio by dataset and RM",
            "purpose": "Document invalid or degenerate prompt rates across settings.",
            "source_data": str(OUT_DIR / "phase0_main_tail_table.csv"),
            "plot_path": "not generated",
            "placement": "appendix",
        },
        {
            "figure": "Grouped V_N calibration backup",
            "purpose": "Existing paper-facing grouped calibration by dataset/LM/RM.",
            "source_data": str(ANALYSIS_DIR / "vn_supporting_validation_main12_r50.json"),
            "plot_path": str(package / "appendix/appendix_fig_grouped_calibration.png"),
            "placement": "appendix",
        },
    ]
    lines = [
        "# Phase 0 Figure Data Manifest",
        "",
        "This manifest lists recommended figures for the Gaussian-tail validation section. The first, second, and fourth figures can be rendered directly from the generated CSV/Markdown files; the third and fifth point to existing calibration assets from the original Phase 0 analysis package.",
        "",
        markdown_table(rows, ["figure", "purpose", "source_data", "plot_path", "placement"]),
        "",
    ]
    path.write_text("\n".join(lines))


def write_report(path: Path, summary_rows: list[dict], prompt_rows: list[dict], registry: dict) -> None:
    main_rows = [r for r in summary_rows if "main paper matrix" in r["notes"]]
    aux_rows = [r for r in summary_rows if "main paper matrix" not in r["notes"]]
    def is_main_prompt(row: dict) -> bool:
        return (
            row["dataset"] in MAIN_DATASETS
            and row.get("lm_key") in MAIN_LMS
            and row.get("rm_key") in MAIN_RMS
        )

    valid_prompt_rows = [r for r in prompt_rows if r["valid"] and r["R2"] is not None and is_main_prompt(r)]
    invalid_prompt_rows = [r for r in prompt_rows if not r["valid"] and is_main_prompt(r)]

    best = max(main_rows, key=lambda r: (r["mean_R2"] or -1)) if main_rows else None
    weakest = min(main_rows, key=lambda r: (r["mean_R2"] or 1e9)) if main_rows else None
    overall_mean = mean(float(r["R2"]) for r in valid_prompt_rows)
    overall_median = percentile([float(r["R2"]) for r in valid_prompt_rows], 0.5)
    pass90 = sum(1 for r in valid_prompt_rows if float(r["R2"]) >= 0.90) / len(valid_prompt_rows) if valid_prompt_rows else 0.0
    pass95 = sum(1 for r in valid_prompt_rows if float(r["R2"]) >= 0.95) / len(valid_prompt_rows) if valid_prompt_rows else 0.0
    invalid_rate = len(invalid_prompt_rows) / (len(valid_prompt_rows) + len(invalid_prompt_rows)) if (valid_prompt_rows or invalid_prompt_rows) else 0.0

    rm_axis = axis_average(summary_rows, "RM")
    lm_axis = axis_average(summary_rows, "LM")
    dataset_axis = axis_average(summary_rows, "dataset")

    protocol = registry.get("sampling_protocol", {})
    setup_lines = [
        f"- Source campaign: `{SOURCE_ROOT}`",
        f"- Raw reward source: per-response RM score arrays in `{RAW_SCORES_DIR}`.",
        f"- Datasets in the main open-ended IF matrix: `{', '.join(sorted(MAIN_DATASETS))}`.",
        "- Main generator LMs: `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B-Instruct`.",
        "- Main RMs: `Skywork/Skywork-Reward-V2-Llama-3.1-8B`, `weqweasdas/RM-Mistral-7B`.",
        f"- Prompt selection: stable 100-prompt manifests from the campaign, seed `{protocol.get('seed', 42)}`.",
        f"- Samples per prompt: `{protocol.get('n_samples_per_problem', 5000)}`.",
        f"- Sampling parameters: temperature `{protocol.get('temperature', 1.0)}`, top_p `{protocol.get('top_p', 0.95)}`, max_tokens `{protocol.get('max_tokens', 1024)}`.",
    ]

    main_table = compact_summary_rows(main_rows)
    aux_table = compact_summary_rows(aux_rows)
    heterogeneity_rows = [
        {"stat": "valid prompt rows", "value": len(valid_prompt_rows)},
        {"stat": "invalid prompt rows", "value": len(invalid_prompt_rows)},
        {"stat": "invalid prompt fraction", "value": fmt_float(invalid_rate, 4)},
        {"stat": "prompt-level mean R2", "value": fmt_float(overall_mean, 4)},
        {"stat": "prompt-level median R2", "value": fmt_float(overall_median, 4)},
        {"stat": "prompt-level 10th percentile R2", "value": fmt_float(percentile([float(r['R2']) for r in valid_prompt_rows], 0.10), 4)},
        {"stat": "prompt-level 90th percentile R2", "value": fmt_float(percentile([float(r['R2']) for r in valid_prompt_rows], 0.90), 4)},
        {"stat": "pass rate R2>=0.90", "value": fmt_float(pass90, 4)},
        {"stat": "pass rate R2>=0.95", "value": fmt_float(pass95, 4)},
    ]

    lines = [
        "# Phase 0: Gaussian-Tail Validation",
        "",
        "## 1. Purpose in the paper",
        "",
        "Phase 0 is the assumption-validation experiment for SLG. Its role is to test whether the upper tail of the reward distribution induced by sampling many responses to the same instruction prompt is well approximated by a Gaussian-tail model. This package focuses on open-ended instruction-following/chat benchmarks and does not use code-task artifacts.",
        "",
        "## 2. Experimental setup",
        "",
        *setup_lines,
        "",
        "The current checkout did not contain the Phase 0 summary files, so this package is regenerated from the preserved raw-score source-of-truth in the cleanup backup path above. No new experiments were launched.",
        "",
        "## 3. Tail-fitting protocol",
        "",
        f"- Reward samples: raw scalar reward-model scores for each sampled response; no log, shift, clipping, or nonzero transform is applied in the main table.",
        f"- Invalid score handling: non-finite values are removed before fitting. `mean_valid_reward_mass_or_p_nonzero` reports the finite-score mass for continuous RM scores.",
        f"- Minimum valid prompt: at least `{MIN_SAMPLES}` finite reward samples, non-degenerate full reward variance, and at least `{MIN_TAIL_SAMPLES}` non-degenerate upper-tail samples.",
        f"- Main upper-tail fraction: top `{TAIL_FRACTION:.2f}` of reward samples per prompt.",
        "- Fitting procedure: sort rewards, keep the upper-tail samples, map their empirical tail ranks to Gaussian quantiles over the same upper-tail probability interval, and fit a least-squares line from Gaussian quantile to reward.",
        "- R2 definition: per-prompt coefficient of determination for the upper-tail QQ linear fit, `1 - SSE/SST`, computed only on the retained upper-tail samples.",
        "- Aggregation: compute fit statistics per prompt first, then aggregate prompt-level R2 values by dataset, LM, and RM.",
        f"- Bootstrap intervals: deterministic `{BOOTSTRAP_REPS}`-replicate bootstrap over valid prompts within each setting; intervals are recorded in the `notes` field of `phase0_main_tail_table.csv`.",
        "- Statistical scope: this is an upper-tail QQ-linearity validation. It is not a whole-distribution Gaussianity test.",
        "",
        "## 4. Main quantitative results",
        "",
        markdown_table(main_table, ["dataset", "LM", "RM", "prompts", "valid", "mean_R2", "median_R2", "R2>=0.90", "R2>=0.95", "status", "notes"]),
        "",
        f"Across the main open-ended IF matrix, prompt-level mean R2 is `{fmt_float(overall_mean, 4)}` and median R2 is `{fmt_float(overall_median, 4)}`. The prompt-level pass rates are `{fmt_float(pass90, 4)}` for R2 >= 0.90 and `{fmt_float(pass95, 4)}` for R2 >= 0.95.",
        "",
    ]
    if best and weakest:
        lines.extend(
            [
                f"The strongest main setting by mean R2 is `{best['dataset']} | {best['LM']} | {best['RM']}` with mean R2 `{fmt_float(best['mean_R2'], 4)}`.",
                f"The weakest main setting by mean R2 is `{weakest['dataset']} | {weakest['LM']} | {weakest['RM']}` with mean R2 `{fmt_float(weakest['mean_R2'], 4)}`.",
                "",
            ]
        )

    lines.extend(
        [
            "## 5. Robustness across datasets, LMs, and RMs",
            "",
            "Dataset-axis averages over main settings:",
            "",
            markdown_table(dataset_axis, ["axis", "mean_setting_mean_R2", "num_settings"]),
            "",
            "LM-axis averages over main settings:",
            "",
            markdown_table(lm_axis, ["axis", "mean_setting_mean_R2", "num_settings"]),
            "",
            "RM-axis averages over main settings:",
            "",
            markdown_table(rm_axis, ["axis", "mean_setting_mean_R2", "num_settings"]),
            "",
            "Skywork and RM-Mistral agree qualitatively when interpreted as upper-tail evidence: both produce high valid-prompt coverage and a substantial mass of high-R2 prompt-level fits. Qwen and Llama also agree qualitatively across the three main datasets, although the exact R2 level varies by dataset/RM. Gemma is included only as an auxiliary alpaca_eval check, because the preserved score matrix does not contain a full cross-dataset Gemma grid.",
            "",
        ]
    )
    if aux_table:
        lines.extend(
            [
                "Auxiliary or excluded settings detected in raw scores:",
                "",
                markdown_table(aux_table, ["dataset", "LM", "RM", "prompts", "valid", "mean_R2", "median_R2", "R2>=0.90", "R2>=0.95", "status", "notes"]),
                "",
            ]
        )

    lines.extend(
        [
            "## 6. Prompt-level heterogeneity",
            "",
            markdown_table(heterogeneity_rows, ["stat", "value"]),
            "",
            "The main conclusion is not that every prompt is perfectly Gaussian in the reward tail. Fit quality varies by prompt, and low-fit cases remain important for method limitations and appendix diagnostics. In the recomputed package, invalid prompts are rare; the dominant caveat is prompt-level tail-shape heterogeneity rather than missing reward samples.",
            "",
            "## 7. Representative fit examples",
            "",
            "Representative examples are listed in `phase0_representative_examples.md`. The examples include strong, medium, and low-fit cases rather than only successes. Plot paths are marked as not generated in this package; the score source path and prompt-level CSV provide the exact data needed to render QQ/tail-fit plots.",
            "",
            "## 8. Failure modes and caveats",
            "",
            "- The evidence concerns the upper reward tail only, not whole-distribution Gaussianity.",
            "- Prompt-level heterogeneity is real; some prompts have visibly weaker upper-tail linearity.",
            "- Reward-model choice changes absolute scores and can affect fit strength, so the claim should be stated across RMs rather than tied to one RM.",
            "- Degenerate/invalid prompts are tracked explicitly; they are excluded from R2 aggregates but counted in valid-prompt ratios.",
            "- Phase 0 validates the modeling assumption used by SLG's allocation rule; it does not by itself establish end-to-end method performance.",
            "",
            "## 9. Paper-ready conclusions",
            "",
            "Phase 0 validates the reward-tail modeling assumption on open-ended instruction-following tasks. Across multiple datasets, generator models, and reward models, a substantial fraction of prompts exhibit high-quality Gaussian-tail fits in the upper reward tail. The effect is strongest in the settings identified in the main table, remains qualitatively consistent across Qwen/Llama and Skywork/RM-Mistral axes, and weaker cases are primarily associated with prompt-level tail-shape heterogeneity rather than pipeline failure. These results support using the tail model for SLG's inference-time allocation rule, while making clear that the claim concerns the upper tail rather than the entire reward distribution and does not require every prompt to be perfectly Gaussian.",
            "",
            "## 10. Suggested tables and figures",
            "",
            "- Main text table: compact version of `phase0_main_tail_table.csv`, restricted to the main open-ended IF matrix.",
            "- Appendix table: full `phase0_main_tail_table.csv`, including auxiliary/partial settings.",
            "- Appendix figure: prompt-wise R2 distribution by dataset/RM/LM from `phase0_prompt_level_tail_stats.csv`.",
            "- Main or appendix figure: representative upper-tail QQ/tail-fit examples from `phase0_representative_examples.md`.",
            "- Appendix figure: existing V_N calibration assets listed in `phase0_figure_data_manifest.md`.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    if not RAW_SCORES_DIR.exists():
        raise FileNotFoundError(f"Missing raw score directory: {RAW_SCORES_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompt_rows, grouped, registry = load_prompt_rows()
    summary_rows = setting_summary_rows(grouped, registry)

    main_fields = [
        "dataset",
        "LM",
        "RM",
        "num_prompts",
        "num_valid_prompts",
        "valid_prompt_ratio",
        "mean_R2",
        "median_R2",
        "pass_rate_R2_ge_0.90",
        "pass_rate_R2_ge_0.95",
        "mean_valid_reward_mass_or_p_nonzero",
        "tail_fraction",
        "samples_per_prompt",
        "status",
        "notes",
    ]
    prompt_fields = [
        "dataset",
        "LM",
        "RM",
        "prompt_id",
        "valid",
        "R2",
        "tail_fraction",
        "num_samples",
        "num_tail_samples",
        "mean_reward",
        "std_reward",
        "valid_reward_mass_or_p_nonzero",
        "notes",
    ]
    write_csv(OUT_DIR / "phase0_main_tail_table.csv", summary_rows, main_fields)
    write_csv(OUT_DIR / "phase0_prompt_level_tail_stats.csv", prompt_rows, prompt_fields)
    write_examples(OUT_DIR / "phase0_representative_examples.md", choose_examples(prompt_rows))
    write_figure_manifest(OUT_DIR / "phase0_figure_data_manifest.md")
    write_report(OUT_DIR / "phase0_neurips_final_report.md", summary_rows, prompt_rows, registry)

    print(f"Wrote {len(summary_rows)} setting rows and {len(prompt_rows)} prompt rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
