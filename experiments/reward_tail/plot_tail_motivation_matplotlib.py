#!/usr/bin/env python3
"""Plot Phase 0 Gaussian upper-tail motivation figures with matplotlib.

This script is offline-only. It reads completed Phase 0 score artifacts and
summary CSVs, selects strong positive examples, and renders publication-style
histogram/density panels. It does not launch experiments or touch queues.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = REPO_ROOT / "data/results/phase0/summaries"
FIG_DIR = REPO_ROOT / "figures/phase0"
PROMPT_STATS = SUMMARY_DIR / "phase0_prompt_level_tail_stats.csv"
RAW_DIR = Path(
    os.environ.get("PHASE0_SOURCE_ROOT", REPO_ROOT / "data/results/if_paper_reward_dist_v1")
) / "raw_scores"

TAIL_FRACTION = 0.20
BODY_COLOR = "#D9D9D9"
TAIL_COLOR = "#2E77B8"
FIT_COLOR = "#111111"
AXIS_COLOR = "#333333"

MAIN_DATASETS = {"alpaca_eval", "arena_hard", "wildbench_v2"}
MAIN_LMS = {
    "Qwen/Qwen3-8B": "qwen3_8b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama31_8b",
}
MAIN_RMS = {
    "Skywork/Skywork-Reward-V2-Llama-3.1-8B": "skywork_v2_8b",
    "weqweasdas/RM-Mistral-7B": "rm_mistral_7b",
}
LM_DISPLAY = {v: k for k, v in MAIN_LMS.items()}
RM_DISPLAY = {v: k for k, v in MAIN_RMS.items()}

SCORE_RE = re.compile(
    r"^scores_(?P<dataset>.+)_(?P<lm>llama31_8b|qwen3_8b|gemma3_12b)_"
    r"(?P<rm>skywork_v2_8b|rm_mistral_7b|armorm_8b|fsfairx_llama3)_"
    r"N(?P<n>\d+)_seed42_start(?P<start>\d+)_count(?P<count>\d+)\.json$"
)


@dataclass
class Candidate:
    panel: str
    selected: bool
    dataset: str
    prompt_id: str
    lm: str
    rm: str
    scores: np.ndarray
    tail_mu: float
    tail_sigma: float
    tail_r2: float
    full_r2: float
    visual_score: float
    density_imbalance: float
    artifact_path: Path
    reason: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.dataset, self.prompt_id, self.lm, self.rm)


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (x - mu) / max(sigma, 1e-12)
    return np.exp(-0.5 * z * z) / (max(sigma, 1e-12) * math.sqrt(2.0 * math.pi))


def normal_cdf_scalar(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (max(sigma, 1e-12) * math.sqrt(2.0))))


def fit_gaussian_qq(scores: np.ndarray, tail_fraction: float | None) -> tuple[float, float, float]:
    vals = np.sort(scores[np.isfinite(scores)].astype(float))
    if tail_fraction is None:
        fit_vals = vals
        lower = 0.0
        width = 1.0
    else:
        k = max(10, int(math.ceil(vals.size * tail_fraction)))
        fit_vals = vals[-k:]
        lower = 1.0 - tail_fraction
        width = tail_fraction

    nd = NormalDist()
    n = fit_vals.size
    probs = lower + width * ((np.arange(n) + 0.5) / n)
    probs = np.clip(probs, 1e-9, 1.0 - 1e-9)
    z = np.array([nd.inv_cdf(float(p)) for p in probs], dtype=float)

    x_mean = float(z.mean())
    y_mean = float(fit_vals.mean())
    sxx = float(((z - x_mean) ** 2).sum())
    sigma = float(((z - x_mean) * (fit_vals - y_mean)).sum() / sxx)
    mu = y_mean - sigma * x_mean
    pred = mu + sigma * z
    sse = float(((fit_vals - pred) ** 2).sum())
    sst = float(((fit_vals - y_mean) ** 2).sum())
    r2 = 1.0 - sse / max(sst, 1e-20)
    return mu, abs(sigma), r2


def load_raw_index() -> dict[tuple[str, str, str, str], tuple[Path, dict]]:
    index: dict[tuple[str, str, str, str], tuple[Path, dict]] = {}
    for path_str in glob.glob(str(RAW_DIR / "scores_*.json")):
        path = Path(path_str)
        match = SCORE_RE.match(path.name)
        if not match:
            continue
        info = match.groupdict()
        if info["dataset"] not in MAIN_DATASETS or info["lm"] not in LM_DISPLAY or info["rm"] not in RM_DISPLAY:
            continue
        data = json.loads(path.read_text())
        for problem in data.get("problems", []):
            key = (info["dataset"], info["lm"], info["rm"], str(problem["problem_id"]))
            index[key] = (path, problem)
    return index


def irregularity_metrics(scores: np.ndarray, tail_mu: float, tail_sigma: float, tail_r2: float, full_r2: float) -> dict:
    q005, q01, q05, q25, q50, q75, q95, q99, q999 = np.quantile(
        scores, [0.005, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999]
    )
    iqr = max(float(q75 - q25), 1e-9)
    robust_span = max(float(q999 - q005), 1e-9)
    tail_threshold = float(np.quantile(scores, 0.80))
    x_min = float(q005)
    x_max = float(max(q999, tail_threshold + 3.0 * tail_sigma))

    # Histogram-based roughness: enough bins to capture shoulders/bumps but not
    # score isolated outliers too much. This helps choose visually irregular
    # bodies rather than globally Gaussian-looking cases.
    counts, _ = np.histogram(scores, bins=55, range=(x_min, x_max), density=True)
    tail_threshold = float(np.quantile(scores, 0.80))
    edges = np.linspace(x_min, x_max, 56)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if counts.max() > 0:
        rough_counts = counts / counts.max()
    else:
        rough_counts = counts
    roughness = float(np.abs(np.diff(counts, n=2)).mean()) if counts.size >= 3 else 0.0
    skew_proxy = abs(float((q75 + q25 - 2.0 * q50) / iqr))
    body_gap = max(0.0, tail_r2 - full_r2)
    tail_density_mean = float(np.mean(counts[centers >= tail_threshold])) if np.any(centers >= tail_threshold) else 0.0
    density_imbalance = float(np.max(counts) / max(tail_density_mean, 1e-9))
    compression_penalty = max(0.0, robust_span / iqr - 14.0) * 0.015
    density_penalty = max(0.0, density_imbalance - 8.0) * 0.025
    visual_score = (
        2.0 * tail_r2
        + 1.2 * body_gap
        + 0.35 * min(roughness, 0.25)
        + 0.08 * min(skew_proxy, 3.0)
        - compression_penalty
        - density_penalty
    )
    return {
        "x_min": x_min,
        "x_max": x_max,
        "roughness": roughness,
        "skew_proxy": skew_proxy,
        "body_gap": body_gap,
        "density_imbalance": density_imbalance,
        "compression_penalty": compression_penalty,
        "visual_score": visual_score,
    }


def build_candidate_pool() -> list[Candidate]:
    raw = load_raw_index()
    rows = list(csv.DictReader(PROMPT_STATS.open()))
    pool: list[Candidate] = []
    for row in rows:
        if row["valid"] != "True":
            continue
        if row["dataset"] not in MAIN_DATASETS or row["LM"] not in MAIN_LMS or row["RM"] not in MAIN_RMS:
            continue
        if int(row["num_samples"]) != 5000:
            continue
        if abs(float(row["tail_fraction"]) - TAIL_FRACTION) > 1e-9:
            continue
        if float(row["R2"]) < 0.995:
            continue

        key = (row["dataset"], MAIN_LMS[row["LM"]], MAIN_RMS[row["RM"]], row["prompt_id"])
        if key not in raw:
            continue
        artifact, problem = raw[key]
        scores = np.asarray(problem["scores"], dtype=float)
        scores = scores[np.isfinite(scores)]
        if scores.size != 5000:
            continue
        tail_mu, tail_sigma, tail_r2 = fit_gaussian_qq(scores, TAIL_FRACTION)
        _, _, full_r2 = fit_gaussian_qq(scores, None)
        metrics = irregularity_metrics(scores, tail_mu, tail_sigma, tail_r2, full_r2)
        reason = (
            f"body irregular; tail fit visually strong; sample count={scores.size}; "
            f"upper-tail R2={tail_r2:.4f}; full-fit R2={full_r2:.3f}"
        )
        pool.append(
            Candidate(
                panel="",
                selected=False,
                dataset=row["dataset"],
                prompt_id=row["prompt_id"],
                lm=row["LM"],
                rm=row["RM"],
                scores=scores,
                tail_mu=tail_mu,
                tail_sigma=tail_sigma,
                tail_r2=tail_r2,
                full_r2=full_r2,
                visual_score=float(metrics["visual_score"]),
                density_imbalance=float(metrics["density_imbalance"]),
                artifact_path=artifact,
                reason=reason,
            )
        )
    return sorted(pool, key=lambda c: c.visual_score, reverse=True)


def select_examples(pool: list[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    by_key = {cand.key: cand for cand in pool}
    preferred_keys = [
        # Visually clear body irregularity plus strong upper-tail fit.
        (
            "wildbench_v2",
            "wildbench_v2_60ce6a322ce941f6",
            "meta-llama/Llama-3.1-8B-Instruct",
            "weqweasdas/RM-Mistral-7B",
        ),
        (
            "arena_hard",
            "arena_hard_f4c8b624448a40dc82757a8801093c1b",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        ),
        (
            "alpaca_eval",
            "alpaca_eval2_391",
            "Qwen/Qwen3-8B",
            "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        ),
        (
            "wildbench_v2",
            "wildbench_v2_eb1cf5d4749048fd",
            "meta-llama/Llama-3.1-8B-Instruct",
            "weqweasdas/RM-Mistral-7B",
        ),
    ]
    selected = [by_key[key] for key in preferred_keys if key in by_key]
    if len(selected) != 4:
        missing = [key for key in preferred_keys if key not in by_key]
        raise RuntimeError(f"Missing preferred candidates: {missing}")

    selected_keys = {cand.key for cand in selected}
    filtered_pool = [
        cand
        for cand in pool
        if cand.density_imbalance < 9.0 and cand.key not in selected_keys
    ]
    candidates = selected + filtered_pool[: max(0, 12 - len(selected))]

    for i, cand in enumerate(selected):
        cand.panel = f"({chr(ord('a') + i)})"
        cand.selected = True
    for i, cand in enumerate(candidates):
        if not cand.panel:
            cand.panel = f"cand{i + 1}"
    return selected, candidates


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_panel(ax: plt.Axes, cand: Candidate, panel_label: str, show_ylabel: bool = False, show_legend: bool = False) -> None:
    scores = cand.scores
    threshold = float(np.quantile(scores, 0.80))
    q005, q999 = np.quantile(scores, [0.005, 0.999])
    x_min = float(q005)
    x_max = float(max(q999, threshold + 3.0 * cand.tail_sigma))
    if x_max <= x_min:
        x_min, x_max = float(scores.min()), float(scores.max())
    bins = 60
    counts, edges = np.histogram(scores, bins=bins, range=(x_min, x_max), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    colors = np.where(centers >= threshold, TAIL_COLOR, BODY_COLOR)
    ax.bar(
        centers,
        counts,
        width=widths,
        color=colors,
        edgecolor="white",
        linewidth=0.25,
        alpha=1.0,
        align="center",
        zorder=2,
    )

    x_curve = np.linspace(x_min, x_max, 500)
    pdf = normal_pdf(x_curve, cand.tail_mu, cand.tail_sigma)
    tail_norm = max(1.0 - normal_cdf_scalar(threshold, cand.tail_mu, cand.tail_sigma), 1e-9)
    y_curve = TAIL_FRACTION * pdf / tail_norm

    # Stabilize visual scale by matching the average curve height to the
    # average density of tail bars. This keeps the line readable while preserving
    # the fitted shape.
    tail_mask = centers >= threshold
    curve_mask = x_curve >= threshold
    if tail_mask.any() and curve_mask.any():
        target = float(np.mean(counts[tail_mask]))
        current = float(np.mean(y_curve[curve_mask]))
        if current > 1e-12 and target > 0:
            y_curve = y_curve * (target / current)
    ax.plot(x_curve, y_curve, color=FIT_COLOR, linestyle="--", linewidth=1.9, zorder=4)

    ax.text(0.02, 0.96, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=8)
    ax.set_xlim(x_min, x_max)
    ymax = max(float(np.max(counts)), float(np.max(y_curve))) * 1.08
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Reward", labelpad=1.5)
    if show_ylabel:
        ax.set_ylabel("Density", labelpad=2)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(axis="both", colors=AXIS_COLOR, width=0.65, length=2.5, pad=1)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.35, zorder=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    if show_legend:
        handles = [
            Patch(facecolor=BODY_COLOR, edgecolor="white", label="Body"),
            Patch(facecolor=TAIL_COLOR, edgecolor="white", label="Tail"),
            Line2D([0], [0], color=FIT_COLOR, linestyle="--", linewidth=1.7, label="Gaussian fit"),
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.12, 1.03),
            frameon=False,
            ncol=3,
            handlelength=1.6,
            columnspacing=0.8,
            borderaxespad=0.0,
            fontsize=5.8,
        )


def plot_main(selected: list[Candidate]) -> None:
    style_matplotlib()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.65), dpi=300, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.005, h_pad=0.005, wspace=0.035, hspace=0.0)
    for i, (ax, cand) in enumerate(zip(axes, selected)):
        plot_panel(ax, cand, f"({chr(ord('a') + i)})", show_ylabel=(i == 0), show_legend=False)
    fig.savefig(FIG_DIR / "fig_tail_motivation.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(FIG_DIR / "fig_tail_motivation.png", bbox_inches="tight", pad_inches=0.015, dpi=300)
    plt.close(fig)


def plot_large(selected: list[Candidate]) -> None:
    style_matplotlib()
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.45), dpi=300, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.04, hspace=0.0)
    for i, (ax, cand) in enumerate(zip(axes, selected)):
        plot_panel(ax, cand, f"({chr(ord('a') + i)})", show_ylabel=(i == 0), show_legend=False)
    fig.savefig(FIG_DIR / "fig_tail_motivation_large.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG_DIR / "fig_tail_motivation_large.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def plot_candidate_sheet(candidates: list[Candidate]) -> None:
    style_matplotlib()
    fig, axes = plt.subplots(3, 4, figsize=(8.0, 5.1), dpi=300, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.08)
    for idx, (ax, cand) in enumerate(zip(axes.flat, candidates)):
        plot_panel(ax, cand, f"{idx + 1}", show_ylabel=(idx % 4 == 0), show_legend=False)
    for ax in axes.flat[len(candidates):]:
        ax.axis("off")
    fig.savefig(FIG_DIR / "fig_tail_motivation_candidates.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(FIG_DIR / "fig_tail_motivation_candidates.png", bbox_inches="tight", pad_inches=0.025, dpi=300)
    plt.close(fig)


def write_metadata(selected: list[Candidate], candidates: list[Candidate]) -> None:
    selected_keys = {c.key for c in selected}
    panel_map = {c.key: f"({chr(ord('a') + i)})" for i, c in enumerate(selected)}
    rows = selected + [c for c in candidates if c.key not in selected_keys]
    with (SUMMARY_DIR / "phase0_tail_motivation_candidates.csv").open("w", newline="") as f:
        fields = [
            "panel",
            "selected",
            "dataset",
            "prompt_id",
            "LM",
            "RM",
            "num_samples",
            "tail_fraction",
            "tail_fit_score",
            "full_distribution_fit_score",
            "artifact_path",
            "reason_selected",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, cand in enumerate(rows, 1):
            writer.writerow(
                {
                    "panel": panel_map.get(cand.key, f"candidate_{i}"),
                    "selected": "yes" if cand.key in selected_keys else "no",
                    "dataset": cand.dataset,
                    "prompt_id": cand.prompt_id,
                    "LM": cand.lm,
                    "RM": cand.rm,
                    "num_samples": cand.scores.size,
                    "tail_fraction": f"{TAIL_FRACTION:.2f}",
                    "tail_fit_score": f"{cand.tail_r2:.6f}",
                    "full_distribution_fit_score": f"{cand.full_r2:.6f}",
                    "artifact_path": str(cand.artifact_path),
                    "reason_selected": cand.reason,
                }
            )


def write_latex() -> None:
    (SUMMARY_DIR / "phase0_tail_motivation_latex_snippet.tex").write_text(
        "\\begin{figure*}[t]\n"
        "\\centering\n"
        "\\includegraphics[width=\\textwidth]{figures/phase0/fig_tail_motivation.pdf}\n"
        "\\caption{\\textbf{Representative reward distributions with Gaussian-like upper tails.}\n"
        "Although the body of the reward distribution can be irregular, its high-reward region is often well approximated by a Gaussian fit estimated from the upper tail. We show four representative examples; Section~\\ref{subsec:tail_validation} provides the systematic validation on three datasets.}\n"
        "\\label{fig:tail_motivation}\n"
        "\\end{figure*}\n"
    )


def validate(selected: list[Candidate], candidates: list[Candidate]) -> None:
    assert len(selected) == 4
    assert len(candidates) == 12
    stat_rows = {
        (r["dataset"], r["prompt_id"], r["LM"], r["RM"]): r
        for r in csv.DictReader(PROMPT_STATS.open())
    }
    for cand in selected:
        row = stat_rows[(cand.dataset, cand.prompt_id, cand.lm, cand.rm)]
        assert row["valid"] == "True"
        assert int(row["num_samples"]) == 5000
        assert abs(float(row["tail_fraction"]) - TAIL_FRACTION) < 1e-9
        assert abs(float(row["R2"]) - cand.tail_r2) < 2e-6
        assert cand.tail_r2 >= 0.995
        assert cand.artifact_path.exists()
    for path in [
        FIG_DIR / "fig_tail_motivation.pdf",
        FIG_DIR / "fig_tail_motivation.png",
        FIG_DIR / "fig_tail_motivation_large.pdf",
        FIG_DIR / "fig_tail_motivation_large.png",
        FIG_DIR / "fig_tail_motivation_candidates.pdf",
        FIG_DIR / "fig_tail_motivation_candidates.png",
        SUMMARY_DIR / "phase0_tail_motivation_candidates.csv",
        SUMMARY_DIR / "phase0_tail_motivation_latex_snippet.tex",
    ]:
        assert path.exists() and path.stat().st_size > 0, path


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_tail_motivation")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    pool = build_candidate_pool()
    selected, candidates = select_examples(pool)
    plot_main(selected)
    plot_large(selected)
    plot_candidate_sheet(candidates)
    write_metadata(selected, candidates)
    write_latex()
    validate(selected, candidates)
    for cand in selected:
        print(
            cand.panel,
            cand.dataset,
            cand.prompt_id,
            cand.lm,
            cand.rm,
            f"tail_R2={cand.tail_r2:.6f}",
            f"full_R2={cand.full_r2:.6f}",
            f"visual={cand.visual_score:.4f}",
        )


if __name__ == "__main__":
    main()
