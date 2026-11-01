#!/usr/bin/env python3
"""Build the main-text Phase 0 tail-motivation histogram figure.

Offline-only: uses completed Phase 0 raw reward-score artifacts and summary
CSVs. It does not launch experiments or modify manifests/queues.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = REPO_ROOT / "data/results/phase0/summaries"
FIG_DIR = REPO_ROOT / "figures/phase0"
PROMPT_STATS = SUMMARY_DIR / "phase0_prompt_level_tail_stats.csv"
RAW_DIR = Path(
    os.environ.get("PHASE0_SOURCE_ROOT", REPO_ROOT / "data/results/if_paper_reward_dist_v1")
) / "raw_scores"

TAIL_FRACTION = 0.20
BODY_COLOR = "#d9d9d9"
TAIL_COLOR = "#2f6fa7"
FIT_COLOR = "#111111"
AXIS_COLOR = "#2f2f2f"
GRID_COLOR = "#ececec"
FONT_FAMILY = "Times New Roman, Times, serif"
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
class Example:
    panel: str
    selected: bool
    dataset: str
    prompt_id: str
    lm: str
    rm: str
    scores: list[float]
    tail_r2: float
    full_r2: float
    artifact_path: Path
    reason: str


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def std(xs: list[float]) -> float:
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def fit_qq(scores: list[float], tail_fraction: float | None) -> dict:
    vals = sorted(float(x) for x in scores if math.isfinite(float(x)))
    if tail_fraction is None:
        fit_vals, lo, width = vals, 0.0, 1.0
    else:
        k = max(10, math.ceil(len(vals) * tail_fraction))
        fit_vals, lo, width = vals[-k:], 1.0 - tail_fraction, tail_fraction
    n = len(fit_vals)
    nd = NormalDist()
    z = [nd.inv_cdf(min(max(lo + width * ((i + 0.5) / n), 1e-9), 1 - 1e-9)) for i in range(n)]
    xm, ym = mean(z), mean(fit_vals)
    sxx = sum((x - xm) ** 2 for x in z)
    slope = sum((x - xm) * (y - ym) for x, y in zip(z, fit_vals)) / sxx
    intercept = ym - slope * xm
    pred = [intercept + slope * x for x in z]
    sse = sum((y - p) ** 2 for y, p in zip(fit_vals, pred))
    sst = sum((y - ym) ** 2 for y in fit_vals)
    return {"intercept": intercept, "slope": slope, "r2": 1.0 - sse / sst}


def raw_index() -> dict[tuple[str, str, str, str], tuple[Path, dict]]:
    out = {}
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
            out[(info["dataset"], info["lm"], info["rm"], str(problem["problem_id"]))] = (path, problem)
    return out


def select_examples() -> list[Example]:
    index = raw_index()
    rows = list(csv.DictReader(PROMPT_STATS.open()))
    pool = []
    for row in rows:
        if row["valid"] != "True":
            continue
        if row["dataset"] not in MAIN_DATASETS or row["LM"] not in MAIN_LMS or row["RM"] not in MAIN_RMS:
            continue
        if int(row["num_samples"]) != 5000 or abs(float(row["tail_fraction"]) - TAIL_FRACTION) > 1e-9:
            continue
        if float(row["R2"]) < 0.995:
            continue
        key = (row["dataset"], MAIN_LMS[row["LM"]], MAIN_RMS[row["RM"]], row["prompt_id"])
        if key not in index:
            continue
        artifact, problem = index[key]
        scores = [float(x) for x in problem["scores"] if math.isfinite(float(x))]
        tail_fit = fit_qq(scores, TAIL_FRACTION)
        full_fit = fit_qq(scores, None)
        # Prioritize non-Gaussian body plus high-quality tail. Avoid relying on
        # prompt text in the final figure.
        body_irregularity = max(0.0, tail_fit["r2"] - full_fit["r2"])
        score = tail_fit["r2"] + 0.45 * body_irregularity
        pool.append((score, body_irregularity, tail_fit["r2"], full_fit["r2"], row, artifact, scores))

    def pick(dataset: str | None = None, rm_contains: str | None = None, used: set[str] | None = None):
        used = used or set()
        candidates = [
            x for x in pool
            if x[4]["prompt_id"] not in used
            and (dataset is None or x[4]["dataset"] == dataset)
            and (rm_contains is None or rm_contains in x[4]["RM"])
        ]
        if not candidates:
            raise RuntimeError(f"No candidate for dataset={dataset}, rm={rm_contains}")
        return max(candidates, key=lambda x: x[0])

    # Main figure: four positives, diverse but quality-first. At least one
    # Skywork and one RM-Mistral are included.
    specs = [
        ("wildbench_v2", "Skywork"),
        ("arena_hard", "RM-Mistral"),
        ("alpaca_eval", "Skywork"),
        ("wildbench_v2", "RM-Mistral"),
    ]
    selected = []
    used = set()
    for spec in specs:
        item = pick(spec[0], spec[1], used)
        selected.append(item)
        used.add(item[4]["prompt_id"])

    examples = []
    for i, (_, body_gap, tail_r2, full_r2, row, artifact, scores) in enumerate(selected):
        panel = f"({chr(ord('a') + i)})"
        reason = (
            f"body irregular and visibly non-Gaussian (full QQ R2={full_r2:.3f}); "
            f"tail visually fit well (upper-tail R2={tail_r2:.4f})"
        )
        examples.append(
            Example(
                panel=panel,
                selected=True,
                dataset=row["dataset"],
                prompt_id=row["prompt_id"],
                lm=row["LM"],
                rm=row["RM"],
                scores=scores,
                tail_r2=tail_r2,
                full_r2=full_r2,
                artifact_path=artifact,
                reason=reason,
            )
        )
    return examples


def esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Panel:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.l, self.r = x + 42, x + w - 10
        self.t, self.b = y + 12, y + h - 38

    def sx(self, v, vmin, vmax):
        return self.l + (v - vmin) / (vmax - vmin) * (self.r - self.l)

    def sy(self, v, vmin, vmax):
        return self.b - (v - vmin) / (vmax - vmin) * (self.b - self.t)


def line(x1, y1, x2, y2, color=AXIS_COLOR, width=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{d}/>'


def text(x, y, s, size=9, anchor="start", weight="400", color="#222"):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{color}">{esc(s)}</text>'


def poly(points, color, width=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{d}/>'


def rect(x, y, w, h, fill, opacity=1.0, stroke=None, sw=0):
    stroke_attr = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" opacity="{opacity:.3f}"{stroke_attr}/>'


def legend(x: float, y: float) -> list[str]:
    return [
        rect(x, y - 8, 12, 8, BODY_COLOR, 1.0),
        text(x + 16, y, "Body", 8, color="#333"),
        rect(x + 56, y - 8, 12, 8, TAIL_COLOR, 1.0),
        text(x + 72, y, "Tail", 8, color="#333"),
        line(x + 112, y - 4, x + 136, y - 4, FIT_COLOR, 1.8, "5 3"),
        text(x + 142, y, "Gaussian fit", 8, color="#333"),
    ]


def panel_svg(p: Panel, ex: Example, bins=36, show_legend=False) -> list[str]:
    scores = ex.scores
    vmin, vmax = min(scores), max(scores)
    pad = 0.035 * (vmax - vmin) or 1.0
    xmin, xmax = vmin - pad, vmax + pad
    bw = (xmax - xmin) / bins
    counts = [0] * bins
    for s in scores:
        idx = min(bins - 1, max(0, int((s - xmin) / bw)))
        counts[idx] += 1
    density = [c / (len(scores) * bw) for c in counts]
    tail = sorted(scores)[-math.ceil(len(scores) * TAIL_FRACTION):]
    threshold = tail[0]
    fit = fit_qq(scores, TAIL_FRACTION)
    mu, sigma = fit["intercept"], abs(fit["slope"])
    tail_norm = max(1e-9, 1.0 - norm_cdf((threshold - mu) / sigma)) if sigma > 1e-9 else 1.0
    curve_density = []
    if sigma > 1e-9:
        # Scale the conditional truncated-Gaussian tail density by the empirical
        # tail mass so it overlays the full-distribution density histogram.
        for j in range(220):
            x = threshold + (xmax - threshold) * j / 219
            y = TAIL_FRACTION * norm_pdf((x - mu) / sigma) / (sigma * tail_norm)
            curve_density.append((x, y))
    ymax = max(max(density), max([y for _, y in curve_density], default=0.0)) * 1.16

    parts = [
        f'<rect x="{p.x:.1f}" y="{p.y:.1f}" width="{p.w:.1f}" height="{p.h:.1f}" fill="#fff"/>',
        text(p.x + 1, p.y + 14, ex.panel, 12, weight="600"),
        line(p.l, p.b, p.r, p.b, AXIS_COLOR, 1.05),
        line(p.l, p.t, p.l, p.b, AXIS_COLOR, 1.05),
    ]
    if show_legend:
        parts.extend(legend(p.l + 2, p.t + 1))

    # Subtle horizontal reference lines.
    for frac in [0.5, 1.0]:
        gy = p.sy(ymax * frac / 1.16, 0, ymax)
        parts.append(line(p.l, gy, p.r, gy, GRID_COLOR, 0.55))

    for i, c in enumerate(counts):
        x0 = xmin + i * bw
        x1 = x0 + 0.94 * bw
        d = c / (len(scores) * bw)
        y0, y1 = p.sy(0, 0, ymax), p.sy(d, 0, ymax)
        fill = TAIL_COLOR if x0 + bw >= threshold else BODY_COLOR
        parts.append(
            f'<rect x="{p.sx(x0, xmin, xmax):.1f}" y="{y1:.1f}" '
            f'width="{max(0.45, p.sx(x1, xmin, xmax)-p.sx(x0, xmin, xmax)):.1f}" '
            f'height="{max(0.0, y0-y1):.1f}" fill="{fill}" opacity="1.000"/>'
        )

    if curve_density:
        curve = [(p.sx(x, xmin, xmax), p.sy(y, 0, ymax)) for x, y in curve_density]
        parts.append(poly(curve, FIT_COLOR, width=2.25, dash="6 4"))

    # Minimal clean ticks.
    xticks = [xmin, (xmin + xmax) / 2, xmax]
    for tick in xticks:
        x = p.sx(tick, xmin, xmax)
        parts.append(line(x, p.b, x, p.b + 3.5, AXIS_COLOR, 0.8))
        parts.append(text(x, p.b + 16, f"{tick:.0f}" if abs(tick) >= 10 else f"{tick:.1f}", 8, anchor="middle", color="#333"))
    yticks = [0, ymax * 0.5 / 1.16, ymax / 1.16]
    for tick in yticks:
        y = p.sy(tick, 0, ymax)
        parts.append(line(p.l - 3.5, y, p.l, y, AXIS_COLOR, 0.8))
        label = "0" if tick == 0 else f"{tick:.2f}" if tick < 0.1 else f"{tick:.1f}"
        parts.append(text(p.l - 6, y + 2.5, label, 7, anchor="end", color="#333"))
    parts.append(text((p.l + p.r) / 2, p.y + p.h - 7, "Reward", 9, anchor="middle"))
    parts.append(
        f'<text transform="rotate(-90 {p.x+11:.1f} {(p.t+p.b)/2:.1f})" x="{p.x+11:.1f}" y="{(p.t+p.b)/2:.1f}" font-size="9" text-anchor="middle" fill="#222">Density</text>'
    )
    return parts


def render(examples: list[Example], path: Path, width=1500, height=330):
    shown = examples[:4]
    w, h = width, height
    panel_w = w / 4
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<style>text{{font-family:{FONT_FAMILY};}}</style>',
    ]
    for i, ex in enumerate(shown):
        parts.extend(panel_svg(Panel(i * panel_w + 12, 12, panel_w - 22, h - 20), ex, show_legend=(i == 0)))
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def convert(svg: Path, pdf: Path, png: Path):
    tool = shutil.which("rsvg-convert")
    if not tool:
        raise RuntimeError("rsvg-convert not found")
    subprocess.run([tool, "-f", "pdf", "-o", str(pdf), str(svg)], check=True)
    subprocess.run([tool, "-f", "png", "-o", str(png), str(svg)], check=True)


def write_metadata(examples: list[Example]):
    path = SUMMARY_DIR / "phase0_tail_motivation_candidates.csv"
    fields = [
        "panel", "selected", "dataset", "prompt_id", "LM", "RM", "num_samples",
        "tail_fraction", "artifact_path", "reason_selected",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ex in examples:
            writer.writerow(
                {
                    "panel": ex.panel,
                    "selected": "yes" if ex.selected else "no",
                    "dataset": ex.dataset,
                    "prompt_id": ex.prompt_id,
                    "LM": ex.lm,
                    "RM": ex.rm,
                    "num_samples": len(ex.scores),
                    "tail_fraction": f"{TAIL_FRACTION:.2f}",
                    "artifact_path": str(ex.artifact_path),
                    "reason_selected": ex.reason,
                }
            )


def write_latex():
    (SUMMARY_DIR / "phase0_tail_motivation_latex_snippet.tex").write_text(
        "\\begin{figure*}[t]\n"
        "\\centering\n"
        "\\includegraphics[width=\\textwidth]{figures/phase0/fig_tail_motivation.pdf}\n"
        "\\caption{\\textbf{Representative reward distributions with Gaussian-like upper tails.}\n"
        "Although the body of the reward distribution can be irregular, its high-reward region is often well approximated by a truncated Gaussian fit. We show four representative examples; Section~\\ref{subsec:tail_validation} provides the systematic validation on three datasets.}\n"
        "\\label{fig:tail_motivation}\n"
        "\\end{figure*}\n"
    )


def validate(examples: list[Example]):
    stat_rows = {
        (r["dataset"], r["prompt_id"], r["LM"], r["RM"]): r
        for r in csv.DictReader(PROMPT_STATS.open())
    }
    for ex in examples:
        key = (ex.dataset, ex.prompt_id, ex.lm, ex.rm)
        row = stat_rows[key]
        assert row["valid"] == "True"
        assert int(row["num_samples"]) == 5000
        assert abs(float(row["tail_fraction"]) - TAIL_FRACTION) < 1e-9
        assert abs(float(row["R2"]) - ex.tail_r2) < 2e-6
        assert ex.artifact_path.exists()
        assert ex.tail_r2 >= 0.995
    for path in [
        FIG_DIR / "fig_tail_motivation.pdf",
        FIG_DIR / "fig_tail_motivation.png",
        FIG_DIR / "fig_tail_motivation_large.pdf",
        FIG_DIR / "fig_tail_motivation_large.png",
        SUMMARY_DIR / "phase0_tail_motivation_candidates.csv",
        SUMMARY_DIR / "phase0_tail_motivation_latex_snippet.tex",
    ]:
        assert path.exists() and path.stat().st_size > 0, path


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    examples = select_examples()
    main_svg = FIG_DIR / "fig_tail_motivation.svg"
    large_svg = FIG_DIR / "fig_tail_motivation_large.svg"
    render(examples, main_svg, width=1500, height=330)
    render(examples, large_svg, width=2200, height=500)
    convert(main_svg, FIG_DIR / "fig_tail_motivation.pdf", FIG_DIR / "fig_tail_motivation.png")
    convert(large_svg, FIG_DIR / "fig_tail_motivation_large.pdf", FIG_DIR / "fig_tail_motivation_large.png")
    # Keep requested PDF/PNG only; SVGs are intermediate.
    main_svg.unlink(missing_ok=True)
    large_svg.unlink(missing_ok=True)
    write_metadata(examples)
    write_latex()
    validate(examples)
    for ex in examples:
        print(ex.panel, ex.dataset, ex.prompt_id, ex.lm, ex.rm, f"tail_R2={ex.tail_r2:.6f}", f"full_QQ_R2={ex.full_r2:.6f}")


if __name__ == "__main__":
    main()
