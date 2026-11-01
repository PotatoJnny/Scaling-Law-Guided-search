# Experiments and evaluation

Run commands from the repository root. Start with the [example configurations](../configs/README.md) for a model-backed run. The research workflows below retain their experiment-specific inputs and settings.

## Main search comparisons

[`experiments/search/`](../experiments/search/) contains manifest generation, configuration building, single-job execution, Slurm submission, and reporting. Shared settings live in [`configs/phase1_generation_defaults.yaml`](../configs/phase1_generation_defaults.yaml) and [`configs/phase1_seeds.json`](../configs/phase1_seeds.json).

The framework implements Best-of-N, Branching BoN, mean-guided search, SLG, and tree-structured SLG. The main manifest generator enumerates the first four methods plus historical `darwin` entries. `darwin` has no dedicated runtime implementation in this checkout; use an implemented method for new runs.

| Script | Role |
| --- | --- |
| [`build_phase1_manifest.py`](../experiments/search/build_phase1_manifest.py) | Generate the broad historical experiment matrix |
| [`run_phase1_job.py`](../experiments/search/run_phase1_job.py) | Build a runtime config and execute one manifest entry |
| [`submit_phase1_jobs.py`](../experiments/search/submit_phase1_jobs.py) | Submit selected entries to Slurm, with optional sharding |
| [`benchmark_eval.py`](../experiments/search/benchmark_eval.py) | Format responses and call benchmark-native evaluation |
| [`build_priority_reports.py`](../experiments/search/build_priority_reports.py) | Summarize the priority experiment families |

For an existing manifest entry, use its `expected_output_dir`:

```bash
python experiments/search/run_phase1_job.py \
  --manifest-path /path/to/manifest.jsonl \
  --expected-output-dir /path/from/the/manifest \
  --skip-official-eval
```

Omit `--skip-official-eval` to invoke the benchmark evaluator. `--strict-official-eval` makes missing evaluation dependencies or failed evaluation stop the run. Read available options with `--help`.

For Slurm, provide `--account` and `--partition`, or set `SLURM_ACCOUNT` and `SLURM_PARTITION`. The submitter uses the active `VIRTUAL_ENV` or `.venv/`, and reads optional module loads from `SLG_MODULE_LOADS`. Override `--gres`, `--venv-path`, and `--module-loads` for your cluster. Dual submission also requires `--mirror-partition` or `SLG_MIRROR_PARTITION`. Queue queries use the current login user. A direct `main.py` run needs no Slurm scheduler.

## Results and metrics

| Artifact | Meaning |
| --- | --- |
| `results.json` | Configuration, running summary, and completed per-prompt records; written throughout generation |
| `phase1_runtime_config.json` | Config constructed by the Phase 1 runner |
| `final_summary.json` | Phase 1 runner's summary after its workflow completes; inspect evaluation status too |
| Benchmark-specific input and output files | Evaluation records written beside the run results when evaluation is enabled |

`main.py` alone writes `results.json`. `final_summary.json` is specific to the Phase 1 workflow and some follow-up workflows.

Search-time reward scores are distinct from benchmark-native metrics. In Phase 1 summaries, `official_eval_status: completed` identifies successful benchmark evaluation. A deferred or unavailable evaluator may leave `final_headline_source: interim_reward_model`; that score is the reward-model proxy. For partial runs, compare methods on the same completed prompt IDs.

AlpacaEval, Arena-Hard, and WildBench evaluations may require benchmark installations and judge access. IFBench uses its instruction-following evaluator. The existing adapter records missing requirements and evaluation status in its output.

## Reward-tail analysis

[`experiments/reward_tail/`](../experiments/reward_tail/) contains the upper-tail analysis and paper-figure scripts. These scripts operate on previously collected rewards. Set `PHASE0_SOURCE_ROOT` to the directory containing `raw_scores/` and `analysis/`; its default is `data/results/if_paper_reward_dist_v1/`. Those inputs are needed to regenerate the summaries and figures. Plotting with `plot_tail_motivation_matplotlib.py` also requires matplotlib.

The [README figure](assets/reward_tail.png) is copied from the existing Phase 0 figure. Its four panels each use 5,000 scores and an upper-tail fraction of 0.20: WildBench v2 with Llama-3.1-8B and RM-Mistral; Arena-Hard with Llama-3.1-8B and Skywork; AlpacaEval with Qwen3-8B and Skywork; and a second WildBench v2 example with Llama-3.1-8B and RM-Mistral.

The underlying reward samples are not bundled with this repository. Provide the collected scores through `PHASE0_SOURCE_ROOT` to regenerate these figures.

## Reproducing a recorded experiment

For exact reproduction, use the particular experiment's saved configuration and prompt subset. The small runnable examples are starting points for using the framework, rather than a claim to regenerate every paper table from a fresh checkout.

For a local WildBench Arrow cache, set `SLG_WILDBENCH_ARROW` when using `build_frozen100_subsets.py`; otherwise it uses the dataset loader.
