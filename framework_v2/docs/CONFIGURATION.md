# Configuration and runtime

Run commands from `framework_v2/`. The CLI takes a JSON file:

```bash
python main.py --config configs/slg_alpaca_eval.json --num_problems 2
python main.py --help
```

Use Python 3.11. `requirements-demo.txt` installs NumPy and SciPy for the CPU example. `requirements.txt` adds the model and dataset runtime, using public package versions from the portable experiment environment. The full cluster inventory remains in `requirements_fir.txt`.

## Experiment fields

| Field | Meaning |
| --- | --- |
| `experiment_name` | Human-readable run name |
| `algorithm` | `slg`, `best_of_n` (or `bon`), `bbon`, `mean_search`, `slg_tree`, or `greedy` |
| `seed` | Python, NumPy, and PyTorch seed |
| `output_mode` | The instruction-following examples use `answer_only_no_think` |
| `output_dir` | Results directory; defaults to `data/results/Results/<experiment_name>` |
| `hardware` | Generator, reward model, and runtime settings |
| `task_setup` | Dataset, action strategy, and task overrides |
| `slg_params` / `bon_params` / `bbon_params` | Parameters for the selected algorithm |

Choose the corresponding parameter block when switching methods. `mean_search` and `slg_tree` use `slg_params`; `greedy` uses `bon_params` and sets `N=1`, temperature 0, and top-p 1.

## Models and hardware

The supplied examples use Qwen3-8B with Skywork-Reward-V2-Llama-3.1-8B on a single H100. Both models are loaded in the same process. Set `hardware.lm_name` and `hardware.rm_name` to the model identifiers or compatible local model directories.

| Setting | Purpose |
| --- | --- |
| `backend` | `vllm` by default; `transformers` is also implemented |
| `lm_tensor_parallel_size` | Tensor parallelism for the generator |
| `gpu_memory_utilization` | Fraction reserved by vLLM; leave room for the reward model |
| `max_model_len` | Generator context limit |
| `lm_max_prompt_batch_size` | Number of prompts per generation call |
| `lm_max_samples_per_call` | Continuations per prompt in each engine call |
| `rm_max_batch_size` | Reward-model scoring batch size |
| `rm_max_length` | Reward-model context limit, default 4,096 |
| `enforce_eager` | vLLM eager execution setting |
| `rm_quantization` | Enable 4-bit reward-model loading; requires bitsandbytes and accelerate |

Lower the prompt, continuation, and reward batches when adapting to less GPU memory. The Transformers backend currently loads the generator from the local Hugging Face cache (`local_files_only=True`); pre-download its weights and tokenizer. The model engine imports vLLM for both backend choices, so both use the full runtime requirements.

## Search parameters

| Parameter | Meaning |
| --- | --- |
| `N` | Total generated-continuation budget per prompt |
| `K` | Candidate prefix count for search |
| `m` | Root sample count and target exploration samples per candidate |
| `tail_fraction` | Upper-tail percentage for the estimator; default `20` means 20% |
| `max_depth` | Maximum state depth used by the search |
| `max_tokens` | Maximum tokens generated in one continuation call |
| `temperature`, `top_p` | Sampling settings |
| `seed` | Base seed passed to generation calls; keep aligned with the experiment seed |
| `verbose` | Print search diagnostics |

In one-layer SLG, root sampling and candidate exploration both consume `N`. The runtime reduces exploration to fit the available budget and assigns the remainder to the selected state. The root can itself be selected. The estimator's `N` argument is the configured total budget.

For SLG and Branching BoN, `K=-1` or `m=-1` enables automatic parameter selection in [`core/runtime.py`](../core/runtime.py). The supplied examples retain the Phase 1 `legacy` rule. `auto_param_strategy: safe_budget` is a separate supported rule; select it explicitly when needed.

`max_tokens` is a per-call limit. A continuation can extend an existing prefix. Use the existing `strict_total_output_tokens` search parameter when the experiment requires a cap on the entire generated response. Historical experiments may also set `strict_prefix_min_ratio` and `strict_prefix_max_ratio`; retain these when reproducing those runs.

## Datasets and state construction

Registered datasets are listed in [`tasks/dataset_configs.py`](../tasks/dataset_configs.py). They include AlpacaEval, Arena-Hard, WildBench v2, IFBench, MATH-500, GSM8K, AIME, AMC, and PIE tasks. Some code tasks require separately prepared local data and execution dependencies.

Set `task_setup.dataset_name` to the registry key, such as `alpaca_eval` or `math500`. `dataset_config` supplies a Hugging Face dataset configuration where needed. `dataset_config_overrides` and `action_strategy_overrides` customize one run without editing registry defaults.

`task_setup.action_strategy` selects response parsing from [`tasks/action_configs.py`](../tasks/action_configs.py). Instruction-following examples use `answer_only_no_think`. Delimiter, fixed-token, thinking-block, and code strategies are also available.

State construction is part of an experiment's method. `double_newline_3step`, answer-only stride variants, and the `paper_default_state` override use different parsing paths. Match the saved action strategy, overrides, and `legacy_prompt_path` when reproducing a historical run.

## Inputs, outputs, and prompt selection

`--num_problems` limits the number of prompts. `--start_index` selects the first dataset position. `task_setup.problem_indices` selects an explicit ordered subset before these CLI limits are applied.

Dataset caches live under `data/local_cache/`. Hugging Face model assets follow its cache configuration, including `HF_HOME`. For AlpacaEval, `ALPACA_EVAL_JSON` can point to an existing dataset JSON file.

`results.json` stores the resolved configuration, summary, and per-prompt records. Running again with the same `output_dir` writes to the same result file; give distinct experiments distinct output directories. Direct `main.py` runs do not create `final_summary.json`; the Phase 1 runner provides that additional summary.
