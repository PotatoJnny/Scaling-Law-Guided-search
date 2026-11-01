# Extending the framework

The search algorithm consumes a generator, a reward engine, and a task. [`examples/toy_search.py`](../examples/toy_search.py) demonstrates all three with the real `SLG_Search` implementation on CPU.

## Add a dataset to an existing task

1. Add its key and column mapping to `DATASET_CONFIGS` in [`tasks/dataset_configs.py`](../tasks/dataset_configs.py). Set `hf_name`, `split`, `task_type`, and `reward_type` as appropriate.
2. For a standard Hugging Face dataset, [`core/runtime.py`](../core/runtime.py) loads it using that configuration. For a custom source, add a loader and route it in `load_task_dataset`. The instruction-following loaders in [`tasks/instruction_following_datasets.py`](../tasks/instruction_following_datasets.py) illustrate normalizing source records.
3. Create a JSON config using the dataset key and a suitable action strategy. Run a few prompts and inspect their formatting, selected responses, and metrics.

Preserve source identifiers when normalizing data so results from different methods can be joined by prompt.

## Add a task

Subclass [`BaseTask`](../tasks/base_task.py) and implement `get_prompt` and `extract_answer`. Register it in [`TASK_REGISTRY`](../tasks/task_registry.py), then reference its `task_type` from the dataset configuration.

Additional hooks cover dataset preparation, problem preparation, response cleaning, task metrics, and serialization. [`LanguageTask`](../tasks/language_task.py) shows chat-template integration and instruction-following output handling; [`MathTask`](../tasks/math_task.py) shows answer extraction.

Action strategies live in [`tasks/action_configs.py`](../tasks/action_configs.py). If an existing delimiter, regex, token-count, thinking, or code parser fits the task, configure it there. New parsing behavior belongs in [`tasks/action_parsers.py`](../tasks/action_parsers.py).

A prefix state must reconstruct the exact text from which generation continues. Verify splitting and reconstruction on both a complete response and a continuation before running a larger experiment.

## Add a reward function

Implement [`BaseRewardEngine.score_states_batch`](../core/base_reward.py). It receives states and returns one numerical score per state, in the same order. Higher scores are better.

Register the implementation in [`core/reward_registry.py`](../core/reward_registry.py) and add its constructor arguments to `create_reward_engine`. `set_problem` can supply problem-specific data. Model-based scoring and code-execution scoring provide existing examples.

For direct use, pass the engine to the algorithm constructor without a registry entry, as the CPU example does.

## Add a generator

Provide `generate(prompts, n, max_tokens, stop_sequences, temperature, top_p, seed)` with the same behavior as [`LLMEngine.generate`](../core/llm_engine.py): return a list of generated-text lists, one inner list per input prompt. The runtime may request many continuations and expects their actual count to be reflected in the result.

For full model-backed runs, expose the tokenizer and prompt-formatting helpers used by the task. For synthetic or custom integrations, the CPU example shows the smaller interface needed by the search itself.

## Add a search method

Subclass [`BaseAlgorithm`](../algorithms/base_algo.py), implement `run(problem_data)` and `clean()`, and register the method in `create_algorithm_runner` in [`core/runtime.py`](../core/runtime.py).

Return the fields consumed by `main.py`: `full_text`, `best_score`, `total_rollouts`, and task-specific answer or trace fields. [`BestOfN`](../algorithms/best_of_n.py) is a compact example. Add configuration handling for the new method; do not rely on the factory's historical fallback to SLG.

For Phase 1 integration, also add the method's mapping and parameter block in [`run_phase1_job.py`](../experiments/search/run_phase1_job.py).

## Verify an integration

Run the smallest complete example for the new component. Check that the returned response can be scored, generated-continuation counts match the stated budget, and task metrics describe the intended objective. The existing CPU path can be run with:

```bash
make check
```

A CPU run checks the search interfaces and budget accounting. A model-backed integration also needs a run with its actual model and data.
