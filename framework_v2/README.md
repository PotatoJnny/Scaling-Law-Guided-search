# SLG research framework

Implementation of **Scaling-Law Guided Search** for [Predicting and Improving Test-time Scaling Laws via Reward Tail-guided Search](https://arxiv.org/abs/2602.01485), NeurIPS 2026, by Muheng Li, Jian Qian, and Wenlong Mou.

See the [repository homepage](../README.md) for the method overview and citation. Run the commands below from this directory.

## Start here

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python examples/toy_search.py
```

This CPU example runs the existing SLG implementation with synthetic responses and rewards. For a model-backed run on GPU:

```bash
python -m pip install -r requirements.txt
python main.py --config configs/slg_alpaca_eval.json --num_problems 2
```

For the matching Best-of-N example:

```bash
python main.py --config configs/best_of_n_alpaca_eval.json --num_problems 2
```

Both configurations use Qwen3-8B, Skywork-Reward-V2-Llama-3.1-8B, and `N=64`. They write to separate directories under `data/results/examples/`. Model-backed examples use the project's single-H100 hardware profile.

## Guides

| Goal | Entry point |
| --- | --- |
| Configure models, datasets, budgets, or output paths | [Configuration](docs/CONFIGURATION.md) |
| Choose a runnable example | [Example configurations](configs/README.md) |
| Run or inspect experiment workflows | [Experiments and evaluation](docs/EXPERIMENTS.md) |
| Add a task, reward function, or algorithm | [Extending the framework](docs/EXTENDING.md) |
| Read the tail estimator | [`core/data_structures.py`](core/data_structures.py) |
| Read the SLG allocation policy | [`algorithms/slg_mcts.py`](algorithms/slg_mcts.py) |

## Code layout

- `algorithms/`: SLG, Best-of-N, Branching BoN, mean-guided search, and tree search.
- `core/`: model engines, reward engines, state objects, and runtime configuration.
- `tasks/`: dataset adapters, prompt construction, action parsers, and task metrics.
- `evaluation/`: per-prompt records and aggregate reporting.
- `configs/`: model-run examples and Phase 1 defaults.
- `artifacts/`: Phase 0 and Phase 1 preparation, execution, and analysis scripts.
- `examples/`: a CPU example independent of model and dataset services.
- `docs/`: usage and extension guides.

## Local checks

```bash
make check
```

Only the demo dependencies are needed. Full model execution additionally requires the GPU runtime, model weights, and benchmark data.
