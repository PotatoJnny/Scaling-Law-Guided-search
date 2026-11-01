# Configurations

Run these commands from `framework_v2/` after installing `requirements.txt`.

| File | Method | Models and data | Budget |
| --- | --- | --- | --- |
| [`slg_alpaca_eval.json`](slg_alpaca_eval.json) | SLG | Qwen3-8B + Skywork-Reward-V2-Llama-3.1-8B on AlpacaEval | 64 |
| [`best_of_n_alpaca_eval.json`](best_of_n_alpaca_eval.json) | Best-of-N | Same | 64 |

```bash
python main.py --config configs/slg_alpaca_eval.json --num_problems 2
python main.py --config configs/best_of_n_alpaca_eval.json --num_problems 2
```

These examples were generated from the existing Phase 1 configuration builder. They use seed 42, temperature 0.7, top-p 0.95, a 1,024-token generation limit per call, and the `answer_only_no_think` action strategy. Each has its own `output_dir` under `data/results/examples/`.

The examples exercise the runtime on two prompts. For a specific paper experiment, use its recorded configuration and prompt selection. See [experiments](../docs/EXPERIMENTS.md) and [parameter details](../docs/CONFIGURATION.md).

[`phase1_generation_defaults.yaml`](phase1_generation_defaults.yaml) contains shared sampling, generation, hardware, and search defaults. [`phase1_seeds.json`](phase1_seeds.json) contains experiment seeds. The Phase 1 runner consumes these files; `main.py` directly consumes the selected JSON file.
