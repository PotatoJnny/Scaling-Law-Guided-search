"""Run the real SLG implementation with synthetic generation and rewards on CPU."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.slg_mcts import SLG_Search
from core.base_reward import BaseRewardEngine
from tasks.base_task import BaseTask


class ToyGenerator:
    """Two synthetic approaches with different reward means and variances."""

    tokenizer = None

    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        self.generated = 0

    def generate(self, prompts, n, **kwargs):
        batches = []
        for prompt in prompts:
            responses = []
            for _ in range(n):
                if "Approach: " in prompt:
                    approach = prompt.split("Approach: ", 1)[1].strip()
                    prefix = ""
                else:
                    approach = str(self.rng.choice(["steady", "variable"]))
                    prefix = f"Approach: {approach}\n\n"
                mean, std = (1.0, 0.2) if approach == "steady" else (0.5, 1.0)
                reward = self.rng.normal(mean, std)
                responses.append(f"{prefix}Reward: {reward:.6f}")
            batches.append(responses)
            self.generated += len(responses)
        return batches


class ToyReward(BaseRewardEngine):
    def __init__(self):
        self.observed = []

    def score_states_batch(self, states, **kwargs):
        rewards = [
            float(state.get_full_response().rsplit("Reward: ", 1)[1])
            for state in states
        ]
        self.observed.extend(rewards)
        return rewards


class ToyTask(BaseTask):
    def __init__(self):
        super().__init__(
            dataset_config={},
            action_strategy={
                "chunking_method": "delimiter",
                "delimiter": "\n\n",
                "stop_sequences": [],
            },
        )

    def get_prompt(self, problem_data):
        return problem_data["prompt"]

    def extract_answer(self, text):
        return text.rsplit("Reward: ", 1)[-1].strip()


def run_demo():
    config = SimpleNamespace(
        N=96, K=2, m=16, max_depth=4, max_tokens=64,
        tail_fraction=20, verbose=False, seed=7,
    )
    generator = ToyGenerator(seed=config.seed)
    reward = ToyReward()
    runner = SLG_Search(generator, reward, ToyTask(), config)
    result = runner.run({"prompt": "Choose an approach.\n"})

    # Check the full search path, including exploration and exploitation.
    assert result["total_rollouts"] == generator.generated == len(reward.observed) == config.N
    assert result["best_score"] == max(reward.observed)

    return {
        "example": "synthetic CPU search",
        "algorithm": "slg",
        "budget": config.N,
        "total_rollouts": result["total_rollouts"],
        "shared_rollouts": runner.last_run_debug["shared_stage"]["shared_rollouts"],
        "selection": runner.last_run_debug["selection"],
        "best_score": result["best_score"],
        "best_response": result["full_text"],
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
