import json
import os

import numpy as np
from datasets import load_dataset, load_from_disk

from algorithms.best_of_n import BestOfN
from algorithms.branching_bon import BranchingBoN
from algorithms.slg_mcts import MeanGuidedSearch, SLG_Search
from algorithms.slg_tree import SLGTreeSearch
from tasks.action_configs import ACTION_STRATEGIES
from tasks.dataset_configs import DATASET_CONFIGS
from tasks.instruction_following_datasets import load_instruction_following_dataset
from tasks.task_registry import create_task


def calculate_auto_params(search_params: dict):
    N = search_params.get("N", 100)
    m = search_params.get("m", -1)
    K = search_params.get("K", -1)
    auto_strategy = search_params.get("auto_param_strategy", "legacy")

    if m == -1:
        if auto_strategy == "safe_budget":
            min_m = search_params.get("min_m", 30)
            m = max(min_m, N // 5)
            if m % 5 != 0:
                m = int(round(m / 5) * 5)
        else:
            calculated_m = (np.log(N) ** 3) / 5
            m = max(20, int(round(calculated_m / 5) * 5))
        search_params["m"] = m
        print(f"[Auto-Config] m calculated from N={N} with strategy={auto_strategy}: {m}")

    if K == -1:
        if auto_strategy == "safe_budget":
            K = search_params.get("default_K", 2)
            max_explore_fraction = search_params.get("max_explore_fraction", 0.6)
            max_explore_rollouts = max(K, int(N * max_explore_fraction))
            if K * m > max_explore_rollouts:
                adjusted_m = max(1, max_explore_rollouts // K)
                if adjusted_m != m:
                    m = adjusted_m
                    search_params["m"] = m
                    print(f"[Auto-Config] Adjusted m to satisfy K*m <= {max_explore_fraction:.2f}N: {m}")
        else:
            calculated_K = int(round(N / (2 * m)))
            max_budget = int(N / (m + 2))
            K = max(2, min(calculated_K, max_budget))
        search_params["K"] = K
        print(f"[Auto-Config] K calculated from N={N}, m={m} with strategy={auto_strategy}: {K}")

    return search_params


class ConfigWrapper:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


def prepare_experiment_config(config_path: str, start_index: int) -> dict:
    with open(config_path, "r") as f:
        cfg = json.load(f)

    if start_index > 0:
        cfg["experiment_name"] = f"{cfg['experiment_name']}_shard{start_index}"

    algorithm_name = cfg.get("algorithm", "slg")
    if algorithm_name in {"slg", "slg_tree", "mean_search"}:
        cfg["slg_params"] = calculate_auto_params(cfg["slg_params"])
    elif algorithm_name == "bbon":
        cfg["bbon_params"] = calculate_auto_params(cfg["bbon_params"])
    elif algorithm_name == "greedy":
        bon_params = dict(cfg.get("bon_params", {}))
        bon_params["N"] = 1
        bon_params["temperature"] = 0.0
        bon_params["top_p"] = 1.0
        cfg["bon_params"] = bon_params

    return cfg


def resolve_task_setup(task_setup: dict):
    dataset_config = DATASET_CONFIGS[task_setup["dataset_name"]]
    action_strategy = ACTION_STRATEGIES[task_setup["action_strategy"]]
    task_type = dataset_config.get("task_type", "math")
    reward_type = task_setup.get("reward_type", dataset_config.get("reward_type", "model"))
    return dataset_config, action_strategy, task_type, reward_type


def create_algorithm_runner(algorithm_name: str, cfg: dict, llm, reward_engine, task):
    if algorithm_name in {"bon", "best_of_n", "greedy"}:
        algo_config = ConfigWrapper(cfg["bon_params"])
        return BestOfN(llm_engine=llm, rm_engine=reward_engine, task=task, config=algo_config)
    if algorithm_name == "bbon":
        algo_config = ConfigWrapper(cfg["bbon_params"])
        return BranchingBoN(llm_engine=llm, rm_engine=reward_engine, task=task, config=algo_config)
    if algorithm_name == "mean_search":
        algo_config = ConfigWrapper(cfg["slg_params"])
        return MeanGuidedSearch(llm_engine=llm, rm_engine=reward_engine, task=task, config=algo_config)
    if algorithm_name == "slg_tree":
        algo_config = ConfigWrapper(cfg["slg_params"])
        return SLGTreeSearch(llm_engine=llm, rm_engine=reward_engine, task=task, config=algo_config)

    algo_config = ConfigWrapper(cfg["slg_params"])
    return SLG_Search(llm_engine=llm, rm_engine=reward_engine, task=task, config=algo_config)


def load_task_dataset(task_setup: dict, dataset_config: dict, cache_root: str):
    dataset_name = task_setup["dataset_name"]
    if dataset_name in {"alpaca_eval", "arena_hard", "wildbench_v2", "ifbench"}:
        return load_instruction_following_dataset(dataset_name, cache_root)

    safe_name = task_setup["dataset_name"].replace("/", "_")
    ds_config = task_setup.get("dataset_config") or None
    config_suffix = ds_config if ds_config else "default"
    local_path = os.path.join(cache_root, f"local_{safe_name}_{config_suffix}_data")

    hf_name = dataset_config.get("hf_name", task_setup["dataset_name"])
    if not os.path.exists(local_path):
        print(f"Downloading {hf_name}...")
        split = dataset_config.get("split", "test")
        if ds_config:
            dataset = load_dataset(hf_name, ds_config, split=split)
        else:
            dataset = load_dataset(hf_name, split=split)
        dataset.save_to_disk(local_path)
    else:
        dataset = load_from_disk(local_path)

    return dataset


def build_task(task_type: str, dataset_config: dict, action_strategy: dict, tokenizer=None):
    return create_task(task_type, dataset_config, action_strategy, tokenizer=tokenizer)
