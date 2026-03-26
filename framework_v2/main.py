import argparse
import json
import time
import os
import datetime
import numpy as np
from datasets import load_dataset, load_from_disk

# Core Framework Imports
from core.llm_engine import LLMEngine
from core.rm_engine import RMEngine
from tasks.math_task import MathTask
from tasks.task_configs import DATASET_CONFIGS, ACTION_STRATEGIES
from algorithms.slg_mcts import SLG_Search
from algorithms.best_of_n import BestOfN
from algorithms.branching_bon import BranchingBoN
from evaluation.evaluator import Evaluator

def calculate_auto_params(slg_params: dict):
    """Auto-calculates K and m if they are set to -1 in the JSON config."""
    N = slg_params.get("N", 100)
    m = slg_params.get("m", -1)
    K = slg_params.get("K", -1)

    if m == -1:
        calculated_m = (np.log(N) ** 3) / 5
        m = max(20, int(round(calculated_m / 5) * 5))
        slg_params["m"] = m
        print(f"[Auto-Config] m calculated from N={N}: {m}")
        
    if K == -1:
        calculated_K = int(round(N / (2 * m)))
        max_budget = int(N / (m + 2))
        K = max(2, min(calculated_K, max_budget))
        slg_params["K"] = K
        print(f"[Auto-Config] K calculated from N={N}, m={m}: {K}")
        
    return slg_params

# We create a simple object to pass the parsed JSON params into your algorithm cleanly
class ConfigWrapper:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON experiment config file")
    parser.add_argument("--num_problems", type=int, default=None, help="Limit to first N problems in the dataset")
    args = parser.parse_args()

    # 1. Load the JSON Config
    with open(args.config, 'r') as f:
        cfg = json.load(f)

    print("="*80)
    print(f"🚀 LAUNCHING EXPERIMENT: {cfg['experiment_name']}")
    print("="*80)

    # 2. Process Auto-Calculations & Init Evaluator
    algorithm_name = cfg.get("algorithm", "slg")
    if algorithm_name == "slg":
        cfg["slg_params"] = calculate_auto_params(cfg["slg_params"])
    elif algorithm_name == "bbon":
        cfg["bbon_params"] = calculate_auto_params(cfg["bbon_params"])

    # Pass the whole config to the evaluator so it gets saved in results.json
    evaluator = Evaluator(experiment_name=cfg['experiment_name'], config=cfg)

    # 3. Boot Hardware 
    print("\nLoading Engines...")
    hw = cfg["hardware"]
    llm = LLMEngine(
        model_name=hw["lm_name"],
        tensor_parallel_size=1,
        gpu_memory_utilization=hw.get("gpu_memory_utilization", 0.6)
    )
    rm = RMEngine(model_name=hw["rm_name"], max_batch_size=hw["rm_max_batch_size"])
    
    task_cfg = cfg["task_setup"]
    task = MathTask(
        dataset_config=DATASET_CONFIGS[task_cfg["dataset_name"]],
        action_strategy=ACTION_STRATEGIES[task_cfg["action_strategy"]],
        tokenizer=llm.tokenizer
    )
    
    # Wrap the raw dictionary into an object so algorithms can do `self.config.N`
    if algorithm_name == "bon":
        algo_config = ConfigWrapper(cfg["bon_params"])
        runner = BestOfN(llm_engine=llm, rm_engine=rm, task=task, config=algo_config)
    elif algorithm_name == "bbon":
        algo_config = ConfigWrapper(cfg["bbon_params"])
        runner = BranchingBoN(llm_engine=llm, rm_engine=rm, task=task, config=algo_config)
    else:
        algo_config = ConfigWrapper(cfg["slg_params"])
        runner = SLG_Search(llm_engine=llm, rm_engine=rm, task=task, config=algo_config)

    # 4. Load Dataset
    safe_name = task_cfg["dataset_name"].replace("/", "_")
    local_path = f"./local_{safe_name}_{task_cfg['dataset_config']}_data"
    
    if not os.path.exists(local_path):
        print(f"Downloading {task_cfg['dataset_name']}...")
        dataset = load_dataset(task_cfg['dataset_name'], task_cfg['dataset_config'], split='test')
        dataset.save_to_disk(local_path)
    else:
        dataset = load_from_disk(local_path)

    # 5. The Execution Loop
    total_problems = min(args.num_problems, len(dataset)) if args.num_problems else len(dataset)
    experiment_start = time.time()

    for idx, question_data in enumerate(dataset):
        if args.num_problems is not None and idx >= args.num_problems:
            break

        elapsed = time.time() - experiment_start
        if idx > 0:
            avg_sec = elapsed / idx
            eta = datetime.timedelta(seconds=int(avg_sec * (total_problems - idx)))
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
            print(f"\n--- Problem {idx + 1}/{total_problems} | Elapsed: {elapsed_str} | ETA: {eta} ---")
        else:
            print(f"\n--- Problem {idx + 1}/{total_problems} ---")
        
        prompt_text = task.get_prompt(question_data)
        true_answer = task.extract_answer(question_data[task.dataset_config["answer_column"]])

        # Run Search
        runner.clean()
        start_time = time.time()
        result = runner.run(question_data)
        search_time = time.time() - start_time

        # Calculate explicit metrics using the Evaluator
        pass_at_all = evaluator.get_pass_at_all(result["all_answers"], true_answer)
        pass_at_1 = evaluator.get_pass_at_1(result["predicted_answer"], true_answer)

        # Package and Record Results (using algorithm-agnostic keys)
        result_dict = {
            "experiment_index": idx + 1,
            "prompt": prompt_text,
            "true_answer": str(true_answer).strip() if true_answer else None,

            "search_time": search_time,
            "best_response": result["full_text"],
            "best_score": result["best_score"],
            "total_rollouts": result["total_rollouts"],

            "pass_at_1": pass_at_1,
            "pass_at_all": pass_at_all
        }
        
        evaluator.record_experiment(result_dict)

    # 6. Finalize and Save Comprehensive Report
    evaluator.generate_final_report()

if __name__ == '__main__':
    main()