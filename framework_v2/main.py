import argparse
import json
import time
import os
import numpy as np
from datasets import load_dataset, load_from_disk

# Core Framework Imports
from core.llm_engine import LLMEngine
from core.rm_engine import RMEngine
from core.data_structures import State
from tasks.math_task import MathTask
from tasks.task_configs import DATASET_CONFIGS, ACTION_STRATEGIES
from algorithms.slg_mcts import SLG_Search
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
    args = parser.parse_args()

    # 1. Load the JSON Config
    with open(args.config, 'r') as f:
        cfg = json.load(f)

    print("="*80)
    print(f"🚀 LAUNCHING EXPERIMENT: {cfg['experiment_name']}")
    print("="*80)

    # 2. Process Auto-Calculations & Init Evaluator
    cfg["slg_params"] = calculate_auto_params(cfg["slg_params"])
    
    # Pass the whole config to the evaluator so it gets saved in results.json
    evaluator = Evaluator(experiment_name=cfg['experiment_name'], config=cfg)

    # 3. Boot Hardware 
    print("\nLoading Engines...")
    hw = cfg["hardware"]
    llm = LLMEngine(model_name=hw["lm_name"], tensor_parallel_size=1)
    rm = RMEngine(model_name=hw["rm_name"], max_batch_size=hw["rm_max_batch_size"])
    
    task_cfg = cfg["task_setup"]
    task = MathTask(
        dataset_config=DATASET_CONFIGS[task_cfg["dataset_name"]],
        action_strategy=ACTION_STRATEGIES[task_cfg["action_strategy"]],
        tokenizer=llm.tokenizer
    )
    
    # Wrap the raw dictionary into an object so SLG_Search can do `self.config.N`
    algo_config = ConfigWrapper(cfg["slg_params"])
    slg_runner = SLG_Search(llm_engine=llm, rm_engine=rm, task=task, config=algo_config)

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
    for idx, question_data in enumerate(dataset):
        print(f"\n--- Problem {idx + 1} ---")
        
        prompt_text = task.get_prompt(question_data)
        true_answer = task.extract_answer(question_data[task.dataset_config["answer_column"]])
        initial_state = State(prompt=prompt_text)

        # Run Search
        slg_runner.clean_tree()
        start_time = time.time()
        
        # one_layer_expand builds the tree and returns the root node
        final_tree_root = slg_runner.one_layer_expand(initial_state=initial_state)
        search_time = time.time() - start_time
        
        # Extract best response found during search
        final_text = slg_runner.best_response.get_full_response() if slg_runner.best_response else ""
        final_answer = task.extract_answer(final_text)
        
        # Calculate explicit metrics using the Evaluator
        pass_at_all = evaluator.get_pass_at_all(final_tree_root.all_answers, true_answer)
        pass_at_1 = evaluator.get_pass_at_1(final_answer, true_answer)

        # Package and Record Results (using algorithm-agnostic keys)
        result_dict = {
            "experiment_index": idx + 1,
            "prompt": prompt_text,
            "true_answer": str(true_answer).strip() if true_answer else None,
            
            "search_time": search_time,
            "best_response": final_text,
            "best_score": slg_runner.best_response_score,
            "total_rollouts": slg_runner.stats['rollouts'],
            
            "pass_at_1": pass_at_1,
            "pass_at_all": pass_at_all
        }
        
        evaluator.record_experiment(result_dict)

    # 6. Finalize and Save Comprehensive Report
    evaluator.generate_final_report()

if __name__ == '__main__':
    main()