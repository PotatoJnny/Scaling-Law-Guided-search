import time
import os
from datetime import datetime
from datasets import load_from_disk, load_dataset
import pandas as pd
import numpy as np
import json
import argparse

from Algorithm.algorithms import SLG_Search
from core.model_config import LMConfig, RMConfig, SLGConfig
from core.data_structures import State
from core.tools import (
    DataclassJSONEncoder, 
    create_prompt_template, 
    get_em_score_for_root, 
    get_problem_text, 
    get_problem_answer,
    extract_model_answer,
    extract_ground_truth
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="math-ai/aime25", help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset configuration (e.g., 'main' for gsm8k)")
    parser.add_argument("--lm_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Language Model Name")
    parser.add_argument("--rm_name", type=str, default="Skywork/Skywork-Reward-V2-Llama-3.1-8B", help="Reward Model Name")
    parser.add_argument("--K", type=int, default=-1, help="Search Width")
    parser.add_argument("--m", type=int, default=-1, help="Number of samples for estimation")
    parser.add_argument("--N", type=int, default=100, help="Total budget")
    
    args = parser.parse_args()

    try:
        N = args.N
        if args.m == -1:
            calculated_m = (np.log(N) ** 3) / 5
            calculated_m = int(round(calculated_m / 5) * 5)
            args.m = max(20, calculated_m)
            print(f"[Auto-Config] m not set. Calculated from N={args.N}: {args.m}")
            
        if args.K == -1:
            calculated_K = int(round(N/(2 * args.m)))
            max_budget = int((N)/(args.m + 2))
            calculated_K = min(calculated_K, max_budget)
            args.K = max(2, calculated_K)
            print(f"[Auto-Config] K not set. Calculated from N={args.N} and m={args.m}: {args.K}")

        print("Loading LLM and RM models...")
        lm_name = args.lm_name if args.lm_name else "meta-llama/Llama-3.2-1B-Instruct"
        rm_name = args.rm_name if args.rm_name else "Skywork/Skywork-Reward-V2-Llama-3.1-8B"

        lm_config = LMConfig(model_name=lm_name, max_batch_size=64, task="others", max_new_tokens=100)
        rm_config = RMConfig(model_name=rm_name, max_batch_size=64)
        ts_config = SLGConfig(K=args.K, m=args.m, N=args.N, max_depth=10, verbose=True, lm_config=lm_config, rm_config=rm_config)
        dataset_name = args.dataset

        results_folder = f"Results/{dataset_name}/SLG_LM-{lm_config.model_name.replace('/', '_')}_RM-{rm_config.model_name.replace('/', '_')}_K-{ts_config.K}_m-{ts_config.m}_N-{ts_config.N}"

        os.makedirs(results_folder, exist_ok=True)
        output_file_path = os.path.join(results_folder, "results.json")
        print(f"Results will be saved in: {output_file_path}")

        print("Loading dataset...")
        # Clean up the folder name so slashes don't create subdirectories
        safe_dataset_name = dataset_name.replace("/", "_")
        config_suffix = f"_{args.dataset_config}" if args.dataset_config else ""
        local_dataset_path = f"./local_{safe_dataset_name}{config_suffix}_data"
        
        if not os.path.exists(local_dataset_path):
            print(f"Local dataset not found. Downloading {dataset_name} from HuggingFace...")
            try:
                if args.dataset_config:
                    dataset = load_dataset(dataset_name, args.dataset_config, split='test')
                else:
                    dataset = load_dataset(dataset_name, split='test')
                dataset.save_to_disk(local_dataset_path)
            except Exception as e:
                print(f"Error downloading dataset: {e}")
                return
        else:
            print(f"Loading dataset from local path: {local_dataset_path}")
            dataset = load_from_disk(local_dataset_path)

        slg_runner = SLG_Search(config=ts_config)
        results = []

        print("\n" + "="*80)
        print("Starting Evaluation")
        print("="*80)

        for idx, question in enumerate(dataset):
            print(f"\n--- Experiment {idx + 1} ---")
            
            problem_text = get_problem_text(question)
            raw_answer = get_problem_answer(question)
            
            true_answer = extract_ground_truth(str(raw_answer)) 
            
            prompt_text = create_prompt_template(problem_text, task=ts_config.lm_config.task)

            print(f"Generated Prompt:\n---\n{prompt_text}\n---")
            
            initial_state = State(prompt=prompt_text, steps=[])
            
 
            slg_runner.clean_tree()
            start_time = time.time()
            root = slg_runner.one_layer_expand(initial_state=initial_state)
            ts_time = time.time() - start_time
            
            print(f"SLG Time Taken: {ts_time:.2f} seconds")
            
            ts_pass_at_n = get_em_score_for_root(root, true_answer)
            ts_best_response = slg_runner.best_response
            ts_best_score = slg_runner.best_response_score
            
            ts_rm_at_1 = 0.0
            if ts_best_response:
                ts_model_ans = extract_model_answer(ts_best_response)
                if ts_model_ans == str(true_answer).strip():
                    ts_rm_at_1 = 1.0


            start_time_bon = time.time()
            bon_response_text, bon_best_score, _, bon_root = slg_runner.BoN_comparison(
                initial_state=initial_state, 
                Sampling_time=ts_config.N
            )
            bon_time = time.time() - start_time_bon
            print(f"BoN Time Taken: {bon_time:.2f} seconds")

            bon_pass_at_n = get_em_score_for_root(bon_root, true_answer)
            
            bon_rm_at_1 = 0.0
            if slg_runner.best_response:
                bon_model_ans = extract_model_answer(slg_runner.best_response)
                if bon_model_ans == str(true_answer).strip():
                    bon_rm_at_1 = 1.0

            result = {
                "experiment_index": idx + 1,
                "prompt": prompt_text,
                "true_answer": str(true_answer).strip() if true_answer else None,

                "ts_time": ts_time,
                "ts_best_response": ts_best_response.get_full_response() if ts_best_response else "N/A",
                "ts_best_score": ts_best_score,
                "ts_pass_at_n": float(ts_pass_at_n),
                "ts_rm_at_1": float(ts_rm_at_1),
                
                "bon_time": bon_time,
                "bon_best_response": bon_response_text,
                "bon_best_score": bon_best_score,
                "bon_pass_at_n": float(bon_pass_at_n),
                "bon_rm_at_1": float(bon_rm_at_1)
            }
            results.append(result)

            if (idx + 1) % 10 == 0:
                with open(output_file_path, 'w') as f:
                    json.dump({"experiments": results}, f, cls=DataclassJSONEncoder, indent=4)


        df_results = pd.DataFrame(results)
        
        summary = {
            "total_experiments": len(results),
            "average_ts_time": df_results['ts_time'].mean(),
            "average_ts_score": df_results['ts_best_score'].mean(),
            "average_ts_pass_at_n": df_results['ts_pass_at_n'].mean(),
            "average_ts_rm_at_1": df_results['ts_rm_at_1'].mean(),
            
            "average_bon_time": df_results['bon_time'].mean(),
            "average_bon_score": df_results['bon_best_score'].mean(),
            "average_bon_pass_at_n": df_results['bon_pass_at_n'].mean(),
            "average_bon_rm_at_1": df_results['bon_rm_at_1'].mean()
        }
        
        final_output = {
            "summary": summary,
            "experiments": results
        }
        
        with open(output_file_path, 'w') as f:
            json.dump(final_output, f, cls=DataclassJSONEncoder, indent=4)
            
        print(f"\nEvaluation Complete. Results saved to {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()