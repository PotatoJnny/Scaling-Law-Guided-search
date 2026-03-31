import argparse
import json
import time
import os
import datetime
import numpy as np
import torch
from datasets import load_dataset, load_from_disk

# Core Framework Imports
from core.llm_engine import LLMEngine
from core.rm_engine import RMEngine
from core.code_executor import CodeExecutor
from tasks.math_task import MathTask
from tasks.language_task import LanguageTask
from tasks.code_task import CodeTask
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
    auto_strategy = slg_params.get("auto_param_strategy", "legacy")

    if m == -1:
        if auto_strategy == "safe_budget":
            min_m = slg_params.get("min_m", 30)
            m = max(min_m, N // 5)
            if m % 5 != 0:
                m = int(round(m / 5) * 5)
        else:
            calculated_m = (np.log(N) ** 3) / 5
            m = max(20, int(round(calculated_m / 5) * 5))
        slg_params["m"] = m
        print(f"[Auto-Config] m calculated from N={N} with strategy={auto_strategy}: {m}")
        
    if K == -1:
        if auto_strategy == "safe_budget":
            K = slg_params.get("default_K", 2)
            max_explore_fraction = slg_params.get("max_explore_fraction", 0.6)
            max_explore_rollouts = max(K, int(N * max_explore_fraction))
            if K * m > max_explore_rollouts:
                adjusted_m = max(1, max_explore_rollouts // K)
                if adjusted_m != m:
                    m = adjusted_m
                    slg_params["m"] = m
                    print(f"[Auto-Config] Adjusted m to satisfy K*m <= {max_explore_fraction:.2f}N: {m}")
        else:
            calculated_K = int(round(N / (2 * m)))
            max_budget = int(N / (m + 2))
            K = max(2, min(calculated_K, max_budget))
        slg_params["K"] = K
        print(f"[Auto-Config] K calculated from N={N}, m={m} with strategy={auto_strategy}: {K}")
        
    return slg_params

# We create a simple object to pass the parsed JSON params into your algorithm cleanly
class ConfigWrapper:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON experiment config file")
    parser.add_argument("--num_problems", type=int, default=None, help="Limit to N problems from start_index")
    parser.add_argument("--start_index", type=int, default=0, help="Dataset index to start from (for sharding)")
    args = parser.parse_args()

    # 1. Load the JSON Config
    with open(args.config, 'r') as f:
        cfg = json.load(f)

    # Append shard suffix when sharding so each shard saves to its own folder
    if args.start_index > 0:
        cfg['experiment_name'] = f"{cfg['experiment_name']}_shard{args.start_index}"

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
    task_cfg = cfg["task_setup"]
    dataset_config = DATASET_CONFIGS[task_cfg["dataset_name"]]
    action_strategy = ACTION_STRATEGIES[task_cfg["action_strategy"]]
    task_type = dataset_config.get("task_type", "math")
    is_language_task = task_type == "language"
    is_code_task = task_type == "code"

    llm = LLMEngine(
        model_name=hw["lm_name"],
        tensor_parallel_size=hw.get("lm_tensor_parallel_size", 1),
        gpu_memory_utilization=hw.get("gpu_memory_utilization", 0.6),
        max_model_len=hw.get("max_model_len", 8192)
    )

    # Code tasks use CodeExecutor (actual execution) instead of RMEngine
    if is_code_task:
        rm = CodeExecutor(
            timeout_secs=hw.get("code_timeout_secs", 10.0),
            n_timing_runs=hw.get("code_timing_runs", 3),
        )
    else:
        rm = RMEngine(
            model_name=hw["rm_name"],
            quantization=hw.get("rm_quantization", False),
            max_batch_size=hw["rm_max_batch_size"]
        )

    if is_code_task:
        task = CodeTask(
            dataset_config=dataset_config,
            action_strategy=action_strategy,
            tokenizer=llm.tokenizer
        )
    elif is_language_task:
        task = LanguageTask(
            dataset_config=dataset_config,
            action_strategy=action_strategy,
            tokenizer=llm.tokenizer
        )
    else:
        task = MathTask(
            dataset_config=dataset_config,
            action_strategy=action_strategy,
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
    dataset_cfg = DATASET_CONFIGS[task_cfg["dataset_name"]]
    hf_name = dataset_cfg.get("hf_name", task_cfg["dataset_name"])  # use hf_name if set, else dataset_name
    safe_name = task_cfg["dataset_name"].replace("/", "_")
    ds_config = task_cfg.get('dataset_config') or None
    config_suffix = ds_config if ds_config else "default"
    local_path = f"./local_{safe_name}_{config_suffix}_data"

    if not os.path.exists(local_path):
        print(f"Downloading {hf_name}...")
        split = dataset_cfg.get("split", "test")
        if ds_config:
            dataset = load_dataset(hf_name, ds_config, split=split)
        else:
            dataset = load_dataset(hf_name, split=split)
        dataset.save_to_disk(local_path)
    else:
        dataset = load_from_disk(local_path)

    # 4a. For code tasks with filter_to_testcases: keep only rows whose problem_id
    # is in the testcases cache, then deduplicate to one row per problem_id.
    if dataset_cfg.get("filter_to_testcases") and is_code_task:
        tc_json = f"./local_{safe_name}_testcases.json"
        if os.path.exists(tc_json):
            with open(tc_json) as _f:
                _avail_pids = set(json.load(_f).keys())
            _seen_pids = set()
            _keep_indices = []
            for _i, _row in enumerate(dataset):
                _pid = str(_row.get("problem_id", ""))
                if _pid in _avail_pids and _pid not in _seen_pids:
                    _keep_indices.append(_i)
                    _seen_pids.add(_pid)
            dataset = dataset.select(_keep_indices)
            print(f"Filtered pie-perf to {len(dataset)} unique problems with test cases.")

    # 4b. For code tasks: load test cases from the raw file-based HF cache.
    # pie-perf-testcases stores each test case as individual files:
    #   public_test_cases/p{ID}/input.{N}.txt  and  output.{N}.txt
    pie_testcases: dict = {}   # problem_id → {"test_inputs": [...], "test_outputs": [...]}
    if is_code_task:
        tc_hf = dataset_config.get("testcases_hf_name")
        if tc_hf:
            tc_json = f"./local_{safe_name}_testcases.json"
            if os.path.exists(tc_json):
                with open(tc_json) as _f:
                    pie_testcases = json.load(_f)
                print(f"Loaded test cases for {len(pie_testcases)} problems from {tc_json}.")
            else:
                # Find the downloaded snapshot directory in the HF cache
                hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                repo_dir_name = tc_hf.replace("/", "--")
                cache_repo = os.path.join(hf_home, "hub", f"datasets--{repo_dir_name}")
                snapshots_dir = os.path.join(cache_repo, "snapshots")
                snapshot = None
                if os.path.isdir(snapshots_dir):
                    snaps = sorted(os.listdir(snapshots_dir))
                    if snaps:
                        snapshot = os.path.join(snapshots_dir, snaps[-1], "public_test_cases")
                if snapshot and os.path.isdir(snapshot):
                    print(f"Reading test cases from HF cache: {snapshot}")
                    for prob_dir in sorted(os.listdir(snapshot)):
                        prob_path = os.path.join(snapshot, prob_dir)
                        if not os.path.isdir(prob_path):
                            continue
                        files = os.listdir(prob_path)
                        idxs = sorted(set(
                            f.split(".")[1] for f in files if f.startswith("input.")
                        ))
                        inputs, outputs = [], []
                        for idx in idxs:
                            inp_f = os.path.join(prob_path, f"input.{idx}.txt")
                            out_f = os.path.join(prob_path, f"output.{idx}.txt")
                            if os.path.exists(inp_f) and os.path.exists(out_f):
                                with open(inp_f) as fi, open(out_f) as fo:
                                    inputs.append(fi.read())
                                    outputs.append(fo.read())
                        if inputs:
                            pie_testcases[prob_dir] = {"test_inputs": inputs, "test_outputs": outputs}
                    print(f"Loaded test cases for {len(pie_testcases)} problems.")
                    with open(tc_json, "w") as _f:
                        json.dump(pie_testcases, _f)
                    print(f"Saved to {tc_json}")
                else:
                    print(f"WARNING: No test case cache found. Download {tc_hf} first by running main.py once with network access.")
                    print(f"  Expected path: {snapshot}")

    # 5. The Execution Loop
    start_idx = args.start_index
    total_problems = min(args.num_problems, len(dataset) - start_idx) if args.num_problems else (len(dataset) - start_idx)
    experiment_start = time.time()

    for idx, question_data in enumerate(dataset):
        if idx < start_idx:
            continue
        if args.num_problems is not None and (idx - start_idx) >= args.num_problems:
            break

        local_idx = idx - start_idx
        elapsed = time.time() - experiment_start
        if local_idx > 0:
            avg_sec = elapsed / local_idx
            eta = datetime.timedelta(seconds=int(avg_sec * (total_problems - local_idx)))
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
            print(f"\n--- Problem {local_idx + 1}/{total_problems} (global #{idx + 1}) | Elapsed: {elapsed_str} | ETA: {eta} ---")
        else:
            print(f"\n--- Problem {local_idx + 1}/{total_problems} (global #{idx + 1}) ---")
        
        # For code tasks: attach test cases and set the executor's current problem
        if is_code_task:
            pid = question_data.get("problem_id", "")
            # testcases dir uses "p00001" format; main dataset may use int or string
            if pid not in pie_testcases:
                try:
                    pid = f"p{int(pid):05d}"
                except (ValueError, TypeError):
                    pass
            tc = pie_testcases.get(pid, {"test_inputs": [], "test_outputs": []})
            question_data = dict(question_data)
            question_data["original_code"] = question_data[dataset_config["question_column"]]
            question_data["test_inputs"] = tc["test_inputs"]
            question_data["test_outputs"] = tc["test_outputs"]
            question_data["measured_runtime_v0"] = question_data.get("measured_runtime_v0") or 1.0
            rm.set_problem(question_data)

        prompt_text = task.get_prompt(question_data)
        # Language/code tasks have no ground-truth answer string
        if is_language_task or is_code_task:
            true_answer = None
        elif task.dataset_config.get("raw_answer_column"):
            true_answer = str(question_data[task.dataset_config["answer_column"]]).strip()
        else:
            true_answer = task.extract_answer(question_data[task.dataset_config["answer_column"]])

        # Run Search
        runner.clean()
        start_time = time.time()
        result = runner.run(question_data)
        search_time = time.time() - start_time

        # Calculate explicit metrics using the Evaluator
        if is_language_task or is_code_task:
            pass_at_1 = pass_at_all = majority_vote = None
            all_answers = []
        else:
            all_answers = result["all_answers"]
            pass_at_all = evaluator.get_pass_at_all(all_answers, true_answer)
            pass_at_1 = evaluator.get_pass_at_1(result["predicted_answer"], true_answer)
            majority_vote = evaluator.get_majority_vote(all_answers, true_answer)

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
            "pass_at_all": pass_at_all,
            "majority_vote": majority_vote,
            "all_answers": all_answers,
            # For language/code tasks skip storing full response text per sample
            # (responses can be thousands of chars; only scores are needed)
            "all_scores_and_answers": [
                {"score": x["score"]}
                for x in result.get("all_scores_and_answers", [])
            ] if (is_language_task or is_code_task) else result.get("all_scores_and_answers", []),
        }
        
        evaluator.record_experiment(result_dict)

    # 6. Optional LLM Judge pass (runs after all search to reuse VRAM)
    if hw.get("judge_name") and evaluator.results:
        print("\n" + "="*80)
        print("Starting LLM Judge evaluation...")
        print("="*80)

        # Free search engines before loading the judge
        del llm, rm
        torch.cuda.empty_cache()

        from evaluation.llm_judge import LLMJudge
        judge = LLMJudge(
            model_name=hw["judge_name"],
            tensor_parallel_size=hw.get("judge_tensor_parallel_size", 1),
            gpu_memory_utilization=hw.get("judge_gpu_memory_utilization", 0.8),
            max_model_len=hw.get("judge_max_model_len", 8192),
            max_tokens=hw.get("judge_max_tokens", 256),
        )

        instructions = [r["prompt"] for r in evaluator.results]
        responses = [r["best_response"] for r in evaluator.results]
        judge_scores = judge.score_batch(instructions, responses)

        for result_entry, score in zip(evaluator.results, judge_scores):
            result_entry["judge_score"] = score

        del judge
        torch.cuda.empty_cache()
        print(f"✅ Judge scoring complete. Scored {len(judge_scores)} responses.")

    # 7. Finalize and Save Comprehensive Report
    evaluator.generate_final_report()

if __name__ == '__main__':
    main()
