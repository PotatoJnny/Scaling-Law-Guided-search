import argparse
import random
import time
import os
import datetime
import torch
import numpy as np

# Core Framework Imports
from core.llm_engine import LLMEngine
from core.reward_registry import create_reward_engine
from core.runtime import (
    build_task,
    create_algorithm_runner,
    load_task_dataset,
    prepare_experiment_config,
    resolve_task_setup,
)
from evaluation.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON experiment config file")
    parser.add_argument("--num_problems", type=int, default=None, help="Limit to N problems from start_index")
    parser.add_argument("--start_index", type=int, default=0, help="Dataset index to start from (for sharding)")
    args = parser.parse_args()

    cfg = prepare_experiment_config(args.config, args.start_index)

    seed = cfg.get("seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    print("="*80)
    print(f"🚀 LAUNCHING EXPERIMENT: {cfg['experiment_name']}")
    print("="*80)

    print("\nLoading Engines...")
    hw = cfg["hardware"]
    task_cfg = cfg["task_setup"]
    dataset_config, action_strategy, task_type, reward_type = resolve_task_setup(task_cfg)
    algorithm_name = cfg.get("algorithm", "slg")

    evaluator_config = dict(cfg)
    evaluator_config["task_type"] = task_type
    evaluator = Evaluator(experiment_name=cfg['experiment_name'], config=evaluator_config)

    llm = LLMEngine(
        model_name=hw["lm_name"],
        tensor_parallel_size=hw.get("lm_tensor_parallel_size", 1),
        is_reasoning_model=hw.get("is_reasoning_model", False),
        gpu_memory_utilization=hw.get("gpu_memory_utilization", 0.6),
        max_model_len=hw.get("max_model_len", 8192),
        enforce_eager=hw.get("enforce_eager", False),
        attention_backend=hw.get("attention_backend"),
        max_num_batched_tokens=hw.get("max_num_batched_tokens"),
        max_batch_size=hw.get("lm_max_batch_size", 16),
    )

    rm = create_reward_engine(reward_type, hw, dataset_config)
    task = build_task(task_type, dataset_config, action_strategy, tokenizer=llm.tokenizer)
    runner = create_algorithm_runner(algorithm_name, cfg, llm, rm, task)


    cache_root = os.path.join("data", "local_cache")
    os.makedirs(cache_root, exist_ok=True)
    dataset = load_task_dataset(task_cfg, dataset_config, cache_root)
    dataset, runtime_context = task.prepare_dataset(dataset, cache_root)

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
        
        question_data = task.prepare_problem_data(
            question_data,
            runtime_context={
                **runtime_context,
                "llm_engine": llm,
                "max_model_len": hw.get("max_model_len", 8192),
                "generation_max_new_tokens": (
                    cfg.get("bon_params", {}).get("max_tokens")
                    or cfg.get("slg_params", {}).get("max_tokens")
                    or cfg.get("bbon_params", {}).get("max_tokens")
                    or 1024
                ),
            },
            reward_source=rm,
        )

        prompt_text = task.get_prompt(question_data)

        # Run Search
        runner.clean()
        start_time = time.time()
        result = runner.run(question_data)
        search_time = time.time() - start_time

        core_result_fields = task.build_core_result_fields(question_data, result, evaluator)
        task_metrics = task.build_result_metrics(question_data, result, reward_source=rm)
        search_artifacts = task.serialize_search_artifacts(result)

        # Package and Record Results (using algorithm-agnostic keys)
        result_dict = {
            "experiment_index": idx + 1,
            "prompt": prompt_text,
            "true_answer": (
                str(core_result_fields.get("true_answer")).strip()
                if core_result_fields.get("true_answer") is not None
                else None
            ),

            "search_time": search_time,
            "best_response": result["full_text"],
            "best_score": result["best_score"],
            "total_rollouts": result["total_rollouts"],

            "pass_at_1": core_result_fields["pass_at_1"],
            "pass_at_all": core_result_fields["pass_at_all"],
            "majority_vote": core_result_fields["majority_vote"],
            "all_answers": core_result_fields["all_answers"],
        }
        for metadata_key in [
            "problem_id",
            "source_index",
            "instruction",
            "messages",
            "uid",
            "category",
            "subcategory",
            "session_id",
            "conversation_input",
            "primary_tag",
            "checklist",
            "key",
            "instruction_id_list",
            "kwargs",
            "dataset",
        ]:
            if metadata_key in question_data:
                result_dict[metadata_key] = question_data[metadata_key]
        result_dict["best_response_num_chars"] = len(result["full_text"])
        result_dict["best_response_num_tokens"] = len(llm.tokenizer.encode(result["full_text"]))
        result_dict.update(search_artifacts)
        result_dict.update(task_metrics)
        
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
