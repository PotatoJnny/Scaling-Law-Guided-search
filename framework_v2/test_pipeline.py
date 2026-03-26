from core.llm_engine import LLMEngine
from core.rm_engine import RMEngine
from tasks.math_task import MathTask
from tasks.task_configs import DATASET_CONFIGS, ACTION_STRATEGIES
from algorithms.slg_mcts import SLG_Search

# 1. A tiny config just for this test so we don't wait hours
class TinyConfig:
    def __init__(self):
        self.N = 8           # Total budget of only 8 rollouts
        self.K = 2           # Expand 2 nodes
        self.m = 2           # 2 rollouts per node for scaling estimation
        self.max_depth = 3   # Don't go deeper than 3 steps
        self.verbose = True

def run_test():
    print("="*60)
    print("🧪 STARTING E2E PIPELINE TEST")
    print("="*60)

    # 2. Boot the engines (we only pay this loading time once!)
    print("\n[1/2] Loading Engines...")
    llm = LLMEngine(model_name="meta-llama/Llama-3.2-1B-Instruct", tensor_parallel_size=1)
    rm = RMEngine(model_name="Skywork/Skywork-Reward-V2-Llama-3.1-8B", max_batch_size=16)

    # 3. Hardcode a single, simple problem
    problem = {
        "question": "If I have 3 apples and buy 4 more, then eat 2, how many do I have left?", 
        "answer": "#### 5"
    }
    
    # 4. Define the strategies we want to compare
    strategies_to_test = ["double_newline", "strict_single_step", "fixed_tokens"]
    
    for strategy in strategies_to_test:
        print("\n" + "!"*60)
        print(f"🚀 TESTING STRATEGY: {strategy.upper()}")
        print("!"*60)
        
        # Initialize the Task with the current strategy
        task = MathTask(
            dataset_config=DATASET_CONFIGS["gsm8k"],
            action_strategy=ACTION_STRATEGIES[strategy],
            tokenizer=llm.tokenizer
        )
        
        # Initialize the algorithm
        algo = SLG_Search(
            llm_engine=llm, 
            rm_engine=rm, 
            task=task, 
            config=TinyConfig()
        )
        
        # ======================================================
        # THE DEEP DIVE: Unpacking the search to expose the tree
        # ======================================================
        from core.data_structures import State
        
        # 1. Format the prompt
        prompt_text = task.get_prompt(problem)
        initial_state = State(prompt=prompt_text)
        
        # 2. Run the search and intercept the root node!
        root_node = algo.one_layer_expand(initial_state)
        
        # 3. Print the raw tree structure
        print("\n" + "*"*60)
        print(f"🌲 FULL MCTS TREE FOR: {strategy.upper()}")
        print("*"*60)
        root_node.print_tree()
        
        # 4. Print every single answer it found across all rollouts
        print("\n" + "*"*60)
        print(f"🔍 ALL EXTRACTED ANSWERS FOUND DURING SEARCH:")
        print(root_node.all_answers)
        print("*"*60)

        # 5. Manually extract the best response
        best_text = algo.best_response.get_full_response() if algo.best_response else ""
        best_answer = task.extract_answer(best_text)
        
        # Print the final summary
        print(f"\n✅ [FINAL SUMMARY FOR {strategy.upper()}]")
        print(f"Predicted Answer: {best_answer}")
        print(f"Best RM Score:    {algo.best_response_score:.4f}")
        print(f"Total Rollouts:   {algo.stats['rollouts']}")
        print(f"Best Trajectory:\n{'-'*30}\n{best_text}\n{'-'*30}")

if __name__ == '__main__':
    run_test()