import torch
from typing import List, Optional, Tuple
from collections import defaultdict

from .base_algo import BaseAlgorithm
from core.data_structures import State, Node, Action

class SLG_Search(BaseAlgorithm):
    def __init__(self, llm_engine, rm_engine, task, config):
        super().__init__(llm_engine, rm_engine, task, config)
        self.stats = defaultdict(int)  
        self.best_response = None
        self.best_response_score = float('-inf')

    def clean_tree(self):
        self.best_response = None
        self.best_response_score = float('-inf')
        self.stats = defaultdict(int)

    def roll_out_to_leaf(self, node: Node, depth: int, num_rollout: Optional[int] = None, num_expand: Optional[int] = None) -> List[State]: 
        if num_rollout is None:
            num_rollout = self.config.m - len(node.children)
            if num_rollout <= 0:
                return []
            
        if num_expand is None:
            num_expand = self.config.K

        prompt_text = node.state.get_full_text()
        
        # vLLM returns a list of lists. We pass 1 prompt, so we grab index [0]
        raw_strings = self.llm_engine.generate(
            prompts=[prompt_text], 
            n=num_rollout,
            max_tokens=2048 # Adjust based on your needs
        )[0]
        
        actual_rollouts = len(raw_strings)
        self.stats['rollouts'] += actual_rollouts

        # 2. Convert raw strings into our optimized State/Action lists
        response_states = []
        for raw_text in raw_strings:
            # Create a clean copy of the current state's history
            new_state = node.state.get_truncated_copy(len(node.state.steps))
            
            # Chunk the vLLM string into distinct steps (assuming \n\n separates steps)
            # You can change this delimiter if your dataset formats steps differently
            chunks = [s for s in raw_text.split("\n\n") if s.strip()]
            
            for i, chunk in enumerate(chunks):
                is_last = (i == len(chunks) - 1)
                new_state.append_step(Action(step_text=chunk + "\n\n", is_final=is_last))
                
            response_states.append(new_state)

        # 3. Score the complete rollouts (passing the task config instruction!)
        rm_instruction = getattr(self.task, "config", {}).get("rm_instruction", None)
        rewards = self.rm_engine.score_states_batch(response_states, rm_instruction=rm_instruction)

        # 4. Filter and select the Top-K
        top_states = []
        sorted_rewards = []
        if rewards:
            sorted_pairs = sorted(zip(rewards, response_states), key=lambda pair: pair[0], reverse=True)
            sorted_rewards, sorted_states = zip(*sorted_pairs)
            
            top_states = list(sorted_states[:num_expand])
            
            if sorted_rewards[0] > self.best_response_score:
                self.best_response_score = sorted_rewards[0]
                self.best_response = sorted_states[0]

        # 5. Extract final math answers using the Config-Driven Task
        extracted_answers = [self.task.extract_answer(state.get_full_response()) for state in response_states]
        
        # 6. Propagate up the tree
        node.all_answers.extend(extracted_answers)
        node.response_list = top_states 
        node.reward_list.extend(list(sorted_rewards))
        node.propagate_reward_list(list(sorted_rewards))
        node.propagate_all_answers(extracted_answers)
        
        return top_states

    def one_layer_expand(self, initial_state: State, num_expand: Optional[int] = None) -> Node:
        if self.config.verbose:
            print("\n" + "="*80)
            print("Starting One Layer Search")
            print("="*80)

        self.clean_tree()
        if num_expand is None: 
            num_expand = self.config.K

        root = Node(state=initial_state)
        self.stats['rollouts'] = 0
        
        # Expand Root
        self.roll_out_to_leaf(root, self.config.max_depth, num_expand=num_expand)
        total_resources = self.config.N
        root.evaluate_value(total_resources)
        root.response_to_children()

        best_value = root.value
        best_node = root

        # Expand Leaves
        leaves = root.get_all_leaves()
        for leaf in leaves:
            self.roll_out_to_leaf(leaf, self.config.max_depth - 1, num_rollout=self.config.m)
            leaf.evaluate_value(total_resources)
            if leaf.value > best_value:
                best_value = leaf.value
                best_node = leaf

        # Exhaust remaining budget on the best node
        while self.stats['rollouts'] < self.config.N:
            remaining_budget = self.config.N - self.stats['rollouts']
            pre_rollout_count = self.stats['rollouts']
            
            self.roll_out_to_leaf(best_node, self.config.max_depth - 1, num_rollout=remaining_budget)
            
            if self.stats['rollouts'] <= pre_rollout_count:
                print("Generation failed to produce new responses. Breaking to prevent infinite loop.")
                break

        if self.config.verbose:
            print("Final Best Response Score:", self.best_response_score)
            
        return root

    def run(self, problem_data: dict) -> dict:
        """
        The standardized entry point. Translates the raw dataset dict into the MCTS search.
        """
        # 1. Use the Task config to create the perfect prompt
        prompt_text = self.task.get_prompt(problem_data)
        initial_state = State(prompt=prompt_text)
        
        # 2. Run the SLG Math Search
        final_tree_root = self.one_layer_expand(initial_state)
        
        # 3. Extract final answer from the absolute best response found
        final_text = self.best_response.get_full_response() if self.best_response else ""
        final_answer = self.task.extract_answer(final_text)
        
        # 4. Return standard results
        return {
            "predicted_answer": final_answer,
            "best_score": self.best_response_score,
            "full_text": final_text,
            "total_rollouts": self.stats['rollouts']
        }