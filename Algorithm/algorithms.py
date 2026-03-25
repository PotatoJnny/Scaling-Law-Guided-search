import torch
from typing import List, Optional, Tuple
from .node import Node
from core.data_structures import State
from core.model_wrapper import LLMWrapper, RMWrapper
from core.model_config import LMConfig, RMConfig, SLGConfig
import numpy as np
from collections import defaultdict
import time
from copy import deepcopy
from core.tools import extract_model_answer

class SLG_Search:
    def __del__(self):
        if hasattr(self, 'llm'): del self.llm
        if hasattr(self, 'rm'): del self.rm
        if hasattr(self, 'stats'): del self.stats
        if hasattr(self, 'best_response'): del self.best_response
        if hasattr(self, 'best_response_score'): del self.best_response_score
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def __init__(self, config: SLGConfig):
        self.llm = LLMWrapper(config.lm_config)
        self.rm = RMWrapper(config.rm_config)
        self.config = config
        self.stats = defaultdict(int)  
        self.best_response = None
        self.best_response_score = float('-inf')
    
    def clean_tree(self):
        self.best_response = None
        self.best_response_score = float('-inf')
        self.stats = defaultdict(int)
        self.llm.num_process = 0

    def roll_out_to_leaf(self, node: Node, depth: int, num_rollout: Optional[int] = None, num_expand: Optional[int] = None) -> List[State]: 
        if num_rollout is None:
            num_rollout = self.config.m - len(node.children)
            if num_rollout <= 0:
                return []
            
        if num_expand is None:
            num_expand = self.config.K

        root_node = deepcopy(node)
        
        response_state = self.llm.perform_n_rollouts(state=root_node.state, horizon=depth, n=num_rollout)
        actual_rollouts = len(response_state)
        if self.stats['rollouts'] is not None:
            self.stats['rollouts'] += actual_rollouts

        rewards = []
        rm_chunk_size = 16 
        for i in range(0, len(response_state), rm_chunk_size):
            batch = response_state[i : i + rm_chunk_size]
            batch_rewards = self.rm.score_states_batch(batch)
            rewards.extend(batch_rewards)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        top_states = []
        sorted_rewards = []
        if rewards:
            sorted_pairs = sorted(zip(rewards, response_state), key=lambda pair: pair[0], reverse=True)
            sorted_rewards, sorted_states = zip(*sorted_pairs)
            if num_expand < len(sorted_states):
                top_states = list(sorted_states[:num_expand])
            else:
                top_states = list(sorted_states)
            if sorted_rewards[0] > self.best_response_score:
                self.best_response_score = sorted_rewards[0]
                self.best_response = sorted_states[0]

        extracted_answers = [extract_model_answer(state) for state in response_state]
        
        node.all_answers.extend(extracted_answers)
        node.response_list = top_states 
        node.reward_list.extend(list(sorted_rewards))
        node.propogate_reward_list(list(sorted_rewards))
        node.propogate_all_answers(extracted_answers)


    def one_layer_expand(self, initial_state: State, num_expand: Optional[int] = None) -> Node:
        if self.config.verbose:
            print("\n" + "="*80)
            print("Starting One Layer Search")
            print("="*80)

        self.clean_tree()
        if num_expand is None: num_expand = self.config.K

        root = Node(state=initial_state)
        self.stats['rollouts'] = 0
        
        self.roll_out_to_leaf(root, self.config.max_depth, num_expand=num_expand)
        total_resouces = self.config.N
        root.evaluate_value(total_resouces)
        root.response_to_children()
        root.print_tree()

        best_value = root.value
        best_node = root

        leaves = root.get_all_leaves()
        for leaf in leaves:
            self.roll_out_to_leaf(leaf, self.config.max_depth - 1, num_rollout=self.config.m)
            leaf.evaluate_value(total_resouces)
            if leaf.value > best_value:
                best_value = leaf.value
                best_node = leaf

        if self.config.verbose:
            print("\n" + "="*80)
            print("One Layer Search Completed")
            print("Root Value:", root.value)
            print("Best Child Value:", best_value)
            print("Current Best Response Score:", self.best_response_score)
            print("Left Resources:", self.config.N - self.stats['rollouts'])
            root.print_tree()

        while self.stats['rollouts'] < self.config.N:
            remaining_budget = self.config.N - self.stats['rollouts']
            pre_rollout_count = self.stats['rollouts']
            
            self.roll_out_to_leaf(best_node, self.config.max_depth - 1, num_rollout=remaining_budget)
            
            if self.stats['rollouts'] <= pre_rollout_count:
                print("Generation failed to produce new responses. Breaking to prevent infinite loop.")
                break

        print("Final Best Response Score:", self.best_response_score)
        print("Best Response:", self.best_response.get_full_response() if self.best_response else "N/A")

        return root
        

    def BoN_comparison(self, initial_state: State, Sampling_time: Optional[int] = None, keep_list: Optional[bool] = False) -> Tuple[str, float, list, Node]:
        if self.config.verbose:
            print("\n" + "="*80)
            print("Starting Best-of-N Baseline")
            print("="*80)
        
        self.best_response = None
        self.best_response_score = float('-inf')
        self.stats = defaultdict(int)

        reward_list = []
        if Sampling_time is None:
            Sampling_time = self.config.N

        root_node = Node(state=deepcopy(initial_state))
        
        chunk_size = Sampling_time 

        for i in range(0, Sampling_time, chunk_size):
            Num_to_sample = min(chunk_size, Sampling_time - i)
            
            self.roll_out_to_leaf(root_node, self.config.max_depth, num_rollout=Num_to_sample, num_expand=Num_to_sample)
            
            if keep_list:
                reward_list.extend(root_node.reward_list[-Num_to_sample:])
                
            print(f"BoN Progress: Generated {i + Num_to_sample}/{Sampling_time} | Current Best Score: {self.best_response_score}")

        if self.config.verbose:
            print("\n" + "="*80)
            print("Best-of-N Baseline Completed")
            print("="*80)

        if not keep_list:
            reward_list = []

        best_text = self.best_response.get_full_response() if self.best_response else "N/A"
        return (best_text, self.best_response_score, reward_list, root_node)


    def tail_index_collector(self, initial_state: Optional[State] = None) -> Tuple[List[float], List[float], List[float], float, float, float]:
        print("\n" + "="*80)
        print("Starting Tail Index Collection")
        print("="*80)
        
        root = Node(state=initial_state)
        self.roll_out_to_leaf(root, 1, num_rollout=self.config.num_actions)
        root.response_to_children()
        leaves = root.get_all_leaves()

        for leaf in leaves:
            self.roll_out_to_leaf(leaf, self.config.max_depth - 1, num_rollout=self.config.K)
            leaf.evaluate_quantile(self.config.K, self.config.N)

        leaves = root.get_all_leaves()
        return_vales = []
        tail_indexes = []
        sigma_values = []
        for leaf in leaves:
            return_vales.append(leaf.value)
            tail_indexes.append(leaf.shape_para)
            sigma_values.append(leaf.scale_para)

        root.evaluate_quantile(self.config.K, self.config.N)
        return return_vales, tail_indexes, sigma_values, root.value, root.shape_para, root.scale_para