# core/mcts.py
import torch
from typing import List, Optional,  Tuple
from .node import Node
from core.data_structures import State
from core.model_wrapper import LLMWrapper, RMWrapper
from core.model_config import LMConfig, RMConfig, SLGConfig
import numpy as np
from collections import defaultdict
import time
from copy import deepcopy






class SLG_Search:
    
    
    def __del__(self):
        if hasattr(self, 'llm'):
            del self.llm
        if hasattr(self, 'rm'):
            del self.rm
        if hasattr(self, 'stats'):
            del self.stats
        if hasattr(self, 'best_response'):
            del self.best_response
        if hasattr(self, 'best_response_score'):
            del self.best_response_score

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def __init__(self, config: SLGConfig):
        self.llm = LLMWrapper(config.lm_config)
        self.rm = RMWrapper(config.rm_config)
        self.config = config
        self.stats = defaultdict(int)  
        self.best_response = None
        self.best_response_score = float('-inf')
    

    def clean_tree(self):
        """
        Clean up the tree to run new search.
        """
        self.best_response = None
        self.best_response_score = float('-inf')
        self.stats = defaultdict(int)
        self.llm.num_process = 0

    def roll_out_to_leaf(self, node: Node, depth: int, num_rollout: Optional[int] = None, num_expand: Optional[int] = None) -> List[State]: 

        """
        Roll out from the given node to leaf nodes. Roll out num_rollout responses and select top num_expand states.
        """

        if num_rollout is None:
            num_rollout = self.config.m - len(node.children)
            if num_rollout <= 0:
                return []
            
        
        if num_expand is None:
            num_expand = self.config.K

        root_node = deepcopy(node)

        
        response_state = self.llm.perform_n_rollouts(state=root_node.state, horizon=depth, n=num_rollout)
        if self.stats['rollouts'] is not None:
            self.stats['rollouts'] += num_rollout

        torch.cuda.empty_cache()
    

        rewards = self.rm.score_states_batch(response_state)

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

        
        node.response_list = top_states
        node.reward_list.extend(list(sorted_rewards))
        node.propogate_reward_list(list(sorted_rewards))
        

    def one_layer_expand(self, initial_state: State, num_expand: Optional[int] = None) -> Node:
        """
        One-layer expansion
        First expand K children, and use m rollouts for each child to evaluate the scaling law.
        Then we put all the remaining rollouts to the best child.
        """

        if self.config.verbose:
            print("\n" + "="*80)
            print("Starting One Layer Search")
            print("="*80)

        self.clean_tree()

        if num_expand is None:
            num_expand = self.config.K

        # Initialize root node
        root = Node(state=initial_state)
        
        self.stats['rollouts'] = 0
        

        # Expand the root
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

        self.roll_out_to_leaf(best_node, self.config.max_depth - 1, num_rollout=self.config.N - self.stats['rollouts'])

        print("Final Best Response Score:", self.best_response_score)
        print("Best Response:", self.best_response.get_full_response() if self.best_response else "N/A")

        return root
        


    
    def BoN_comparison(self, initial_state: State, Sampling_time: Optional[int] = None, keep_list: Optional[bool] = False) -> Tuple[str, float, list]:
        """
        Best-of-N baseline for comparison.
        
        Args:
            initial_state: The starting state (prompt with no steps)
        """
        if self.config.verbose:
            print("\n" + "="*80)
            print("Starting Best-of-N Baseline")
            print("="*80)
        
        # self.clean_tree()
        self.best_response = None
        self.best_response_score = float('-inf')
        self.stats = defaultdict(int)

        if keep_list:
            reward_list = []


        if Sampling_time is None:
            Sampling_time = self.config.N


        batch_size = min(500, Sampling_time)
        for _ in range(0, self.config.N, batch_size):
            # deal with the last batch which might be smaller than batch_size
            initial_state = deepcopy(initial_state)
            Num_to_sample = min(batch_size, self.config.N - _)
            node = Node(state=initial_state)
            self.roll_out_to_leaf(node, self.config.max_depth, num_rollout=Num_to_sample, num_expand=Num_to_sample)
            new_reward_list = node.reward_list
            if keep_list:
                reward_list.extend(new_reward_list)
            print(f"Current Best Score after {_ + batch_size} responses: {self.best_response_score}")


            
        

        # for start in tqdm.tqdm(range(0, self.config.N, batch_size)):
        #     end = min(start + batch_size, self.config.N)
        #     current_batch_size = end - start
        #     if self.config.verbose:
        #         print(f"Generating responses {start + 1} to {end}...")
        #     responses = self.llm.perform_n_rollouts(state=initial_state, horizon=self.config.max_depth, n=current_batch_size)
        #     if start == 0:
        #         all_responses = responses
        #     else:
        #         all_responses.extend(responses)
        #     scores = self.rm.score_states_batch(responses)
        #     for state, score in zip(responses, scores):
        #             if score > best_score:
        #                 best_score = score
        #                 best_response = state
        #     print(f"Current Best Score after {end} responses: {best_score}")
            
        if self.config.verbose:
            print("\n" + "="*80)
            print("Best-of-N Baseline Completed")
            print("="*80)
            print(f"Best-of-N Best Score: {self.best_response_score}")
            print("Best-of-N Best Response:")
            if self.best_response:
                print(self.best_response.get_full_response())
            else:
                print("No valid response found.")

        if not keep_list:
            reward_list = []

        return (self.best_response.get_full_response() if self.best_response else "N/A", self.best_response_score, reward_list)




    def tail_index_collector(self, initial_state: Optional[State] = None) -> Tuple[List[float], List[float], List[float], float, float, float]:


        print("\n" + "="*80)
        print("Starting Tail Index Collection")
        print("="*80)

        
        root = Node(state=initial_state)
        # Expand the root for one level with self.num_actions children
        self.roll_out_to_leaf(root, 1, num_rollout=self.config.num_actions)
        root.response_to_children()
        leaves = root.get_all_leaves()

        # for each leaf, we roll out K responses to estimate the tail index and scale parameter

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



         
        


