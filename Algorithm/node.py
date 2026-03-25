from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
from core.data_structures import State, Action
from copy import deepcopy
import copy
import numpy as np
import scipy
from scipy.stats import genpareto
from scipy.optimize import minimize

@dataclass
class Node:
    """
    Represents a node in the MCTS tree. Each node corresponds to a State.
    """
    state: State
    parent: Optional['Node'] = None
    children: List['Node'] = field(default_factory=list)
    response_list: List['State'] = field(default_factory=list) 
    reward_list: List[float] = field(default_factory=list)
    value: Optional[float] = None  
    is_leaf: bool = True
    is_complete: bool = False  
    all_answers: List[str] = field(default_factory=list)  
    
    def  __copy__(self):
        return Node(
            state=self.state,
            reward_list=self.reward_list,
            value = self.value
        )

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def response_to_children(self):
        if self.response_list:
            self.is_leaf = False
        for state in self.response_list:
                state = deepcopy(state)
                if len(state.steps) < len(self.state.steps) + 1:
                    response = state.get_full_response()
                    print(f"Warning: Response shorter than expected when creating child nodes. Response: {response}")
                    continue
                state.steps = state.steps[: len(self.state.steps) + 1]
                state.is_complete = False
                new_child = Node(state=state, parent=self, response_list=[state])
                self.children.append(new_child)
        if len(self.children) == 0:
            self.is_complete = True

    def add_child(self, child_node: 'Node'):
        child_node.parent = self
        self.children.append(child_node)
        self.is_leaf = False
        
    def propogate_reward_list(self, reward_list: List[float]):
        current = self
        reward_list = reward_list
        while current.reward_list is not None and current.parent is not None:
            if current.parent.reward_list is not current.reward_list:
                current.parent.reward_list.extend(reward_list)
            current = current.parent

    def propogate_all_answers(self, answer_list: List[str]):
        current = self
        while current.all_answers is not None and current.parent is not None:
            if current.parent.all_answers is not current.all_answers:
                current.parent.all_answers.extend(answer_list)
            current = current.parent

    def evaluate_value(self, N: int):
        data = np.array(deepcopy(self.reward_list))
        data = np.array(data)

        tail_quantile_threshold = 20
        threshold = np.percentile(data, 100 - tail_quantile_threshold)
        truncated_data = data[data >= threshold]

        data_variance = np.var(truncated_data, ddof=1)
        data_mean = np.mean(truncated_data)
        normal_quantile = scipy.stats.norm.ppf(1 - tail_quantile_threshold / 100.0)
        hat_variance = data_variance / (1 + normal_quantile * scipy.stats.norm.pdf(normal_quantile) / (tail_quantile_threshold / 100.0) - (scipy.stats.norm.pdf(normal_quantile) / (tail_quantile_threshold / 100.0))**2)
        hat_std = np.sqrt(hat_variance)
        hat_mean = data_mean - hat_std * (scipy.stats.norm.pdf(normal_quantile) / (tail_quantile_threshold / 100.0))
        
        expected_max = hat_mean + hat_std * scipy.stats.norm.ppf(1 - 1/N)

        self.value = expected_max
        self.mean_para = hat_mean
        self.std_para = hat_std
        
    def assign_value(self, score: float):
        self.value = score
    
    def get_all_leaves(self) -> List['Node']:
        if self.is_leaf and not self.is_complete:
            return [self]
        
        leaves = []
        for child in self.children:
            leaves.extend(child.get_all_leaves())
        return leaves

    def remove_leaf(self):
        if self.parent:
            self.parent.children.remove(self)
            if not self.parent.children:
                self.parent.remove_leaf()
    
    def prune_leaves(self, K: int):
        leaves = self.get_all_leaves()
        if len(leaves) <= K:
            return
        leaves = [leaf for leaf in leaves if leaf.value is not None]
        if not leaves:
            return
        leaves.sort(key=lambda x: x.value, reverse=True)
        pruned_leaves = leaves[K:]
        for leaf in pruned_leaves:
            leaf.remove_leaf()

    def get_path_from_root(self) -> List['Node']:
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def get_depth(self) -> int:
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth
    
    def print_node_info(self, indent: int = 0):
        prefix = "  " * indent
        print(f"{prefix}Node (Depth {self.get_depth()}):")
        print(f"{prefix}  Is Leaf: {self.is_leaf}")
        print(f"{prefix}  Value: {self.value}")
        print(f"{prefix}  Children: {len(self.children)}")
        print(f"{prefix}  Response List: {len(self.response_list)}")
        print(f"{prefix}  Steps: {len(self.state.steps)}")
        print(f"{prefix}  Reward List: {len(self.reward_list)}")
        print(f"{prefix}  Best Reward: {max(self.reward_list) if self.reward_list else 'N/A'}")
        print(f"{prefix} Mean_para: {getattr(self, 'mean_para', 'N/A')}, Scale_para: {getattr(self, 'std_para', 'N/A')}")
        
    def print_tree(self, max_depth: Optional[int] = None, current_depth: int = 0):
        self.print_node_info(indent=current_depth)
        if max_depth is not None and current_depth >= max_depth:
            return
        for child in self.children:
            child.print_tree(max_depth=max_depth, current_depth=current_depth + 1)