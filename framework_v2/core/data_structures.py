from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import scipy.stats

@dataclass(frozen=True, slots=True)
class Action:
    """Represents one reasoning step."""
    step_text: str
    is_final: bool = False

@dataclass
class State:
    """Represents the current state of the generation process."""
    prompt: str
    steps: List[Action] = field(default_factory=list)
    is_complete: bool = False

    def get_full_text(self) -> str:
        return self.prompt + self.get_full_response()

    def get_full_response(self) -> str:
        return "".join([action.step_text for action in self.steps])

    def append_step(self, action: Action):
        self.steps.append(action)
        if action.is_final:
            self.is_complete = True

    def get_truncated_copy(self, target_length: int) -> 'State':
        truncated_steps = self.steps[:target_length]
        is_complete = any(a.is_final for a in truncated_steps)
        return State(
            prompt=self.prompt,
            steps=truncated_steps,
            is_complete=is_complete
        )


class Node:
    """Represents a node in the tree."""
    def __init__(self, state: State, parent: Optional['Node'] = None):
        self.state = state
        self.parent = parent
        self.children: List['Node'] = []
        
        # Tracking lists
        self.response_list: List[State] = [] 
        self.reward_list: List[float] = []
        self.all_answers: List[str] = []  
        
        self.value: Optional[float] = None
        self.is_leaf: bool = True
        self.is_complete: bool = state.is_complete

    def response_to_children(self):
        current_len = len(self.state.steps)

        for state in self.response_list:
            if len(state.steps) < current_len + 1:
                response = state.get_full_response()
                print(f"Warning: Response shorter than expected. Response: {response}")
                continue

            child_state = state.get_truncated_copy(target_length=current_len + 1)
            new_child = Node(state=child_state, parent=self)
            new_child.response_list = [child_state]
            self.children.append(new_child)

        if self.children:
            self.is_leaf = False
        else:
            self.is_complete = True

    def add_child(self, child_node: 'Node'):
        child_node.parent = self
        self.children.append(child_node)
        self.is_leaf = False

    def propagate_reward_list(self, reward_list: List[float]):
        current = self
        while current.parent is not None:
            if current.parent.reward_list is not current.reward_list:
                current.parent.reward_list.extend(reward_list)
            current = current.parent

    def propagate_all_answers(self, answer_list: List[str]):
        current = self
        while current.parent is not None:
            if current.parent.all_answers is not current.all_answers:
                current.parent.all_answers.extend(answer_list)
            current = current.parent

    def evaluate_value(self, N: int):
        if not self.reward_list:
            return
            
        data = np.array(self.reward_list)
        tail_quantile_threshold = 20
        threshold = np.percentile(data, 100 - tail_quantile_threshold)
        truncated_data = data[data >= threshold]

        if len(truncated_data) < 2 or np.var(truncated_data) == 0:
            self.value = float(np.max(data))
            return

        data_variance = np.var(truncated_data, ddof=1)
        data_mean = np.mean(truncated_data)
        
        normal_quantile = scipy.stats.norm.ppf(1 - tail_quantile_threshold / 100.0)
        pdf_val = scipy.stats.norm.pdf(normal_quantile)
        q_val = tail_quantile_threshold / 100.0
        
        variance_adj = 1 + normal_quantile * pdf_val / q_val - (pdf_val / q_val)**2
        hat_variance = max(0.0, data_variance / variance_adj)
        hat_std = np.sqrt(hat_variance)
        hat_mean = data_mean - hat_std * (pdf_val / q_val)
        
        expected_max = hat_mean + hat_std * scipy.stats.norm.ppf(1 - 1/N)

        self.value = float(expected_max)
        self.mean_para = float(hat_mean)
        self.std_para = float(hat_std)

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

    def get_path_from_root(self) -> List['Node']:
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    

    def get_depth(self) -> int:
        """Returns the depth of this node in the tree (root is 0)."""
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth

    def print_node_info(self, indent: int = 0):
        """Prints information about this node."""
        prefix = "  " * indent
        print(f"{prefix}Node (Depth {self.get_depth()}):")
        print(f"{prefix}  Is Leaf: {self.is_leaf}")
        print(f"{prefix}  Value: {self.value}")
        print(f"{prefix}  Children: {len(self.children)}")
        print(f"{prefix}  Steps: {len(self.state.steps)}")
        print(f"{prefix}  Reward List: {len(self.reward_list)}")
        print(f"{prefix}  Best Reward: {max(self.reward_list) if self.reward_list else 'N/A'}")
        
    def print_tree(self, max_depth: Optional[int] = None, current_depth: int = 0):
        """Recursively prints the entire tree structure."""
        self.print_node_info(indent=current_depth)
        
        if max_depth is not None and current_depth >= max_depth:
            return
        
        for child in self.children:
            child.print_tree(max_depth=max_depth, current_depth=current_depth + 1)