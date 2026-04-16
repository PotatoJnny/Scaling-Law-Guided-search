from typing import List

from core.data_structures import Node, State

from .slg_mcts import SLG_Search


class BranchingBoN(SLG_Search):
    """
    Budget-balanced branching baseline.

    It uses the same branch-discovery stage as SLG:
    1. root exploration with m samples
    2. keep K branches
    3. one exploration round for each branch

    The only difference is the final allocation rule:
    instead of sending the remaining budget to one selected branch,
    split the remaining rollouts as evenly as possible across all discovered branches.
    """

    def _even_split(self, total: int, count: int) -> List[int]:
        if count <= 0 or total <= 0:
            return [0] * max(count, 0)
        base = total // count
        remainder = total % count
        return [base + (1 if idx < remainder else 0) for idx in range(count)]

    def one_layer_expand(self, initial_state: State, num_expand=None) -> Node:
        root, leaves = self._prepare_one_layer_tree(initial_state, num_expand=num_expand)
        targets = leaves if leaves else [root]
        allocations = self._even_split(self._remaining_budget(), len(targets))

        for node, num_rollout in zip(targets, allocations):
            if num_rollout <= 0:
                continue
            self.roll_out_to_leaf(
                node,
                self._branch_depth_limit(node),
                num_rollout=num_rollout,
            )

        return root
