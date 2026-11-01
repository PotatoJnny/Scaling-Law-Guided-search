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

    def finish_from_prepared_tree(self, root: Node, leaves: List[Node]) -> Node:
        targets = leaves if leaves else [root]
        remaining_budget_before_finish = int(self._remaining_budget())
        allocations = self._even_split(remaining_budget_before_finish, len(targets))
        allocations_before_finish = [
            {
                **self._node_debug_payload(node, leaf_index=idx if leaves else None),
                "allocated_rollouts": int(num_rollout),
            }
            for idx, (node, num_rollout) in enumerate(zip(targets, allocations))
        ]

        for node, num_rollout in zip(targets, allocations):
            if num_rollout <= 0:
                continue
            self.roll_out_to_leaf(
                node,
                self._branch_depth_limit(node),
                num_rollout=num_rollout,
            )

        self.last_run_debug.update(
            {
                "variant": self.__class__.__name__,
                "selection": {
                    "metric_name": "even_split",
                    "remaining_budget_before_finish": remaining_budget_before_finish,
                    "num_targets": len(targets),
                    "total_assigned_rollouts": int(sum(allocations)),
                    "allocations_before_finish": allocations_before_finish,
                    "allocations_after_finish": [
                        {
                            **self._node_debug_payload(node, leaf_index=idx if leaves else None),
                            "allocated_rollouts": int(num_rollout),
                        }
                        for idx, (node, num_rollout) in enumerate(zip(targets, allocations))
                    ],
                    "final_total_rollouts": int(self.stats["rollouts"]),
                },
            }
        )
        return root

    def one_layer_expand(self, initial_state: State, num_expand=None) -> Node:
        root, leaves = self._prepare_one_layer_tree(initial_state, num_expand=num_expand)
        return self.finish_from_prepared_tree(root, leaves)
