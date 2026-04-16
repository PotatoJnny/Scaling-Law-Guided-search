from collections import defaultdict
from math import inf, log, sqrt
from typing import List, Optional

from .base_algo import BaseAlgorithm
from core.data_structures import Action, Node, State


class SLGTreeSearch(BaseAlgorithm):
    """
    A tree-structured variant of SLG.

    Relative to the current one-shot SLG implementation, this version repeatedly:
    1. selects a frontier leaf from the current tree,
    2. expands that leaf with fresh rollouts,
    3. converts top continuations into child nodes,
    4. continues until the rollout budget is exhausted.

    The value model is kept identical to SLG (tail-based V_N estimation); the main
    change is the search policy over a growing tree.
    """

    def __init__(self, llm_engine, rm_engine, task, config):
        super().__init__(llm_engine, rm_engine, task, config)
        self.stats = defaultdict(int)
        self.best_response = None
        self.best_response_score = float("-inf")

    def clean_tree(self):
        self.best_response = None
        self.best_response_score = float("-inf")
        self.stats = defaultdict(int)

    def clean(self):
        self.clean_tree()

    def _get_stop_seqs_for_state(self, state: State) -> list:
        think_end = self.task.action_strategy.get("think_end_token", None)
        if think_end is not None and think_end in state.get_full_response():
            return self.task.action_strategy.get(
                "completion_stop_sequences",
                self.task.action_strategy.get("stop_sequences", []),
            )
        return self.task.action_strategy.get("stop_sequences", [])

    def _in_completion_phase(self, state: State) -> bool:
        think_end = self.task.action_strategy.get("think_end_token", None)
        return think_end is not None and think_end in state.get_full_response()

    def _fix_completion_actions(self, actions: List[Action]) -> List[Action]:
        think_end = self.task.action_strategy.get("think_end_token", "</think>")
        fixed = []
        for action in actions:
            text = action.step_text
            if not action.is_final and text.endswith(think_end):
                text = text[:-len(think_end)].rstrip()
            fixed.append(Action(step_text=text, is_final=True))
        return fixed

    def _is_think_block(self) -> bool:
        return self.task.action_strategy.get("chunking_method") == "think_block"

    def _ensure_tree_stats(self, node: Node) -> None:
        if not hasattr(node, "visit_count"):
            node.visit_count = 0
        if not hasattr(node, "expand_count"):
            node.expand_count = 0

    def _mark_path_visit(self, node: Node) -> None:
        current = node
        while current is not None:
            self._ensure_tree_stats(current)
            current.visit_count += 1
            current = current.parent

    def _expand_root_with_full_completions(self, root: Node, num_expand: int) -> None:
        prompt_text = root.state.get_full_text()
        stop_seqs = self.task.action_strategy.get(
            "completion_stop_sequences",
            self.task.action_strategy.get("stop_sequences", []),
        )

        raw_strings = self.llm_engine.generate(
            prompts=[prompt_text],
            n=self.config.m,
            max_tokens=getattr(self.config, "max_tokens", 2048),
            stop_sequences=stop_seqs,
            temperature=getattr(self.config, "temperature", 1.0),
            top_p=getattr(self.config, "top_p", 0.95),
        )[0]

        self.stats["rollouts"] += len(raw_strings)

        response_states = []
        for raw_text in raw_strings:
            new_state = root.state.get_truncated_copy(len(root.state.steps))
            actions = self.task.parse_response_to_actions(raw_text)
            for action in actions:
                new_state.append_step(action)
            response_states.append(new_state)

        if not response_states:
            root.is_complete = True
            return

        rewards = self.rm_engine.score_states_batch(response_states, **self._get_rm_kwargs())
        sorted_pairs = sorted(zip(rewards, response_states), key=lambda pair: pair[0], reverse=True)
        sorted_rewards, sorted_states = zip(*sorted_pairs)

        root.response_list = list(sorted_states[:num_expand])
        root.reward_list.extend(list(sorted_rewards))
        root.all_answers.extend(
            [self.task.extract_answer(state.get_full_response()) for state in response_states]
        )

        if sorted_rewards[0] > self.best_response_score:
            self.best_response_score = sorted_rewards[0]
            self.best_response = sorted_states[0]

    def roll_out_to_leaf(
        self,
        node: Node,
        depth: int,
        num_rollout: Optional[int] = None,
        num_expand: Optional[int] = None,
    ) -> List[State]:
        if depth <= 0:
            node.is_complete = True
            return []

        if num_rollout is None:
            num_rollout = max(0, self.config.m - len(node.children))
            if num_rollout <= 0:
                return []

        if num_expand is None:
            num_expand = self.config.K

        prompt_text = node.state.get_full_text()
        stop_seqs = self._get_stop_seqs_for_state(node.state)
        in_completion = self._in_completion_phase(node.state)

        raw_strings = self.llm_engine.generate(
            prompts=[prompt_text],
            n=num_rollout,
            max_tokens=getattr(self.config, "max_tokens", 2048),
            stop_sequences=stop_seqs,
            temperature=getattr(self.config, "temperature", 1.0),
            top_p=getattr(self.config, "top_p", 0.95),
        )[0]

        actual_rollouts = len(raw_strings)
        self.stats["rollouts"] += actual_rollouts
        self._ensure_tree_stats(node)
        node.expand_count += 1

        response_states = []
        for raw_text in raw_strings:
            new_state = node.state.get_truncated_copy(len(node.state.steps))
            actions = self.task.parse_response_to_actions(raw_text)
            if in_completion:
                actions = self._fix_completion_actions(actions)
            for action in actions:
                new_state.append_step(action)
            response_states.append(new_state)

        if not response_states:
            node.is_complete = True
            return []

        rewards = self.rm_engine.score_states_batch(response_states, **self._get_rm_kwargs())
        sorted_pairs = sorted(zip(rewards, response_states), key=lambda pair: pair[0], reverse=True)
        sorted_rewards, sorted_states = zip(*sorted_pairs)

        top_states = list(sorted_states[:num_expand])
        node.response_list = top_states

        node.reward_list.extend(list(sorted_rewards))
        node.propagate_reward_list(list(sorted_rewards))

        extracted_answers = [self.task.extract_answer(state.get_full_response()) for state in response_states]
        node.all_answers.extend(extracted_answers)
        node.propagate_all_answers(extracted_answers)

        if sorted_rewards[0] > self.best_response_score:
            self.best_response_score = sorted_rewards[0]
            self.best_response = sorted_states[0]

        return top_states

    def _bootstrap_root_children(self, root: Node) -> None:
        children = list(root.children)
        if not children:
            return

        for leaf in children:
            if self.stats["rollouts"] >= self.config.N:
                break
            remaining = self.config.N - self.stats["rollouts"]
            num_rollout = min(getattr(self.config, "tree_expand_batch", self.config.m), remaining)
            if num_rollout <= 0:
                break
            self.roll_out_to_leaf(
                leaf,
                self.config.max_depth - leaf.get_depth(),
                num_rollout=num_rollout,
                num_expand=self.config.K,
            )
            leaf.evaluate_value(self.config.N, tail_fraction=getattr(self.config, "tail_fraction", 20))
            if leaf.response_list:
                self._append_children_from_responses(leaf)
            else:
                leaf.is_complete = True

    def _append_children_from_responses(self, node: Node) -> None:
        current_len = len(node.state.steps)
        existing = {child.state.get_full_response() for child in node.children}
        new_children = []
        for state in node.response_list:
            if len(state.steps) < current_len + 1:
                continue
            child_state = state.get_truncated_copy(target_length=current_len + 1)
            signature = child_state.get_full_response()
            if signature in existing:
                continue
            child = Node(state=child_state, parent=node)
            child.response_list = [child_state]
            self._ensure_tree_stats(child)
            node.children.append(child)
            new_children.append(child)
            existing.add(signature)

        if node.children:
            node.is_leaf = False
        elif not new_children:
            node.is_complete = True

    def _leaf_score(self, node: Node) -> float:
        self._ensure_tree_stats(node)
        selection_policy = getattr(self.config, "tree_selection_policy", "ucb")
        if node.value is None and node.parent is not None and node.parent.value is not None:
            node_value = node.parent.value
        else:
            node_value = node.value if node.value is not None else float("-inf")

        if node.visit_count == 0:
            return inf

        if selection_policy == "greedy":
            return node_value

        parent_visits = getattr(node.parent, "visit_count", 1) if node.parent is not None else 1
        c = getattr(self.config, "tree_ucb_c", 0.5)
        explore = c * sqrt(log(parent_visits + 1.0) / (node.visit_count + 1.0))
        return node_value + explore

    def _should_revisit_parent(self, node: Node) -> bool:
        if not getattr(self.config, "tree_allow_parent_resample", True):
            return False
        if not node.children or node.value is None:
            return False
        child_values = [child.value for child in node.children if child.value is not None]
        if not child_values:
            return True
        margin = getattr(self.config, "tree_parent_margin", 0.0)
        return max(child_values) < (node.value - margin)

    def _collect_expandable_nodes(self, node: Node) -> List[Node]:
        candidates: List[Node] = []
        if node.get_depth() >= self.config.max_depth or node.is_complete:
            return candidates

        is_leaf = (not node.children) or node.is_leaf
        if is_leaf or self._should_revisit_parent(node):
            candidates.append(node)

        for child in node.children:
            candidates.extend(self._collect_expandable_nodes(child))
        return candidates

    def _select_expand_target(self, root: Node) -> Optional[Node]:
        candidates = self._collect_expandable_nodes(root)
        if not candidates:
            return None

        best_leaf = None
        best_score = float("-inf")
        for leaf in candidates:
            score = self._leaf_score(leaf)
            if score > best_score:
                best_score = score
                best_leaf = leaf
        return best_leaf

    def search(self, initial_state: State) -> Node:
        self.clean_tree()
        root = Node(state=initial_state)
        self._ensure_tree_stats(root)

        if self._is_think_block():
            self._expand_root_with_full_completions(root, num_expand=self.config.K)
        else:
            self.roll_out_to_leaf(root, self.config.max_depth, num_rollout=self.config.m, num_expand=self.config.K)

        root.evaluate_value(self.config.N, tail_fraction=getattr(self.config, "tail_fraction", 20))
        self._append_children_from_responses(root)

        if getattr(self.config, "tree_bootstrap_root_children", True):
            self._bootstrap_root_children(root)

        while self.stats["rollouts"] < self.config.N:
            target = self._select_expand_target(root)
            if target is None:
                break

            self._mark_path_visit(target)
            remaining = self.config.N - self.stats["rollouts"]
            num_rollout = min(getattr(self.config, "tree_expand_batch", self.config.m), remaining)
            if num_rollout <= 0:
                break

            before = self.stats["rollouts"]
            self.roll_out_to_leaf(
                target,
                self.config.max_depth - target.get_depth(),
                num_rollout=num_rollout,
                num_expand=self.config.K,
            )
            if self.stats["rollouts"] <= before:
                target.is_complete = True
                continue

            target.evaluate_value(self.config.N, tail_fraction=getattr(self.config, "tail_fraction", 20))
            if target.response_list:
                self._append_children_from_responses(target)
            else:
                target.is_complete = True

        return root

    def run(self, problem_data: dict) -> dict:
        prompt_text = self.task.get_prompt(problem_data)
        initial_state = State(prompt=prompt_text)
        root = self.search(initial_state)
        final_text = self.best_response.get_full_response() if self.best_response else ""
        final_answer = self.task.extract_answer(final_text)
        return {
            "predicted_answer": final_answer,
            "best_score": self.best_response_score,
            "full_text": final_text,
            "total_rollouts": self.stats["rollouts"],
            "all_answers": root.all_answers,
        }
