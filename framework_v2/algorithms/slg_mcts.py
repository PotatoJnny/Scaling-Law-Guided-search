import torch
from typing import List, Optional
from collections import defaultdict

from .base_algo import BaseAlgorithm
from core.data_structures import State, Node, Action


class SLG_Search(BaseAlgorithm):
    def __init__(self, llm_engine, rm_engine, task, config):
        super().__init__(llm_engine, rm_engine, task, config)
        self.stats = defaultdict(int)
        self.best_response = None
        self.best_response_score = float("-inf")
        self.all_answers = []
        self.all_scores_and_answers = []

    def clean_tree(self):
        self.best_response = None
        self.best_response_score = float("-inf")
        self.stats = defaultdict(int)
        self.all_answers = []
        self.all_scores_and_answers = []

    def clean(self):
        self.clean_tree()

    def _remaining_budget(self) -> int:
        return max(0, int(self.config.N) - int(self.stats["rollouts"]))

    def _cap_requested_rollouts(self, requested: int) -> int:
        requested = max(0, int(requested))
        return min(requested, self._remaining_budget())

    def _branch_depth_limit(self, node: Node) -> int:
        return max(1, int(self.config.max_depth) - node.get_depth())

    def _record_scored_states(self, rewards: List[float], states: List[State]) -> None:
        answers = [self.task.extract_answer(state.get_full_response()) for state in states]
        self.all_answers.extend(answers)
        self.all_scores_and_answers.extend(
            {"score": float(score), "answer": answer}
            for score, answer in zip(rewards, answers)
        )
        for score, state in zip(rewards, states):
            if score > self.best_response_score:
                self.best_response_score = score
                self.best_response = state

    def _get_stop_seqs_for_state(self, state: State) -> list:
        """Returns probe or completion stop sequences depending on whether a thinking chain is present."""
        think_end = self.task.action_strategy.get("think_end_token", None)
        if think_end is not None and think_end in state.get_full_response():
            return self.task.action_strategy.get(
                "completion_stop_sequences",
                self.task.action_strategy.get("stop_sequences", [])
            )
        return self.task.action_strategy.get("stop_sequences", [])

    def _in_completion_phase(self, state: State) -> bool:
        """True when the state already has a completed thinking chain (probe done)."""
        think_end = self.task.action_strategy.get("think_end_token", None)
        return think_end is not None and think_end in state.get_full_response()

    def _fix_completion_actions(self, actions: List[Action]) -> List[Action]:
        """
        In the completion phase, parse_response_to_actions may incorrectly re-append
        </think> and mark is_final=False (its probe-phase fallback). Fix: strip the
        spurious </think> suffix and mark every action as final.
        """
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

    def _expand_root_with_full_completions(self, root: Node, num_expand: int) -> None:
        """
        For think-block search, the intended SLG behavior is:
        1. generate m full thinking+answer responses from the root,
        2. score those full responses,
        3. keep the top-k underlying thinking chains as branches.

        This avoids ranking root branches using RM scores on partial thinking-only states.
        """
        prompt_text = root.state.get_full_text()
        stop_seqs = self.task.action_strategy.get(
            "completion_stop_sequences",
            self.task.action_strategy.get("stop_sequences", [])
        )
        num_rollout = self._cap_requested_rollouts(self.config.m)
        if num_rollout <= 0:
            return

        raw_strings = self.llm_engine.generate(
            prompts=[prompt_text],
            n=num_rollout,
            max_tokens=2048,
            stop_sequences=stop_seqs,
            temperature=getattr(self.config, 'temperature', 1.0),
            top_p=getattr(self.config, 'top_p', 0.95),
            seed=self._next_sampling_seed(),
        )[0]

        actual_rollouts = len(raw_strings)
        self.stats['rollouts'] += actual_rollouts

        response_states = []
        for raw_text in raw_strings:
            new_state = root.state.get_truncated_copy(len(root.state.steps))
            actions = self.task.parse_response_to_actions(raw_text)
            for action in actions:
                new_state.append_step(action)
            response_states.append(new_state)

        if not response_states:
            return

        rewards = self.rm_engine.score_states_batch(response_states, **self._get_rm_kwargs())
        self._record_scored_states(rewards, response_states)

        sorted_pairs = sorted(zip(rewards, response_states), key=lambda pair: pair[0], reverse=True)
        sorted_rewards, sorted_states = zip(*sorted_pairs)

        root.response_list = list(sorted_states[: min(num_expand, len(sorted_states))])
        root.reward_list.extend(list(sorted_rewards))
        root.all_answers.extend([self.task.extract_answer(state.get_full_response()) for state in response_states])

    def roll_out_to_leaf(self, node: Node, depth: int, num_rollout: Optional[int] = None, num_expand: Optional[int] = None) -> List[State]:
        if depth <= 0:
            if getattr(self.config, "verbose", False):
                print(f"⚠️ Reached max_depth limit. Halting expansion for this branch.")
            return []

        if num_rollout is None:
            num_rollout = self.config.m - len(node.children)
        num_rollout = self._cap_requested_rollouts(num_rollout)
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
            max_tokens=2048,
            stop_sequences=stop_seqs,
            temperature=getattr(self.config, 'temperature', 1.0),
            top_p=getattr(self.config, 'top_p', 0.95),
            seed=self._next_sampling_seed(),
        )[0]

        actual_rollouts = len(raw_strings)
        self.stats['rollouts'] += actual_rollouts

        response_states = []
        for raw_text in raw_strings:
            new_state = node.state.get_truncated_copy(len(node.state.steps))

            actions = self.task.parse_response_to_actions(raw_text)
            if in_completion:
                actions = self._fix_completion_actions(actions)

            for action in actions:
                new_state.append_step(action)

            response_states.append(new_state)

        rewards = self.rm_engine.score_states_batch(response_states, **self._get_rm_kwargs())
        self._record_scored_states(rewards, response_states)

        top_states = []
        sorted_rewards = []
        if rewards:
            sorted_pairs = sorted(zip(rewards, response_states), key=lambda pair: pair[0], reverse=True)
            sorted_rewards, sorted_states = zip(*sorted_pairs)
            
            top_states = list(sorted_states[: min(num_expand, len(sorted_states))])

        extracted_answers = [self.task.extract_answer(state.get_full_response()) for state in response_states]
        
        node.all_answers.extend(extracted_answers)
        node.response_list = top_states 
        node.reward_list.extend(list(sorted_rewards))
        node.propagate_reward_list(list(sorted_rewards))
        node.propagate_all_answers(extracted_answers)
        
        return top_states
    

    def _expand_leaves_batched(self, leaves: List[Node], n_per_leaf: int) -> None:
        """Expands all leaves in one batched LM call and one batched RM call."""
        if not leaves or n_per_leaf <= 0:
            return

        # All leaves at the same depth share the same phase; use first leaf to determine stop seqs
        stop_seqs = self._get_stop_seqs_for_state(leaves[0].state)
        in_completion = self._in_completion_phase(leaves[0].state)
        leaf_prompts = [leaf.state.get_full_text() for leaf in leaves]

        # Single LM call for all leaves
        all_raw = self.llm_engine.generate(
            prompts=leaf_prompts,
            n=n_per_leaf,
            max_tokens=2048,
            stop_sequences=stop_seqs,
            temperature=getattr(self.config, 'temperature', 1.0),
            top_p=getattr(self.config, 'top_p', 0.95),
            seed=self._next_sampling_seed(),
        )

        # Parse responses and record slice indices per leaf
        all_states = []
        leaf_slices = []
        for leaf, raw_strings in zip(leaves, all_raw):
            self.stats['rollouts'] += len(raw_strings)
            start = len(all_states)
            for raw_text in raw_strings:
                new_state = leaf.state.get_truncated_copy(len(leaf.state.steps))
                actions = self.task.parse_response_to_actions(raw_text)
                if in_completion:
                    actions = self._fix_completion_actions(actions)
                for action in actions:
                    new_state.append_step(action)
                all_states.append(new_state)
            leaf_slices.append((start, len(all_states)))

        if not all_states:
            return

        # Single RM call for all states across all leaves
        all_rewards = self.rm_engine.score_states_batch(all_states, **self._get_rm_kwargs())
        self._record_scored_states(all_rewards, all_states)

        # Distribute results back to each leaf node
        for leaf, (start, end) in zip(leaves, leaf_slices):
            rewards = all_rewards[start:end]
            response_states = all_states[start:end]
            if not rewards:
                continue

            sorted_pairs = sorted(zip(rewards, response_states), key=lambda p: p[0], reverse=True)
            sorted_rewards, sorted_states = zip(*sorted_pairs)

            extracted_answers = [self.task.extract_answer(s.get_full_response()) for s in response_states]

            leaf.all_answers.extend(extracted_answers)
            leaf.response_list = list(sorted_states[:self.config.K])
            leaf.reward_list.extend(list(sorted_rewards))
            leaf.propagate_reward_list(list(sorted_rewards))
            leaf.propagate_all_answers(extracted_answers)

    def _prepare_one_layer_tree(self, initial_state: State, num_expand: Optional[int] = None):
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
        if self._is_think_block():
            self._expand_root_with_full_completions(root, num_expand=num_expand)
        else:
            self.roll_out_to_leaf(root, self.config.max_depth, num_expand=num_expand)

        total_resources = self.config.N
        root.evaluate_value(total_resources, tail_fraction=getattr(self.config, 'tail_fraction', 20))
        root.response_to_children()
        leaves = root.get_all_leaves()

        # Budget-aware leaf exploration. This keeps the SLG family strictly within N rollouts.
        remaining_budget = self._remaining_budget()
        if leaves and remaining_budget > 0:
            n_per_leaf = min(int(self.config.m), remaining_budget // len(leaves))
            if n_per_leaf > 0:
                self._expand_leaves_batched(leaves, n_per_leaf=n_per_leaf)
        for leaf in leaves:
            leaf.evaluate_value(total_resources, tail_fraction=getattr(self.config, 'tail_fraction', 20))
        return root, leaves

    def _choose_focus_node(self, root: Node, leaves: List[Node]) -> Node:
        best_value = root.value if root.value is not None else float("-inf")
        best_node = root
        for leaf in leaves:
            if leaf.value is not None and leaf.value > best_value:
                best_value = leaf.value
                best_node = leaf
        return best_node

    def _consume_remaining_on_single_node(self, node: Node) -> None:
        while self.stats['rollouts'] < self.config.N:
            remaining_budget = self.config.N - self.stats['rollouts']
            pre_rollout_count = self.stats['rollouts']
            self.roll_out_to_leaf(
                node,
                self._branch_depth_limit(node),
                num_rollout=remaining_budget,
            )
            if self.stats['rollouts'] <= pre_rollout_count:
                print("Generation failed to produce new responses. Breaking to prevent infinite loop.")
                break

    def one_layer_expand(self, initial_state: State, num_expand: Optional[int] = None) -> Node:
        root, leaves = self._prepare_one_layer_tree(initial_state, num_expand=num_expand)
        best_node = self._choose_focus_node(root, leaves)

        # Exhaust remaining budget on the best node
        self._consume_remaining_on_single_node(best_node)

        if self.config.verbose:
            print("Final Best Response Score:", self.best_response_score)
            
        return root

    def run(self, problem_data: dict) -> dict:
        """
        The standardized entry point. Translates the raw dataset dict into the MCTS search.
        """
        prompt_text = self.task.get_prompt(problem_data)
        initial_state = State(prompt=prompt_text)
        final_tree_root = self.one_layer_expand(initial_state)
        final_text = self.best_response.get_full_response() if self.best_response else ""
        final_answer = self.task.extract_answer(final_text)
        return {
            "predicted_answer": final_answer,
            "best_score": self.best_response_score,
            "full_text": final_text,
            "total_rollouts": self.stats['rollouts'],
            "all_answers": final_tree_root.all_answers,
            "all_scores_and_answers": self.all_scores_and_answers,
        }


class MeanGuidedSearch(SLG_Search):
    def _choose_focus_node(self, root: Node, leaves: List[Node]) -> Node:
        best_node = root
        best_mean = float("-inf")
        for leaf in leaves:
            if leaf.reward_list:
                mean_score = float(sum(leaf.reward_list) / len(leaf.reward_list))
                if mean_score > best_mean:
                    best_mean = mean_score
                    best_node = leaf
        return best_node
