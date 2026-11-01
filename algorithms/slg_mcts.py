import copy
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
        self.last_run_debug = {}

    def clean_tree(self):
        self.best_response = None
        self.best_response_score = float("-inf")
        self.stats = defaultdict(int)
        self.all_answers = []
        self.all_scores_and_answers = []
        self.last_run_debug = {}

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

    @staticmethod
    def _safe_float(value):
        if value is None:
            return None
        return float(value)

    def _node_mean_reward(self, node: Node) -> float | None:
        if not node.reward_list:
            return None
        return float(sum(node.reward_list) / len(node.reward_list))

    def _node_debug_payload(self, node: Node, *, leaf_index: int | None = None) -> dict:
        response_tokens = self._state_response_tokens(node.state)
        payload = {
            "is_root": node.parent is None,
            "depth": node.get_depth(),
            "num_steps": len(node.state.steps),
            "response_tokens": response_tokens,
            "reward_count": len(node.reward_list),
            "best_reward": self._safe_float(max(node.reward_list)) if node.reward_list else None,
            "mean_reward": self._node_mean_reward(node),
            "value": self._safe_float(node.value),
            "is_complete": bool(node.is_complete),
        }
        if leaf_index is not None:
            payload["leaf_index"] = int(leaf_index)
        return payload

    def _selection_metric_name(self) -> str:
        return "potential"

    def _selection_metric_value(self, node: Node) -> float | None:
        return self._safe_float(node.value)

    def _capture_prepared_tree_snapshot(self, root: Node) -> dict:
        return {
            "root": copy.deepcopy(root),
            "stats": dict(self.stats),
            "best_response": copy.deepcopy(self.best_response),
            "best_response_score": float(self.best_response_score),
            "all_answers": list(self.all_answers),
            "all_scores_and_answers": copy.deepcopy(self.all_scores_and_answers),
            "sampling_counter": int(self._sampling_counter),
            "shared_stage_debug": copy.deepcopy(self.last_run_debug.get("shared_stage", {})),
        }

    def _restore_prepared_tree_snapshot(self, snapshot: dict) -> tuple[Node, List[Node]]:
        self.clean_tree()
        self.stats = defaultdict(int, snapshot["stats"])
        self.best_response = copy.deepcopy(snapshot["best_response"])
        self.best_response_score = float(snapshot["best_response_score"])
        self.all_answers = list(snapshot["all_answers"])
        self.all_scores_and_answers = copy.deepcopy(snapshot["all_scores_and_answers"])
        self._sampling_counter = int(snapshot["sampling_counter"])
        self.last_run_debug = {
            "shared_stage": copy.deepcopy(snapshot.get("shared_stage_debug", {})),
        }
        root = copy.deepcopy(snapshot["root"])
        leaves = root.get_all_leaves()
        return root, leaves

    def capture_prepared_tree_snapshot(self, root: Node) -> dict:
        return self._capture_prepared_tree_snapshot(root)

    def restore_prepared_tree_snapshot(self, snapshot: dict) -> tuple[Node, List[Node]]:
        return self._restore_prepared_tree_snapshot(snapshot)

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

    def _build_root_branch_state(self, root_state: State, candidate_state: State) -> State:
        if self._root_prefix_postselect_disabled():
            return candidate_state
        total_limit = self._strict_total_output_tokens()
        min_ratio = getattr(self.config, "strict_prefix_min_ratio", None)
        max_ratio = getattr(self.config, "strict_prefix_max_ratio", None)
        if total_limit is None or min_ratio is None or max_ratio is None:
            return candidate_state
        return self.task.build_postselected_root_branch_state(
            root_state,
            candidate_state,
            prefix_min_ratio=float(min_ratio),
            prefix_max_ratio=float(max_ratio),
        )

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
        max_tokens = self._max_generation_tokens_for_state(root.state)
        if max_tokens <= 0:
            return

        raw_strings = self.llm_engine.generate(
            prompts=[prompt_text],
            n=num_rollout,
            max_tokens=max_tokens,
            stop_sequences=stop_seqs,
            temperature=getattr(self.config, 'temperature', 1.0),
            top_p=getattr(self.config, 'top_p', 0.95),
            seed=self._next_sampling_seed(),
        )[0]

        actual_rollouts = len(raw_strings)
        self.stats['rollouts'] += actual_rollouts

        response_states = []
        for raw_text in raw_strings:
            new_state = self.task.build_state_from_raw_response(root.state, raw_text)
            response_states.append(new_state)

        if not response_states:
            return

        rewards = self.rm_engine.score_states_batch(response_states, **self._get_rm_kwargs())
        self._record_scored_states(rewards, response_states)

        sorted_pairs = sorted(zip(rewards, response_states), key=lambda pair: pair[0], reverse=True)
        sorted_rewards, sorted_states = zip(*sorted_pairs)

        top_states = []
        for state in sorted_states[: min(num_expand, len(sorted_states))]:
            top_states.append(self._build_root_branch_state(root.state, state))
        root.response_list = top_states
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
        max_tokens = self._max_generation_tokens_for_state(node.state)
        if max_tokens <= 0:
            return []

        stop_seqs = self._get_stop_seqs_for_state(node.state)
        in_completion = self._in_completion_phase(node.state)

        raw_strings = self.llm_engine.generate(
            prompts=[prompt_text],
            n=num_rollout,
            max_tokens=max_tokens,
            stop_sequences=stop_seqs,
            temperature=getattr(self.config, 'temperature', 1.0),
            top_p=getattr(self.config, 'top_p', 0.95),
            seed=self._next_sampling_seed(),
        )[0]

        actual_rollouts = len(raw_strings)
        self.stats['rollouts'] += actual_rollouts

        response_states = []
        for raw_text in raw_strings:
            if self._is_think_block():
                new_state = node.state.get_truncated_copy(len(node.state.steps))
                actions = self.task.parse_response_to_actions(raw_text)
                if in_completion:
                    actions = self._fix_completion_actions(actions)
                for action in actions:
                    new_state.append_step(action)
            else:
                new_state = self.task.build_state_from_raw_response(node.state, raw_text)
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

        leaves_by_max_tokens = defaultdict(list)
        for leaf in leaves:
            max_tokens = self._max_generation_tokens_for_state(leaf.state)
            if max_tokens > 0:
                leaves_by_max_tokens[max_tokens].append(leaf)

        for max_tokens, token_group in sorted(leaves_by_max_tokens.items()):
            # All leaves at the same depth share the same phase; use first leaf to determine stop seqs
            stop_seqs = self._get_stop_seqs_for_state(token_group[0].state)
            in_completion = self._in_completion_phase(token_group[0].state)
            leaf_prompts = [leaf.state.get_full_text() for leaf in token_group]

            all_raw = self.llm_engine.generate(
                prompts=leaf_prompts,
                n=n_per_leaf,
                max_tokens=max_tokens,
                stop_sequences=stop_seqs,
                temperature=getattr(self.config, 'temperature', 1.0),
                top_p=getattr(self.config, 'top_p', 0.95),
                seed=self._next_sampling_seed(),
            )

            all_states = []
            leaf_slices = []
            for leaf, raw_strings in zip(token_group, all_raw):
                self.stats['rollouts'] += len(raw_strings)
                start = len(all_states)
                for raw_text in raw_strings:
                    if self._is_think_block():
                        new_state = leaf.state.get_truncated_copy(len(leaf.state.steps))
                        actions = self.task.parse_response_to_actions(raw_text)
                        if in_completion:
                            actions = self._fix_completion_actions(actions)
                        for action in actions:
                            new_state.append_step(action)
                    else:
                        new_state = self.task.build_state_from_raw_response(leaf.state, raw_text)
                    all_states.append(new_state)
                leaf_slices.append((start, len(all_states)))

            if not all_states:
                continue

            all_rewards = self.rm_engine.score_states_batch(all_states, **self._get_rm_kwargs())
            self._record_scored_states(all_rewards, all_states)

            for leaf, (start, end) in zip(token_group, leaf_slices):
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
        n_per_leaf = 0
        if leaves and remaining_budget > 0:
            n_per_leaf = min(int(self.config.m), remaining_budget // len(leaves))
            if n_per_leaf > 0:
                self._expand_leaves_batched(leaves, n_per_leaf=n_per_leaf)
        for leaf in leaves:
            leaf.evaluate_value(total_resources, tail_fraction=getattr(self.config, 'tail_fraction', 20))
        self.last_run_debug["shared_stage"] = {
            "budget": int(self.config.N),
            "shared_rollouts": int(self.stats["rollouts"]),
            "remaining_budget_after_shared_stage": int(self._remaining_budget()),
            "num_root_children": len(root.children),
            "num_leaves": len(leaves),
            "n_per_leaf": int(n_per_leaf),
            "leaf_summaries": [
                self._node_debug_payload(leaf, leaf_index=leaf_index)
                for leaf_index, leaf in enumerate(leaves)
            ],
        }
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

    def finish_from_prepared_tree(self, root: Node, leaves: List[Node]) -> Node:
        best_node = self._choose_focus_node(root, leaves)
        selected_leaf_index = next((idx for idx, leaf in enumerate(leaves) if leaf is best_node), None)
        remaining_budget_before_finish = int(self._remaining_budget())
        selected_metric_before_finish = self._selection_metric_value(best_node)
        selected_node_before_finish = self._node_debug_payload(
            best_node,
            leaf_index=selected_leaf_index,
        )

        # Exhaust remaining budget on the best node
        self._consume_remaining_on_single_node(best_node)

        self.last_run_debug.update(
            {
                "variant": self.__class__.__name__,
                "selection": {
                    "metric_name": self._selection_metric_name(),
                    "selected_leaf_index": selected_leaf_index,
                    "metric_value_before_finish": selected_metric_before_finish,
                    "metric_value_after_finish": self._selection_metric_value(best_node),
                    "selected_node_before_finish": selected_node_before_finish,
                    "selected_node_after_finish": self._node_debug_payload(
                        best_node,
                        leaf_index=selected_leaf_index,
                    ),
                    "remaining_budget_before_finish": remaining_budget_before_finish,
                    "final_total_rollouts": int(self.stats["rollouts"]),
                },
            }
        )

        if self.config.verbose:
            print("Final Best Response Score:", self.best_response_score)
        return root

    def one_layer_expand(self, initial_state: State, num_expand: Optional[int] = None) -> Node:
        root, leaves = self._prepare_one_layer_tree(initial_state, num_expand=num_expand)
        return self.finish_from_prepared_tree(root, leaves)

    def build_result_payload(self, final_tree_root: Node) -> dict:
        final_text = self.best_response.get_full_response() if self.best_response else ""
        final_answer = self.task.extract_answer(final_text)
        return {
            "predicted_answer": final_answer,
            "best_score": self.best_response_score,
            "full_text": final_text,
            "raw_response": self.best_response.raw_response if self.best_response else "",
            "clean_response": self.best_response.clean_response if self.best_response else final_text,
            "state_metadata": dict(self.best_response.state_metadata) if self.best_response else {},
            "total_rollouts": self.stats['rollouts'],
            "all_answers": final_tree_root.all_answers,
            "all_scores_and_answers": self.all_scores_and_answers,
        }

    def run(self, problem_data: dict) -> dict:
        """
        The standardized entry point. Translates the raw dataset dict into the MCTS search.
        """
        prompt_text = self.task.get_prompt(problem_data)
        initial_state = State(prompt=prompt_text)
        final_tree_root = self.one_layer_expand(initial_state)
        return self.build_result_payload(final_tree_root)


class MeanGuidedSearch(SLG_Search):
    def _selection_metric_name(self) -> str:
        return "mean_reward"

    def _selection_metric_value(self, node: Node) -> float | None:
        return self._node_mean_reward(node)

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
