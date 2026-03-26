from collections import defaultdict
from typing import List

from .base_algo import BaseAlgorithm
from core.data_structures import State


class BranchingBoN(BaseAlgorithm):
    """
    Branching Best-of-N: ablation of SLG without the scaling-law value estimation.

    Step 1: generate K responses from root (randomly sampled, no RM-guided selection).
    Step 2: give each of K branches an equal budget of floor((N - K) / K) completions.
            Total rollouts = K + K * floor((N-K)/K) <= N always.
    Step 3: score all branch completions with RM, return the best.
    """

    def __init__(self, llm_engine, rm_engine, task, config):
        super().__init__(llm_engine, rm_engine, task, config)
        self.stats = defaultdict(int)
        self.best_response = None
        self.best_response_score = float('-inf')
        self.all_answers = []

    def clean(self):
        self.stats = defaultdict(int)
        self.best_response = None
        self.best_response_score = float('-inf')
        self.all_answers = []

    def _update_best(self, rewards: List[float], states: List[State]):
        for score, state in zip(rewards, states):
            if score > self.best_response_score:
                self.best_response_score = score
                self.best_response = state

    def run(self, problem_data: dict) -> dict:
        self.clean()

        prompt_text = self.task.get_prompt(problem_data)
        initial_state = State(prompt=prompt_text)

        stop_seqs = self.task.action_strategy.get("stop_sequences", [])
        rm_instruction = (
            self.task.action_strategy.get("rm_instruction", None)
            or self.task.dataset_config.get("rm_instruction", None)
        )
        max_tokens = getattr(self.config, "max_tokens", 2048)

        # ── Step 1: root expansion — generate exactly K samples (random, no RM selection) ──
        raw_root = self.llm_engine.generate(
            prompts=[initial_state.get_full_text()],
            n=self.config.K,
            max_tokens=max_tokens,
            stop_sequences=stop_seqs,
        )[0]

        self.stats["rollouts"] += len(raw_root)

        # Build full response states from root
        root_states = []
        for raw_text in raw_root:
            state = initial_state.get_truncated_copy(len(initial_state.steps))
            for action in self.task.parse_response_to_actions(raw_text):
                state.append_step(action)
            root_states.append(state)

        # Score root responses for tracking only (not used for branch selection)
        root_rewards = self.rm_engine.score_states_batch(root_states, rm_instruction=rm_instruction)
        self._update_best(root_rewards, root_states)
        self.all_answers.extend(
            self.task.extract_answer(s.get_full_response()) for s in root_states
        )

        # All K root responses become branch prefixes (random sampling = use all K)
        top_k = [s.get_truncated_copy(1) for s in root_states]

        # ── Step 2: equal-budget branch expansion — remaining budget = N - K ──────────
        branch_budget = (self.config.N - self.config.K) // self.config.K
        if branch_budget <= 0 or not top_k:  # K >= N: budget exhausted at root
            # Budget exhausted at root; return what we have
            final_text = self.best_response.get_full_response() if self.best_response else ""
            return {
                "predicted_answer": self.task.extract_answer(final_text),
                "best_score": self.best_response_score,
                "full_text": final_text,
                "total_rollouts": self.stats["rollouts"],
                "all_answers": self.all_answers,
            }

        if getattr(self.config, "verbose", False):
            print(f"BBoN | K={self.config.K}, branch_budget={branch_budget}, total={self.config.K + self.config.K * branch_budget}")

        # One batched LLM call for all K branches
        branch_prompts = [s.get_full_text() for s in top_k]
        all_branch_raw = self.llm_engine.generate(
            prompts=branch_prompts,
            n=branch_budget,
            max_tokens=max_tokens,
            stop_sequences=stop_seqs,
        )

        # Build branch states and collect for one batched RM call
        all_branch_states = []
        for branch_prefix, raw_strings in zip(top_k, all_branch_raw):
            self.stats["rollouts"] += len(raw_strings)
            for raw_text in raw_strings:
                state = branch_prefix.get_truncated_copy(len(branch_prefix.steps))
                for action in self.task.parse_response_to_actions(raw_text):
                    state.append_step(action)
                all_branch_states.append(state)

        # ── Step 3: score all branch completions, return best ─────────────────
        branch_rewards = self.rm_engine.score_states_batch(
            all_branch_states, rm_instruction=rm_instruction
        )
        self._update_best(branch_rewards, all_branch_states)
        self.all_answers.extend(
            self.task.extract_answer(s.get_full_response()) for s in all_branch_states
        )

        final_text = self.best_response.get_full_response() if self.best_response else ""
        return {
            "predicted_answer": self.task.extract_answer(final_text),
            "best_score": self.best_response_score,
            "full_text": final_text,
            "total_rollouts": self.stats["rollouts"],
            "all_answers": self.all_answers,
        }
