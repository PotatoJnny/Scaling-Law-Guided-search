from collections import defaultdict
from typing import Optional

from .base_algo import BaseAlgorithm
from core.data_structures import State


class BestOfN(BaseAlgorithm):
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

    def search(self, initial_state: State) -> None:
        """Generates N responses and picks the best by reward model score."""
        if getattr(self.config, 'verbose', False):
            print(f"\n{'='*80}")
            print(f"Starting Best-of-N (N={self.config.N})")
            print('='*80)

        stop_seqs = self.task.action_strategy.get("stop_sequences", [])

        raw_strings = self.llm_engine.generate(
            prompts=[initial_state.get_full_text()],
            n=self.config.N,
            max_tokens=getattr(self.config, 'max_tokens', 2048),
            stop_sequences=stop_seqs
        )[0]

        self.stats['rollouts'] = len(raw_strings)

        # Build states from raw responses
        response_states = []
        for raw_text in raw_strings:
            state = initial_state.get_truncated_copy(len(initial_state.steps))
            for action in self.task.parse_response_to_actions(raw_text):
                state.append_step(action)
            response_states.append(state)

        if not response_states:
            return

        rm_instruction = self.task.action_strategy.get("rm_instruction", None) or \
                         self.task.dataset_config.get("rm_instruction", None)
        rewards = self.rm_engine.score_states_batch(response_states, rm_instruction=rm_instruction)

        # Find best response
        for score, state in zip(rewards, response_states):
            if score > self.best_response_score:
                self.best_response_score = score
                self.best_response = state

        self.all_answers = [
            self.task.extract_answer(s.get_full_response()) for s in response_states
        ]

        if getattr(self.config, 'verbose', False):
            print(f"Best score: {self.best_response_score:.4f}")

    def run(self, problem_data: dict) -> dict:
        """Standardized entry point."""
        self.clean()

        prompt_text = self.task.get_prompt(problem_data)
        initial_state = State(prompt=prompt_text)

        self.search(initial_state)

        final_text = self.best_response.get_full_response() if self.best_response else ""
        final_answer = self.task.extract_answer(final_text)

        return {
            "predicted_answer": final_answer,
            "best_score": self.best_response_score,
            "full_text": final_text,
            "total_rollouts": self.stats['rollouts'],
            "all_answers": self.all_answers,
        }
