import re
import torch
from typing import List, Optional
from vllm import LLM, SamplingParams

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator assessing the quality of an AI assistant's response to a user instruction.\n\n"
    "Rate the response on a scale from 1 to 10, considering:\n"
    "- **Helpfulness**: Does it fully address what was asked?\n"
    "- **Accuracy**: Is the information correct and trustworthy?\n"
    "- **Clarity**: Is it well-written and easy to follow?\n"
    "- **Depth**: Does it provide sufficient detail without unnecessary padding?\n\n"
    "First write a brief evaluation (2-4 sentences), then give your score in this exact format:\n"
    "Score: [[X]] (where X is an integer from 1 to 10)"
)

JUDGE_USER_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}\n\n"
    "Evaluate the response above."
)

SCORE_PATTERN = re.compile(r'\[\[(\d{1,2})\]\]')


class LLMJudge:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 8192,
        max_tokens: int = 256,
    ):
        print(f"Loading LLM Judge: {model_name}...")
        self.model_name = model_name
        self.max_tokens = max_tokens

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        print("✅ LLM Judge loaded.")

    def _build_prompt(self, instruction: str, response: str) -> str:
        chat = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                instruction=instruction,
                response=response[:4000]  # truncate long responses to stay within context
            )},
        ]
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def _parse_score(self, text: str) -> Optional[float]:
        matches = SCORE_PATTERN.findall(text)
        if matches:
            score = int(matches[-1])
            if 1 <= score <= 10:
                return float(score)
        return None

    def score_batch(self, instructions: List[str], responses: List[str]) -> List[Optional[float]]:
        """Score a batch of (instruction, response) pairs. Returns list of scores (1–10) or None on parse failure."""
        if not instructions:
            return []

        prompts = [self._build_prompt(inst, resp) for inst, resp in zip(instructions, responses)]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_tokens,
            stop=None,
        )
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=True)

        scores = []
        for output in outputs:
            text = output.outputs[0].text
            score = self._parse_score(text)
            if score is None:
                print(f"⚠️  Judge parse failure. Raw output: {text[:200]!r}")
            scores.append(score)

        return scores

    def score(self, instruction: str, response: str) -> Optional[float]:
        return self.score_batch([instruction], [response])[0]
