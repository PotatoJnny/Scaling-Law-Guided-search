# 1. THE DATASET CONFIGS (Only cares about the problem and the final answer)
# task_type "language": no ground-truth answer; quality measured by RM + LLM judge
DATASET_CONFIGS = {
    "gsm8k": {
        "question_column": "question",
        "answer_column": "answer",
        "base_prompt": "Solve this step-by-step. Put your final answer after ####.\nProblem: {question}\nSolution:\n",
        "primary_delimiter": "####",
        "exact_match_type": "string"
    },
    "aime24": {
        "hf_name": "HuggingFaceH4/aime_2024",
        "split": "train",
        "question_column": "problem",
        "answer_column": "answer",
        "base_prompt": "Solve the following AIME problem. Show your work step by step. Put your final integer answer inside \\boxed{{}}.\nProblem: {question}\nSolution:\n",
        "primary_delimiter": "\\boxed{",
        "fallback_regex": r"(\d+)",
        "raw_answer_column": True,
        "exact_match_type": "integer"
    },
    "aime25": {
        "hf_name": "math-ai/AIME25",
        "split": "test",
        "question_column": "problem",
        "answer_column": "answer",
        "base_prompt": "Solve the following AIME problem. Show your work step by step. Put your final integer answer inside \\boxed{{}}.\nProblem: {question}\nSolution:\n",
        "primary_delimiter": "\\boxed{",
        "fallback_regex": r"(\d+)",
        "raw_answer_column": True,
        "exact_match_type": "integer"
    },
    "math500": {
        "hf_name": "HuggingFaceH4/MATH-500",  # actual HuggingFace dataset ID for loading
        "question_column": "problem",
        "answer_column": "answer",
        "base_prompt": "Solve this math problem step-by-step. Put your final answer inside \\boxed{{}}.\nProblem: {question}\nSolution:\n",
        "primary_delimiter": "\\boxed{",
        "answer_type": "latex",
        "raw_answer_column": True,
        "exact_match_type": "latex"
    },
    "amc23": {
        "hf_name": "math-ai/amc23",
        "split": "test",
        "question_column": "question",
        "answer_column": "answer",
        "base_prompt": "Solve the following AMC competition problem. Show your work step by step. Put your final numerical answer inside \\boxed{{}}.\nProblem: {question}\nSolution:\n",
        "primary_delimiter": "\\boxed{",
        "fallback_regex": r"(\d+)",
        "raw_answer_column": True,
        "exact_match_type": "integer"
    },
    # ── Code optimization datasets ────────────────────────────────────────────
    "pie": {
        # PIE: Performance-Improving Edits (Shypula et al., 2023)
        # input = slow code, target = fast code, test cases in pie-perf-testcases
        # NOTE: use "train" split and filter to problems with test cases.
        # The "test" split uses high-numbered problem IDs (p03xxx) that are NOT
        # covered by pie-perf-testcases (which only has p00000–p00961).
        "hf_name": "rootacess/pie-perf",
        "testcases_hf_name": "rootacess/pie-perf-testcases",
        "split": "train",
        "filter_to_testcases": True,    # filter+deduplicate rows to those with test cases
        "question_column": "input",    # slow code
        "answer_column": None,         # no string answer; scored by execution
        "task_type": "code",
    },
    # ── Language task datasets ─────────────────────────────────────────────────
    "alpaca_eval": {
        "hf_name": "tatsu-lab/alpaca_eval",
        "split": "eval",
        "question_column": "instruction",
        "answer_column": None,          # no ground truth
        "task_type": "language",
    },
    "ultrafeedback": {
        "hf_name": "openbmb/UltraFeedback",
        "split": "train",               # UltraFeedback has no dedicated test split
        "question_column": "instruction",
        "answer_column": None,
        "task_type": "language",
    },
    # ── Thinking-model variants: prompt ends with <think>\n to prime the model into thinking mode
    "aime24_thinking": {
        "hf_name": "HuggingFaceH4/aime_2024",
        "split": "train",
        "question_column": "problem",
        "answer_column": "answer",
        "base_prompt": "Solve the following AIME problem. Put your final integer answer inside \\boxed{{}}.\nProblem: {question}\n<think>\n",
        "primary_delimiter": "\\boxed{",
        "fallback_regex": r"(\d+)",
        "raw_answer_column": True,
        "exact_match_type": "integer"
    },
    "aime25_thinking": {
        "hf_name": "math-ai/AIME25",
        "split": "test",
        "question_column": "problem",
        "answer_column": "answer",
        "base_prompt": "Solve the following AIME problem. Put your final integer answer inside \\boxed{{}}.\nProblem: {question}\n<think>\n",
        "primary_delimiter": "\\boxed{",
        "fallback_regex": r"(\d+)",
        "raw_answer_column": True,
        "exact_match_type": "integer"
    },
    "amc23_thinking": {
        "hf_name": "math-ai/amc23",
        "split": "test",
        "question_column": "question",
        "answer_column": "answer",
        "base_prompt": "Solve the following AMC competition problem. Put your final numerical answer inside \\boxed{{}}.\nProblem: {question}\n<think>\n",
        "primary_delimiter": "\\boxed{",
        "fallback_regex": r"(\d+)",
        "raw_answer_column": True,
        "exact_match_type": "integer"
    }
}


ACTION_STRATEGIES = {
    "double_newline": {
        "prompt_injection": "Separate each logical step with a double newline (\\n\\n). ",
        "stop_sequences": ["<|EOR|>"], # Generate full responses
        "chunking_method": "delimiter",
        "delimiter": "\n\n"
    },
    "double_newline_2step": {
        "prompt_injection": "Separate each logical step with a double newline (\\n\\n). ",
        "stop_sequences": ["<|EOR|>"],
        "chunking_method": "delimiter",
        "delimiter": "\n\n",
        "min_chunks_per_action": 2
    },
    "double_newline_3step": {
        "prompt_injection": "Separate each logical step with a double newline (\\n\\n). ",
        "stop_sequences": ["<|EOR|>"],
        "chunking_method": "delimiter",
        "delimiter": "\n\n",
        "min_chunks_per_action": 3
    },
    "step_prefix": {
        "prompt_injection": "Begin every single step with 'Step X: '. ",
        "stop_sequences": ["<|EOR|>"], 
        "chunking_method": "regex",
        "regex_pattern": r'(?=Step \d+:)' 
    },
    "fixed_tokens": {
        "prompt_injection": "",
        "stop_sequences": ["<|EOR|>"],
        "chunking_method": "token_count",
        "token_count": 50
    },
    "fixed_tokens_100": {
        "prompt_injection": "",
        "stop_sequences": ["<|EOR|>"],
        "chunking_method": "token_count",
        "token_count": 100
    },
    "fixed_tokens_200": {
        "prompt_injection": "",
        "stop_sequences": ["<|EOR|>"],
        "chunking_method": "token_count",
        "token_count": 200
    },
    "fixed_tokens_500": {
        "prompt_injection": "",
        "stop_sequences": ["<|EOR|>"],
        "chunking_method": "token_count",
        "token_count": 500
    },
    "strict_single_step": {
        "prompt_injection": "Generate ONLY ONE logical step. Do not write the full solution.",
        "stop_sequences": ["\n", "***"], # Forces vLLM to stop early
        "chunking_method": "delimiter",
        "delimiter": "\n"
    },
    "code_plan": {
        # Two-action structure: # PLAN: comment block (action 1) → code (action 2)
        "prompt_injection": (
            "First write a comment block starting with `# PLAN:` explaining your optimization strategy. "
            "Then provide the optimized code.\n\n"
        ),
        "stop_sequences": [],
        "completion_stop_sequences": [],
        "chunking_method": "code_block",
    },
    "think_block": {
        "prompt_injection": "",
        "stop_sequences": ["</think>"],       # probe phase: stop after thinking chain
        "completion_stop_sequences": [],      # completion phase: run to EOS
        "chunking_method": "think_block",
        "think_end_token": "</think>",
        "rm_response_mode": "answer_only_after_think",
        "rm_prompt_suffix_to_strip": "<think>\n",
    }
}
