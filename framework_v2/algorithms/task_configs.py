# 1. THE DATASET CONFIGS (Only cares about the problem and the final answer)
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
    }
}
