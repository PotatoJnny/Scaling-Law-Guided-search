# 1. THE DATASET CONFIGS (Only cares about the problem and the final answer)
DATASET_CONFIGS = {
    "gsm8k": {
        "question_column": "question",
        "answer_column": "answer",
        "base_prompt": "Solve this step-by-step. Put your final answer after ####.\nProblem: {question}\nSolution:\n",
        "primary_delimiter": "####",
        "exact_match_type": "string"
    },
    "aime25": {
        "question_column": "problem",
        "answer_column": "answer",
        "base_prompt": "Solve the AIME problem. Put your answer inside \\boxed{{}}.\nProblem: {question}\nSolution:\n",
        "primary_delimiter": "\\boxed{",
        "fallback_regex": r"(\d+)",
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
    "strict_single_step": {
        "prompt_injection": "Generate ONLY ONE logical step. Do not write the full solution.",
        "stop_sequences": ["\n", "***"], # Forces vLLM to stop early
        "chunking_method": "delimiter",
        "delimiter": "\n"
    }
}