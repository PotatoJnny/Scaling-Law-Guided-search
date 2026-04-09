ACTION_STRATEGIES = {
    "double_newline": {
        "prompt_injection": "Separate each logical step with a double newline (\\n\\n). ",
        "stop_sequences": ["<|EOR|>"],
        "chunking_method": "delimiter",
        "delimiter": "\n\n",
    },
    "double_newline_2step": {
        "prompt_injection": "Separate each logical step with a double newline (\\n\\n). ",
        "stop_sequences": ["<|EOR|>"],
        "chunking_method": "delimiter",
        "delimiter": "\n\n",
        "min_chunks_per_action": 2,
    },
    "double_newline_3step": {
        "prompt_injection": "Separate each logical step with a double newline (\\n\\n). ",
        "stop_sequences": ["<|EOR|>"],
        "chunking_method": "delimiter",
        "delimiter": "\n\n",
        "min_chunks_per_action": 3,
    },
    "step_prefix": {
        "prompt_injection": "Begin every single step with 'Step X: '. ",
        "stop_sequences": ["<|EOR|>"],
        "chunking_method": "regex",
        "regex_pattern": r"(?=Step \d+:)",
    },
    "fixed_tokens": {
        "prompt_injection": "",
        "stop_sequences": ["<|EOR|>"],
        "chunking_method": "token_count",
        "token_count": 50,
    },
    "fixed_tokens_100": {
        "prompt_injection": "",
        "stop_sequences": ["<|EOR|>"],
        "chunking_method": "token_count",
        "token_count": 100,
    },
    "fixed_tokens_200": {
        "prompt_injection": "",
        "stop_sequences": ["<|EOR|>"],
        "chunking_method": "token_count",
        "token_count": 200,
    },
    "fixed_tokens_500": {
        "prompt_injection": "",
        "stop_sequences": ["<|EOR|>"],
        "chunking_method": "token_count",
        "token_count": 500,
    },
    "strict_single_step": {
        "prompt_injection": "Generate ONLY ONE logical step. Do not write the full solution.",
        "stop_sequences": ["\n", "***"],
        "chunking_method": "delimiter",
        "delimiter": "\n",
    },
    "code_plan": {
        "prompt_injection": (
            "First write a comment block starting with `// PLAN:` explaining your optimization strategy "
            "(e.g., algorithmic improvement, data structure change, faster I/O). "
            "Then provide the optimized code.\n\n"
        ),
        "stop_sequences": [],
        "completion_stop_sequences": [],
        "chunking_method": "code_block",
    },
    "code_plain": {
        "prompt_injection": "",
        "stop_sequences": [],
        "completion_stop_sequences": [],
        "chunking_method": "code_block",
    },
    "think_block": {
        "prompt_injection": "",
        "stop_sequences": ["</think>"],
        "completion_stop_sequences": [],
        "chunking_method": "think_block",
        "think_end_token": "</think>",
        "rm_response_mode": "answer_only_after_think",
        "rm_prompt_suffix_to_strip": "<think>\n",
    },
}
