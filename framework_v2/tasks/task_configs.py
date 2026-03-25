MATH_PROOF_CONFIG = {
    "step_delimiter": "\n", # Or '***' based on your preference
    "stop_sequences": ["\n", "***"], 
    "prompt_template": (
        "You are a mathematical assistant. Generate ONLY ONE logical step at a time.\n"
        "Rules:\n"
        "- Generate exactly one proof step\n"
        "- Use proper notation\n"
        "- If complete, end with '***'\n\n"
        "Question: {question}\n"
        "Solution:\n"
    ),
    "cleanup_rules": {
        "remove_step_prefix": True,
        "enforce_terminal": "***"
    }
}



MATH_CONFIGS = {
    "gsm8k": {
        "question_column": "question",
        "answer_column": "answer",
        "prompt_template": "Solve this step-by-step. Put your final answer after ####.\nProblem: {question}\nSolution:",
        "primary_delimiter": "####",
        "fallback_regex": r'-?\d+(?:\.\d+)?',
        "exact_match_type": "string",
        "rm_instruction": "Evaluate the following mathematical reasoning. Focus strictly on logical rigor, step-by-step correctness, and the accuracy of the final answer."
    },
    "aime25": {
        "question_column": "problem",
        "answer_column": "solution",  
        "prompt_template": "Solve the AIME problem. The answer is an integer between 0 and 999. Put your answer inside \\boxed{{}}.\nProblem: {question}\nSolution:",
        "primary_delimiter": "\\boxed{",
        "fallback_regex": r'\b\d{1,3}\b',
        "exact_match_type": "integer",
        "rm_instruction": "Evaluate the following mathematical reasoning. Focus strictly on logical rigor, step-by-step correctness, and the accuracy of the final answer."
    }
}