# framework_v2/tasks/math_task.py
import re
from .base_task import BaseTask
from core.data_structures import Action
from typing import List

class MathTask(BaseTask):
    def __init__(self, config: dict):
        """Initializes the task with a specific dataset's rules."""
        self.config = config

    def get_prompt(self, problem_data: dict) -> str:
        q_text = problem_data.get(self.config["question_column"], "")
        
        return self.config["prompt_template"].format(question=q_text)

    def extract_answer(self, raw_response: str) -> str:
        text = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
        
        delimiter = self.config["primary_delimiter"]

        if delimiter == "\\boxed{":
            match = re.search(r'\\boxed\{([^{}]+)\}', text)
            if match:
                return match.group(1).strip()

        elif delimiter in text:
            return text.split(delimiter)[-1].strip()
            
        numbers = re.findall(self.config["fallback_regex"], text.replace(',', ''))
        return numbers[-1] if numbers else ""


    def evaluate_correctness(self, model_answer: str, ground_truth: str) -> float:
        clean_truth = self.extract_answer(str(ground_truth))
        clean_model = str(model_answer).strip()
        
        if self.config["exact_match_type"] == "integer":
            try:
                return 1.0 if int(clean_model) == int(clean_truth) else 0.0
            except ValueError:
                return 0.0
        if self.config["exact_match_type"] == "float":
            try:
                return 1.0 if float(clean_model) == float(clean_truth) else 0.0
            except ValueError:
                return 0.0
                
        return 1.0 if clean_model == clean_truth else 0.0
    

    def parse_response_to_actions(self, full_response: str) -> List[Action]:
            
            delimiter = self.config.get("step_delimiter", "\n")
            terminal_token = self.config.get("cleanup_rules", {}).get("enforce_terminal", "***")
            
            # 1. Split by the logical delimiter
            raw_lines = full_response.split(delimiter)
            
            actions = []
            for line in raw_lines:
                clean_line = line.strip()
                if not clean_line: continue
                
                # 2. Unified Terminal Check (replaces work_on_last_step)
                is_final = terminal_token in clean_line
                
                # 3. Text Clean up (Remove 'Step X:', etc)
                # This regex finds "Step 1:", "step 2.", "1." at the start
                clean_line = re.sub(r'^(?i)step\s*\d+[:.]*\s*', '', clean_line)
                clean_line = clean_line.replace(terminal_token, "").strip()
                
                if clean_line:
                    actions.append(Action(step_text=clean_line, is_final=is_final))
                
                if is_final: break # Stop parsing if we hit the end marker
                
            return actions