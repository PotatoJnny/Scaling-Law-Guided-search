import os
import re
import json
import pandas as pd
from typing import Dict, Any, List
from core.data_structures import Node

class Evaluator:
    # ---> NEW: We pass the entire config dictionary here
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        self.experiment_name = experiment_name
        self.config = config 
        self.results_folder = f"Results/{experiment_name}"
        self.output_file_path = os.path.join(self.results_folder, "results.json")
        self.results = []
        
        os.makedirs(self.results_folder, exist_ok=True)
        print(f"📁 Logger initialized. Saving to: {self.output_file_path}")

    def _normalize(self, s: str) -> str:
        """Normalize numeric strings: strip commas, unify 72.0 -> 72."""
        s = str(s).strip().replace(',', '')
        try:
            f = float(s)
            return str(int(f)) if f == int(f) else str(f)
        except (ValueError, OverflowError):
            return s

    def get_pass_at_1(self, best_answer: str, true_answer: str) -> float:
        if not best_answer or not true_answer:
            return 0.0
        return 1.0 if self._normalize(best_answer) == self._normalize(true_answer) else 0.0

    def get_pass_at_all(self, all_answers: List[str], true_answer: str) -> float:
        if not all_answers or not true_answer:
            return 0.0
        norm_true = self._normalize(true_answer)
        return 1.0 if any(self._normalize(a) == norm_true for a in all_answers) else 0.0

    def record_experiment(self, result_dict: Dict[str, Any]):
        self.results.append(result_dict)
        if len(self.results) % 5 == 0:
            self._save_to_disk()

    def generate_final_report(self):
        if not self.results:
            return

        df = pd.DataFrame(self.results)
        
        # ---> NEW: Generalized, algorithm-agnostic metrics!
        summary = {
            "total_experiments": len(self.results),
            "average_time": float(df.get('search_time', pd.Series(dtype=float)).mean()),
            "average_score": float(df.get('best_score', pd.Series(dtype=float)).mean()),
            "average_pass_at_1": float(df.get('pass_at_1', pd.Series(dtype=float)).mean()),
            "average_pass_at_all": float(df.get('pass_at_all', pd.Series(dtype=float)).mean()),
        }
        
        self._save_to_disk(summary=summary)
        print(f"\n✅ Final Evaluation Complete. Report saved to {self.output_file_path}")

    def _save_to_disk(self, summary: Dict = None):
        output_data = {
            "experiment_config": self.config,
            "experiments": self.results
        }
        if summary:
            output_data["summary"] = summary
            
        with open(self.output_file_path, 'w') as f:
            json.dump(output_data, f, indent=4)