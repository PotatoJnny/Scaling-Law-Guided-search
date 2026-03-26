import os
import json
import pandas as pd
from typing import Dict, Any, List
from core.data_structures import Node

class Evaluator:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.results_folder = f"Results/{experiment_name}"
        self.output_file_path = os.path.join(self.results_folder, "results.json")
        self.results = []
        
        # Create directory immediately
        os.makedirs(self.results_folder, exist_ok=True)
        print(f"📁 Logger initialized. Saving to: {self.output_file_path}")

    def get_em_score_for_root(self, root: Node, true_answer: str) -> float:
        """
        Traverses the MCTS tree to check if the true answer exists 
        anywhere in the extracted answers. Returns 1.0 if found, else 0.0.
        """
        if not root or not true_answer:
            return 0.0
            
        for ans in root.all_answers:
            if str(ans).strip() == str(true_answer).strip():
                return 1.0
                
        return 0.0

    def record_experiment(self, result_dict: Dict[str, Any]):
        """Stores a single problem's result and triggers an incremental save."""
        self.results.append(result_dict)
        
        # Save incrementally every 5 steps
        if len(self.results) % 5 == 0:
            self._save_to_disk()

    def generate_final_report(self):
        """Calculates averages and saves the final comprehensive JSON."""
        if not self.results:
            return

        df = pd.DataFrame(self.results)
        
        summary = {
            "total_experiments": len(self.results),
            "average_ts_time": float(df.get('ts_time', pd.Series(dtype=float)).mean()),
            "average_ts_score": float(df.get('ts_best_score', pd.Series(dtype=float)).mean()),
            "average_ts_pass_at_n": float(df.get('ts_pass_at_n', pd.Series(dtype=float)).mean()),
            "average_ts_rm_at_1": float(df.get('ts_rm_at_1', pd.Series(dtype=float)).mean()),
        }
        
        self._save_to_disk(summary=summary)
        print(f"\n✅ Final Evaluation Complete. Report saved to {self.output_file_path}")

    def _save_to_disk(self, summary: Dict = None):
        """Internal method to handle the actual file writing."""
        output_data = {"experiments": self.results}
        if summary:
            output_data["summary"] = summary
            
        with open(self.output_file_path, 'w') as f:
            json.dump(output_data, f, indent=4)