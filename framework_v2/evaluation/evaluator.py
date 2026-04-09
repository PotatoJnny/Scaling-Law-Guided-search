import os
import re
import json
import pandas as pd
from typing import Dict, Any, List

class Evaluator:
    # ---> NEW: We pass the entire config dictionary here
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        self.experiment_name = experiment_name
        self.config = config
        self.task_type = config.get("task_setup", {}).get("task_type") or config.get("task_type")
        if self.task_type is None:
            dataset_name = config.get("task_setup", {}).get("dataset_name")
            self.task_type = "language" if dataset_name in {"alpaca_eval", "ultrafeedback"} else "math"
        self.results_folder = os.path.join("data", "results", "Results", experiment_name)
        self.output_file_path = os.path.join(self.results_folder, "results.json")
        self.results = []
        
        os.makedirs(self.results_folder, exist_ok=True)
        print(f"📁 Logger initialized. Saving to: {self.output_file_path}")

    def _normalize(self, s: str) -> str:
        """Normalize a math answer (number or LaTeX) for comparison."""
        s = str(s).strip()

        # Strip LaTeX whitespace and cosmetic wrappers
        s = re.sub(r'\s+', '', s)
        # Degree symbols: \circ, ^\circ, °
        s = re.sub(r'\^\\?circ', '', s)
        s = re.sub(r'°', '', s)
        s = re.sub(r'\\left', '', s)
        s = re.sub(r'\\right', '', s)
        s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
        s = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', s)
        s = s.replace('$', '')
        # Only strip thousand-separator commas (between digits), not coordinate commas
        s = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', s)
        # Normalize \sqrt{n} and \sqrt n to the same form
        s = re.sub(r'\\sqrt\{(\d+)\}', r'\\sqrt\1', s)

        # Handle \frac{a}{b} and -\frac{a}{b}
        frac_pat = re.fullmatch(r'(-?)\\frac\{([^}]+)\}\{([^}]+)\}', s)
        if frac_pat:
            sign, num, den = frac_pat.group(1), frac_pat.group(2), frac_pat.group(3)
            try:
                val = (float('-1') if sign == '-' else 1.0) * float(num) / float(den)
                return str(int(val)) if val == int(val) else f"{val:.10g}"
            except (ValueError, ZeroDivisionError):
                pass

        # Try direct float conversion
        try:
            f = float(s)
            return str(int(f)) if f == int(f) else str(f)
        except (ValueError, OverflowError):
            pass

        return s.lower()

    def get_pass_at_1(self, best_answer: str, true_answer: str) -> float:
        if not best_answer or not true_answer:
            return 0.0
        return 1.0 if self._normalize(best_answer) == self._normalize(true_answer) else 0.0

    def get_pass_at_all(self, all_answers: List[str], true_answer: str) -> float:
        if not all_answers or not true_answer:
            return 0.0
        norm_true = self._normalize(true_answer)
        return 1.0 if any(self._normalize(a) == norm_true for a in all_answers) else 0.0

    def get_majority_vote(self, all_answers: List[str], true_answer: str) -> float:
        """Returns 1.0 if the plurality answer (by count) matches true_answer."""
        if not all_answers or not true_answer:
            return 0.0
        from collections import Counter
        norm_counts = Counter(self._normalize(a) for a in all_answers if a)
        if not norm_counts:
            return 0.0
        majority_answer = norm_counts.most_common(1)[0][0]
        return 1.0 if majority_answer == self._normalize(true_answer) else 0.0

    def record_experiment(self, result_dict: Dict[str, Any]):
        self.results.append(result_dict)
        if len(self.results) % 5 == 0:
            self._save_to_disk()

    def _build_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {
                "total_experiments": 0,
                "average_time": 0.0,
                "average_score": 0.0,
            }

        df = pd.DataFrame(self.results)
        summary = {
            "total_experiments": len(self.results),
            "average_time": float(df.get('search_time', pd.Series(dtype=float)).mean()),
            "average_score": float(df.get('best_score', pd.Series(dtype=float)).mean()),
        }

        average_fields = {
            "pass_at_1": "average_pass_at_1",
            "pass_at_all": "average_pass_at_all",
            "majority_vote": "average_majority_vote",
            "best_correctness": "average_best_correctness",
            "best_speedup": "average_best_speedup",
            "best_runtime": "average_best_runtime",
            "best_original_runtime": "average_best_original_runtime",
            "num_testcases": "average_num_testcases",
        }
        for column, summary_key in average_fields.items():
            if column in df.columns:
                valid = pd.to_numeric(df[column], errors="coerce").dropna()
                if not valid.empty:
                    summary[summary_key] = float(valid.mean())

        fraction_fields = {
            "has_extracted_code": "fraction_with_extracted_code",
            "is_fully_correct": "fraction_fully_correct",
            "beats_original_runtime": "fraction_beats_original_runtime",
            "score_gt_1": "fraction_score_gt_1",
        }
        for column, summary_key in fraction_fields.items():
            if column in df.columns:
                valid = pd.to_numeric(df[column], errors="coerce").dropna()
                if not valid.empty:
                    summary[summary_key] = float(valid.mean())

        if "correct_speedup" in df.columns:
            valid = pd.to_numeric(df["correct_speedup"], errors="coerce").dropna()
            if not valid.empty:
                summary["median_correct_speedup"] = float(valid.median())

        if 'judge_score' in df.columns:
            valid_scores = df['judge_score'].dropna()
            summary["average_judge_score"] = float(valid_scores.mean()) if not valid_scores.empty else None
            summary["judge_score_count"] = int(valid_scores.count())
        return summary

    def generate_final_report(self):
        if not self.results:
            return

        summary = self._build_summary()
        self._save_to_disk(summary=summary)
        print(f"\n✅ Final Evaluation Complete. Report saved to {self.output_file_path}")

    def _save_to_disk(self, summary: Dict = None):
        summary = summary or self._build_summary()
        output_data = {
            "experiment_name": self.experiment_name,
            "experiment_config": self.config,
            "total_experiments": len(self.results),
            "summary": summary,
            "experiments": self.results
        }
            
        with open(self.output_file_path, 'w') as f:
            json.dump(output_data, f, indent=4)
