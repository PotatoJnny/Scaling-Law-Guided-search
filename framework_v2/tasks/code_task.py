import json
import os
import re
from .base_task import BaseTask

_FENCED_BLOCK_CPP = re.compile(r'```(?:cpp|c\+\+|cxx)?\s*\n(.*?)```', re.DOTALL)
_FENCED_BLOCK_PY  = re.compile(r'```(?:python)?\s*\n(.*?)```', re.DOTALL)

CODE_PROMPT_TEMPLATE_CPP = (
    "You are given a C++ program that is correct but slow. "
    "Your task is to optimize it to run faster while keeping the exact same input/output behavior.\n\n"
    "Original program:\n"
    "```cpp\n{original_code}\n```\n\n"
    "{injection}"
    "Provide the complete optimized C++ program in a fenced ```cpp``` block."
)

CODE_PROMPT_TEMPLATE_PY = (
    "You are given a Python program that is correct but slow. "
    "Your task is to optimize it to run faster while keeping the exact same input/output behavior.\n\n"
    "Original program:\n"
    "```python\n{original_code}\n```\n\n"
    "{injection}"
    "Provide the optimized Python program in a fenced ```python``` block."
)

CODE_PLAN_INJECTION = (
    "First write a comment block starting with `// PLAN:` explaining your optimization strategy "
    "(e.g., algorithmic improvement, data structure change, use of faster I/O). "
    "Then provide the optimized code.\n\n"
)


class CodeTask(BaseTask):
    """
    Task for code optimization using the PIE dataset.
    Score comes from CodeExecutor (actual execution), not a reward model.
    """

    def get_prompt(self, problem_data: dict) -> str:
        original_code = problem_data["original_code"]
        injection = self.action_strategy.get("prompt_injection", CODE_PLAN_INJECTION)
        language = self.dataset_config.get("language", "python")
        template = CODE_PROMPT_TEMPLATE_CPP if language == "cpp" else CODE_PROMPT_TEMPLATE_PY
        return template.format(
            original_code=original_code,
            injection=injection,
        )

    def prepare_dataset(self, dataset, cache_root: str):
        dataset = self._filter_by_language(dataset)
        dataset = self._filter_to_problems_with_testcases(dataset, cache_root)
        pie_testcases = self._load_pie_testcases(cache_root)
        return dataset, {"pie_testcases": pie_testcases}

    def prepare_problem_data(self, problem_data: dict, runtime_context=None, reward_source=None) -> dict:
        runtime_context = runtime_context or {}
        pie_testcases = runtime_context.get("pie_testcases", {})

        pid = problem_data.get("problem_id", "")
        if pid not in pie_testcases:
            try:
                pid = f"p{int(pid):05d}"
            except (ValueError, TypeError):
                pass

        tc = pie_testcases.get(pid, {"test_inputs": [], "test_outputs": []})
        prepared = dict(problem_data)
        prepared["original_code"] = prepared[self.dataset_config["question_column"]]
        prepared["test_inputs"] = tc["test_inputs"]
        prepared["test_outputs"] = tc["test_outputs"]
        prepared["measured_runtime_v0"] = prepared.get("measured_runtime_v0") or 1.0

        if reward_source is not None:
            reward_source.set_problem(prepared)
        return prepared

    def extract_answer(self, text: str) -> str:
        """Extract the last fenced code block, or the raw text if it looks like code."""
        language = self.dataset_config.get("language", "python")
        pattern = _FENCED_BLOCK_CPP if language == "cpp" else _FENCED_BLOCK_PY
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip()
        return self.sanitize_response_text(text)

    def build_result_metrics(self, problem_data: dict, search_result: dict, reward_source=None) -> dict:
        if reward_source is None:
            return {}

        metrics = reward_source.inspect_text(search_result.get("full_text", ""))
        return {
            "best_correctness": metrics["correctness"],
            "best_runtime": metrics["runtime"],
            "best_speedup": metrics["speedup"],
            "best_original_runtime": metrics["original_runtime"],
            "num_testcases": metrics["num_testcases"],
            "has_extracted_code": metrics["has_extracted_code"],
            "is_fully_correct": metrics["is_fully_correct"],
            "beats_original_runtime": metrics["beats_original_runtime"],
            "score_gt_1": metrics["score_gt_1"],
            "correct_speedup": metrics["correct_speedup"],
        }

    def serialize_search_artifacts(self, search_result: dict) -> dict:
        return {
            "all_scores_and_answers": [
                {"score": x["score"]}
                for x in search_result.get("all_scores_and_answers", [])
            ]
        }

    def _filter_by_language(self, dataset):
        lang = self.dataset_config.get("filter_language")
        if not lang:
            return dataset
        keep = [i for i, row in enumerate(dataset) if row.get("language") == lang]
        filtered = dataset.select(keep)
        print(f"Filtered pie-perf to {len(filtered)} rows with language='{lang}'.")
        return filtered

    def _filter_to_problems_with_testcases(self, dataset, cache_root: str):
        if not self.dataset_config.get("filter_to_testcases"):
            return dataset

        tc_json = os.path.join(cache_root, "local_pie_testcases.json")
        if not os.path.exists(tc_json):
            return dataset

        with open(tc_json) as f:
            available_pids = set(json.load(f).keys())

        seen_pids = set()
        keep_indices = []
        for index, row in enumerate(dataset):
            pid = str(row.get("problem_id", ""))
            if pid in available_pids and pid not in seen_pids:
                keep_indices.append(index)
                seen_pids.add(pid)

        filtered = dataset.select(keep_indices)
        print(f"Filtered pie-perf to {len(filtered)} unique problems with test cases.")
        return filtered

    def _load_pie_testcases(self, cache_root: str):
        tc_hf = self.dataset_config.get("testcases_hf_name")
        if not tc_hf:
            return {}

        tc_json = os.path.join(cache_root, "local_pie_testcases.json")
        if os.path.exists(tc_json):
            with open(tc_json) as f:
                testcases = json.load(f)
            print(f"Loaded test cases for {len(testcases)} problems from {tc_json}.")
            return testcases

        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        repo_dir_name = tc_hf.replace("/", "--")
        cache_repo = os.path.join(hf_home, "hub", f"datasets--{repo_dir_name}")
        snapshots_dir = os.path.join(cache_repo, "snapshots")
        snapshot = None
        if os.path.isdir(snapshots_dir):
            snaps = sorted(os.listdir(snapshots_dir))
            if snaps:
                snapshot = os.path.join(snapshots_dir, snaps[-1], "public_test_cases")

        if not snapshot or not os.path.isdir(snapshot):
            print(f"WARNING: No test case cache found. Download {tc_hf} first by running main.py once with network access.")
            print(f"  Expected path: {snapshot}")
            return {}

        print(f"Reading test cases from HF cache: {snapshot}")
        testcases = {}
        for prob_dir in sorted(os.listdir(snapshot)):
            prob_path = os.path.join(snapshot, prob_dir)
            if not os.path.isdir(prob_path):
                continue
            files = os.listdir(prob_path)
            idxs = sorted(set(f.split(".")[1] for f in files if f.startswith("input.")))
            inputs, outputs = [], []
            for idx in idxs:
                inp_f = os.path.join(prob_path, f"input.{idx}.txt")
                out_f = os.path.join(prob_path, f"output.{idx}.txt")
                if os.path.exists(inp_f) and os.path.exists(out_f):
                    with open(inp_f) as fi, open(out_f) as fo:
                        inputs.append(fi.read())
                        outputs.append(fo.read())
            if inputs:
                testcases[prob_dir] = {"test_inputs": inputs, "test_outputs": outputs}

        print(f"Loaded test cases for {len(testcases)} problems.")
        with open(tc_json, "w") as f:
            json.dump(testcases, f)
        print(f"Saved to {tc_json}")
        return testcases
