"""
PieCppTask — CodeTask subclass for the original PIE C++ benchmark.

Dataset layout (after running data/pie_cpp/download.sh):
  data/pie_cpp/
    test.jsonl            — official test split (978 rows, 41 problems)
    val.jsonl             — validation split
    train.jsonl           — full training split
    train_hq_only.jsonl   — high-quality training pairs only
    merged_test_cases/    — extracted from merged_test_cases.json.tar.gz
      {problem_id}/
        input.{idx}.txt
        output.{idx}.txt

JSONL fields used:
  src_code    — original slow C++ program
  tgt_code    — reference fast C++ program
  problem_id  — problem identifier (e.g. "p02676")
  tests       — list of string indices into the test case directory
"""

import json
import os
from typing import Any, Dict, List, Optional

from .code_task import CodeTask


class PieCppTask(CodeTask):

    def prepare_dataset(self, dataset, cache_root: str):
        data_dir = self.dataset_config["local_data_dir"]
        split = self.dataset_config.get("split", "test")

        jsonl_path = os.path.join(data_dir, f"{split}.jsonl")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"PIE C++ JSONL not found at {jsonl_path}. "
                f"Run data/pie_cpp/download.sh first."
            )

        rows = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        print(f"Loaded {len(rows)} rows from {jsonl_path}.")

        # Locate test case directory
        tc_dir = os.path.join(data_dir, "merged_test_cases")
        if not os.path.isdir(tc_dir):
            tc_dir = os.path.join(data_dir, "public_test_cases", "codenet", "public_test_cases")
        if not os.path.isdir(tc_dir):
            raise FileNotFoundError(
                f"Test case directory not found under {data_dir}. "
                f"Run data/pie_cpp/download.sh and extract the archives."
            )
        print(f"Using test cases from: {tc_dir}")

        processed = []
        skipped = 0
        for row in rows:
            pid = str(row.get("problem_id", ""))
            prob_tc_dir = os.path.join(tc_dir, pid)

            test_indices = [str(t) for t in row.get("tests", [])]
            test_inputs, test_outputs = [], []

            if os.path.isdir(prob_tc_dir) and test_indices:
                for idx in test_indices:
                    inp_f = os.path.join(prob_tc_dir, f"input.{idx}.txt")
                    out_f = os.path.join(prob_tc_dir, f"output.{idx}.txt")
                    if os.path.exists(inp_f) and os.path.exists(out_f):
                        with open(inp_f) as fi, open(out_f) as fo:
                            test_inputs.append(fi.read())
                            test_outputs.append(fo.read())

            if not test_inputs:
                skipped += 1
                continue

            processed.append({
                "problem_id": pid,
                "src_code": row.get("src_code", ""),
                "tgt_code": row.get("tgt_code", ""),
                "test_inputs": test_inputs,
                "test_outputs": test_outputs,
                "measured_runtime_v0": row.get("src_agg_runtime"),  # pre-measured if available
                "language": "cpp",
            })

        print(f"Loaded {len(processed)} rows with test cases ({skipped} skipped, no test cases found).")

        # One row per problem (keep the first occurrence per problem_id)
        seen_pids = set()
        unique = []
        for row in processed:
            pid = row["problem_id"]
            if pid not in seen_pids:
                seen_pids.add(pid)
                unique.append(row)
        print(f"Deduplicated to {len(unique)} unique problems.")
        processed = unique

        dataset_obj = _ListDataset(processed)
        return dataset_obj, {}

    def prepare_problem_data(self, problem_data: dict, runtime_context=None, reward_source=None) -> dict:
        prepared = dict(problem_data)
        prepared["original_code"] = prepared["src_code"]
        if reward_source is not None:
            reward_source.set_problem(prepared)
        return prepared

    # No-ops — PIE C++ handles everything in prepare_dataset
    def _filter_by_language(self, dataset):
        return dataset

    def _filter_to_problems_with_testcases(self, dataset, cache_root: str):
        return dataset

    def _load_pie_testcases(self, cache_root: str):
        return {}


class _ListDataset:
    """Minimal list-backed dataset compatible with the framework's iteration interface."""

    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]

    def __iter__(self):
        return iter(self._rows)

    def select(self, indices):
        return _ListDataset([self._rows[i] for i in indices])
