"""
CodeExecutor: drop-in replacement for RMEngine for code optimization tasks.
Same interface: score_states_batch(states) -> List[float].

Score = correctness_fraction × max(time_original / time_new, ε)
  - correct + 10× faster  → 10.0
  - correct + same speed  → 1.0
  - wrong                 → ≈ 0.0
  - timed out             → 0.0
"""

import os
import re
import time
import statistics
import subprocess
import tempfile
import hashlib
from typing import List, Optional

from .data_structures import State

_FENCED_BLOCK = re.compile(r'```(?:python)?\s*\n(.*?)```', re.DOTALL)
_SCORE_EPSILON = 0.01   # floor for speedup ratio
_SCORE_CAP    = 100.0  # ceiling for speedup ratio (avoids inf when original code times out)


def _looks_like_code(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    return bool(re.match(
        r"^(?:from\s+\S+\s+import\s+\S+|import\s+\S+|def\s+\w+|class\s+\w+|if\s+__name__\s*==|"
        r"for\s+.+:|while\s+.+:|with\s+.+:|try:|except\b|elif\b|else:|return\b|break\b|continue\b|"
        r"print\s*\(|[A-Za-z_][A-Za-z0-9_,\s]*=\s*|sys\.stdout|sys\.stdin)",
        stripped,
    ))


def _extract_code(text: str) -> Optional[str]:
    """Extract the most plausible executable Python snippet from a model response."""
    if not text:
        return None

    matches = _FENCED_BLOCK.findall(text)
    if matches:
        for block in reversed(matches):
            block = block.strip()
            if block and any(_looks_like_code(line) for line in block.splitlines()):
                return block

    cleaned = text.strip()
    if not cleaned:
        return None

    lines = cleaned.splitlines()
    start = None
    end = len(lines)
    for idx, line in enumerate(lines):
        if _looks_like_code(line):
            start = idx
            break
    if start is None:
        return None

    for idx in range(start + 1, len(lines)):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        if stripped.startswith("```") or stripped.startswith("### ") or stripped.startswith("**"):
            end = idx
            break
        if not _looks_like_code(lines[idx]) and not stripped.startswith((")", "]", "}", ".")):
            end = idx
            break

    snippet = "\n".join(lines[start:end]).strip()
    if snippet:
        return snippet

    return None


def _run_program(code_file: str, test_input: str, timeout: float) -> tuple[bool, float]:
    """Run code_file with test_input on stdin. Returns (stdout_matches_expected, elapsed_sec)."""
    # NOTE: caller handles expected-output comparison; we just return elapsed time and stdout
    try:
        start = time.perf_counter()
        result = subprocess.run(
            ["python3", code_file],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        return result.returncode == 0, elapsed, result.stdout
    except subprocess.TimeoutExpired:
        return False, float('inf'), ""
    except Exception:
        return False, float('inf'), ""


class CodeExecutor:
    """
    Evaluates LM-generated code by actually running it.

    Usage in main.py:
        executor.set_problem(problem_data)   # once per problem before running the algorithm
        scores = executor.score_states_batch(states)
    """

    def __init__(
        self,
        timeout_secs: float = 10.0,
        n_timing_runs: int = 3,
    ):
        self.timeout_secs = timeout_secs
        self.n_timing_runs = n_timing_runs

        # Set per problem via set_problem()
        self._test_inputs: List[str] = []
        self._test_outputs: List[str] = []
        self._time_original: float = 1.0   # baseline; re-measured when set_problem() is called

        self._exec_cache: dict[str, tuple[float, float]] = {}   # code_hash → (correctness, runtime)

    def set_problem(self, problem_data: dict) -> None:
        """Call once per problem before calling score_states_batch."""
        self._test_inputs = problem_data.get("test_inputs", [])
        self._test_outputs = problem_data.get("test_outputs", [])
        self._exec_cache.clear()

        original_code = problem_data.get("original_code", "")
        dataset_runtime = float(problem_data.get("measured_runtime_v0") or 1.0)
        if original_code and self._test_inputs:
            measured = self._measure_runtime(original_code)
            # If timing times out (inf), fall back to dataset-recorded runtime
            self._time_original = measured if measured != float('inf') else dataset_runtime
        else:
            self._time_original = dataset_runtime

    def _measure_runtime(self, code: str) -> float:
        """Median wall time across n_timing_runs on the first test input."""
        if not self._test_inputs:
            return 1.0
        with tempfile.TemporaryDirectory() as tmpdir:
            code_file = os.path.join(tmpdir, "sol.py")
            with open(code_file, "w") as f:
                f.write(code)
            times = []
            test_in = self._test_inputs[0]
            for _ in range(self.n_timing_runs):
                ok, elapsed, _ = _run_program(code_file, test_in, self.timeout_secs)
                if ok:
                    times.append(elapsed)
            return statistics.median(times) if times else float('inf')

    def _evaluate_code(self, code: str) -> tuple[float, float]:
        """Returns (correctness_fraction, median_runtime_sec). Cached by code hash."""
        key = hashlib.md5(code.encode()).hexdigest()
        if key in self._exec_cache:
            return self._exec_cache[key]

        with tempfile.TemporaryDirectory() as tmpdir:
            code_file = os.path.join(tmpdir, "sol.py")
            with open(code_file, "w") as f:
                f.write(code)

            correct = 0
            runtimes = []
            for test_in, test_out in zip(self._test_inputs, self._test_outputs):
                ok, elapsed, stdout = _run_program(code_file, test_in, self.timeout_secs)
                if ok and stdout.strip() == test_out.strip():
                    correct += 1
                    runtimes.append(elapsed)

        n = len(self._test_inputs)
        correctness = correct / n if n > 0 else 0.0
        runtime = statistics.median(runtimes) if runtimes else float('inf')
        result = (correctness, runtime)
        self._exec_cache[key] = result
        return result

    def score_states_batch(self, states: List[State], **kwargs) -> List[float]:
        """Score each state by running its extracted code. Compatible with RMEngine interface."""
        scores = []
        for state in states:
            response = state.get_full_response()
            code = _extract_code(response)
            if code is None:
                scores.append(0.0)
                continue

            correctness, runtime_new = self._evaluate_code(code)
            if runtime_new == float('inf') or correctness == 0.0:
                scores.append(0.0)
                continue

            speedup = self._time_original / max(runtime_new, 1e-6)
            score = correctness * min(max(speedup, _SCORE_EPSILON), _SCORE_CAP)
            scores.append(score)

        return scores
