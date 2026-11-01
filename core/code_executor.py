"""
CodeExecutor: drop-in replacement for RMEngine for code optimization tasks.
Same interface: score_states_batch(states) -> List[float].

Score = time_original / time_new if and only if all tests pass; otherwise 0.0
  - fully correct + 10× faster  → 10.0
  - fully correct + same speed   → 1.0
  - partially correct / wrong    → 0.0
  - timed out / compile failure → 0.0

Supports both Python and C++ (auto-detected from language= config, or from
the CodeTask.dataset_config["language"] field passed at construction).
"""

import hashlib
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

from .data_structures import State
from .base_reward import BaseRewardEngine

_FENCED_BLOCK_CPP = re.compile(r'```(?:cpp|c\+\+|cxx)?\s*\n(.*?)```', re.DOTALL)
_FENCED_BLOCK_PY  = re.compile(r'```(?:python)?\s*\n(.*?)```', re.DOTALL)


# ---------------------------------------------------------------------------
# Code extraction helpers
# ---------------------------------------------------------------------------

def _looks_like_python(line: str) -> bool:
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


def _looks_like_cpp(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(re.match(
        r"^(?:#include\b|#define\b|using\s+namespace\b|int\s+main\s*\(|"
        r"void\s+\w+\s*\(|std::|cout\b|cin\b|printf\s*\(|scanf\s*\(|"
        r"//|/\*|\*|struct\s+\w+|class\s+\w+|template\s*<|typedef\b|namespace\b)",
        stripped,
    ))


def _extract_code(text: str, language: str = "python") -> Optional[str]:
    """Extract the most plausible executable code snippet from a model response."""
    if not text:
        return None

    if language == "cpp":
        pattern = _FENCED_BLOCK_CPP
        looks_like = _looks_like_cpp
    else:
        pattern = _FENCED_BLOCK_PY
        looks_like = _looks_like_python

    matches = pattern.findall(text)
    if matches:
        for block in reversed(matches):
            block = block.strip()
            if block and any(looks_like(line) for line in block.splitlines()):
                return block

    # Fallback: heuristic scan for un-fenced code
    cleaned = text.strip()
    if not cleaned:
        return None
    lines = cleaned.splitlines()
    start = None
    end = len(lines)
    for idx, line in enumerate(lines):
        if looks_like(line):
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
        if not looks_like(lines[idx]) and not stripped.startswith((")", "]", "}", ".", "else", "elif")):
            end = idx
            break
    snippet = "\n".join(lines[start:end]).strip()
    return snippet if snippet else None


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def _run_python(code_file: str, test_input: str, timeout: float):
    """Returns (success, elapsed_sec, stdout)."""
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


def _compile_cpp(src_file: str, binary_file: str, timeout: float = 30.0) -> bool:
    """Compile src_file to binary_file with g++. Returns True on success."""
    try:
        result = subprocess.run(
            ["g++", "-O2", "-o", binary_file, src_file,
             "-std=c++17", "-lm"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except Exception:
        return False


def _run_binary(binary_file: str, test_input: str, timeout: float):
    """Returns (success, elapsed_sec, stdout)."""
    try:
        start = time.perf_counter()
        result = subprocess.run(
            [binary_file],
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


# ---------------------------------------------------------------------------
# Main executor class
# ---------------------------------------------------------------------------

class CodeExecutor(BaseRewardEngine):
    """
    Evaluates LM-generated code by actually running it.

    Usage in main.py:
        executor.set_problem(problem_data)   # once per problem before running the algorithm
        scores = executor.score_states_batch(states)

    Supports language="python" (default) and language="cpp".
    The language is taken from the dataset_config passed at construction or
    from problem_data["language"] at set_problem() time.
    """

    def __init__(
        self,
        timeout_secs: float = 10.0,
        n_timing_runs: int = 3,
        language: str = "python",
    ):
        self.timeout_secs = timeout_secs
        self.n_timing_runs = n_timing_runs
        self.language = language

        self._test_inputs: List[str] = []
        self._test_outputs: List[str] = []
        self._time_original: float = 1.0

        self._exec_cache: dict = {}   # code_hash → (correctness, runtime)
        self._tmpdir: Optional[str] = None  # persistent tmpdir for compiled binaries

    def _get_tmpdir(self) -> str:
        if self._tmpdir is None or not os.path.isdir(self._tmpdir):
            self._tmpdir = tempfile.mkdtemp(prefix="code_executor_")
        return self._tmpdir

    def set_problem(self, problem_data: dict) -> None:
        self._test_inputs = problem_data.get("test_inputs", [])
        self._test_outputs = problem_data.get("test_outputs", [])
        self._exec_cache.clear()
        # Allow per-problem language override
        self.language = problem_data.get("language", self.language) or self.language

        original_code = problem_data.get("original_code", "")
        dataset_runtime = float(problem_data.get("measured_runtime_v0") or 1.0)
        if original_code and self._test_inputs:
            measured = self._measure_runtime(original_code)
            self._time_original = measured if measured != float('inf') else dataset_runtime
        else:
            self._time_original = dataset_runtime

    def _measure_runtime(self, code: str) -> float:
        """Median wall time across n_timing_runs on the first test input."""
        if not self._test_inputs:
            return 1.0
        tmpdir = self._get_tmpdir()
        times = []
        test_in = self._test_inputs[0]

        if self.language == "cpp":
            src = os.path.join(tmpdir, "_orig.cpp")
            binary = os.path.join(tmpdir, "_orig_bin")
            with open(src, "w") as f:
                f.write(code)
            if not _compile_cpp(src, binary):
                return float('inf')
            for _ in range(self.n_timing_runs):
                ok, elapsed, _ = _run_binary(binary, test_in, self.timeout_secs)
                if ok:
                    times.append(elapsed)
        else:
            src = os.path.join(tmpdir, "_orig.py")
            with open(src, "w") as f:
                f.write(code)
            for _ in range(self.n_timing_runs):
                ok, elapsed, _ = _run_python(src, test_in, self.timeout_secs)
                if ok:
                    times.append(elapsed)

        return statistics.median(times) if times else float('inf')

    def _evaluate_code(self, code: str) -> tuple:
        """Returns (correctness_fraction, median_runtime_sec). Cached by code hash."""
        key = hashlib.md5(code.encode()).hexdigest()
        if key in self._exec_cache:
            return self._exec_cache[key]

        tmpdir = self._get_tmpdir()

        if self.language == "cpp":
            src = os.path.join(tmpdir, f"{key[:8]}.cpp")
            binary = os.path.join(tmpdir, f"{key[:8]}_bin")
            with open(src, "w") as f:
                f.write(code)
            if not _compile_cpp(src, binary):
                result = (0.0, float('inf'))
                self._exec_cache[key] = result
                return result
            run_fn = lambda inp: _run_binary(binary, inp, self.timeout_secs)
        else:
            src = os.path.join(tmpdir, f"{key[:8]}.py")
            with open(src, "w") as f:
                f.write(code)
            run_fn = lambda inp: _run_python(src, inp, self.timeout_secs)

        correct = 0
        runtimes = []
        for test_in, test_out in zip(self._test_inputs, self._test_outputs):
            ok, elapsed, stdout = run_fn(test_in)
            if ok and stdout.strip() == test_out.strip():
                correct += 1
                runtimes.append(elapsed)

        n = len(self._test_inputs)
        correctness = correct / n if n > 0 else 0.0
        runtime = statistics.median(runtimes) if runtimes else float('inf')
        result = (correctness, runtime)
        self._exec_cache[key] = result
        return result

    def inspect_text(self, text: str) -> Dict[str, Any]:
        code = _extract_code(text, self.language)
        metrics: Dict[str, Any] = {
            "has_extracted_code": code is not None,
            "num_testcases": len(self._test_inputs),
            "original_runtime": self._time_original,
            "correctness": 0.0,
            "runtime": None,
            "speedup": None,
            "score": 0.0,
            "is_fully_correct": False,
            "beats_original_runtime": False,
            "score_gt_1": False,
            "correct_speedup": None,
        }

        if code is None:
            return metrics

        correctness, runtime_new = self._evaluate_code(code)
        metrics["correctness"] = correctness
        metrics["is_fully_correct"] = math.isclose(correctness, 1.0, rel_tol=0.0, abs_tol=1e-9)

        if runtime_new != float('inf'):
            metrics["runtime"] = runtime_new
            speedup = self._time_original / max(runtime_new, 1e-6)
            metrics["speedup"] = speedup
            if metrics["is_fully_correct"]:
                metrics["score"] = speedup
            metrics["beats_original_runtime"] = speedup > 1.0

        metrics["score_gt_1"] = metrics["score"] > 1.0
        metrics["correct_speedup"] = metrics["speedup"] if metrics["is_fully_correct"] else None
        return metrics

    def score_states_batch(self, states: List[State], **kwargs) -> List[float]:
        scores = []
        for state in states:
            metrics = self.inspect_text(state.get_full_response())
            scores.append(float(metrics["score"]))
        return scores

    def __del__(self):
        if self._tmpdir and os.path.isdir(self._tmpdir):
            try:
                shutil.rmtree(self._tmpdir)
            except Exception:
                pass
