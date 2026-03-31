"""
API-based LLM judge for evaluating language task responses.
Supports OpenAI and Anthropic. Designed to run as a post-processing step
on an existing results.json produced by main.py.

Usage:
    python -m evaluation.llm_judge_api \
        --results Results/my_experiment/results.json \
        --provider openai --model gpt-4o \
        --api_key sk-...

    python -m evaluation.llm_judge_api \
        --results Results/my_experiment/results.json \
        --provider anthropic --model claude-opus-4-6 \
        --api_key sk-ant-...
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

SCORE_PATTERN = re.compile(r'\[\[(\d{1,2})\]\]')

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator assessing the quality of an AI assistant's response to a user instruction.\n\n"
    "Rate the response on a scale from 1 to 10, considering:\n"
    "- Helpfulness: Does it fully address what was asked?\n"
    "- Accuracy: Is the information correct and trustworthy?\n"
    "- Clarity: Is it well-written and easy to follow?\n"
    "- Depth: Does it provide sufficient detail without unnecessary padding?\n\n"
    "First write a brief evaluation (2-4 sentences), then give your score in this exact format:\n"
    "Score: [[X]] (where X is an integer from 1 to 10)"
)

JUDGE_USER_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}\n\n"
    "Evaluate the response above."
)


def _parse_score(text: str) -> Optional[float]:
    matches = SCORE_PATTERN.findall(text)
    if matches:
        score = int(matches[-1])
        if 1 <= score <= 10:
            return float(score)
    return None


def _judge_openai(client, model: str, instruction: str, response: str) -> tuple[Optional[float], str]:
    reply = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                instruction=instruction,
                response=response[:6000]
            )},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    text = reply.choices[0].message.content
    return _parse_score(text), text


def _judge_anthropic(client, model: str, instruction: str, response: str) -> tuple[Optional[float], str]:
    reply = client.messages.create(
        model=model,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": JUDGE_USER_TEMPLATE.format(
            instruction=instruction,
            response=response[:6000]
        )}],
        temperature=0.0,
        max_tokens=256,
    )
    text = reply.content[0].text
    return _parse_score(text), text


def run_judge(results_path: str, provider: str, model: str, api_key: str,
              retry_delay: float = 5.0, max_retries: int = 3):
    results_path = Path(results_path)
    with open(results_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("No results found in file.")
        return

    # Set up client
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        judge_fn = lambda inst, resp: _judge_openai(client, model, inst, resp)
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        judge_fn = lambda inst, resp: _judge_anthropic(client, model, inst, resp)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")

    already_scored = sum(1 for r in results if r.get("judge_score") is not None)
    if already_scored:
        print(f"Resuming: {already_scored}/{len(results)} already scored.")

    scores = []
    for i, entry in enumerate(results):
        if entry.get("judge_score") is not None:
            scores.append(entry["judge_score"])
            continue

        instruction = entry.get("prompt", "")
        response = entry.get("best_response", "")

        score = None
        for attempt in range(max_retries):
            try:
                score, raw_text = judge_fn(instruction, response)
                if score is None:
                    print(f"  [#{i+1}] Parse failure. Raw: {raw_text[:120]!r}")
                break
            except Exception as e:
                print(f"  [#{i+1}] API error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))

        entry["judge_score"] = score
        entry["judge_model"] = f"{provider}/{model}"
        scores.append(score)

        valid = [s for s in scores if s is not None]
        print(f"[{i+1}/{len(results)}] score={score}  running_avg={sum(valid)/len(valid):.2f} ({len(valid)} valid)")

        # Save incrementally after every 10 problems
        if (i + 1) % 10 == 0:
            _save(data, results_path, model, scores)

    _save(data, results_path, model, scores)


def _save(data: dict, results_path: Path, model: str, scores: list):
    valid = [s for s in scores if s is not None]
    data["summary"]["average_judge_score"] = round(sum(valid) / len(valid), 4) if valid else None
    data["summary"]["judge_score_count"] = len(valid)
    data["summary"]["judge_model"] = model
    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved to {results_path}  (avg={data['summary']['average_judge_score']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results.json")
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic"])
    parser.add_argument("--model", default="gpt-4o",
                        help="Model name, e.g. gpt-4o, claude-opus-4-6")
    parser.add_argument("--api_key", default=None,
                        help="API key (falls back to OPENAI_API_KEY / ANTHROPIC_API_KEY env vars)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get(
        "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
    )
    if not api_key:
        raise ValueError(f"No API key provided for {args.provider}.")

    run_judge(args.results, args.provider, args.model, api_key)
