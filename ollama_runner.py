#!/usr/bin/env python3
"""
AmnesiaBench Ollama Runner — Run binary-search context-window experiments
against local models via Ollama API.

Usage:
    python3 ollama_runner.py --model qwen3:32b --problem crt_three_congruences
    python3 ollama_runner.py --model qwen3:32b --problem-type arc --max-problems 5
    python3 ollama_runner.py --model qwen3:32b --list-problems

Author: Sherlock (2026-04-04)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import urllib.request
import urllib.error

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
ARC_PROBLEMS = REPO_ROOT / "amnesia_bench" / "arc_problems.json"

# Add amnesia_bench so we can import evaluator
sys.path.insert(0, str(REPO_ROOT / "amnesia_bench"))
from arc_evaluator import evaluate_arc_answer


# ── Built-in problems (math/number theory) ────────────────────────────────────

MATH_PROBLEMS = {
    "crt_three_congruences": {
        "problem_id": "crt_three_congruences",
        "problem_text": (
            "Find the smallest positive integer $n$ satisfying all three "
            "congruences simultaneously:\n"
            "$$n \\equiv 3 \\pmod{7}$$\n"
            "$$n \\equiv 5 \\pmod{11}$$\n"
            "$$n \\equiv 8 \\pmod{13}$$"
        ),
        "correct_answer": 346,
        "topic": "number_theory",
    },
    "digit_sum_ten": {
        "problem_id": "digit_sum_ten",
        "problem_text": (
            "How many three-digit positive integers have digits that sum to 10?"
        ),
        "correct_answer": 54,
        "topic": "combinatorics",
    },
}

# Load AIMO3 problems from existing results (extract problem text from conversations)
def _load_aimo3_problems() -> dict:
    """Scan results/ for aimo3 problems and extract problem text from conversations."""
    probs = {}
    for fn in sorted(RESULTS_DIR.iterdir()):
        if "aimo3" in fn.name and fn.name.endswith("_Compact.json"):
            try:
                with open(fn) as f:
                    data = json.load(f)
                pid = data["problem_id"]
                if pid in probs:
                    continue
                # Extract problem text from first trial's conversation
                conv = data["binary_search"][0]["trials"][0]["conversation"]
                user_msg = next((m["content"] for m in conv if m["role"] == "user"), None)
                correct = data["binary_search"][0]["trials"][0].get("correct_answer")
                if user_msg and correct is not None:
                    probs[pid] = {
                        "problem_id": pid,
                        "problem_text": user_msg,
                        "correct_answer": correct,
                        "topic": "aimo3",
                    }
            except (json.JSONDecodeError, KeyError, StopIteration):
                continue
    return probs


def load_all_problems() -> dict:
    """Load all available problems."""
    all_probs = dict(MATH_PROBLEMS)

    # Load AIMO3 from results
    all_probs.update(_load_aimo3_problems())

    # Load ARC problems
    if ARC_PROBLEMS.exists():
        with open(ARC_PROBLEMS) as f:
            arc = json.load(f)
        for p in arc:
            all_probs[p["problem_id"]] = p

    return all_probs


# ── Ollama API ────────────────────────────────────────────────────────────────

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def ollama_generate(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> dict:
    """Single-turn generation via Ollama /api/chat endpoint."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read().decode())
    wall_time = time.time() - start

    msg = result.get("message", {})
    response_text = msg.get("content", "")
    thinking_text = msg.get("thinking", "") if "thinking" in msg else ""

    # Token counts from Ollama
    prompt_tokens = result.get("prompt_eval_count", 0)
    eval_tokens = result.get("eval_count", 0)
    total_tokens = prompt_tokens + eval_tokens

    return {
        "response": response_text,
        "thinking": thinking_text,
        "prompt_tokens": prompt_tokens,
        "eval_tokens": eval_tokens,
        "total_tokens": total_tokens,
        "wall_time_s": wall_time,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def extract_boxed_answer(text: str) -> Optional[int]:
    """Extract integer from \\boxed{...}."""
    # Try \\boxed{N} first
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if matches:
        try:
            return int(matches[-1].strip().replace(",", ""))
        except ValueError:
            pass
    # Try just a standalone number at the end
    matches = re.findall(r"\b(\d+)\b", text)
    if matches:
        try:
            return int(matches[-1])
        except ValueError:
            pass
    return None


def evaluate_math_answer(response: str, correct_answer: int) -> dict:
    """Evaluate a math problem response."""
    predicted = extract_boxed_answer(response)
    return {
        "correct": predicted == correct_answer,
        "predicted": predicted,
        "expected": correct_answer,
    }


# ── System prompts ────────────────────────────────────────────────────────────

def make_system_prompt(token_limit: int, compaction: bool, topic: str) -> str:
    """Build the system prompt for a given config."""
    if topic == "arc":
        # ARC uses its own system prompt
        from arc_prompts import ARC_SYSTEM_PROMPT_SIMPLE
        base = ARC_SYSTEM_PROMPT_SIMPLE
        if compaction:
            base += (
                "\n\nYour context window is limited. "
                "To compact, write:\n<compact>your summary here</compact>\n\n"
                "When you call compact, the conversation resets to:\n"
                "  [this system prompt] + [the problem] + [your summary]\n"
                "You may compact at most 5 times."
            )
        return base

    # Math / number theory system prompt
    base = "You are a mathematical problem solver."
    if compaction:
        base += (
            f"\nYour context window is {token_limit} tokens total. "
            "If you exceed it without compacting, you FAIL with score 0.\n"
            "You do not have access to any tools except compact.\n\n"
            "To compact, write:\n<compact>your summary here</compact>\n\n"
            "When you call compact, the conversation resets to:\n"
            "  [this system prompt] + [the problem] + [your summary]\n"
            f"You get a fresh {token_limit} budget, but the reset prompt eats into it.\n"
            "The compact call itself costs tokens. You may compact at most 5 times."
        )
    else:
        base += (
            f"\nYour context window is {token_limit} tokens total. "
            "Work within this limit."
        )
    base += "\n\nGive your final answer as \\boxed{integer}."
    return base


# ── Single trial ──────────────────────────────────────────────────────────────

def run_trial(
    model: str,
    problem: dict,
    token_limit: int,
    compaction: bool,
    trial_idx: int,
) -> dict:
    """Run a single trial of a problem at a given token limit."""
    topic = problem.get("topic", "math")
    is_arc = topic == "arc"

    system = make_system_prompt(token_limit, compaction, topic)
    user_msg = problem["problem_text"]

    # Estimate how many tokens we can generate
    # Rough: 4 chars per token for the prompt
    prompt_est = (len(system) + len(user_msg)) // 4
    max_gen = max(token_limit - prompt_est, 256)

    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    n_compactions = 0
    total_wall_time = 0.0

    # Simple single-turn for now (compaction would need multi-turn loop)
    result = ollama_generate(
        model=model,
        system=system,
        user=user_msg,
        max_tokens=max_gen,
        temperature=0.7,
    )

    total_wall_time = result["wall_time_s"]
    response_text = result["response"]
    thinking_text = result.get("thinking", "")
    full_text = thinking_text + "\n" + response_text if thinking_text else response_text

    conversation.append({
        "role": "assistant",
        "content": response_text,
        "thinking": thinking_text,
        "tokens": result["eval_tokens"],
        "prompt_tokens": result["prompt_tokens"],
        "total_tokens": result["total_tokens"],
    })

    # Evaluate
    if is_arc:
        eval_result = evaluate_arc_answer(full_text, problem["ground_truth"])
        success = eval_result["correct"]
        answer = None
    else:
        eval_result = evaluate_math_answer(full_text, problem["correct_answer"])
        success = eval_result["correct"]
        answer = eval_result["predicted"]

    return {
        "problem_id": problem["problem_id"],
        "correct_answer": problem.get("correct_answer"),
        "token_limit": token_limit,
        "compaction": compaction,
        "trial_idx": trial_idx,
        "success": success,
        "answer": answer,
        "total_tokens_peak": result["total_tokens"],
        "n_turns": 1,
        "n_compactions": n_compactions,
        "wall_time_s": round(total_wall_time, 2),
        "finish_reason": "stop",
        "conversation": conversation,
        "eval_detail": eval_result if is_arc else None,
    }


# ── Binary search ─────────────────────────────────────────────────────────────

def binary_search_window(
    model: str,
    problem: dict,
    compaction: bool = True,
    trials_per_step: int = 3,
    threshold: float = 0.6,  # fraction of trials that must succeed
    initial_window: int = 32768,
    min_window: int = 256,
    verbose: bool = True,
) -> dict:
    """Binary search for minimum context window size."""

    lo = min_window
    hi = initial_window
    steps = []
    min_success_window = None

    # First: verify the model can solve it at max window
    if verbose:
        print(f"\n{'='*60}")
        print(f"Problem: {problem['problem_id']}")
        print(f"Model: {model}")
        print(f"Config: {'Compact' if compaction else 'HardCut'}")
        print(f"{'='*60}")

    while lo < hi:
        mid = (lo + hi) // 2
        # Round to nearest 128 for cleaner numbers
        mid = max(min_window, (mid // 128) * 128)

        if verbose:
            print(f"\n  Window: {mid} tokens (range [{lo}, {hi}])")

        trial_results = []
        successes = 0
        for t in range(trials_per_step):
            if verbose:
                print(f"    Trial {t+1}/{trials_per_step}...", end=" ", flush=True)

            trial = run_trial(model, problem, mid, compaction, t)
            trial_results.append(trial)

            if trial["success"]:
                successes += 1
                if verbose:
                    print(f"✅ (answer={trial['answer']}, {trial['wall_time_s']:.1f}s)")
            else:
                if verbose:
                    print(f"❌ (answer={trial['answer']}, {trial['wall_time_s']:.1f}s)")

        success_rate = successes / trials_per_step
        steps.append({
            "window": mid,
            "trials": trial_results,
            "success_rate": success_rate,
        })

        if verbose:
            print(f"    → {successes}/{trials_per_step} = {success_rate:.0%}")

        if success_rate >= threshold:
            hi = mid
            min_success_window = mid
        else:
            lo = mid + 128  # step up by at least 128

        # Convergence check
        if hi - lo < 128:
            break

    result = {
        "problem_id": problem["problem_id"],
        "model": model,
        "config": {
            "compaction": compaction,
            "name": "Compact" if compaction else "HardCut",
        },
        "binary_search": steps,
        "minimum_window": min_success_window,
        "search_range_final": [lo, hi],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if verbose:
        print(f"\n  Result: minimum window = {min_success_window}")
        print(f"  Final range: [{lo}, {hi}]")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AmnesiaBench Ollama Runner")
    parser.add_argument("--model", default="qwen3:32b", help="Ollama model name")
    parser.add_argument("--problem", help="Specific problem ID to run")
    parser.add_argument("--problem-type", help="Run all problems of a type (arc, aimo3, math)")
    parser.add_argument("--max-problems", type=int, default=999, help="Max problems to run")
    parser.add_argument("--trials", type=int, default=3, help="Trials per binary search step")
    parser.add_argument("--threshold", type=float, default=0.6, help="Success threshold")
    parser.add_argument("--initial-window", type=int, default=32768, help="Starting window size")
    parser.add_argument("--no-compact", action="store_true", help="Disable compaction (HardCut mode)")
    parser.add_argument("--list-problems", action="store_true", help="List available problems")
    parser.add_argument("--single-shot", action="store_true", help="Just run one trial, no binary search")
    parser.add_argument("--window", type=int, default=32768, help="Token window for single-shot")

    args = parser.parse_args()
    all_problems = load_all_problems()

    if args.list_problems:
        print(f"Available problems ({len(all_problems)}):\n")
        by_topic = {}
        for pid, p in sorted(all_problems.items()):
            topic = p.get("topic", "unknown")
            by_topic.setdefault(topic, []).append(pid)
        for topic in sorted(by_topic):
            print(f"  [{topic}]")
            for pid in by_topic[topic]:
                print(f"    {pid}")
        return

    # Select problems
    problems = []
    if args.problem:
        if args.problem not in all_problems:
            print(f"Unknown problem: {args.problem}")
            print(f"Use --list-problems to see available problems")
            sys.exit(1)
        problems = [all_problems[args.problem]]
    elif args.problem_type:
        for pid, p in sorted(all_problems.items()):
            if p.get("topic", "") == args.problem_type or pid.startswith(args.problem_type):
                problems.append(p)
        problems = problems[:args.max_problems]
    else:
        print("Specify --problem or --problem-type (or --list-problems)")
        sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Problems: {len(problems)}")
    print(f"Config: {'HardCut' if args.no_compact else 'Compact'}")

    compaction = not args.no_compact

    if args.single_shot:
        # Quick single trial
        for p in problems:
            print(f"\n--- {p['problem_id']} ---")
            trial = run_trial(args.model, p, args.window, compaction, 0)
            print(f"  Success: {trial['success']}")
            print(f"  Answer: {trial['answer']}")
            print(f"  Tokens: {trial['total_tokens_peak']}")
            print(f"  Time: {trial['wall_time_s']:.1f}s")
        return

    # Full binary search
    RESULTS_DIR.mkdir(exist_ok=True)
    model_safe = args.model.replace("/", "_").replace(":", "_")

    for p in problems:
        result = binary_search_window(
            model=args.model,
            problem=p,
            compaction=compaction,
            trials_per_step=args.trials,
            threshold=args.threshold,
            initial_window=args.initial_window,
        )

        # Save results
        config_name = "Compact" if compaction else "HardCut"
        fn = f"{model_safe}_{p['problem_id']}_{config_name}.json"
        out_path = RESULTS_DIR / fn
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved: {out_path}")

        # Also save summary
        summary = {
            "problem_id": p["problem_id"],
            "model": args.model,
            "config": config_name,
            "minimum_window": result["minimum_window"],
            "search_range_final": result["search_range_final"],
            "steps": len(result["binary_search"]),
        }
        sfn = f"{model_safe}_{p['problem_id']}_summary.json"
        sout = RESULTS_DIR / sfn
        # Append to summary if exists
        existing = []
        if sout.exists():
            with open(sout) as f:
                existing = json.load(f)
        existing.append(summary)
        with open(sout, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  Summary: {sout}")


if __name__ == "__main__":
    main()
