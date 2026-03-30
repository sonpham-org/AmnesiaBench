# Author: Claude Sonnet 4.6 (Bubba)
# Date: 29-March-2026
# PURPOSE: Evaluation job for AmnesiaBench v3. Implements nested binary search to find
#   n_reliable — the smallest context window N where the model can solve a problem at
#   >=66.7% success rate (2/3 passes). Saves results to {model}_{problem}_evaluation.json.
#   Integration points: called by cli.py; imports clients, prompts, utils, backoff.
#   Checks prediction file first — if attempt=False, skips evaluation entirely.
#   Resume-friendly: skips if evaluation file already exists.
# SRP/DRY check: Pass — binary search logic is isolated here; no prediction or scoring
#   code. Prompt construction delegates to prompts.py.

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .backoff import ResumptionQueue
from .prompts import build_evaluation_prompt
from .utils import extract_final_answer, prediction_filename, evaluation_filename

_PACKAGE_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = _PACKAGE_DIR.parent / "results"

# Search parameters
OUTER_CHECKS_PER_N = 1          # single trial in outer search
INNER_CHECKS_PER_N = 3          # 3 trials in inner search
INNER_PASS_THRESHOLD = 2        # need 2/3 = 66.7% in inner search
OUTER_STOP_RATIO = 0.05         # outer stops when step < 5% of current N
INNER_STOP_ABS = 1              # inner stops when hi - lo <= 1 token
N_MIN = 1
TEMPERATURE = 0.7


def run_evaluation(
    client,
    model_name: str,
    problem: dict,
    context_max: int,
    results_dir: Optional[Path] = None,
    queue: Optional[ResumptionQueue] = None,
    force: bool = False,
) -> dict:
    """
    Run the nested binary search evaluation for one (model, problem) pair.

    Flow:
      1. Check prediction file — if attempt=False, skip.
      2. Test at full context_max (unbounded). If fails → n_while_unbounded=None.
      3. Outer binary search: find the rough transition point.
      4. Inner binary search: 3-trial refinement around the transition.
      5. Save and return result.

    Result schema:
    {
        "model_name": str,
        "problem_id": str,
        "timestamp": ISO-8601 str,
        "n_while_unbounded": int or null,
        "n_reliable": int or null,
        "outer_search_log": [...],
        "inner_search_log": [...],
        "search_range_final": [lo, hi],
        "total_api_calls": int,
        "total_input_tokens": int,
        "total_output_tokens": int,
        "total_cost_usd": float,
        "wall_time_s": float,
    }
    """
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    problem_id = problem["problem_id"]
    out_path = evaluation_filename(results_dir, model_name, problem_id)

    # Resume-friendly: check for completed or in-progress evaluation
    checkpoint = None
    if out_path.exists() and not force:
        existing = json.loads(out_path.read_text())
        status = existing.get("status", "completed")
        if status == "completed":
            print(f"  [evaluate] SKIP {model_name} / {problem_id} — completed: {out_path.name}")
            return existing
        elif status == "running":
            print(f"  [evaluate] RESUMING {model_name} / {problem_id} from checkpoint")
            checkpoint = existing

    # Check prediction — if attempt=False, skip evaluation
    pred_path = prediction_filename(results_dir, model_name, problem_id)
    if pred_path.exists():
        pred = json.loads(pred_path.read_text())
        if not pred.get("attempt", True):
            print(
                f"  [evaluate] SKIP {model_name} / {problem_id} "
                f"— prediction says attempt=False"
            )
            result = _build_result(
                model_name, problem_id,
                n_while_unbounded=None,
                n_reliable=None,
                outer_log=[],
                inner_log=[],
                search_range=[N_MIN, context_max],
                api_calls=0,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                wall_time=0.0,
                skipped=True,
            )
            out_path.write_text(json.dumps(result, indent=2))
            return result

    print(f"\n  [evaluate] {model_name} / {problem_id} | context_max={context_max}")

    t_start = time.time()
    state = {
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }

    problem_text = problem["problem_text"]
    ground_truth = str(problem.get("ground_truth", ""))

    # ── Resume from checkpoint if available ───────────────────────────────────
    if checkpoint:
        n_while_unbounded = checkpoint.get("n_while_unbounded")
        outer_log = checkpoint.get("outer_search_log", [])
        inner_log = checkpoint.get("inner_search_log", [])
        phase = checkpoint.get("phase", "outer")
        state["api_calls"] = checkpoint.get("total_api_calls", 0)
        state["input_tokens"] = checkpoint.get("total_input_tokens", 0)
        state["output_tokens"] = checkpoint.get("total_output_tokens", 0)

        if n_while_unbounded is None and phase == "unbounded":
            # Unbounded test hadn't passed yet — redo it
            pass  # fall through to unbounded test below
        elif phase == "outer":
            # Replay outer log to reconstruct lo/hi
            lo, hi = _replay_search_log(outer_log, N_MIN, n_while_unbounded or context_max)
            print(f"  [evaluate] Resumed outer search at [{lo}, {hi}] (step {len(outer_log)})")
            outer_log, transition_lo, transition_hi = _outer_binary_search(
                client, problem_text, ground_truth,
                lo=lo, hi=hi, state=state,
                existing_log=outer_log, checkpoint_path=out_path,
                checkpoint_data={"model_name": model_name, "problem_id": problem_id,
                                 "n_while_unbounded": n_while_unbounded},
            )
            # Continue to inner search
            mid = (transition_lo + transition_hi) // 2
            inner_lo = max(N_MIN, mid - (transition_hi - transition_lo) * 3 // 2)
            inner_hi = min(n_while_unbounded or context_max, mid + (transition_hi - transition_lo) * 3 // 2)
            inner_hi = max(inner_hi, transition_hi)
            print(f"  [evaluate] Inner binary search [{inner_lo}, {inner_hi}]")
            inner_log, n_reliable = _inner_binary_search(
                client, problem_text, ground_truth,
                lo=inner_lo, hi=inner_hi, state=state,
                existing_log=[], checkpoint_path=out_path,
                checkpoint_data={"model_name": model_name, "problem_id": problem_id,
                                 "n_while_unbounded": n_while_unbounded, "outer_search_log": outer_log},
            )
            wall_time = round(time.time() - t_start, 2)
            result = _build_result(
                model_name, problem_id,
                n_while_unbounded=n_while_unbounded, n_reliable=n_reliable,
                outer_log=outer_log, inner_log=inner_log,
                search_range=[inner_lo, inner_hi],
                api_calls=state["api_calls"], input_tokens=state["input_tokens"],
                output_tokens=state["output_tokens"], cost=0.0, wall_time=wall_time,
                status="completed",
            )
            out_path.write_text(json.dumps(result, indent=2))
            print(f"  [evaluate] DONE — n_reliable={n_reliable} | {wall_time}s → {out_path.name}")
            return result
        elif phase == "inner":
            # Replay inner log
            outer_log = checkpoint.get("outer_search_log", [])
            lo, hi = _replay_search_log(inner_log, checkpoint.get("inner_lo", N_MIN), checkpoint.get("inner_hi", context_max))
            print(f"  [evaluate] Resumed inner search at [{lo}, {hi}] (step {len(inner_log)})")
            inner_log, n_reliable = _inner_binary_search(
                client, problem_text, ground_truth,
                lo=lo, hi=hi, state=state,
                existing_log=inner_log, checkpoint_path=out_path,
                checkpoint_data={"model_name": model_name, "problem_id": problem_id,
                                 "n_while_unbounded": n_while_unbounded, "outer_search_log": outer_log},
            )
            wall_time = round(time.time() - t_start, 2)
            result = _build_result(
                model_name, problem_id,
                n_while_unbounded=n_while_unbounded, n_reliable=n_reliable,
                outer_log=outer_log, inner_log=inner_log,
                search_range=[lo, hi],
                api_calls=state["api_calls"], input_tokens=state["input_tokens"],
                output_tokens=state["output_tokens"], cost=0.0, wall_time=wall_time,
                status="completed",
            )
            out_path.write_text(json.dumps(result, indent=2))
            print(f"  [evaluate] DONE — n_reliable={n_reliable} | {wall_time}s → {out_path.name}")
            return result

    # ── Step 1: Unbounded test ────────────────────────────────────────────────
    print(f"  [evaluate] Unbounded test at N={context_max} ...")
    unbounded_pass, unbounded_log = _test_n(
        client, problem_text, ground_truth, context_max, n_trials=1, state=state
    )
    if not unbounded_pass:
        print(f"  [evaluate] UNSOLVABLE at context_max={context_max} — skipping search")
        result = _build_result(
            model_name, problem_id,
            n_while_unbounded=None,
            n_reliable=None,
            outer_log=unbounded_log,
            inner_log=[],
            search_range=[N_MIN, context_max],
            api_calls=state["api_calls"],
            input_tokens=state["input_tokens"],
            output_tokens=state["output_tokens"],
            cost=0.0,
            wall_time=round(time.time() - t_start, 2),
            status="completed",
        )
        out_path.write_text(json.dumps(result, indent=2))
        return result

    n_while_unbounded = context_max

    # ── Step 2: Outer binary search (with checkpointing) ─────────────────────
    print(f"  [evaluate] Outer binary search [{N_MIN}, {n_while_unbounded}] ...")
    outer_log, transition_lo, transition_hi = _outer_binary_search(
        client, problem_text, ground_truth,
        lo=N_MIN, hi=n_while_unbounded,
        state=state,
        checkpoint_path=out_path,
        checkpoint_data={"model_name": model_name, "problem_id": problem_id,
                         "n_while_unbounded": n_while_unbounded},
    )

    # ── Step 3: Inner binary search (with checkpointing) ─────────────────────
    mid = (transition_lo + transition_hi) // 2
    inner_lo = max(N_MIN, mid - (transition_hi - transition_lo) * 3 // 2)
    inner_hi = min(n_while_unbounded, mid + (transition_hi - transition_lo) * 3 // 2)
    inner_hi = max(inner_hi, transition_hi)

    print(
        f"  [evaluate] Inner binary search [{inner_lo}, {inner_hi}] "
        f"(3× expansion around transition [{transition_lo}, {transition_hi}])"
    )
    inner_log, n_reliable = _inner_binary_search(
        client, problem_text, ground_truth,
        lo=inner_lo, hi=inner_hi,
        state=state,
        checkpoint_path=out_path,
        checkpoint_data={"model_name": model_name, "problem_id": problem_id,
                         "n_while_unbounded": n_while_unbounded, "outer_search_log": outer_log},
    )

    wall_time = round(time.time() - t_start, 2)
    result = _build_result(
        model_name, problem_id,
        n_while_unbounded=n_while_unbounded,
        n_reliable=n_reliable,
        outer_log=outer_log,
        inner_log=inner_log,
        search_range=[inner_lo, inner_hi],
        api_calls=state["api_calls"],
        input_tokens=state["input_tokens"],
        output_tokens=state["output_tokens"],
        cost=0.0,
        wall_time=wall_time,
        status="completed",
    )

    out_path.write_text(json.dumps(result, indent=2))
    print(
        f"  [evaluate] DONE — n_reliable={n_reliable} | "
        f"api_calls={state['api_calls']} | {wall_time}s → {out_path.name}"
    )
    return result


# ─── Outer Binary Search ──────────────────────────────────────────────────────

def _replay_search_log(log: list, initial_lo: int, initial_hi: int) -> tuple:
    """Replay a search log to reconstruct current lo/hi state."""
    lo, hi = initial_lo, initial_hi
    for entry in log:
        n = entry["N"]
        if entry["passed"]:
            hi = n
        else:
            lo = n
    return lo, hi


def _write_checkpoint(
    checkpoint_path: Optional[Path],
    checkpoint_data: dict,
    phase: str,
    outer_log: list,
    inner_log: list,
    state: dict,
    extra: Optional[dict] = None,
):
    """Write a running checkpoint to disk after each search step."""
    if checkpoint_path is None:
        return
    data = {
        **checkpoint_data,
        "status": "running",
        "phase": phase,
        "outer_search_log": outer_log,
        "inner_search_log": inner_log,
        "total_api_calls": state["api_calls"],
        "total_input_tokens": state["input_tokens"],
        "total_output_tokens": state["output_tokens"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        data.update(extra)
    checkpoint_path.write_text(json.dumps(data, indent=2))


def _outer_binary_search(
    client, problem_text: str, ground_truth: str,
    lo: int, hi: int, state: dict,
    existing_log: Optional[list] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_data: Optional[dict] = None,
) -> tuple:
    """
    Find the rough fail→pass transition zone using a single trial per N.
    Stop when step < 5% of current N.
    Returns (log, transition_lo, transition_hi).
    Writes checkpoint after each step for resumability.
    """
    log = existing_log if existing_log else []
    step = len(log)
    last_fail_lo = lo
    first_pass_hi = hi

    while True:
        mid = (lo + hi) // 2
        if mid <= lo or mid >= hi:
            break
        step_size = hi - lo
        if step_size < OUTER_STOP_RATIO * mid:
            break

        step += 1
        print(f"  [outer step {step}] N={mid}  range=[{lo},{hi}]")
        passed, trial_log = _test_n(
            client, problem_text, ground_truth, mid,
            n_trials=OUTER_CHECKS_PER_N, state=state,
        )
        log.append({"N": mid, "passed": passed, "trials": trial_log})

        # Checkpoint after every step
        _write_checkpoint(
            checkpoint_path, checkpoint_data or {},
            phase="outer", outer_log=log, inner_log=[], state=state,
        )

        if passed:
            first_pass_hi = mid
            hi = mid
        else:
            last_fail_lo = mid
            lo = mid

    print(f"  [outer] transition zone: [{last_fail_lo}, {first_pass_hi}]")
    return log, last_fail_lo, first_pass_hi


# ─── Inner Binary Search ──────────────────────────────────────────────────────

def _inner_binary_search(
    client, problem_text: str, ground_truth: str,
    lo: int, hi: int, state: dict,
    existing_log: Optional[list] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_data: Optional[dict] = None,
) -> tuple:
    """
    Refine the transition zone using 3 trials per N; require 2/3 to pass.
    Stop when hi - lo <= 1 token.
    Returns (log, n_reliable) where n_reliable is the smallest passing N.
    Writes checkpoint after each step for resumability.
    """
    log = existing_log if existing_log else []
    step = len(log)
    n_reliable = hi  # conservative default

    while hi - lo > INNER_STOP_ABS:
        mid = (lo + hi) // 2
        if mid <= lo or mid >= hi:
            break

        step += 1
        print(f"  [inner step {step}] N={mid}  range=[{lo},{hi}]")
        passed, trial_log = _test_n(
            client, problem_text, ground_truth, mid,
            n_trials=INNER_CHECKS_PER_N, state=state,
            pass_threshold=INNER_PASS_THRESHOLD,
        )
        log.append({"N": mid, "passed": passed, "n_trials": INNER_CHECKS_PER_N, "trials": trial_log})

        # Checkpoint after every step
        _write_checkpoint(
            checkpoint_path, checkpoint_data or {},
            phase="inner", outer_log=(checkpoint_data or {}).get("outer_search_log", []),
            inner_log=log, state=state,
            extra={"inner_lo": lo, "inner_hi": hi},
        )

        if passed:
            n_reliable = mid
            hi = mid
        else:
            lo = mid

    print(f"  [inner] n_reliable={n_reliable}")
    return log, n_reliable


# ─── Single N Test ────────────────────────────────────────────────────────────

def _test_n(
    client, problem_text: str, ground_truth: str,
    N: int, n_trials: int, state: dict,
    pass_threshold: int = 1,
) -> tuple:
    """
    Run n_trials trials at context window N in parallel.
    Returns (passed: bool, trial_log: list).
    passed = True if successes >= pass_threshold.
    """
    results = [None] * n_trials

    def _run_one(idx):
        return _run_trial(client, problem_text, ground_truth, N, idx)

    with ThreadPoolExecutor(max_workers=n_trials) as pool:
        futures = {pool.submit(_run_one, i): i for i in range(n_trials)}
        for future in as_completed(futures):
            idx = futures[future]
            r = future.result()
            results[idx] = r
            status = "PASS" if r["success"] else "FAIL"
            print(
                f"    trial {idx}: {status} | ans={r['answer']!r} | "
                f"{r['finish_reason']} | {r['total_tokens']} tok | {r['wall_time_s']:.1f}s"
            )
            state["api_calls"] += 1
            state["input_tokens"] += r.get("input_tokens", 0)
            state["output_tokens"] += r.get("output_tokens", 0)

    n_pass = sum(1 for r in results if r["success"])
    passed = n_pass >= pass_threshold
    print(f"    [{n_pass}/{n_trials} passed — {'PASS' if passed else 'FAIL'}]")
    return passed, results


# ─── Single Trial ─────────────────────────────────────────────────────────────

def _run_trial(
    client, problem_text: str, ground_truth: str,
    N: int, trial_idx: int,
) -> dict:
    """Run one single evaluation trial at context window N."""
    t0 = time.time()
    prompt = build_evaluation_prompt(N, problem_text)
    messages = [{"role": "user", "content": prompt}]

    try:
        resp = client.generate(messages, max_tokens=N)
    except Exception as e:
        return {
            "trial_idx": trial_idx,
            "N": N,
            "success": False,
            "answer": None,
            "expected": ground_truth,
            "finish_reason": "error",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "wall_time_s": round(time.time() - t0, 2),
            "error": str(e),
        }

    raw = resp.get("content", "") or ""
    answer = extract_final_answer(raw)
    success = answer is not None and str(answer).strip() == str(ground_truth).strip()

    return {
        "trial_idx": trial_idx,
        "N": N,
        "success": success,
        "answer": answer,
        "expected": ground_truth,
        "finish_reason": resp.get("finish_reason", "unknown"),
        "input_tokens": resp.get("input_tokens", 0),
        "output_tokens": resp.get("output_tokens", 0),
        "total_tokens": resp.get("total_tokens", 0),
        "wall_time_s": round(time.time() - t0, 2),
    }


# ─── Result Builder ───────────────────────────────────────────────────────────

def _build_result(
    model_name: str,
    problem_id: str,
    n_while_unbounded,
    n_reliable,
    outer_log: list,
    inner_log: list,
    search_range: list,
    api_calls: int,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    wall_time: float,
    skipped: bool = False,
    status: str = "completed",
) -> dict:
    return {
        "status": status,
        "model_name": model_name,
        "problem_id": problem_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_while_unbounded": n_while_unbounded,
        "n_reliable": n_reliable,
        "outer_search_log": outer_log,
        "inner_search_log": inner_log,
        "search_range_final": search_range,
        "total_api_calls": api_calls,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_cost_usd": cost,
        "wall_time_s": wall_time,
        "skipped": skipped,
    }


def run_evaluations_for_problems(
    client,
    model_name: str,
    problems: list,
    context_max: int,
    results_dir: Optional[Path] = None,
    queue: Optional[ResumptionQueue] = None,
    force: bool = False,
) -> list:
    """Run evaluation job for a list of problems. Returns list of result dicts."""
    results = []
    for problem in problems:
        try:
            result = run_evaluation(
                client, model_name, problem, context_max,
                results_dir=results_dir,
                queue=queue,
                force=force,
            )
        except Exception as e:
            err = str(e)
            print(f"  [evaluate] FAILED {model_name} / {problem['problem_id']}: {err}")
            if queue:
                queue.push(model_name, problem["problem_id"], "evaluation", err)
            continue
        results.append(result)
    return results
