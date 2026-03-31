#!/usr/bin/env python3
# Author: Claude Sonnet 4.6 (Bubba)
# Date: 28-March-2026
# PURPOSE: AmnesiaBench v2 — multi-model, multi-problem benchmark runner that binary-searches
#   for the minimum context window (n_reliable) at which each LLM can solve competition-math
#   problems at 60% success rate. Supports 10 problems × N models for overnight runs.
#   Features: prediction phase, composite Scott scoring, --model / --model-name flags,
#   --run-all-models mode reading models.json, per-model result namespacing, full scoring table.
#   Supports llama.cpp (http://), Google Gemini (gemini://), OpenRouter (openrouter://),
#   and Anthropic OAuth (anthropic://) backends via create_client().
#   ARC puzzle support: uses arc_evaluator for grid answer evaluation, arc_prompts for system prompts.
#   Exponential backoff applied to all external API calls (429/503 retry with jitter).
#   Integration points: run_prediction_phase() → run_problem() → binary_search() → run_trial().
# SRP/DRY check: Pass — prediction phase, scoring, model iteration all isolated. No duplication
#   of result I/O. calculate_scores() is the single scoring engine. run_all_models() delegates
#   to run_problem() so the multi-model path is just a loop around the single-model path.
#   with_exponential_backoff() is the single retry engine used by both LLMClient and GeminiClient.
"""
AmnesiaBench v2 — How much context does a model actually need?

Binary-searches (log scale) for the minimum context window at which an LLM
can solve competition-math problems at a 60% success rate. Runs prediction
phase, computes composite Scott scores, supports multi-model overnight runs.
Supports both llama.cpp (http://) and Google Gemini (gemini://) backends.

Usage:
    # Single problem, single model (llama.cpp):
    python3 amnesia_bench.py --problem ab507a9f --model http://localhost:8080 --model-name Qwen35B

    # All problems, single model:
    python3 amnesia_bench.py --all --model http://localhost:8080 --model-name Qwen35B

    # All problems, Gemini:
    python3 amnesia_bench.py --all --model gemini://gemini-2.0-flash-lite --api-key $GEMINI_API_KEY

    # All problems, all models from models.json:
    python3 amnesia_bench.py --all --run-all-models

    # Analysis:
    python3 amnesia_bench.py --analyze
    python3 amnesia_bench.py --scores
"""

import argparse
import contextlib
import glob
import io
import json
import math
import os
import random
import re
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import requests

from arc.arc_evaluator import evaluate_arc_answer
from arc.arc_prompts import ARC_SYSTEM_PROMPT

# ─── Defaults ────────────────────────────────────────────────────────────────

SERVER_URL = "http://localhost:8080"
MIN_WINDOW = 1
MAX_WINDOW = 32768
TRIALS_PER_WINDOW = 1
SUCCESS_THRESHOLD = 0.6          # 60%
CONVERGENCE_RATIO = 1.05         # stop when hi/lo < 5% (fallback)
CONVERGENCE_ABS = 50             # stop when hi - lo < 50 tokens (primary)
MAX_COMPLETION_TOKENS = 16384    # hard cap per generation turn
MAX_COMPACTIONS = 5
COMPACTION_TRIGGER = 0.70
MAX_TURNS = 40
CODE_TIMEOUT = 30
TEMPERATURE = 0.7
PROBLEMS_DIR = Path(__file__).parent / "problems"
RESULTS_DIR = Path(__file__).parent / "results"
MODELS_JSON = Path(__file__).parent / "models.json"

DEFAULT_COMPACTION_PROMPT = "Compact your context window to continue."

# ─── Prompt Templates ────────────────────────────────────────────────────────

SYSTEM_HARD = """\
You are a mathematical problem solver.
Your context window is {token_limit} tokens total (this prompt + your output).
If you run out, generation stops and I take your last \\boxed{{}} answer.
You do not have access to any tools. Reason through the problem using only text.
Plan your reasoning to fit. Give your final answer as \\boxed{{integer}}."""

SYSTEM_COMPACT = """\
You are a mathematical problem solver.
Your context window is {token_limit} tokens total. If you exceed it without compacting, you FAIL with score 0.
You do not have access to any tools except compact.

To compact, write:
<compact>your summary here</compact>

When you call compact, the conversation resets to:
  [this system prompt] + [the problem] + [your summary]
You get a fresh {token_limit} budget, but the reset prompt eats into it.
The compact call itself costs tokens. You may compact at most {max_compactions} times.

Give your final answer as \\boxed{{integer}}."""

POST_COMPACT_USER = """\
{problem_text}

Your previous progress (from compact call):
---
{summary}
---
Continue solving. Give your final answer as \\boxed{{integer}}."""

PREDICTION_PROMPT = """\
You are about to solve a math problem under context window constraints.

--- PROBLEM ---
{problem_text}

--- SCORING ---
Your score depends on:
- n_reliable: minimum context window where you solve this 60%+ of the time
- n_reliable_prediction: your prediction of n_reliable (before we test)
- Lower n_reliable_prediction = better score IF correct; if too low and you fail = infinity penalty
- success_prediction: whether you can solve this at all

You may opt out (success_prediction=False) if you think you cannot solve this problem.

--- INSTRUCTIONS ---
Respond in 300 tokens or less. Include these tags:
<success_prediction>True or False</success_prediction>
<n_reliable_prediction>integer (tokens)</n_reliable_prediction>
<compaction_prompt>one sentence describing what to preserve when compacting</compaction_prompt>"""


# ─── Python Sandbox ──────────────────────────────────────────────────────────

class PythonSandbox:
    """In-process Python executor with persistent namespace."""

    def __init__(self, timeout: int = CODE_TIMEOUT):
        self.timeout = timeout
        self.namespace = {"__builtins__": __builtins__}

    def execute(self, code: str) -> str:
        stdout = io.StringIO()
        stderr = io.StringIO()

        def _alarm_handler(signum, frame):
            raise TimeoutError(f"Code execution exceeded {self.timeout}s")

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(self.timeout)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(code, self.namespace)
            out = stdout.getvalue()
            err = stderr.getvalue()
            result = out if out else "(no output)"
            if err:
                result += f"\nSTDERR: {err}"
            return result
        except TimeoutError as e:
            return f"Error: {e}"
        except Exception:
            return f"Error: {traceback.format_exc()}"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def reset(self):
        self.namespace = {"__builtins__": __builtins__}


# ─── Exponential Backoff ─────────────────────────────────────────────────────

def with_exponential_backoff(fn, max_retries=20, base_delay=2.0, max_delay=120.0):
    """
    Wrap any API call with exponential backoff on 429/503 errors.
    Respects Retry-After header when present.
    Uses full jitter: delay = min(base * 2^attempt + uniform(0,1), max_delay).
    Raises immediately on non-retriable errors or when retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except requests.HTTPError as e:
            if e.response.status_code in (429, 503) and attempt < max_retries - 1:
                # Respect Retry-After header if present
                retry_after = e.response.headers.get("Retry-After") or e.response.headers.get("x-ratelimit-reset-requests")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 2), max_delay)
                else:
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 2), max_delay)
                print(f"    [backoff] {e.response.status_code} — retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                raise


# ─── LLM Client ─────────────────────────────────────────────────────────────

class LLMClient:
    """Wrapper for llama.cpp or any OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, server_url: str = SERVER_URL, temperature: float = TEMPERATURE, api_key: str = None, model_name: str = None):
        self.server_url = server_url.rstrip("/")
        self.temperature = temperature
        self.model_name = model_name  # passed to API as model field (required by OpenRouter)
        self.auth_header = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def generate(self, messages: list[dict], max_tokens: int) -> dict:
        """
        Send messages to the model. Returns usage + content dict.
        Uses exponential backoff on 429/503 errors.
        """
        max_tokens = max(1, max_tokens)
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

        def _do_request():
            if self.model_name:
                payload["model"] = self.model_name
            headers = dict(self.auth_header)
            # Enable prompt caching for OpenRouter (reduces cost + latency on repeated prompts)
            if "openrouter.ai" in self.server_url:
                headers["X-OpenRouter-Cache"] = "true"
            resp = requests.post(
                f"{self.server_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=3600,
                stream=True,
            )
            resp.raise_for_status()
            return resp

        resp = with_exponential_backoff(_do_request)

        full_content = ""
        reasoning = ""
        content = ""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        finish_reason = "unknown"

        print("    [stream] ", end="", flush=True)
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            r_piece = delta.get("reasoning_content", "") or delta.get("reasoning", "") or ""
            c_piece = delta.get("content", "") or ""
            if r_piece:
                reasoning += r_piece
                sys.stdout.write(r_piece)
                sys.stdout.flush()
            if c_piece:
                content += c_piece
                sys.stdout.write(c_piece)
                sys.stdout.flush()
            finish_reason = choice.get("finish_reason") or finish_reason
            usage = chunk.get("usage", {})
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                total_tokens = usage.get("total_tokens", total_tokens)
        print()

        if reasoning:
            full_content = f"<think>\n{reasoning}\n</think>\n{content}"
        else:
            full_content = content

        return {
            "content": full_content,
            "reasoning_content": reasoning,
            "final_content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,
        }

    def ping(self) -> bool:
        # For remote APIs (OpenRouter, etc.) skip /health check — just assume reachable
        if "openrouter.ai" in self.server_url or "localhost" not in self.server_url and not self.server_url.startswith("http://192."):
            return True
        try:
            r = requests.get(f"{self.server_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


# ─── Gemini Client ───────────────────────────────────────────────────────────

class GeminiClient:
    """Client for Google Gemini API (gemini-2.0-flash-lite or similar).

    Accepts OpenAI-style message lists and converts them to Gemini's
    generateContent format. Returns the same dict shape as LLMClient.generate()
    so the rest of the benchmark code is backend-agnostic.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-lite",
        temperature: float = TEMPERATURE,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _convert_messages(self, messages: list[dict]) -> tuple[Optional[dict], list[dict]]:
        """Convert OpenAI-style messages to Gemini format.

        Returns (system_instruction, contents) where:
          - system_instruction is None or {"parts": [{"text": "..."}]}
          - contents is a list of {"role": "user"|"model", "parts": [{"text": "..."}]}
        """
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")

            if role == "system":
                system_instruction = {"parts": [{"text": text}]}
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})
            else:
                contents.append({"role": "user", "parts": [{"text": text}]})

        return system_instruction, contents

    def generate(self, messages: list[dict], max_tokens: int) -> dict:
        """
        Send messages to Gemini generateContent endpoint.
        Returns same dict format as LLMClient: content, prompt_tokens,
        completion_tokens, total_tokens, finish_reason.
        Uses exponential backoff on 429/503 errors.
        """
        max_tokens = max(1, max_tokens)
        system_instruction, contents = self._convert_messages(messages)

        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": self.temperature,
            },
        }
        if system_instruction is not None:
            payload["systemInstruction"] = system_instruction

        url = (
            f"{self.base_url}/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )

        def _do_request():
            resp = requests.post(url, json=payload, timeout=3600)
            resp.raise_for_status()
            return resp

        resp = with_exponential_backoff(_do_request)
        data = resp.json()

        # Parse response
        candidates = data.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts", [])
            content = "".join(p.get("text", "") for p in parts)
            finish_reason_raw = candidate.get("finishReason", "STOP")
            # Normalise Gemini finish reasons to llama.cpp style
            finish_reason_map = {
                "STOP": "stop",
                "MAX_TOKENS": "length",
                "SAFETY": "stop",
                "RECITATION": "stop",
                "OTHER": "stop",
            }
            finish_reason = finish_reason_map.get(finish_reason_raw, "stop")
        else:
            content = ""
            finish_reason = "stop"

        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = usage.get("totalTokenCount", prompt_tokens + completion_tokens)

        # Print a brief stream-alike indicator for consistency
        print(f"    [gemini] {completion_tokens} tokens | finish={finish_reason}")
        print(content[:120].replace("\n", " ") + ("..." if len(content) > 120 else ""))

        return {
            "content": content,
            "reasoning_content": "",
            "final_content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,
        }

    def ping(self) -> bool:
        """Health check — try a minimal generation."""
        try:
            resp = self.generate(
                messages=[{"role": "user", "content": "Say OK."}],
                max_tokens=10,
            )
            return bool(resp.get("content"))
        except Exception:
            return False


# ─── Anthropic OAuth Client ──────────────────────────────────────────────────

class AnthropicOAuthClient:
    """Client for Anthropic API using OAuth tokens (sk-ant-oat prefix).

    Uses ANTHROPIC_OAUTHTOKEN env var. Requires anthropic-beta header.
    System prompt is always "You are Claude Code, Anthropic's official CLI for Claude."
    ARC/AIMO strategies go in the user message, NOT in system.
    Does NOT pass temperature parameter (omit entirely — 0.0 gets rejected).
    """

    FIXED_SYSTEM = "You are Claude Code, Anthropic's official CLI for Claude."

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.token = os.environ.get("ANTHROPIC_OAUTHTOKEN")
        if not self.token:
            raise ValueError(
                "AnthropicOAuthClient requires ANTHROPIC_OAUTHTOKEN env var (sk-ant-oat prefix)."
            )
        self.base_url = "https://api.anthropic.com/v1/messages"

    def generate(self, messages: list[dict], max_tokens: int) -> dict:
        """Send messages to Anthropic messages API. Returns same dict shape as LLMClient."""
        max_tokens = max(1, max_tokens)

        # Convert OpenAI-style messages to Anthropic format.
        # System goes in top-level 'system' field, not in messages array.
        anthropic_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role == "system":
                # Skip — we use fixed system prompt
                continue
            elif role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": text})
            else:
                anthropic_messages.append({"role": "user", "content": text})

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": self.FIXED_SYSTEM,
            "messages": anthropic_messages,
            # NOTE: temperature intentionally omitted — 0.0 gets rejected by Anthropic
        }

        headers = {
            "Authorization": f"Bearer {self.token}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "oauth-2025-04-20",
            "content-type": "application/json",
        }

        def _do_request():
            resp = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=3600,
            )
            resp.raise_for_status()
            return resp

        resp = with_exponential_backoff(_do_request)
        data = resp.json()

        # Extract text from content[0].text
        content_blocks = data.get("content", [])
        content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        stop_reason = data.get("stop_reason", "end_turn")
        finish_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        finish_reason = finish_reason_map.get(stop_reason, "stop")

        print(f"    [anthropic] {completion_tokens} tokens | finish={finish_reason}")
        print(content[:120].replace("\n", " ") + ("..." if len(content) > 120 else ""))

        return {
            "content": content,
            "reasoning_content": "",
            "final_content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,
        }

    def ping(self) -> bool:
        """Health check — try a minimal generation."""
        try:
            resp = self.generate(
                messages=[{"role": "user", "content": "Say OK."}],
                max_tokens=10,
            )
            return bool(resp.get("content"))
        except Exception:
            return False


# ─── Client Factory ──────────────────────────────────────────────────────────

def create_client(
    server_url: str,
    api_key: str = None,
    model_name: str = None,
    temperature: float = TEMPERATURE,
) -> Union[LLMClient, GeminiClient]:
    """
    Create appropriate client based on server_url scheme.

    - gemini:// or google:// → GeminiClient
      model extracted from the URL path (e.g. gemini://gemini-2.0-flash-lite)
    - http:// or https://   → LLMClient (llama.cpp)

    api_key is required for GeminiClient. Raises ValueError if missing.
    """
    if server_url.startswith("gemini://") or server_url.startswith("google://"):
        # Extract model from URL: gemini://gemini-2.0-flash-lite → gemini-2.0-flash-lite
        scheme = "gemini://" if server_url.startswith("gemini://") else "google://"
        gemini_model = server_url[len(scheme):].strip("/") or "gemini-2.0-flash-lite"
        # Allow model_name override
        if model_name and not model_name.startswith("gemini"):
            # model_name is just a label — still use gemini_model for the actual API call
            pass
        if not api_key:
            raise ValueError(
                "GeminiClient requires an API key. Pass --api-key or set GEMINI_API_KEY env var."
            )
        return GeminiClient(api_key=api_key, model=gemini_model, temperature=temperature)
    elif server_url.startswith("openrouter://"):
        # openrouter://openai/gpt-oss-120b:free → https://openrouter.ai/api/v1
        or_model = server_url[len("openrouter://"):].strip("/")
        if not api_key:
            raise ValueError("OpenRouter requires an API key. Pass --api-key or set api_key_env.")
        return LLMClient(
            server_url="https://openrouter.ai/api",
            temperature=temperature,
            api_key=api_key,
            model_name=or_model,
        )
    elif server_url.startswith("anthropic://"):
        # anthropic://claude-sonnet-4-6 → model = claude-sonnet-4-6
        anthropic_model = server_url[len("anthropic://"):].strip("/") or "claude-sonnet-4-6"
        return AnthropicOAuthClient(model=anthropic_model)
    elif server_url.startswith("http"):
        return LLMClient(server_url=server_url, temperature=temperature)
    else:
        raise ValueError(
            f"Unrecognised server URL scheme: '{server_url}'. "
            "Use http://, https://, openrouter://, anthropic://, gemini://, or google://"
        )


# ─── Parsing Helpers ─────────────────────────────────────────────────────────

def extract_python_blocks(text: str) -> list[str]:
    pattern = r"```python\s*\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)


def extract_compact_call(text: str) -> Optional[str]:
    match = re.search(r"<compact>(.*?)</compact>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_boxed_answer(text: str) -> Optional[int]:
    """Extract the last \\boxed{...} answer from text, ignoring <think> blocks."""
    non_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    target = non_think if non_think.strip() else text

    matches = re.findall(r"\\boxed\{([^{}]+)\}", target)
    if not matches:
        matches = re.findall(r"\\boxed\{(.+?)\}", target)
    if not matches:
        return None

    raw = matches[-1].strip()
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        f = float(raw)
        if f == int(f):
            return int(f)
    except ValueError:
        pass
    try:
        cleaned = raw.replace("^", "**").replace(",", "")
        return int(eval(cleaned))
    except Exception:
        pass
    return None


# ─── Prediction Phase ────────────────────────────────────────────────────────

def run_prediction_phase(
    client,
    problem: dict,
    max_tokens: int = 300,
) -> dict:
    """
    Ask the model to predict its own performance before testing begins.
    Returns parsed prediction dict with keys: success_prediction, n_reliable_prediction,
    compaction_prompt, raw_response. Falls back to safe defaults on parse failure.
    """
    problem_text = problem.get("problem_text", "")
    prompt = PREDICTION_PROMPT.format(problem_text=problem_text)
    messages = [{"role": "user", "content": prompt}]

    print(f"\n  [Prediction Phase] Asking model to predict performance...")
    try:
        resp = client.generate(messages, max_tokens=max_tokens)
    except Exception as e:
        print(f"  [Prediction Phase] API error: {e} — using defaults")
        return _prediction_defaults(raw_response=f"ERROR: {e}")

    raw = resp.get("content", "")
    completion_tokens = resp.get("completion_tokens", 0)

    if completion_tokens > max_tokens:
        print(f"  [Prediction Phase] Response too long ({completion_tokens} > {max_tokens}) — using defaults")
        return _prediction_defaults(raw_response=raw)

    success_match = re.search(
        r"<success_prediction>\s*(True|False)\s*</success_prediction>",
        raw, re.IGNORECASE
    )
    if not success_match:
        print("  [Prediction Phase] Missing <success_prediction> tag — using defaults")
        return _prediction_defaults(raw_response=raw)
    success_prediction = success_match.group(1).strip().lower() == "true"

    n_reliable_match = re.search(
        r"<n_reliable_prediction>\s*(\d+)\s*</n_reliable_prediction>",
        raw
    )
    if not n_reliable_match:
        print("  [Prediction Phase] Missing <n_reliable_prediction> tag — using defaults")
        return _prediction_defaults(raw_response=raw)
    n_reliable_prediction = int(n_reliable_match.group(1))

    compaction_match = re.search(
        r"<compaction_prompt>(.*?)</compaction_prompt>",
        raw, re.DOTALL
    )
    if not compaction_match:
        print("  [Prediction Phase] Missing <compaction_prompt> tag — using defaults")
        return _prediction_defaults(raw_response=raw)
    compaction_prompt = compaction_match.group(1).strip() or DEFAULT_COMPACTION_PROMPT

    print(f"  [Prediction Phase] success={success_prediction}, n_reliable={n_reliable_prediction}, compaction='{compaction_prompt[:60]}'")

    return {
        "success_prediction": success_prediction,
        "n_reliable_prediction": n_reliable_prediction,
        "compaction_prompt": compaction_prompt,
        "raw_response": raw,
    }


def _prediction_defaults(raw_response: str = "") -> dict:
    """Return safe prediction defaults (n_reliable=None means infinity)."""
    return {
        "success_prediction": True,
        "n_reliable_prediction": None,
        "compaction_prompt": DEFAULT_COMPACTION_PROMPT,
        "raw_response": raw_response,
    }


# ─── Scoring Engine ──────────────────────────────────────────────────────────

def calculate_scores(results_dir: Optional[Path] = None) -> None:
    """
    Load all per-model result files and compute Scott's composite benchmark scores.

    Scott's formula:
        Per-problem score:
            problem_score = baseline_n_reliable / n_reliable
            prediction_score = baseline_n_reliable_prediction / n_reliable_prediction

        Where baseline = lowest n_reliable (or n_reliable_prediction) across all models
        that solved that problem (i.e. the best-performing model sets the baseline).

        Composite scores:
            composite = mean(problem_scores over all solved problems)
            prediction_composite = mean(prediction_scores over all problems)

        Coverage = problems_attempted / problems_eligible
            eligible = problems where model context_max >= baseline_n_reliable

        Accuracy = problems_solved / problems_attempted

        Prediction accuracy = correct_success_predictions / total_problems
            (correct = predicted True and solved, OR predicted False and unsolvable)

        Final score = composite * prediction_composite * coverage * accuracy
                    * prediction_accuracy * 10000

        NOTE: FLOPs not tracked yet — omitted from scoring, noted in output.

    Prints both a per-problem table and a per-model composite score table.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    results_dir = Path(results_dir)

    # ── Load all per-config result files (not summary files) ──
    # File naming: results/{model_name}_{problem_id}_{config}.json
    # or legacy: results/{problem_id}_{config}.json (no model prefix)
    result_files = sorted(results_dir.glob("*.json"))
    result_files = [f for f in result_files if not f.name.endswith("_summary.json")]

    if not result_files:
        print("No result files found. Run experiments first.")
        return

    # Structure: {model_name: {problem_id: {config_name: result_dict}}}
    by_model: dict[str, dict[str, dict[str, dict]]] = {}

    for rf in result_files:
        try:
            data = json.loads(rf.read_text())
        except Exception as e:
            print(f"  [scores] Could not read {rf.name}: {e}")
            continue

        model_name = data.get("model_name") or data.get("model") or "unknown"
        pid = data.get("problem_id", rf.stem)
        config = data.get("config", {})
        config_name = config.get("name", "unknown") if isinstance(config, dict) else str(config)

        by_model.setdefault(model_name, {}).setdefault(pid, {})[config_name] = data

    if not by_model:
        print("No parseable result files found.")
        return

    all_problem_ids = sorted({pid for m in by_model.values() for pid in m})

    # ── Load fixed baselines from problem JSON files (gptoss_120b_correct_token_avg) ──
    # This is the GPT-4o 120B reference — fixed, not dynamic across models.
    # Using a fixed baseline ensures scores are stable and comparable across runs.
    baseline_n_reliable: dict[str, Optional[int]] = {}
    baseline_n_pred: dict[str, Optional[int]] = {}

    problems_dir = Path(__file__).parent / "problems"
    for pid in all_problem_ids:
        prob_file = problems_dir / f"{pid}.json"
        if prob_file.exists():
            try:
                prob_data = json.loads(prob_file.read_text())
                token_avg = prob_data.get("gptoss_120b_correct_token_avg")
                if token_avg is not None:
                    baseline_n_reliable[pid] = int(token_avg)
                    baseline_n_pred[pid] = int(token_avg)
                else:
                    baseline_n_reliable[pid] = None
                    baseline_n_pred[pid] = None
            except Exception:
                baseline_n_reliable[pid] = None
                baseline_n_pred[pid] = None
        else:
            baseline_n_reliable[pid] = None
            baseline_n_pred[pid] = None

    # ── Per-problem detail table ──
    print(f"\n{'='*110}")
    print(f"  AmnesiaBench v2 — Per-Problem Detail")
    print(f"{'='*110}")
    print(f"{'Model':<25} {'Problem':<28} {'Config':<22} {'MinWin':>7} {'Baseline':>8} {'ProbScore':>10} {'N_Pred':>8} {'PredScore':>10}")
    print(f"{'-'*110}")

    # ── Per-model composite score computation ──
    model_scores = {}

    for model_name in sorted(by_model.keys()):
        model_data = by_model[model_name]
        problem_scores = []
        prediction_scores = []
        total_problems = len(all_problem_ids)
        problems_attempted = 0
        problems_solved = 0
        problems_eligible = 0
        correct_success_preds = 0

        for pid in all_problem_ids:
            if pid not in model_data:
                continue

            baseline = baseline_n_reliable.get(pid)
            base_pred = baseline_n_pred.get(pid)

            # Count as eligible if baseline exists (some model solved it)
            if baseline is not None:
                problems_eligible += 1

            # Use the best config for this problem (lowest minimum_window)
            configs_for_pid = model_data[pid]
            best_result = None
            best_mw = None
            for config_name, result in configs_for_pid.items():
                mw = result.get("minimum_window")
                if mw is not None:
                    if best_mw is None or mw < best_mw:
                        best_mw = mw
                        best_result = result

            if best_result is None:
                # Model didn't solve this problem in any config
                pred = list(configs_for_pid.values())[0].get("prediction", {}) or {}
                success_pred = pred.get("success_prediction", True)
                if not success_pred and baseline is None:
                    correct_success_preds += 1  # correctly predicted failure
                # Still attempted
                problems_attempted += 1
                continue

            problems_attempted += 1
            problems_solved += 1

            # Problem score
            if baseline is not None and best_mw is not None:
                prob_score = baseline / best_mw
            else:
                prob_score = 0.0
            problem_scores.append(prob_score)

            # Prediction score
            pred = best_result.get("prediction", {}) or {}
            n_pred_val = pred.get("n_reliable_prediction")
            success_pred = pred.get("success_prediction", True)

            if success_pred:
                correct_success_preds += 1  # correctly predicted success (and solved)

            if n_pred_val is not None and base_pred is not None and n_pred_val > 0:
                pred_score = base_pred / n_pred_val
            else:
                pred_score = 0.0
            prediction_scores.append(pred_score)

            prob_score_str = f"{prob_score:.3f}"
            pred_score_str = f"{pred_score:.3f}" if n_pred_val is not None else "N/A"
            baseline_str = str(baseline) if baseline is not None else "N/A"
            n_pred_str = str(n_pred_val) if n_pred_val is not None else "inf"

            # Use config name from best result
            cfg = best_result.get("config", {})
            cfg_name = cfg.get("name", "unknown") if isinstance(cfg, dict) else str(cfg)

            print(
                f"{model_name:<25} {pid:<28} {cfg_name:<22} {str(best_mw):>7} "
                f"{baseline_str:>8} {prob_score_str:>10} {n_pred_str:>8} {pred_score_str:>10}"
            )

        # ── Composite scores ──
        composite = sum(problem_scores) / len(problem_scores) if problem_scores else 0.0
        pred_composite = sum(prediction_scores) / len(prediction_scores) if prediction_scores else 0.0
        # Coverage: fraction of eligible problems the model attempted (capped at 1.0)
        coverage = min(1.0, problems_solved / problems_eligible) if problems_eligible > 0 else 0.0
        accuracy = problems_solved / problems_attempted if problems_attempted > 0 else 0.0
        pred_accuracy = correct_success_preds / total_problems if total_problems > 0 else 0.0

        final_score = composite * pred_composite * coverage * accuracy * pred_accuracy * 10000

        model_scores[model_name] = {
            "composite": composite,
            "pred_composite": pred_composite,
            "coverage": coverage,
            "accuracy": accuracy,
            "pred_accuracy": pred_accuracy,
            "final_score": final_score,
            "problems_attempted": problems_attempted,
            "problems_solved": problems_solved,
            "problems_eligible": problems_eligible,
            "total_problems": total_problems,
        }

    # ── Per-model composite table ──
    print(f"\n{'='*100}")
    print(f"  AmnesiaBench v2 — Composite Scores (Scott's Formula)")
    print(f"  NOTE: FLOPs not tracked — omitted from scoring.")
    print(f"{'='*100}")
    print(f"{'Model':<25} {'Composite':>10} {'PredComp':>10} {'Coverage':>9} {'Accuracy':>9} {'PredAcc':>8} {'FinalScore':>12}")
    print(f"{'-'*100}")

    for model_name in sorted(model_scores.keys()):
        s = model_scores[model_name]
        print(
            f"{model_name:<25} "
            f"{s['composite']:>10.4f} "
            f"{s['pred_composite']:>10.4f} "
            f"{s['coverage']:>9.3f} "
            f"{s['accuracy']:>9.3f} "
            f"{s['pred_accuracy']:>8.3f} "
            f"{s['final_score']:>12.2f}"
        )
    print(f"{'='*100}")
    print(f"\nFormula: final_score = composite × pred_composite × coverage × accuracy × pred_accuracy × 10000")
    print(f"  composite = mean(baseline_n_reliable / model_n_reliable) over solved problems")
    print(f"  pred_composite = mean(baseline_n_pred / model_n_pred) over all problems")
    print(f"  coverage = attempted / eligible (eligible: baseline exists for problem)")
    print(f"  accuracy = solved / attempted")
    print(f"  pred_accuracy = correct_success_predictions / total_problems\n")


# ─── Single Trial ─────────────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str
    content: str
    tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    code_executed: Optional[str] = None
    code_output: Optional[str] = None
    compact_summary: Optional[str] = None


@dataclass
class TrialResult:
    problem_id: str
    correct_answer: object  # int for math, list for ARC
    token_limit: int
    tir: bool
    compaction: bool
    trial_idx: int
    success: bool
    answer: object  # Optional[int] for math, str marker for ARC
    total_tokens_peak: int
    n_turns: int
    n_compactions: int
    n_code_calls: int
    n_code_errors: int
    wall_time_s: float
    error: Optional[str]
    finish_reason: str
    conversation: list = field(default_factory=list)


def run_trial(
    client,
    problem_id: str,
    problem_text: str,
    correct_answer,
    token_limit: int,
    tir: bool,
    compaction: bool,
    trial_idx: int,
    compaction_hint: str = "",
    topic: str = "math",
) -> TrialResult:
    t0 = time.time()
    sandbox = PythonSandbox() if tir else None
    conversation: list[Turn] = []
    messages: list[dict] = []
    n_compactions = 0
    n_code_calls = 0
    n_code_errors = 0
    peak_tokens = 0
    error_msg = None
    finish = "max_turns"

    active_compaction_hint = compaction_hint.strip() if compaction_hint else DEFAULT_COMPACTION_PROMPT

    is_arc = (topic == "arc")
    is_anthropic = isinstance(client, AnthropicOAuthClient)

    if is_arc:
        if is_anthropic:
            # Anthropic OAuth: fixed system prompt, ARC strategy in user message
            sys_prompt = AnthropicOAuthClient.FIXED_SYSTEM
            user_content = ARC_SYSTEM_PROMPT + "\n\n" + problem_text
        else:
            # All other providers: ARC system prompt, problem in user message
            sys_prompt = ARC_SYSTEM_PROMPT
            user_content = problem_text
    else:
        # AIMO3 math — existing behavior
        if compaction:
            sys_prompt = SYSTEM_COMPACT.format(
                token_limit=token_limit, max_compactions=MAX_COMPACTIONS
            )
        else:
            sys_prompt = SYSTEM_HARD.format(token_limit=token_limit)
        user_content = problem_text

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]
    conversation.append(Turn(role="system", content=sys_prompt))
    conversation.append(Turn(role="user", content=user_content))

    for turn_i in range(MAX_TURNS):
        if peak_tokens > 0:
            remaining = token_limit - peak_tokens
        else:
            remaining = token_limit

        if remaining <= 0:
            finish = "budget_exceeded" if compaction else "truncated"
            break

        capped_tokens = min(remaining, MAX_COMPLETION_TOKENS)
        try:
            resp = client.generate(messages, max_tokens=capped_tokens)
        except Exception as e:
            error_msg = f"API error: {e}"
            finish = "error"
            break

        if resp["finish_reason"] in ("length", "truncated"):
            if is_arc:
                from arc.arc_evaluator import extract_all_numbered_answers
                if not extract_all_numbered_answers(resp["content"]):
                    finish = "truncated"
            elif extract_boxed_answer(resp["content"]) is None:
                finish = "truncated"

        content = resp["content"]
        total_now = resp["total_tokens"]
        peak_tokens = max(peak_tokens, total_now)

        turn = Turn(
            role="assistant",
            content=content,
            tokens=resp["completion_tokens"],
            prompt_tokens=resp["prompt_tokens"],
            total_tokens=total_now,
            finish_reason=resp["finish_reason"],
        )
        conversation.append(turn)

        if is_arc:
            arc_result = evaluate_arc_answer(content, correct_answer)
            if arc_result["correct"]:
                finish = "solved"
                break
            elif arc_result["num_answers_found"] > 0:
                # Model provided answer tags but got it wrong
                finish = "wrong_answer"
                break
        else:
            answer = extract_boxed_answer(content)
            if answer is not None:
                finish = "solved"
                break

        compact_summary = extract_compact_call(content) if compaction else None
        if compact_summary is not None:
            turn.compact_summary = compact_summary
            n_compactions += 1
            if n_compactions > MAX_COMPACTIONS:
                finish = "max_compactions"
                break
            hint_line = f"\nHint: {active_compaction_hint}" if active_compaction_hint else ""
            restart_user_msg = POST_COMPACT_USER.format(
                problem_text=problem_text + hint_line,
                summary=compact_summary,
            )
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": restart_user_msg},
            ]
            peak_tokens = 0
            conversation.append(Turn(role="user", content=f"[COMPACTION #{n_compactions} — context reset]"))
            continue

        if total_now >= token_limit:
            finish = "budget_exceeded" if compaction else "truncated"
            break

        code_blocks = extract_python_blocks(content) if tir else []
        if code_blocks:
            all_outputs = []
            for code in code_blocks:
                n_code_calls += 1
                output = sandbox.execute(code)
                if output.startswith("Error:"):
                    n_code_errors += 1
                all_outputs.append(output)
            combined_output = "\n---\n".join(all_outputs)
            if len(combined_output) > 2000:
                combined_output = combined_output[:2000] + "\n... (truncated)"
            code_turn = Turn(
                role="user",
                content=f"Code output:\n{combined_output}",
                code_executed="\n---\n".join(code_blocks),
                code_output=combined_output,
            )
            conversation.append(code_turn)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Code output:\n{combined_output}"})
            continue

        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": "Continue solving."})
        conversation.append(Turn(role="user", content="Continue solving."))

    # Extract final answer from conversation
    answer = None
    if is_arc:
        # For ARC, evaluate against ground truth grids
        for t in reversed(conversation):
            if t.role == "assistant":
                arc_result = evaluate_arc_answer(t.content, correct_answer)
                if arc_result["num_answers_found"] > 0:
                    answer = "arc_answer_found"
                    break
        elapsed = time.time() - t0
        # Re-evaluate the last assistant response for correctness
        success = False
        for t in reversed(conversation):
            if t.role == "assistant":
                arc_result = evaluate_arc_answer(t.content, correct_answer)
                success = arc_result["correct"]
                break
    else:
        for t in reversed(conversation):
            if t.role == "assistant":
                answer = extract_boxed_answer(t.content)
                if answer is not None:
                    break
        elapsed = time.time() - t0
        success = answer is not None and answer == correct_answer

    return TrialResult(
        problem_id=problem_id,
        correct_answer=correct_answer,
        token_limit=token_limit,
        tir=tir,
        compaction=compaction,
        trial_idx=trial_idx,
        success=success,
        answer=answer,
        total_tokens_peak=peak_tokens,
        n_turns=len([t for t in conversation if t.role == "assistant"]),
        n_compactions=n_compactions,
        n_code_calls=n_code_calls,
        n_code_errors=n_code_errors,
        wall_time_s=round(elapsed, 2),
        error=error_msg,
        finish_reason=finish,
        conversation=[asdict(t) for t in conversation],
    )


# ─── Binary Search ───────────────────────────────────────────────────────────

@dataclass
class WindowTest:
    window: int
    trials: list
    n_success: int
    n_trials: int
    pass_rate: float
    passed: bool


def binary_search(
    client,
    problem_id: str,
    problem_text: str,
    correct_answer,
    tir: bool,
    compaction: bool,
    min_window: int = MIN_WINDOW,
    max_window: int = MAX_WINDOW,
    trials: int = TRIALS_PER_WINDOW,
    compaction_hint: str = "",
    topic: str = "math",
) -> dict:
    config_name = f"{'Compact' if compaction else 'HardCut'}"
    print(f"\n{'='*60}")
    print(f"  {problem_id} | {config_name}")
    print(f"  Search range: [{min_window}, {max_window}]")
    print(f"{'='*60}")

    search_log: list[WindowTest] = []

    # Verify solvable at max window
    print(f"\n  [Verify] Testing max window = {max_window} ...")
    test = _test_window(
        client, problem_id, problem_text, correct_answer,
        max_window, tir, compaction, trials, compaction_hint, topic
    )
    search_log.append(test)
    print(f"  [Verify] {test.n_success}/{test.n_trials} passed ({test.pass_rate:.0%})")

    if not test.passed:
        print(f"  UNSOLVABLE at max window. Skipping binary search.")
        return _build_result(
            problem_id, tir, compaction, search_log,
            minimum_window=None,
            search_range_final=(min_window, max_window),
        )

    lo, hi = min_window, max_window
    step = 0
    while hi / lo > CONVERGENCE_RATIO and (hi - lo) > CONVERGENCE_ABS:
        step += 1
        mid = (lo + hi) // 2
        mid = max(min_window, max(1, (mid // 16) * 16))
        if mid == lo or mid == hi:
            break

        print(f"\n  [Step {step}] Testing window = {mid}  (range [{lo}, {hi}], gap {hi-lo}, ratio {hi/lo:.3f})")
        test = _test_window(
            client, problem_id, problem_text, correct_answer,
            mid, tir, compaction, trials, compaction_hint, topic
        )
        search_log.append(test)
        print(f"  [Step {step}] {test.n_success}/{test.n_trials} passed ({test.pass_rate:.0%}) → {'hi=mid' if test.passed else 'lo=mid'}")

        if test.passed:
            hi = mid
        else:
            lo = mid

    # Snap to nearest multiple of 16 (round up)
    snapped = ((hi + 15) // 16) * 16
    print(f"\n  RESULT: minimum window ≈ {hi} tokens → snapped to {snapped} (range [{lo}, {hi}])")
    return _build_result(
        problem_id, tir, compaction, search_log,
        minimum_window=snapped,
        search_range_final=(lo, hi),
    )


def _test_window(
    client, problem_id, problem_text, correct_answer,
    window, tir, compaction, n_trials,
    compaction_hint: str = "",
    topic: str = "math",
) -> WindowTest:
    t0 = time.time()

    def _run_one(i):
        return run_trial(
            client, problem_id, problem_text, correct_answer,
            token_limit=window, tir=tir, compaction=compaction,
            trial_idx=i, compaction_hint=compaction_hint,
            topic=topic,
        )

    trials_results = [None] * n_trials
    n_success = 0
    with ThreadPoolExecutor(max_workers=n_trials) as pool:
        futures = {pool.submit(_run_one, i): i for i in range(n_trials)}
        for future in as_completed(futures):
            i = futures[future]
            result = future.result()
            trials_results[i] = asdict(result)
            if result.success:
                n_success += 1
            status = "OK" if result.success else "FAIL"
            ans_str = f"ans={result.answer}" if result.answer is not None else "no answer"
            print(f"    trial {i}: {status} | {ans_str} | {result.finish_reason} | {result.total_tokens_peak} tok | {result.wall_time_s}s")

    elapsed = time.time() - t0
    pass_rate = n_success / n_trials
    print(f"    [{n_trials} trials in {elapsed:.1f}s wall, {n_success}/{n_trials} passed]")
    return WindowTest(
        window=window, trials=trials_results,
        n_success=n_success, n_trials=n_trials,
        pass_rate=pass_rate, passed=pass_rate >= SUCCESS_THRESHOLD,
    )


def _build_result(problem_id, tir, compaction, search_log, minimum_window, search_range_final):
    return {
        "problem_id": problem_id,
        "config": {
            "tir": tir,
            "compaction": compaction,
            "name": f"{'TIR' if tir else 'NoTIR'}_{'Compact' if compaction else 'HardCut'}",
        },
        "binary_search": [asdict(w) for w in search_log],
        "minimum_window": minimum_window,
        "search_range_final": list(search_range_final),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─── Problem Loading ─────────────────────────────────────────────────────────

def load_problem(problem_id: str) -> dict:
    """Load a problem JSON from problems/. Matches exact stem or substring."""
    path = PROBLEMS_DIR / f"{problem_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    for p in PROBLEMS_DIR.glob("*.json"):
        if problem_id in p.stem:
            return json.loads(p.read_text())
    raise FileNotFoundError(f"No problem matching '{problem_id}' in {PROBLEMS_DIR}")


def load_all_problems() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(PROBLEMS_DIR.glob("*.json"))]


# ─── Result Filename Helpers ─────────────────────────────────────────────────

def result_filename(model_name: str, problem_id: str, config_name: str) -> Path:
    """
    Build result file path for a given model/problem/config combination.
    Format: results/{model_name}_{problem_id}_{config_name}.json
    Model name is sanitized (spaces → underscores, slashes → dashes).
    """
    safe_model = re.sub(r"[^\w\-]", "_", model_name)
    return RESULTS_DIR / f"{safe_model}_{problem_id}_{config_name}.json"


def summary_filename(model_name: str, problem_id: str) -> Path:
    safe_model = re.sub(r"[^\w\-]", "_", model_name)
    return RESULTS_DIR / f"{safe_model}_{problem_id}_summary.json"


# ─── Single-Problem Runner ───────────────────────────────────────────────────

def run_problem(
    client,
    problem: dict,
    model_name: str = "unknown",
    configs: list[tuple[bool, bool]] = None,
    min_window: int = MIN_WINDOW,
    max_window: int = MAX_WINDOW,
    trials: int = TRIALS_PER_WINDOW,
):
    """
    Run binary search for all configs on one problem. Save per-config and summary results.
    Results namespaced by model_name to prevent multi-model collisions.
    """
    if configs is None:
        configs = [
            (False, False),  # NoTIR + HardCut
            (False, True),   # NoTIR + Compact
        ]

    pid = problem["problem_id"]
    RESULTS_DIR.mkdir(exist_ok=True)

    all_results = []
    for tir, compaction in configs:
        config_name = f"{'TIR' if tir else 'NoTIR'}_{'Compact' if compaction else 'HardCut'}"
        outpath = result_filename(model_name, pid, config_name)

        # Resume: skip if valid completed result exists
        if outpath.exists():
            try:
                existing = json.loads(outpath.read_text())
                if existing.get("minimum_window") is not None or existing.get("binary_search"):
                    print(f"\n  [SKIP] {model_name} | {pid} | {config_name} — result exists at {outpath.name}")
                    all_results.append(existing)
                    continue
            except Exception:
                pass

        # Prediction phase
        prediction = run_prediction_phase(client, problem, max_tokens=300)
        compaction_hint = prediction.get("compaction_prompt", DEFAULT_COMPACTION_PROMPT)

        if not prediction.get("success_prediction", True):
            print(f"\n  [Prediction Phase] Model opted out. Skipping binary search for {pid} | {config_name}.")
            result = _build_result(
                pid, tir, compaction, [],
                minimum_window=None,
                search_range_final=(min_window, max_window),
            )
            result["prediction"] = prediction
            result["model_name"] = model_name
            all_results.append(result)
            outpath.write_text(json.dumps(result, indent=2, default=str))
            print(f"\n  Saved (opt-out): {outpath.name}")
            continue

        # Binary search
        problem_topic = problem.get("topic", "math")
        result = binary_search(
            client,
            problem_id=pid,
            problem_text=problem["problem_text"],
            correct_answer=problem["ground_truth"],
            tir=tir,
            compaction=compaction,
            min_window=min_window,
            max_window=max_window,
            trials=trials,
            compaction_hint=compaction_hint,
            topic=problem_topic,
        )
        result["model_name"] = model_name
        result["prediction"] = prediction
        all_results.append(result)

        outpath.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n  Saved: {outpath.name}")

    # Save combined summary (compact, no conversation traces)
    summary = []
    for r in all_results:
        entry = {
            "model_name": model_name,
            "problem_id": r["problem_id"],
            "config": r["config"]["name"] if isinstance(r.get("config"), dict) else r.get("config"),
            "minimum_window": r["minimum_window"],
            "search_range_final": r.get("search_range_final"),
            "steps": len(r.get("binary_search", [])),
        }
        pred = r.get("prediction")
        if pred:
            entry["n_reliable_prediction"] = pred.get("n_reliable_prediction")
            entry["success_prediction"] = pred.get("success_prediction")
        summary.append(entry)

    sp = summary_filename(model_name, pid)
    sp.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary: {sp.name}")

    return all_results


# ─── Multi-Model Runner ──────────────────────────────────────────────────────

def load_models_json() -> list[dict]:
    """Load models.json from the AmnesiaBench directory. Returns list of {name, url} dicts."""
    if not MODELS_JSON.exists():
        raise FileNotFoundError(
            f"models.json not found at {MODELS_JSON}. "
            "Create it with a list of {{name, url}} entries."
        )
    models = json.loads(MODELS_JSON.read_text())
    if not isinstance(models, list) or not models:
        raise ValueError("models.json must be a non-empty list of {name, url} objects.")
    for m in models:
        if "name" not in m or "url" not in m:
            raise ValueError(f"Each model entry must have 'name' and 'url' keys. Got: {m}")
    return models


def run_all_models(
    problems: list[dict],
    configs: list[tuple[bool, bool]] = None,
    min_window: int = MIN_WINDOW,
    max_window: int = MAX_WINDOW,
    trials: int = TRIALS_PER_WINDOW,
    temperature: float = TEMPERATURE,
    cli_api_key: str = None,
):
    """
    Iterate over all models in models.json, run all problems for each model.
    Models are run sequentially (one model at a time, all problems per model).
    If a model's server is unreachable, it is skipped with a warning.
    Supports api_key_env field in models.json for Gemini-style API key lookup.
    """
    models = load_models_json()
    print(f"\n{'#'*70}")
    print(f"  --run-all-models: {len(models)} model(s) × {len(problems)} problem(s)")
    for m in models:
        print(f"    {m['name']} → {m['url']}")
    print(f"{'#'*70}\n")

    for model_entry in models:
        mname = model_entry["name"]
        murl = model_entry["url"]
        print(f"\n{'#'*70}")
        print(f"  MODEL: {mname}")
        print(f"  URL:   {murl}")
        print(f"{'#'*70}")

        # Resolve API key: cli flag > api_key_env field > env var GEMINI_API_KEY
        api_key = cli_api_key
        api_key_env = model_entry.get("api_key_env")
        if not api_key and api_key_env:
            api_key = os.environ.get(api_key_env)
            if api_key:
                print(f"  API key resolved from env var: {api_key_env}")
            else:
                print(f"  WARNING: api_key_env='{api_key_env}' not found in environment")

        try:
            client = create_client(
                server_url=murl,
                api_key=api_key,
                model_name=mname,
                temperature=temperature,
            )
        except ValueError as e:
            print(f"  ERROR: Could not create client for {mname}: {e} — skipping")
            continue

        if not client.ping():
            print(f"  WARNING: Cannot reach server at {murl} — skipping {mname}")
            continue

        print(f"  Server OK: {murl}")
        for problem in problems:
            print(f"\n{'='*60}")
            print(f"  PROBLEM: {problem['problem_id']}")
            print(f"  Answer: {problem['ground_truth']}")
            print(f"{'='*60}")
            run_problem(
                client, problem,
                model_name=mname,
                configs=configs,
                min_window=min_window,
                max_window=max_window,
                trials=trials,
            )

    print("\n\nAll models done. Run --scores for composite scoring table.")


# ─── Analysis ────────────────────────────────────────────────────────────────

def analyze_results():
    """Print a per-model summary table of all completed results."""
    summary_files = sorted(RESULTS_DIR.glob("*_summary.json"))
    if not summary_files:
        print("No results found. Run experiments first.")
        return

    print(f"\n{'Model':<25} {'Problem':<30} {'Config':<24} {'Min Window':>10} {'Range':>18} {'Steps':>6}")
    print("-" * 118)

    current_model = None
    for f in summary_files:
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"  [analyze] Could not read {f.name}: {e}")
            continue

        for entry in data:
            model = entry.get("model_name", "unknown")
            pid = entry.get("problem_id", "?")
            config = entry.get("config", "?")
            mw = entry.get("minimum_window")
            mw_str = str(mw) if mw is not None else "UNSOLVABLE"
            sr = entry.get("search_range_final", ["-", "-"])
            lo, hi = sr if sr else ("-", "-")
            steps = entry.get("steps", "?")

            if model != current_model:
                if current_model is not None:
                    print()
                current_model = model

            print(f"{model:<25} {pid:<30} {config:<24} {mw_str:>10} [{str(lo):>6}, {str(hi):>6}] {str(steps):>6}")


# ─── Main ────────────────────────────────────────────────────────────────────

def derive_model_name(url: str) -> str:
    """Derive a short model name from the server URL."""
    url = url.rstrip("/")
    # For gemini:// URLs extract the model name directly
    if url.startswith("gemini://") or url.startswith("google://"):
        scheme = "gemini://" if url.startswith("gemini://") else "google://"
        return url[len(scheme):].strip("/") or "gemini"
    # For anthropic:// URLs extract the model name directly
    if url.startswith("anthropic://"):
        return url[len("anthropic://"):].strip("/") or "claude-sonnet-4-6"
    # Extract host:port, replace dots/colons with underscores
    host_port = url.split("//")[-1]
    return re.sub(r"[^\w]", "_", host_port)


def main():
    parser = argparse.ArgumentParser(description="AmnesiaBench v2 — multi-model context window benchmark")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--problem", type=str, help="Problem ID (or substring) to test")
    group.add_argument("--all", action="store_true", help="Run all problems")
    group.add_argument("--analyze", action="store_true", help="Analyze existing results")
    group.add_argument("--scores", action="store_true", help="Print composite Scott scoring table")

    parser.add_argument("--model", type=str, default=SERVER_URL,
                        help=f"Server URL (default: {SERVER_URL}). Use gemini://MODEL for Gemini.")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Label for this model in results (default: derived from --model URL)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for Gemini backends (overrides GEMINI_API_KEY env var)")
    parser.add_argument("--run-all-models", action="store_true",
                        help="Iterate over all models in models.json (overrides --model/--model-name)")

    parser.add_argument("--min-window", type=int, default=MIN_WINDOW)
    parser.add_argument("--max-window", type=int, default=MAX_WINDOW)
    parser.add_argument("--trials", type=int, default=TRIALS_PER_WINDOW)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--config", type=str, default=None,
                        help="Run specific config only: NoTIR_HardCut, TIR_HardCut, NoTIR_Compact, TIR_Compact")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Results directory for --scores / --analyze (default: ./results)")

    args = parser.parse_args()

    # Redirect results dir if specified
    if args.results_dir:
        global RESULTS_DIR
        RESULTS_DIR = Path(args.results_dir)

    if args.analyze:
        analyze_results()
        return

    if args.scores:
        rd = Path(args.results_dir) if args.results_dir else None
        calculate_scores(rd)
        return

    min_window = args.min_window
    max_window = args.max_window
    trials_per_window = args.trials

    # Config filter
    configs = None
    if args.config:
        config_map = {
            "NoTIR_HardCut": (False, False),
            "TIR_HardCut": (True, False),
            "NoTIR_Compact": (False, True),
            "TIR_Compact": (True, True),
            # Legacy short names
            "HardCut": (False, False),
            "Compact": (False, True),
        }
        if args.config not in config_map:
            print(f"ERROR: Unknown config '{args.config}'. Choose from: {list(config_map.keys())}")
            sys.exit(1)
        configs = [config_map[args.config]]

    # Load problems
    if args.all:
        problems = load_all_problems()
    else:
        problems = [load_problem(args.problem)]

    # Resolve API key: --api-key > scheme-specific env var > GEMINI_API_KEY fallback
    _model_url = args.model
    if args.api_key:
        api_key = args.api_key
    elif _model_url.startswith("openrouter://"):
        api_key = os.environ.get("OPENROUTER_API_KEY")
    elif _model_url.startswith("anthropic://"):
        api_key = os.environ.get("ANTHROPIC_OAUTHTOKEN")
    else:
        api_key = os.environ.get("GEMINI_API_KEY")

    # Multi-model mode
    if args.run_all_models:
        run_all_models(
            problems=problems,
            configs=configs,
            min_window=min_window,
            max_window=max_window,
            trials=trials_per_window,
            temperature=args.temperature,
            cli_api_key=api_key,
        )
        return

    # Single-model mode
    model_url = args.model
    model_name = args.model_name or derive_model_name(model_url)

    try:
        client = create_client(
            server_url=model_url,
            api_key=api_key,
            model_name=model_name,
            temperature=args.temperature,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if not client.ping():
        print(f"ERROR: Cannot reach server at {model_url}")
        if model_url.startswith("http"):
            print(f"Start it first:\n  llama-server --model <path> --host 0.0.0.0 --port 8080 --ctx-size 65536")
        else:
            print(f"Check your API key and model name.")
        sys.exit(1)
    print(f"Server OK: {model_url}  (model_name: {model_name})")

    print(f"Problems: {[p['problem_id'] for p in problems]}")
    print(f"Search range: [{min_window}, {max_window}]")
    print(f"Trials per window: {trials_per_window}")
    print(f"Configs: {configs or [(False,False),(False,True)]}")
    print()

    for problem in problems:
        print(f"\n{'#'*60}")
        print(f"  PROBLEM: {problem['problem_id']}")
        print(f"  Answer: {problem['ground_truth']}")
        print(f"  120B pass rate: {problem.get('gptoss_120b_pass_rate', '?')}")
        print(f"{'#'*60}")
        run_problem(
            client, problem,
            model_name=model_name,
            configs=configs,
            min_window=min_window,
            max_window=max_window,
            trials=trials_per_window,
        )

    print("\n\nAll done. Run --analyze to see summary or --scores for composite score table.")


if __name__ == "__main__":
    main()
