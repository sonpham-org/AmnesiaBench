# Author: Claude Sonnet 4.6 (Bubba)
# Date: 29-March-2026
# PURPOSE: Shared utility functions for AmnesiaBench v3. Covers answer extraction,
#   result file path helpers, and model name sanitization. Imported by predict.py,
#   evaluate.py, score.py, and cli.py.
#   Integration points: no circular imports — this module imports nothing from the package.
# SRP/DRY check: Pass — every utility here is used in >=2 modules; nothing is duplicated.

import re
from pathlib import Path
from typing import Optional


# ─── Answer Extraction ────────────────────────────────────────────────────────

def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the model's final answer from {final_answer: "..."} format.
    Returns the raw string inside the quotes, or None if not found.
    Strips leading/trailing whitespace from the captured value.
    """
    # Match {final_answer: "VALUE"} with or without surrounding braces in text
    pattern = r'\{final_answer:\s*"([^"]+)"\s*\}'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    # Fallback: bare format without outer braces (defensive)
    pattern2 = r'final_answer:\s*"([^"]+)"'
    match2 = re.search(pattern2, text)
    if match2:
        return match2.group(1).strip()
    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    r"""
    Extract the last \boxed{...} answer from text, ignoring <think> blocks.
    Returns the raw boxed content as a string, or None if not found.
    Used for legacy v2 problem sets that use \boxed{} format.
    """
    # Strip think blocks first
    non_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    target = non_think if non_think.strip() else text
    matches = re.findall(r"\\boxed\{([^{}]+)\}", target)
    if not matches:
        matches = re.findall(r"\\boxed\{(.+?)\}", target)
    if not matches:
        return None
    return matches[-1].strip()


# ─── Result File Helpers ──────────────────────────────────────────────────────

def sanitize_model_name(model_name: str) -> str:
    """Replace non-alphanumeric chars (except dash/underscore) with underscores."""
    return re.sub(r"[^\w\-]", "_", model_name)


def prediction_filename(results_dir: Path, model_name: str, problem_id: str) -> Path:
    """Return path for a prediction result file."""
    safe = sanitize_model_name(model_name)
    return Path(results_dir) / f"{safe}_{problem_id}_prediction.json"


def evaluation_filename(results_dir: Path, model_name: str, problem_id: str) -> Path:
    """Return path for an evaluation result file."""
    safe = sanitize_model_name(model_name)
    return Path(results_dir) / f"{safe}_{problem_id}_evaluation.json"


def derive_model_name(url: str) -> str:
    """
    Derive a short human-readable model name from a backend URL.
    Examples:
      anthropic://claude-sonnet-4-6   → claude-sonnet-4-6
      gemini://gemini-2.0-flash-lite  → gemini-2.0-flash-lite
      openrouter://openai/gpt-4o      → openai_gpt-4o
      http://localhost:8080           → localhost_8080
    """
    for scheme in ("anthropic://", "gemini://", "google://", "openrouter://"):
        if url.startswith(scheme):
            remainder = url[len(scheme):].strip("/")
            return re.sub(r"[^\w\-.]", "_", remainder) or scheme.rstrip("://")
    # http/https: strip scheme, replace non-word chars
    host_part = re.sub(r"^https?://", "", url).rstrip("/")
    return re.sub(r"[^\w\-]", "_", host_part) or "local"
