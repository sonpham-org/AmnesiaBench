# Author: Claude Sonnet 4.6 (Bubba)
# Date: 29-March-2026
# PURPOSE: Problem loading utilities for AmnesiaBench v3. Loads problem JSON files from
#   the problems/ directory adjacent to the package. Supports exact ID match and
#   substring match for convenience.
#   Integration points: imported by predict.py, evaluate.py, and cli.py.
# SRP/DRY check: Pass — single source of problem I/O; no duplication with result I/O.

import json
from pathlib import Path
from typing import List

# Default problems directory: one level up from this file (the package root), then problems/
_PACKAGE_DIR = Path(__file__).parent
PROBLEMS_DIR = _PACKAGE_DIR.parent / "problems"


def set_problems_dir(path: Path) -> None:
    """Override the problems directory (e.g. for testing)."""
    global PROBLEMS_DIR
    PROBLEMS_DIR = Path(path)


def load_problem(problem_id: str) -> dict:
    """
    Load a single problem JSON file from PROBLEMS_DIR.
    Matches on exact stem first, then substring.
    Raises FileNotFoundError if no match found.

    Expected problem JSON schema:
    {
        "problem_id": str,
        "problem_text": str,
        "ground_truth": str,          # expected answer string
        ... (any additional metadata)
    }
    """
    exact = PROBLEMS_DIR / f"{problem_id}.json"
    if exact.exists():
        return json.loads(exact.read_text())

    # Substring fallback
    for p in sorted(PROBLEMS_DIR.glob("*.json")):
        if problem_id in p.stem:
            return json.loads(p.read_text())

    raise FileNotFoundError(
        f"No problem matching '{problem_id}' found in {PROBLEMS_DIR}"
    )


def load_all_problems() -> List[dict]:
    """Load every .json file in PROBLEMS_DIR, sorted by filename."""
    if not PROBLEMS_DIR.exists():
        raise FileNotFoundError(
            f"Problems directory not found: {PROBLEMS_DIR}\n"
            "Create it and add problem JSON files."
        )
    files = sorted(PROBLEMS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No problem files found in {PROBLEMS_DIR}")
    return [json.loads(p.read_text()) for p in files]


def list_problem_ids() -> List[str]:
    """Return a sorted list of all problem IDs (stems) available."""
    if not PROBLEMS_DIR.exists():
        return []
    return sorted(p.stem for p in PROBLEMS_DIR.glob("*.json"))
