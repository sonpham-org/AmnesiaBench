#!/usr/bin/env python3
# Author: Claude Opus 4.6 (Bubba subagent)
# Date: 30-March-2026
# PURPOSE: Flatten arc/arc_problems.json into individual problem files in problems/,
#          matching the AIMO3 format so the runner discovers them alongside math problems.
# SRP/DRY check: Pass — single script, reads one file, writes N files, no duplication.

import json
from pathlib import Path

ARC_PROBLEMS = Path(__file__).parent / "arc" / "arc_problems.json"
PROBLEMS_DIR = Path(__file__).parent / "problems"


def main():
    with open(ARC_PROBLEMS) as f:
        problems = json.load(f)

    print(f"Read {len(problems)} ARC problems from {ARC_PROBLEMS}")

    PROBLEMS_DIR.mkdir(exist_ok=True)

    written = 0
    for p in problems:
        pid = p["problem_id"]
        out = {
            "problem_id": pid,
            "problem_text": p["problem_text"],
            "ground_truth": p["ground_truth"],
            "topic": "arc",
            "source": "ARC1-Eval",
        }
        outpath = PROBLEMS_DIR / f"{pid}.json"
        outpath.write_text(json.dumps(out, indent=2))
        written += 1
        print(f"  Wrote {outpath.name}")

    print(f"\nDone: {written} ARC problem files written to {PROBLEMS_DIR}")
    print(f"Total files in problems/: {len(list(PROBLEMS_DIR.glob('*.json')))}")


if __name__ == "__main__":
    main()
