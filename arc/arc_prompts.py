# Author: Claude Sonnet 4.6
# Date: 30-March-2026
# PURPOSE: ARC system prompts for AmnesiaBench. Two variants for A/B testing:
#          SIMPLE — minimal, lets the model think on its own.
#          GUIDED — teaches concrete solving strategies from arc-explainer.
#          Both go in the system message; problem_text goes in the user message.
# SRP/DRY check: Pass — prompt definitions only, no generation or evaluation logic.

from __future__ import annotations

# ── SIMPLE — Son's preferred version ──────────────────────────────────────────
#
# Minimal instruction. Explains what grids are and what the task is.
# Lets the model use its own reasoning approach.

ARC_SYSTEM_PROMPT_SIMPLE = """You are solving a pattern transformation puzzle.

You will see grids of numbers (0-9). Each number represents a different colored element. Training examples show input grids and their corresponding output grids. Every training example follows the SAME deterministic transformation rule.

Your job is to discover this rule and apply it to the test input(s) to predict the correct output grid(s). The rule must explain ALL training examples — if your hypothesis fails on any example, revise it.

Give each answer inside numbered tags: <answer_1> for test 1, <answer_2> for test 2, etc. Use space-separated values, one row per line."""


# ── GUIDED — Mark's preferred version ────────────────────────────────────────
#
# Teaches 7 concrete strategies derived from arc-explainer's GEPA mode,
# visual solver, and basePrompts.ts. For models with limited or no ARC
# training data that need scaffolding on how to approach these puzzles.

ARC_SYSTEM_PROMPT_GUIDED = """You are solving a pattern transformation puzzle.

You will see grids of numbers (0-9). Each number represents a different colored element. Training examples show input grids and their corresponding output grids. Every training example follows the SAME deterministic transformation rule. Your job is to discover this rule and apply it to the test input(s) to predict the correct output grid(s).

Approach the puzzle systematically:

1. START SIMPLE. Check for global transformations first: rotation, reflection, transposition, uniform color replacement, cropping, or scaling. Many puzzles have surprisingly simple rules.

2. LOOK FOR STRUCTURE. Check if rows or columns of a single value divide the grid into sections. The transformation may apply independently to each section.

3. IDENTIFY OBJECTS. Group connected cells of the same non-zero value into objects. Track how objects change between input and output: do they move, change color, change shape, grow, shrink, duplicate, or disappear?

4. FIND MARKERS. Look for cells with unique values or positions that might define where or how a transformation applies — like anchor points, corners of a region, or signals that trigger a specific operation.

5. CONSIDER COMPOSITION. The transformation may require multiple steps applied in sequence. One operation may need to happen before another makes sense. Try decomposing complex changes into simpler sub-rules applied in order.

6. VERIFY ACROSS ALL EXAMPLES. The rule MUST explain every training example, not just one. If your hypothesis fails on any example, it is wrong — revise it. Do not fixate on patterns that only appear in a single example.

7. IGNORE NOISE. Some properties may vary between examples without being part of the rule. If something is not consistent across all input-output pairs, it is not relevant to the transformation.

Give each answer inside numbered tags: <answer_1> for test 1, <answer_2> for test 2, etc. Use space-separated values, one row per line."""


# ── Default alias ─────────────────────────────────────────────────────────────

ARC_SYSTEM_PROMPT = ARC_SYSTEM_PROMPT_SIMPLE


if __name__ == "__main__":
    print("ARC System Prompts — A/B Variants")
    print("=" * 60)
    print(f"SIMPLE: {len(ARC_SYSTEM_PROMPT_SIMPLE)} chars, ~{len(ARC_SYSTEM_PROMPT_SIMPLE) // 4} tokens")
    print(f"GUIDED: {len(ARC_SYSTEM_PROMPT_GUIDED)} chars, ~{len(ARC_SYSTEM_PROMPT_GUIDED) // 4} tokens")
    print()
    print("--- SIMPLE ---")
    print(ARC_SYSTEM_PROMPT_SIMPLE)
    print()
    print("--- GUIDED ---")
    print(ARC_SYSTEM_PROMPT_GUIDED)
