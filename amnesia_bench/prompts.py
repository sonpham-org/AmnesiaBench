# Author: Claude Sonnet 4.6 (Bubba)
# Date: 29-March-2026
# PURPOSE: All prompt templates for AmnesiaBench v3. Single source of truth for every
#   string presented to models. Imported by predict.py and evaluate.py.
#   Integration points: predict.py uses PREDICTION_PROMPT; evaluate.py uses
#   INSTRUCTIONS + EVALUATION_PROMPT.
# SRP/DRY check: Pass — no prompt text lives outside this file.

# ─── Evaluation Templates ─────────────────────────────────────────────────────

INSTRUCTIONS = """\
Your context window is restricted to N = {N} tokens (including these instructions and the problem statement). When you reach 50% of this limit, you will be asked to compact your context so that you have room to continue working on the problem.

When you finish the problem, output your answer as a string in this exact format: {{final_answer: "YOUR_ANSWER_HERE"}}"""

EVALUATION_PROMPT = """\
<instructions>
{instructions}
</instructions>

<problem>
{problem}
</problem>"""

# ─── Prediction Template ──────────────────────────────────────────────────────

# NOTE: N is intentionally left as the literal text "N tokens" here —
# the model is NOT given a concrete number during prediction.
PREDICTION_PROMPT = """\
You are about to be evaluated on the problem below. Your context window will be restricted to N tokens (including the instructions and problem statement). When you reach 50% of this limit, you will be asked to compact your context to continue working.

<evaluation_prompt>
<instructions>
Your context window is restricted to N tokens (including these instructions and the problem statement). When you reach 50% of this limit, you will be asked to compact your context so that you have room to continue working on the problem.

When you finish the problem, output your answer as a string in this exact format: {{final_answer: "YOUR_ANSWER_HERE"}}
</instructions>

<problem>
{problem}
</problem>
</evaluation_prompt>

Do you want to attempt the problem? You should opt out from solving the above problem if you don't think you can solve it reliably.

What value of N do you want to use? Choose the lowest value for which you feel confident that you can still solve the problem correctly.

Make your determinations in 300 tokens or less, and output them as strings in the format:
{{attempt: "output_True_or_False_here", N: "output_N_value_here_or_output_0_if_attempt_is_False"}}"""


def build_instructions(N: int) -> str:
    """Return INSTRUCTIONS with the concrete token count filled in."""
    return INSTRUCTIONS.format(N=N)


def build_evaluation_prompt(N: int, problem: str) -> str:
    """Return the full evaluation prompt with N and problem text filled in."""
    instructions = build_instructions(N)
    return EVALUATION_PROMPT.format(instructions=instructions, problem=problem)


def build_prediction_prompt(problem: str) -> str:
    """Return the prediction prompt with problem text filled in."""
    return PREDICTION_PROMPT.format(problem=problem)
