# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liceron
"""The buyer agent: decides whether another round of labels is worth the money, and what to buy.

Decisions are made by explicit rules over measured numbers (validation accuracy, money left, gain per dollar);
the language model only writes the explanation shown to the customer. Hard limits (budget, hold remaining,
round size) are enforced again by the settlement code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from blindspot.learner import pretty
from blindspot.money import show


@dataclass
class Decision:
    buy: bool
    n: int
    reason: str  # code: start | gain_ok | target_reached | low_value | budget | pool_empty
    detail: str  # plain-English rule outcome
    est_cost_cents: int
    gain_per_10: float | None  # validation points gained per $10 in the last round
    picks: list[dict[str, Any]] = field(default_factory=list)
    explanation: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def decide(
    *,
    val_acc: float,
    target: float,
    history: list[tuple[int, float]],
    price_cents: int,
    money_left_cents: int,
    round_size: int,
    min_gain_per_10: float,
    pool_left: int,
) -> Decision:
    """history: [(cents spent so far, validation accuracy)] after each settled round, starting with (0, seed acc)."""
    gain = None
    if len(history) >= 2:
        (c0, a0), (c1, a1) = history[-2], history[-1]
        spent = c1 - c0
        if spent <= 0:
            spent = 1
        gain = (a1 - a0) * 100 / (spent / 1000)  # points per $10
    affordable = money_left_cents // price_cents
    n = int(min(round_size, affordable, pool_left))
    if val_acc >= target:
        return Decision(
            False, 0, "target_reached", f"Validation accuracy {val_acc:.1%} reached the target {target:.0%}.", 0, gain
        )
    if n <= 0:
        why = "budget" if affordable <= 0 else "pool_empty"
        text = "Not enough budget left for another label." if why == "budget" else "No unlabelled messages left."
        return Decision(False, 0, why, text, 0, gain)
    if gain is not None and gain < min_gain_per_10:
        return Decision(
            False,
            0,
            "low_value",
            f"The last round added only {gain:.2f} points per $10 (minimum {min_gain_per_10:.2f}); "
            "buying more would waste money.",
            0,
            gain,
        )
    reason = "start" if gain is None else "gain_ok"
    detail = f"Validation accuracy is {val_acc:.1%}, below the {target:.0%} target. " + (
        "First purchase round." if gain is None else f"The last round paid off at {gain:.2f} points per $10."
    )
    return Decision(True, n, reason, detail, n * price_cents, gain)


def summarize_picks(picks: list[dict[str, Any]], intents: list[str], top: int = 4) -> list[dict[str, Any]]:
    """Most common confusions among the picks, for the explanation and the UI."""
    counts: dict[tuple[int, int], int] = {}
    for p in picks:
        a, b = sorted(p["confused"])
        counts[(a, b)] = counts.get((a, b), 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    return [{"a": pretty(intents[a]), "b": pretty(intents[b]), "count": c} for (a, b), c in ranked]


def template_explanation(d: Decision, confusions: list[dict[str, Any]]) -> str:
    if not d.buy:
        return d.detail
    areas = "; ".join(f"{c['a']} vs {c['b']} ({c['count']})" for c in confusions[:3])
    return (
        f"{d.detail} Buying {d.n} labels for {show(d.est_cost_cents)}, aimed at the areas where the model "
        f"is most confused: {areas}."
    )


EXPLAIN_INSTRUCTIONS = """You are Blindspot's buyer agent explaining a purchase decision to the customer who funds it.
The decision is already made by rules; do not change numbers or the decision. In 2-3 short sentences: say what the model
is getting wrong (use the confusion areas given), what is being bought and for how much, and why it is worth it.
Plain English, no hype."""


async def llm_explanation(
    api_key: str, d: Decision, confusions: list[dict[str, Any]], model: str = "gemini-3.8-flash"
) -> str:
    """Optional: Gemini writes the explanation through a Nexura agent (no tools). Falls back to the template on error."""
    from nexura_agents import Agent, Runner
    from nexura_agents.providers.gemini import GeminiNativeProvider

    facts = {
        "decision": "buy" if d.buy else "stop",
        "labels": d.n,
        "cost": show(d.est_cost_cents),
        "rule": d.detail,
        "confusions": confusions,
    }
    agent = Agent(
        name="blindspot-explainer",
        instructions=EXPLAIN_INSTRUCTIONS,
        model=GeminiNativeProvider(model, api_key=api_key, timeout=30.0),
        max_turns=1,
    )
    result = await Runner().run_async(agent, "Facts (JSON):\n" + json.dumps(facts), timeout=40)
    text = str(result.final_output or "").strip()
    return text if result.status == "completed" and text else template_explanation(d, confusions)
