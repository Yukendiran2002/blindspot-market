# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liceron
"""LIVE SANDBOX run of a whole learning job with SIMULATED annotators (labelled as such). No real money."""

import asyncio
import sys
import time
from pathlib import Path

from blindspot.db import Database
from blindspot.learner import load_corpus
from blindspot.paypal import PayPal
from blindspot.service import Blindspot
from blindspot.settings import load_env

env = load_env()  # PAYPAL_CLIENT_ID / PAYPAL_CLIENT_SECRET (+ PAYPAL_SANDBOX_BUYER_EMAIL for live_job)


async def main(db_path, rounds):
    pp = PayPal(env["PAYPAL_CLIENT_ID"], env["PAYPAL_CLIENT_SECRET"])
    app = Blindspot(Database(db_path), pp, load_corpus(Path("data")))
    human = env["PAYPAL_SANDBOX_BUYER_EMAIL"]
    anns = [
        app.add_annotator("Ana (simulated, 97%)", human, simulated_accuracy=0.97),
        app.add_annotator("Ben (simulated, 93%)", "ben.annotator@example.com", simulated_accuracy=0.93),
        app.add_annotator("Spam bot (simulated, 10%)", "spam.bot@example.com", simulated_accuracy=0.10),
        app.add_annotator(
            "Sleeper (simulated, 95% then 15%)",
            "sleeper@example.com",
            simulated_accuracy=0.95,
            simulated_late_accuracy=0.15,
        ),
    ]
    job = app.create_job("Support intent classifier", budget="60.00", round_size=40, seed_count=200)
    print("job", job, "| hold:", await app.fund_with_card(job))
    for r in range(rounds):
        plan = await app.plan_round(job, anns)
        if not plan["buy"]:
            print("agent stops:", plan["decision"]["detail"])
            break
        print(f"\nround {plan['number']}: {plan['decision']['explanation'][:220]}")
        app.simulate_answers(plan["round_id"], seed=r)
        t = time.time()
        res = await app.settle_round(plan["round_id"])
        print(f"  settled in {time.time() - t:.1f}s: {res}")
    print("\nclose:", await app.close_job(job))
    s = app.summary(job)
    print(
        "spent",
        s["spent"],
        "| curve:",
        [(c["labels"], round(c["validation"], 3), round(c["sealed"], 3), c["spent"]) for c in s["curve"]],
    )
    print("annotators:", s["annotators"])
    for e in app.db.all(
        "SELECT kind, data FROM events WHERE job_id=? AND kind IN ('charged','annotator_revoked','refunded','budget_released') ORDER BY seq",
        job,
    ):
        print(" ", e["kind"], e["data"][:200])
    await pp.aclose()


asyncio.run(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3))
