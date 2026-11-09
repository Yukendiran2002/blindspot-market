# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liceron
"""Blindspot money and quality tests against the LOCAL FAKE PayPal and a small synthetic corpus."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from fake_paypal import FakePayPal

from blindspot.db import Database
from blindspot.learner import Corpus
from blindspot.paypal import PayPal
from blindspot.service import Blindspot

CLASSES = 6


def corpus(seed: int = 0) -> Corpus:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(CLASSES, 16)) * 3

    def make(n):
        y = rng.integers(CLASSES, size=n)
        X = centers[y] + rng.normal(size=(n, 16)) * 5.5
        return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32), y

    X, y = make(1500)
    vX, vy = make(300)
    tX, ty = make(400)
    return Corpus(
        texts=[f"message {i}" for i in range(len(y))],
        labels=y,
        X=X,
        val_X=vX,
        val_y=vy,
        test_X=tX,
        test_y=ty,
        intents=[f"intent_{k}" for k in range(CLASSES)],
    )


class Env:
    def __init__(self, tmp_path) -> None:
        self.fake = FakePayPal()
        self.path = tmp_path / "b.db"
        self.c = corpus()
        self.app = Blindspot(Database(self.path), PayPal("id", "secret", transport=self.fake.transport), self.c)

    def restart(self) -> Blindspot:
        self.app = Blindspot(Database(self.path), PayPal("id", "secret", transport=self.fake.transport), self.c)
        return self.app

    def annotators(self, *specs):
        return [
            self.app.add_annotator(name, email, simulated_accuracy=acc, simulated_late_accuracy=late)
            for name, email, acc, late in specs
        ]

    def job(self, budget="30.00", round_size=20, strategy="blindspot"):
        jid = self.app.create_job(
            "test",
            budget=budget,
            round_size=round_size,
            seed_count=60,
            strategy=strategy,
            target_accuracy=0.99,
            min_gain_per_10=-100,
        )
        run(self.app.fund_with_card(jid))
        return jid


def run(coro):
    return asyncio.run(coro)


GOOD = ("Ana", "ana@x.test", 1.0, None)
SPAM = ("Spam", "spam@x.test", 0.0, None)
SLEEPER = ("Sleeper", "sleep@x.test", 1.0, 0.0)


def one_round(env, jid, anns, seed=0):
    plan = run(env.app.plan_round(jid, anns))
    assert plan["buy"]
    env.app.simulate_answers(plan["round_id"], seed=seed)
    return plan, run(env.app.settle_round(plan["round_id"]))


def test_round_charges_only_accepted_labels_and_pays_annotators(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD, SPAM)
    jid = env.job()
    _plan, res = one_round(env, jid, anns)
    assert res["accepted"] == 10 and res["rejected"] == 10  # the spammer's half is rejected
    assert res["charged"] == "5.00" and env.fake.captured_total() == 500
    assert env.fake.paid_to("ana@x.test") > 0 and env.fake.paid_to("spam@x.test") == 0
    assert res["capture"] == "DONE" and res["payout"] == "SUBMITTED"
    s = env.app.summary(jid)
    spam = next(a for a in s["annotators"] if a["name"] == "Spam")
    assert spam["banned"] and spam["earned"] == "0.00"


def test_model_learns_from_bought_labels(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job(budget="60.00")
    before = env.app.summary(jid)["curve"][0]["validation"]
    for r in range(3):
        one_round(env, jid, anns, seed=r)
    after = env.app.summary(jid)["curve"][-1]["validation"]
    assert after > before


def test_lost_capture_response_is_found_by_custom_id(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job()
    env.fake.fault("capture", "lose_response")
    _plan, res = one_round(env, jid, anns)
    assert res["capture"] == "UNCERTAIN" and res["payout"] == "NONE"  # never pay before the charge is confirmed
    app = env.restart()
    run(app.settle.reconcile(jid))
    s = app.db.one("SELECT * FROM settlements WHERE id=?", res["settlement_id"])
    assert s["capture_state"] == "DONE" and s["payout_state"] == "DONE"  # submitted, then every item SUCCESS
    assert len(env.fake.captures) == 1 and env.fake.captured_total() == 1000


def test_capture_timeout_before_processing_resends_same_id(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job()
    env.fake.fault("capture", "timeout_before")
    _, res = one_round(env, jid, anns)
    assert res["capture"] == "UNCERTAIN" and not env.fake.captures
    run(env.restart().settle.reconcile(jid))
    assert len(env.fake.captures) == 1


def test_lost_payout_response_never_pays_twice(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job()
    env.fake.fault("payout", "lose_response")
    _, res = one_round(env, jid, anns)
    assert res["payout"] == "UNCERTAIN" and len(env.fake.payouts) == 1
    app = env.restart()
    run(app.settle.reconcile(jid))
    s = app.db.one("SELECT payout_state, payout_batch_id FROM settlements WHERE id=?", res["settlement_id"])
    # the duplicate check refused a second batch and pointed at the original, which is then tracked to SUCCESS
    assert s["payout_state"] == "DONE" and len(env.fake.payouts) == 1
    assert s["payout_batch_id"] == next(iter(env.fake.payouts))


def test_payout_server_error_is_retried_safely(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job()
    env.fake.fault("payout", "500_before")
    _, res = one_round(env, jid, anns)
    assert res["payout"] == "UNCERTAIN" and not env.fake.payouts
    run(env.app.settle.reconcile(jid))
    assert len(env.fake.payouts) == 1


def test_refused_charge_pays_nobody_and_untrains_labels(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job()
    env.fake.fault("capture", "422")
    _plan, res = one_round(env, jid, anns)
    assert res["capture"] == "FAILED" and not env.fake.payouts
    assert env.app.db.one("SELECT status FROM jobs WHERE id=?", jid)["status"] == "PAYMENT_PROBLEM"
    active = env.app.db.one("SELECT COUNT(*) n FROM labels WHERE job_id=? AND source='bought' AND active=1", jid)["n"]
    assert active == 0
    with pytest.raises(RuntimeError):
        run(env.app.plan_round(jid, anns))


def test_sleeper_is_revoked_and_customer_refunded(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD, SLEEPER)
    jid = env.job(budget="60.00")
    _, r1 = one_round(env, jid, anns, seed=1)
    assert r1["accepted"] == 20 and env.fake.paid_to("sleep@x.test") > 0
    _, _r2 = one_round(env, jid, anns, seed=2)
    refunds = list(env.fake.refunds.values())
    assert len(refunds) == 1 and refunds[0]["amount"]["value"] == "5.00"  # 10 earlier labels x $0.50
    job = env.app.db.one("SELECT * FROM jobs WHERE id=?", jid)
    assert job["refunded_cents"] == 500
    revoked = env.app.db.one("SELECT COUNT(*) n FROM tasks WHERE job_id=? AND status='REVOKED'", jid)["n"]
    inactive = env.app.db.one("SELECT COUNT(*) n FROM labels WHERE job_id=? AND active=0", jid)["n"]
    assert revoked == 10 and inactive == 10
    plan3 = run(env.app.plan_round(jid, anns))
    assert plan3["buy"]
    owners = {
        r["annotator_id"] for r in env.app.db.all("SELECT annotator_id FROM tasks WHERE round_id=?", plan3["round_id"])
    }
    assert owners == {anns[0]}  # banned annotator gets no more work


def test_budget_is_never_exceeded(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job(budget="12.00", round_size=20)  # 24 labels max
    one_round(env, jid, anns)
    plan2, _res2 = one_round(env, jid, anns, seed=1)
    assert plan2["decision"]["n"] == 4
    plan3 = run(env.app.plan_round(jid, anns))
    assert not plan3["buy"] and plan3["decision"]["reason"] == "budget"
    assert env.fake.captured_total() == 1200


def test_close_voids_the_rest_once(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job()
    one_round(env, jid, anns)
    out = run(env.app.close_job(jid))
    assert out["released"] == "20.00" and out["hold"] == "VOIDED"
    assert run(env.app.close_job(jid)) == {"status": "CLOSED"}
    aid = env.app.db.one("SELECT authorization_id FROM jobs WHERE id=?", jid)["authorization_id"]
    assert env.fake.auths[aid]["status"] == "VOIDED"


def test_annotator_cannot_answer_someone_elses_task(tmp_path):
    env = Env(tmp_path)
    a, b = env.annotators(GOOD, ("Bo", "bo@x.test", None, None))
    jid = env.job()
    run(env.app.plan_round(jid, [a]))
    task = env.app.next_task(a)
    with pytest.raises(ValueError):
        env.app.answer(task["task_id"], b, 0)
    with pytest.raises(ValueError):
        env.app.answer(task["task_id"], a, 99)


def test_random_strategy_job_runs(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job(strategy="random")
    plan, res = one_round(env, jid, anns)
    assert res["accepted"] == 20 and plan["confusions"] == []


def test_only_sandbox_host_allowed():
    for url in (
        "https://api-m.paypal.com",
        "http://api-m.sandbox.paypal.com",
        "https://api-m.sandbox.paypal.com.evil.test",
    ):
        with pytest.raises(ValueError):
            PayPal("id", "secret", url)


def test_capture_found_by_custom_id_even_if_paypal_forgot_the_request_id(tmp_path):
    """Long outage: PayPal no longer remembers our PayPal-Request-Id. The custom_id lookup must stop a second charge."""
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job()
    env.fake.fault("capture", "lose_response")
    _, _res = one_round(env, jid, anns)
    env.fake.idem.clear()
    run(env.restart().settle.reconcile(jid))
    assert len(env.fake.captures) == 1


def test_capture_not_yet_visible_is_resent_with_the_same_id(tmp_path):
    """The capture went through but is not yet visible in GET order: resending must reuse the same request id."""
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job()
    env.fake.fault("capture", "lose_response")
    _, res = one_round(env, jid, anns)
    env.fake.hide_captures = 1
    run(env.restart().settle.reconcile(jid))
    assert len(env.fake.captures) == 1
    s = env.app.db.one("SELECT capture_state FROM settlements WHERE id=?", res["settlement_id"])
    assert s["capture_state"] == "DONE"


def test_settlement_refuses_to_charge_beyond_the_hold(tmp_path):
    """Second line of defence: even if planning was wrong, settlement never charges more than the hold has left."""
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job(budget="30.00", round_size=20)
    plan = run(env.app.plan_round(jid, anns))
    env.app.simulate_answers(plan["round_id"])
    with env.app.db.tx() as conn:  # someone captured most of the hold elsewhere
        conn.execute("UPDATE jobs SET captured_cents=2900 WHERE id=?", (jid,))
    with pytest.raises(RuntimeError, match="exceeds hold"):
        run(env.app.settle_round(plan["round_id"]))
    assert not env.fake.captures


# --- settlement state machine (review findings) --------------------------------------------------


def test_pending_capture_pays_nobody_trains_nothing_and_blocks_buying(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job()
    env.fake.fault("capture", "pending")
    plan = run(env.app.plan_round(jid, anns))
    env.app.simulate_answers(plan["round_id"])
    res = run(env.app.settle_round(plan["round_id"]))
    assert res["pending"] and res["capture"] == "PENDING" and not env.fake.payouts
    active = env.app.db.one("SELECT COUNT(*) n FROM labels WHERE job_id=? AND source='bought' AND active=1", jid)["n"]
    assert active == 0
    assert env.app.db.one("SELECT status FROM rounds WHERE id=?", plan["round_id"])["status"] == "SETTLING"
    with pytest.raises(RuntimeError):
        run(env.app.plan_round(jid, anns))
    assert run(env.app.close_job(jid))["status"] != "CLOSED"  # cannot close over an unconfirmed charge
    for c in env.fake.captures.values():
        c["status"] = "COMPLETED"  # PayPal completes the capture later
    app = env.restart()
    run(app.resume())
    s = app.db.one("SELECT * FROM settlements WHERE id=?", res["settlement_id"])
    assert s["capture_state"] == "DONE" and s["payout_state"] in ("SUBMITTED", "DONE")
    assert app.db.one("SELECT status FROM rounds WHERE id=?", plan["round_id"])["status"] == "SETTLED"
    active = app.db.one("SELECT COUNT(*) n FROM labels WHERE job_id=? AND source='bought' AND active=1", jid)["n"]
    assert active == 20


def test_declined_pending_capture_releases_the_reservation(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job()
    env.fake.fault("capture", "pending")
    plan = run(env.app.plan_round(jid, anns))
    env.app.simulate_answers(plan["round_id"])
    run(env.app.settle_round(plan["round_id"]))
    assert env.app._job(jid)["captured_cents"] == 1000
    for c in env.fake.captures.values():
        c["status"] = "DECLINED"
    run(env.app.resume())
    job = env.app._job(jid)
    assert job["captured_cents"] == 0 and job["status"] == "PAYMENT_PROBLEM" and not env.fake.payouts


def test_crash_after_sending_looks_up_before_resending(tmp_path):
    """Process died after the capture request left: the restart must find it, not send a second one."""
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job()
    plan = run(env.app.plan_round(jid, anns))
    env.app.simulate_answers(plan["round_id"])
    sid = env.app.settle.prepare_round(env.app._job(jid), plan["round_id"])
    aid = env.app._job(jid)["authorization_id"]
    run(env.app.paypal.capture_authorization(aid, amount="10.00", request_id="other-id", custom_id=sid, invoice_id=sid))
    with env.app.db.tx() as conn:  # the request had been counted as sent before the crash
        conn.execute("UPDATE settlements SET capture_tries=1 WHERE id=?", (sid,))
    run(env.restart().resume())
    assert len(env.fake.captures) == 1
    assert env.app.db.one("SELECT capture_state FROM settlements WHERE id=?", sid)["capture_state"] == "DONE"


def test_unreadable_order_never_triggers_a_resend(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job()
    env.fake.fault("capture", "lose_response")
    plan = run(env.app.plan_round(jid, anns))
    env.app.simulate_answers(plan["round_id"])
    run(env.app.settle_round(plan["round_id"]))
    env.fake.idem.clear()
    env.fake.fault("order", "500")
    run(env.app.settle.reconcile(jid))
    assert len(env.fake.captures) == 1
    s = env.app.db.one("SELECT capture_state FROM settlements WHERE round_id=?", plan["round_id"])
    assert s["capture_state"] == "UNCERTAIN"


def test_too_few_answered_test_questions_is_not_accepted(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job()
    plan = run(env.app.plan_round(jid, anns))
    for t in env.app.db.all("SELECT id, pool_index FROM tasks WHERE round_id=? AND kind='learn'", plan["round_id"]):
        env.app.answer(t["id"], anns[0], int(env.c.labels[t["pool_index"]]))
    env.app.expire_open_tasks(plan["round_id"])  # the test questions were never answered
    res = run(env.app.settle_round(plan["round_id"]))
    assert res["accepted"] == 0 and not env.fake.captures and not env.fake.payouts


def test_pending_refund_is_not_counted_until_completed(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD, SLEEPER)
    jid = env.job(budget="60.00")
    one_round(env, jid, anns, seed=1)
    env.fake.refund_status = "PENDING"
    one_round(env, jid, anns, seed=2)
    assert env.app._job(jid)["refunded_cents"] == 0
    for r in env.fake.refunds.values():
        r["status"] = "COMPLETED"
    run(env.app.settle.reconcile(jid))
    assert env.app._job(jid)["refunded_cents"] == 500


def test_failed_payout_item_is_reported_as_failed(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD, ("Bo", "bo@x.test", 1.0, None))
    jid = env.job()
    _, res = one_round(env, jid, anns)
    env.fake.item_status["bo@x.test"] = "RETURNED"
    run(env.app.settle.reconcile(jid))
    s = env.app.db.one("SELECT payout_state FROM settlements WHERE id=?", res["settlement_id"])
    assert s["payout_state"] == "ITEM_FAILED"


def test_round_left_settling_after_a_crash_is_finished_on_restart(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job()
    plan = run(env.app.plan_round(jid, anns))
    env.app.simulate_answers(plan["round_id"])
    env.app.settle.prepare_round(env.app._job(jid), plan["round_id"])  # crash right after the audit
    app = env.restart()
    assert run(app.resume()) == [plan["round_id"]]
    assert app.db.one("SELECT status FROM rounds WHERE id=?", plan["round_id"])["status"] == "SETTLED"
    assert len(env.fake.captures) == 1 and len(env.fake.payouts) == 1


def test_refused_void_is_not_reported_as_released(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job()
    one_round(env, jid, anns)
    aid = env.app._job(jid)["authorization_id"]
    env.fake.fault("void", "422")
    out = run(env.app.close_job(jid))
    assert out["status"] != "CLOSED" and env.fake.auths[aid]["status"] != "VOIDED"
    assert not env.app.db.one("SELECT 1 FROM events WHERE job_id=? AND kind='budget_released'", jid)


def test_a_job_cannot_be_funded_twice(tmp_path):
    env = Env(tmp_path)
    jid = env.job()
    with pytest.raises(RuntimeError):
        run(env.app.fund_with_card(jid))
    assert len(env.fake.orders) == 1


def test_refused_void_on_an_unused_hold_is_not_reported_as_released(tmp_path):
    env = Env(tmp_path)
    jid = env.job()
    env.fake.fault("void", "422")
    out = run(env.app.close_job(jid))
    assert out["status"] != "CLOSED" and "nothing was released" in out["note"]


def test_annotator_with_one_test_answer_is_not_accepted_even_if_trust_bar_were_lower(tmp_path, monkeypatch):
    import blindspot.settlement as st

    monkeypatch.setattr(st, "TRUST_MIN", 0.5)  # one correct answer would pass the bar: evidence must still be enough
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job()
    plan = run(env.app.plan_round(jid, anns))
    for t in env.app.db.all("SELECT * FROM tasks WHERE round_id=?", plan["round_id"]):
        if t["kind"] == "learn":
            env.app.answer(t["id"], anns[0], int(env.c.labels[t["pool_index"]]))
    gold = env.app.db.one("SELECT * FROM tasks WHERE round_id=? AND kind='gold' LIMIT 1", plan["round_id"])
    env.app.answer(gold["id"], anns[0], gold["gold_label"])
    env.app.expire_open_tasks(plan["round_id"])
    res = run(env.app.settle_round(plan["round_id"]))
    assert res["accepted"] == 0 and not env.fake.captures
