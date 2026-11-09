# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liceron
"""Ledger audit against PayPal's records (reader backed by the LOCAL FAKE PayPal) and the toolkit reader's limits."""

from __future__ import annotations

import pytest
from test_blindspot import GOOD, SLEEPER, Env, one_round, run

from blindspot.records import ToolkitReader, audit_job


class FakeReader:
    """Stands in for ToolkitReader: returns the order as the local fake PayPal reports it."""

    def __init__(self, env: Env) -> None:
        self.env = env

    async def order(self, order_id: str) -> dict:
        return self.env.fake._order_view(order_id)


def test_audit_matches_after_rounds_refund_and_close(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD, SLEEPER)
    jid = env.job(budget="60.00")
    one_round(env, jid, anns, seed=1)
    one_round(env, jid, anns, seed=2)  # sleeper revoked -> refund
    run(env.app.close_job(jid))
    res = run(audit_job(env.app.db, FakeReader(env), jid))
    assert res["ok"], res["checks"]
    names = [c["check"] for c in res["checks"]]
    assert "Total refunded" in names and "Unused budget released" in names


def test_audit_flags_a_ledger_that_disagrees_with_paypal(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job()
    one_round(env, jid, anns)
    with env.app.db.tx() as conn:
        conn.execute("UPDATE jobs SET captured_cents=captured_cents+50 WHERE id=?", (jid,))
    res = run(audit_job(env.app.db, FakeReader(env), jid))
    assert not res["ok"]
    assert [c["check"] for c in res["checks"] if not c["ok"]] == ["Total charged"]


def test_audit_flags_a_charge_blindspot_does_not_know(tmp_path):
    env = Env(tmp_path)
    anns = env.annotators(GOOD)
    jid = env.job()
    one_round(env, jid, anns)
    aid = env.app.db.one("SELECT authorization_id FROM jobs WHERE id=?", jid)["authorization_id"]
    run(env.app.paypal.capture_authorization(aid, amount="1.00", request_id="stray", custom_id="x", invoice_id="x"))
    res = run(audit_job(env.app.db, FakeReader(env), jid))
    failed = {c["check"] for c in res["checks"] if not c["ok"]}
    assert failed == {"No charges Blindspot does not know about", "Total charged"}


def test_toolkit_reader_is_sandbox_and_read_only():
    r = ToolkitReader("id", "secret")
    assert r._api._paypal_client.base_url == "https://api-m.sandbox.paypal.com"
    for method in ("pay_order", "create_order", "create_refund", "accept_dispute_claim", "cancel_subscription"):
        with pytest.raises(PermissionError):
            run(r.run(method, {}))
