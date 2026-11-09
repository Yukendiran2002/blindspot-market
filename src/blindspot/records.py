# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liceron
"""Independent check of a job's money against PayPal's own records, through the PayPal Agent Toolkit.

Blindspot moves money with its own client (:mod:`blindspot.paypal`): every charge, payout and refund carries an
idempotency id and is recorded before it is sent, which the toolkit does not do. This module is the second pair of
eyes. It reads the job's order back through PayPal's official Agent Toolkit (``paypal-agent-toolkit``) and compares
what PayPal says with Blindspot's ledger:

- :func:`audit_job`: deterministic checks (hold amount, every capture and its custom_id, captured and refunded
  totals, hold released). Code decides pass or fail; no language model is involved.
- :func:`ask_about_money`: a Gemini agent (Nexura SDK) answers the customer's questions about their money. It gets
  READ-ONLY tools only: the toolkit's ``get_order_details`` for this job's order, the toolkit's
  ``list_transactions``, and Blindspot's ledger. It cannot create, capture, refund or pay anything.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from blindspot.db import Database
from blindspot.money import fmt, to_cents

SANDBOX = "https://api-m.sandbox.paypal.com"
READ_ONLY = frozenset({"get_order_details", "list_transactions"})
TOOLKIT_TIMEOUT = 30.0


class ToolkitReader:
    """The PayPal Agent Toolkit's framework-neutral core, limited to read-only methods and the sandbox."""

    def __init__(self, client_id: str, client_secret: str) -> None:
        from paypal_agent_toolkit.shared.api import PayPalAPI
        from paypal_agent_toolkit.shared.configuration import Context

        # Context(sandbox=None) would mean LIVE in the toolkit: always pass True and check the result.
        self._api = PayPalAPI(client_id, client_secret, Context(sandbox=True, source="BLINDSPOT"))
        if self._api._paypal_client.base_url != SANDBOX:
            raise RuntimeError("PayPal Agent Toolkit is not pointed at the sandbox")

    async def run(self, method: str, params: dict[str, Any]) -> Any:
        if method not in READ_ONLY:
            raise PermissionError(f"{method} is not a read-only toolkit method")
        # the toolkit uses blocking requests without a timeout; keep it off the event loop and bounded
        out = await asyncio.wait_for(asyncio.to_thread(self._api.run, method, params), TOOLKIT_TIMEOUT)
        return json.loads(out) if isinstance(out, str) else out

    async def order(self, order_id: str) -> dict[str, Any]:
        res = await self.run("get_order_details", {"order_id": order_id})
        return res.get("raw", res)


def _payments(order: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"authorizations": [], "captures": [], "refunds": []}
    for unit in order.get("purchase_units", []):
        for k, items in out.items():
            items.extend(unit.get("payments", {}).get(k) or [])
    return out


async def audit_job(db: Database, reader: Any, job_id: str, paypal: Any = None) -> dict[str, Any]:
    """Compare Blindspot's ledger for one job with the order as PayPal reports it. Records an ``audit_completed`` event."""
    job = db.one("SELECT * FROM jobs WHERE id=?", job_id)
    if job is None:
        raise KeyError(job_id)
    if not job["paypal_order_id"]:
        raise RuntimeError("job was never funded")
    order = await reader.order(job["paypal_order_id"])
    pay = _payments(order)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, ours: str, paypal: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "blindspot": ours, "paypal": paypal})

    auth = next((a for a in pay["authorizations"] if a.get("id") == job["authorization_id"]), None)
    check(
        "Budget hold exists for the job's amount",
        auth is not None and to_cents(auth["amount"]["value"]) == job["budget_cents"],
        f"{fmt(job['budget_cents'])} on {job['authorization_id']}",
        "missing" if auth is None else f"{auth['amount']['value']} {auth.get('status')}",
    )
    caps = {c["id"]: c for c in pay["captures"] if c.get("status") in ("COMPLETED", "PARTIALLY_REFUNDED", "REFUNDED")}
    ours = db.all(
        "SELECT id, capture_id, charge_cents FROM settlements WHERE job_id=? AND kind='round' AND capture_state='DONE' "
        "AND charge_cents>0",
        job_id,
    )
    for s in ours:
        c = caps.get(s["capture_id"])
        check(
            f"Charge {s['capture_id']} matches",
            c is not None and to_cents(c["amount"]["value"]) == s["charge_cents"] and c.get("custom_id") == s["id"],
            f"{fmt(s['charge_cents'])} for settlement {s['id']}",
            "not found" if c is None else f"{c['amount']['value']} custom_id={c.get('custom_id')}",
        )
    unknown = sorted(set(caps) - {s["capture_id"] for s in ours})
    check("No charges Blindspot does not know about", not unknown, "none", ", ".join(unknown) or "none")
    paypal_captured = sum(to_cents(c["amount"]["value"]) for c in caps.values())
    check("Total charged", paypal_captured == job["captured_cents"], fmt(job["captured_cents"]), fmt(paypal_captured))
    refunds = [r for r in pay["refunds"] if r.get("status") in ("COMPLETED", "PENDING")]
    if refunds or job["refunded_cents"]:
        paypal_refunded = sum(to_cents(r["amount"]["value"]) for r in refunds)
        check(
            "Total refunded", paypal_refunded == job["refunded_cents"], fmt(job["refunded_cents"]), fmt(paypal_refunded)
        )
    if job["status"] == "CLOSED" and job["captured_cents"] < job["budget_cents"]:
        check(
            "Unused budget released",
            auth is not None and auth.get("status") in ("VOIDED", "CAPTURED"),
            f"released {fmt(job['budget_cents'] - job['captured_cents'])}",
            "missing" if auth is None else str(auth.get("status")),
        )
    if paypal is not None:  # the toolkit has no payouts tool: read each annotator payout with the Payouts API
        for s in db.all(
            "SELECT * FROM settlements WHERE job_id=? AND kind='round' AND payout_batch_id IS NOT NULL", job_id
        ):
            expected = {i["sender_item_id"]: i["amount"]["value"] for i in json.loads(s["payout_items"])}
            try:
                res = await paypal.get_payout(s["payout_batch_id"])
            except Exception:  # noqa: BLE001 - reported as a failed check
                check(f"Payout batch {s['payout_batch_id']}", False, "sent", "could not read")
                continue
            got = {
                i.get("payout_item", {}).get("sender_item_id"): (
                    i.get("payout_item", {}).get("amount", {}).get("value"),
                    i.get("transaction_status"),
                )
                for i in res.get("items", [])
            }
            amounts_ok = all(got.get(k, (None,))[0] == v for k, v in expected.items())
            statuses = sorted({st for _, st in got.values()})
            check(
                f"Payout batch {s['payout_batch_id']} (Payouts API)",
                amounts_ok and len(got) == len(expected),
                f"{len(expected)} people, {fmt(s['payout_cents'])}",
                f"{len(got)} items, {', '.join(statuses) or 'none'}",
            )
    result = {
        "ok": all(c["ok"] for c in checks),
        "order_id": job["paypal_order_id"],
        "checks": checks,
        "source": "PayPal Agent Toolkit get_order_details; Payouts API for annotator payouts",
    }
    db.event("audit_completed", job_id, ok=result["ok"], checks=checks, source=result["source"])
    return result


ASK_INSTRUCTIONS = """You answer a customer's questions about the money in their Blindspot learning job.
Blindspot is a sandbox demo: an agent buys labels from people with the customer's PayPal budget hold.
Use the tools to look things up; never guess amounts or ids. Prefer PayPal's own records (paypal_order) and say so when
you quote them. Blindspot's ledger (blindspot_ledger) explains why each amount was charged, paid or refunded.
You cannot move money and must not offer to. Answer in at most 5 short sentences or a short list, in plain English."""


async def ask_about_money(
    db: Database, reader: ToolkitReader, job_id: str, question: str, *, api_key: str, model: str = "gemini-3.8-flash"
) -> dict[str, Any]:
    from nexura_agents import Agent, Runner, tool
    from nexura_agents.providers.gemini import GeminiNativeProvider

    job = db.one("SELECT * FROM jobs WHERE id=?", job_id)
    if job is None:
        raise KeyError(job_id)
    order_id = job["paypal_order_id"]
    used: list[str] = []

    @tool(description="PayPal's own record of this job's order (via the PayPal Agent Toolkit): hold, charges, refunds.")
    async def paypal_order() -> str:
        used.append("paypal_order")
        if not order_id:
            return "This job has no PayPal order yet."
        order = await reader.order(order_id)
        pay = _payments(order)
        slim = {
            "order_id": order.get("id"),
            "status": order.get("status"),
            "authorizations": [
                {k: a.get(k) for k in ("id", "status", "amount", "expiration_time")} for a in pay["authorizations"]
            ],
            "captures": [
                {k: c.get(k) for k in ("id", "status", "amount", "custom_id", "create_time")} for c in pay["captures"]
            ],
            "refunds": [{k: r.get(k) for k in ("id", "status", "amount", "create_time")} for r in pay["refunds"]],
        }
        return json.dumps(slim)

    @tool(description="Blindspot's ledger for this job: each round's accepted labels, charge, payout and refunds.")
    def blindspot_ledger() -> str:
        used.append("blindspot_ledger")
        rows = db.all(
            "SELECT s.kind, r.number round, s.accepted, s.rejected, s.charge_cents, s.capture_id, s.capture_state, "
            "s.payout_cents, s.payout_batch_id, s.payout_state, s.refund_cents, s.refund_id, s.refund_state "
            "FROM settlements s LEFT JOIN rounds r ON r.id=s.round_id WHERE s.job_id=? ORDER BY s.created_at",
            job_id,
        )
        out = []
        for r in rows:
            d = dict(r)
            for k in ("charge_cents", "payout_cents", "refund_cents"):
                d[k.replace("_cents", "")] = fmt(d.pop(k))
            out.append(d)
        revocations = [
            {k: json.loads(e["data"]).get(k) for k in ("reason", "labels")}
            for e in db.all("SELECT data FROM events WHERE job_id=? AND kind='annotator_revoked' ORDER BY seq", job_id)
        ]
        return json.dumps(
            {
                "revocations (annotators caught cheating later; their earlier labels removed and refunded)": revocations,
                "budget": fmt(job["budget_cents"]),
                "price_per_label": fmt(job["price_cents"]),
                "annotator_pay_per_label": fmt(job["worker_cents"]),
                "status": job["status"],
                "settlements": out,
            }
        )

    agent = Agent(
        name="blindspot-money-questions",
        instructions=ASK_INSTRUCTIONS,
        model=GeminiNativeProvider(model, api_key=api_key, timeout=40.0),
        tools=[paypal_order, blindspot_ledger],
        max_turns=6,
    )
    result = await Runner().run_async(agent, question[:500], timeout=90)
    answer = str(result.final_output or "").strip()
    if result.status != "completed" or not answer:
        answer = "I could not answer that right now. The audit table shows PayPal's records for this job."
    db.event("money_question", job_id, question=question[:500], answer=answer[:2000], tools=used)
    return {"answer": answer, "tools": used}
