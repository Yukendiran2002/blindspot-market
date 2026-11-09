# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liceron
"""Money movement for Blindspot. The only module that calls PayPal's money-moving endpoints.

A round settles in this order, each step recorded before it is attempted:
1. Quality audit: annotators answer hidden gold questions; an annotator below the trust bar has the round's labels rejected.
2. Charge: the customer pays only for accepted labels, as a partial capture of their budget hold
   (PayPal-Request-Id = custom_id = settlement id, so a retry returns the same capture).
3. Pay: one Payouts batch pays every annotator for accepted work (sender_batch_id = settlement id; a replay is
   refused by PayPal, so nobody is paid twice). Payouts happen only after the capture is COMPLETED, and each
   recipient's item is tracked until SUCCESS.
Accepted labels train the model only once their charge is COMPLETED. A charge whose outcome is not final blocks new
purchases and closing.
Later evidence can revoke an annotator: their earlier labels leave the training set and the customer is refunded
for them (refund of that round's capture). Closing a job voids whatever is left of the hold.
Outcomes PayPal does not confirm are UNCERTAIN and settled by :func:`reconcile`, never by sending a new request.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any

from blindspot.db import Database, new_id
from blindspot.money import fmt
from blindspot.paypal import PayPal, PayPalDefinite, PayPalUncertain, captures_of, payout_batch_from_links

TRUST_MIN = 0.70  # smoothed gold accuracy an annotator must keep
MIN_GOLD = 2  # gold answers needed before trust can reject


def trust(correct: int, total: int) -> float:
    return (correct + 1) / (total + 2)


def gold_record(db: Database, job_id: str, annotator_id: str) -> tuple[int, int]:
    rows = db.all(
        "SELECT answer, gold_label FROM tasks WHERE job_id=? AND annotator_id=? AND kind='gold' AND answer IS NOT NULL",
        job_id,
        annotator_id,
    )
    return sum(1 for r in rows if r["answer"] == r["gold_label"]), len(rows)


def audit_round(db: Database, job: Any, round_id: str) -> dict[str, Any]:
    """Accept or reject every answered task of the round; returns per-annotator results.

    An annotator's work is accepted only with enough evidence: at least MIN_GOLD answered test questions over the job,
    and a smoothed test accuracy at or above TRUST_MIN. Too few answered test questions is INSUFFICIENT_EVIDENCE: the
    work is neither charged nor paid (it fails closed, never open)."""
    tasks = db.all("SELECT * FROM tasks WHERE round_id=? AND status='ANSWERED'", round_id)
    by_annotator: dict[str, list[Any]] = defaultdict(list)
    for t in tasks:
        by_annotator[t["annotator_id"]].append(t)
    result = {}
    for annotator_id, items in by_annotator.items():
        correct, total = gold_record(db, job["id"], annotator_id)
        score = trust(correct, total)
        if total < MIN_GOLD:
            verdict = "INSUFFICIENT_EVIDENCE"
        elif score < TRUST_MIN:
            verdict = "FAILED_QUALITY_CHECKS"
        else:
            verdict = "PASSED"
        ok = verdict == "PASSED"
        learn = [t for t in items if t["kind"] == "learn"]
        gold_ok = [t for t in items if t["kind"] == "gold" and t["answer"] == t["gold_label"]]
        accepted_learn = [t for t in learn if ok and t["answer"] is not None and t["answer"] >= 0]
        result[annotator_id] = {
            "trust": round(score, 3),
            "gold": f"{correct}/{total}",
            "ok": ok,
            "verdict": verdict,
            "accepted": [t["id"] for t in accepted_learn],
            "rejected": [t["id"] for t in learn if t not in accepted_learn],
            "paid_tasks": len(accepted_learn) + (len(gold_ok) if ok else 0),
        }
    return result


TERMINAL_ITEM_FAILURES = {"FAILED", "RETURNED", "BLOCKED", "REFUNDED", "REVERSED", "DENIED"}
UNRESOLVED_CAPTURE = ("SENDING", "UNCERTAIN", "PENDING", "REVIEW")
RESEND_WINDOW_S = 24 * 3600  # after this, an unknown capture is quarantined for a person instead of resent


class Settlement:
    def __init__(self, db: Database, paypal: PayPal, clock=time.time) -> None:
        self.db, self.paypal, self.clock = db, paypal, clock

    # --- round settlement -----------------------------------------------------------------------
    def prepare_round(self, job: Any, round_id: str) -> str:
        """Audit, record accepted labels (inactive until paid for) and the money to move. Idempotent per round."""
        existing = self.db.one("SELECT id FROM settlements WHERE round_id=? AND kind='round'", round_id)
        if existing:
            return existing["id"]
        audit = audit_round(self.db, job, round_id)
        sid = new_id("stl")
        now = self.clock()
        accepted = [tid for a in audit.values() for tid in a["accepted"]]
        rejected = [tid for a in audit.values() for tid in a["rejected"]]
        charge = len(accepted) * job["price_cents"]
        hold_left = job["budget_cents"] - job["captured_cents"]
        if charge > hold_left:
            raise RuntimeError(f"charge {charge} exceeds hold remaining {hold_left}; refusing")
        items = []
        for annotator_id, a in audit.items():
            if a["paid_tasks"] <= 0:
                continue
            ann = self.db.one("SELECT * FROM annotators WHERE id=?", annotator_id)
            items.append(
                {
                    "recipient_type": "EMAIL",
                    "receiver": ann["paypal_email"],
                    "amount": {"value": fmt(a["paid_tasks"] * job["worker_cents"]), "currency": "USD"},
                    "note": f"Blindspot: {a['paid_tasks']} accepted labels for job {job['name']}"[:200],
                    "sender_item_id": f"{sid}-{annotator_id}"[:63],
                    "_annotator": annotator_id,
                    "_tasks": a["paid_tasks"],
                    "_status": "OWED",
                }
            )
        payout = sum(a["paid_tasks"] for a in audit.values()) * job["worker_cents"]
        with self.db.tx() as conn:
            for tid in accepted:
                conn.execute("UPDATE tasks SET status='ACCEPTED' WHERE id=?", (tid,))
                t = conn.execute("SELECT pool_index, answer FROM tasks WHERE id=?", (tid,)).fetchone()
                # inactive until the customer's charge for it is COMPLETED: unpaid labels never train the model
                conn.execute(
                    "INSERT OR REPLACE INTO labels(job_id, pool_index, label, source, task_id, active) "
                    "VALUES (?,?,?,?,?,0)",
                    (job["id"], t["pool_index"], t["answer"], "bought", tid),
                )
            for tid in rejected:
                conn.execute("UPDATE tasks SET status='REJECTED' WHERE id=?", (tid,))
            for a_id, a in audit.items():
                # gold answers: paid (ACCEPTED) only when correct and the annotator passed; never charged to the customer
                conn.execute(
                    "UPDATE tasks SET status=CASE WHEN ?=1 AND answer=gold_label THEN 'ACCEPTED' ELSE 'REJECTED' END "
                    "WHERE round_id=? AND annotator_id=? AND kind='gold' AND status='ANSWERED'",
                    (1 if a["ok"] else 0, round_id, a_id),
                )
            conn.execute(
                "INSERT INTO settlements(id, job_id, round_id, kind, accepted, rejected, charge_cents, capture_state, "
                "payout_cents, payout_items, payout_state, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sid,
                    job["id"],
                    round_id,
                    "round",
                    len(accepted),
                    len(rejected),
                    charge,
                    "SENDING" if charge else "DONE",
                    payout,
                    json.dumps(items),
                    "NONE",
                    now,
                    now,
                ),
            )
            conn.execute("UPDATE rounds SET status='SETTLING' WHERE id=?", (round_id,))
            self.db.event(
                "round_audited",
                job["id"],
                conn=conn,
                round_id=round_id,
                settlement_id=sid,
                accepted=len(accepted),
                rejected=len(rejected),
                charge=fmt(charge),
                payout=fmt(payout),
                annotators={
                    k: {kk: v[kk] for kk in ("trust", "gold", "ok", "verdict", "paid_tasks")} for k, v in audit.items()
                },
            )
        return sid

    async def charge(self, sid: str) -> None:
        """Capture the accepted labels from the hold. Safe to call repeatedly in any state.

        A crash or timeout after a request was sent is never answered by a blind resend: the order is read first and
        the capture looked up by custom_id. Only a successful read that shows no capture allows a resend, with the
        same PayPal-Request-Id, and only inside RESEND_WINDOW_S; after that the settlement goes to REVIEW."""
        s = self._s(sid)
        state = s["capture_state"]
        if state == "PENDING":
            await self._poll_capture(s)
            return
        if state not in ("SENDING", "UNCERTAIN"):
            return
        job = self.db.one("SELECT * FROM jobs WHERE id=?", s["job_id"])
        if s["capture_tries"] > 0:  # a request may have reached PayPal
            found = await self._find_capture(job, sid)
            if found != "absent":
                return  # recorded, or PayPal could not be read: try again later, never guess
            if self.clock() - s["created_at"] > RESEND_WINDOW_S:
                self._set(sid, capture_state="REVIEW", error="capture outcome unknown past the safe resend window")
                self.db.event("charge_needs_review", s["job_id"], settlement_id=sid)
                return
        with self.db.tx() as conn:
            conn.execute("UPDATE settlements SET capture_tries=capture_tries+1 WHERE id=?", (sid,))
        try:
            cap = await self.paypal.capture_authorization(
                job["authorization_id"],
                amount=fmt(s["charge_cents"]),
                request_id=sid,
                custom_id=sid,
                invoice_id=sid,
                note=f"Blindspot: {s['accepted']} accepted labels",
            )
        except PayPalUncertain as exc:
            self._set(sid, capture_state="UNCERTAIN", error=str(exc))
            self.db.event("charge_uncertain", s["job_id"], settlement_id=sid, error=str(exc))
            return
        except PayPalDefinite as exc:
            self._fail_charge(self._s(sid), exc.name, exc.issues)
            return
        self._record_capture(job, self._s(sid), cap)

    async def pay(self, sid: str) -> None:
        s = self._s(sid)
        if s["capture_state"] != "DONE" or s["payout_state"] not in ("NONE", "SENDING", "UNCERTAIN"):
            return
        items = json.loads(s["payout_items"])
        if not items:
            self._set(sid, payout_state="DONE")
            return
        self._set(sid, payout_state="SENDING")
        job = self.db.one("SELECT name FROM jobs WHERE id=?", s["job_id"])
        clean = [{k: v for k, v in i.items() if not k.startswith("_")} for i in items]
        try:
            res = await self.paypal.create_payout(
                sender_batch_id=sid,
                items=clean,
                subject="You were paid for teaching an AI",
                message=f"Thanks for your labels on '{job['name']}'. Your accepted work is paid below.",
            )
        except PayPalUncertain as exc:
            self._set(sid, payout_state="UNCERTAIN", error=str(exc))
            self.db.event("payout_uncertain", s["job_id"], settlement_id=sid, error=str(exc))
            return
        except PayPalDefinite as exc:
            if any("already exists" in m for m in (exc.issues + [str(exc)])):
                # Our earlier attempt reached PayPal: never pay twice. PayPal links the original batch in the error.
                batch = payout_batch_from_links(exc.links)
                if batch:
                    self._set(sid, payout_state="SUBMITTED", payout_batch_id=batch, error=None)
                    self.db.event("payout_already_submitted", s["job_id"], settlement_id=sid, payout_batch_id=batch)
                else:
                    self._set(sid, payout_state="REVIEW", error="duplicate batch but original batch id not returned")
                    self.db.event("payout_needs_review", s["job_id"], settlement_id=sid)
                return
            self._set(
                sid,
                payout_state="FAILED",
                error=json.dumps({"status": exc.status, "name": exc.name, "issues": exc.issues}),
            )
            self.db.event("payout_refused", s["job_id"], settlement_id=sid, name=exc.name, issues=exc.issues)
            return
        batch = res.get("batch_header", {})
        self._set(sid, payout_state="SUBMITTED", payout_batch_id=batch.get("payout_batch_id"), error=None)
        self.db.event(
            "payout_submitted",
            s["job_id"],
            settlement_id=sid,
            payout_batch_id=batch.get("payout_batch_id"),
            batch_status=batch.get("batch_status"),
            amount=fmt(s["payout_cents"]),
            recipients=[{"annotator": i["_annotator"], "amount": i["amount"]["value"]} for i in items],
        )

    async def refresh_payout(self, sid: str) -> dict[str, Any] | None:
        """Track every expected recipient, not the batch: SUBMITTED -> ITEMS_PENDING / ITEM_FAILED / DONE."""
        s = self._s(sid)
        if not s["payout_batch_id"] or s["payout_state"] not in ("SUBMITTED", "ITEMS_PENDING"):
            return None
        try:
            res = await self.paypal.get_payout(s["payout_batch_id"])
        except (PayPalUncertain, PayPalDefinite):
            return None
        batch_status = res.get("batch_header", {}).get("batch_status")
        got = {
            i.get("payout_item", {}).get("sender_item_id"): {
                "status": i.get("transaction_status"),
                "item_id": i.get("payout_item_id"),
                "amount": i.get("payout_item", {}).get("amount", {}).get("value"),
            }
            for i in res.get("items", [])
        }
        items = json.loads(s["payout_items"])
        for it in items:
            seen = got.get(it["sender_item_id"])
            if seen and seen["amount"] == it["amount"]["value"]:
                it["_status"], it["_item_id"] = seen["status"], seen["item_id"]
            elif seen:
                it["_status"] = "AMOUNT_MISMATCH"
        statuses = [it.get("_status") for it in items]
        if batch_status in ("DENIED", "CANCELED") or any(st in TERMINAL_ITEM_FAILURES for st in statuses):
            state = "ITEM_FAILED"
        elif "AMOUNT_MISMATCH" in statuses:
            state = "REVIEW"
        elif statuses and all(st == "SUCCESS" for st in statuses):
            state = "DONE"
        else:
            state = "ITEMS_PENDING"  # UNCLAIMED, PENDING, ONHOLD, not yet listed...
        changed = state != s["payout_state"] or json.dumps(items) != s["payout_items"]
        if changed:
            self._set(sid, payout_state=state, payout_items=json.dumps(items))
            self.db.event(
                "payout_checked",
                s["job_id"],
                settlement_id=sid,
                batch_status=batch_status,
                state=state,
                items=[{"annotator": it["_annotator"], "status": it.get("_status")} for it in items],
            )
        return {"status": batch_status, "state": state, "items": statuses}

    # --- revocation (late evidence) --------------------------------------------------------------
    async def revoke_annotator(self, job: Any, annotator_id: str, reason: str) -> list[str]:
        """Ban an annotator; deactivate their previously accepted labels and refund the customer for them."""
        rows = self.db.all(
            "SELECT t.id, t.round_id, t.pool_index FROM tasks t WHERE t.job_id=? AND t.annotator_id=? "
            "AND t.kind='learn' AND t.status='ACCEPTED'",
            job["id"],
            annotator_id,
        )
        by_round: dict[str, list[Any]] = defaultdict(list)
        for r in rows:
            by_round[r["round_id"]].append(r)
        created = []
        now = self.clock()
        with self.db.tx() as conn:
            conn.execute("UPDATE annotators SET banned=1 WHERE id=?", (annotator_id,))
            for round_id, items in by_round.items():
                orig = conn.execute(
                    "SELECT * FROM settlements WHERE round_id=? AND kind='round'", (round_id,)
                ).fetchone()
                if orig is None or orig["capture_state"] != "DONE" or not orig["capture_id"]:
                    continue
                for r in items:
                    conn.execute("UPDATE tasks SET status='REVOKED' WHERE id=?", (r["id"],))
                    conn.execute("UPDATE labels SET active=0 WHERE job_id=? AND task_id=?", (job["id"], r["id"]))
                sid = new_id("rev")
                cents = len(items) * job["price_cents"]
                conn.execute(
                    "INSERT INTO settlements(id, job_id, round_id, kind, rejected, refund_cents, refund_capture_id, "
                    "refund_state, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        sid,
                        job["id"],
                        round_id,
                        "revocation",
                        len(items),
                        cents,
                        orig["capture_id"],
                        "SENDING",
                        now,
                        now,
                    ),
                )
                created.append(sid)
            self.db.event(
                "annotator_revoked",
                job["id"],
                conn=conn,
                annotator_id=annotator_id,
                reason=reason,
                labels=len(rows),
                refunds=created,
            )
        for sid in created:
            await self.refund(sid)
        return created

    async def refund(self, sid: str) -> None:
        """Refund revoked labels. Counted as refunded only when PayPal reports the refund COMPLETED."""
        s = self._s(sid)
        if s["refund_state"] == "PENDING" and s["refund_id"]:
            try:
                res = await self.paypal.get_refund(s["refund_id"])
            except (PayPalUncertain, PayPalDefinite):
                return
            self._record_refund(s, res)
            return
        if s["refund_state"] not in ("SENDING", "UNCERTAIN"):
            return
        try:
            res = await self.paypal.refund_capture(
                s["refund_capture_id"],
                amount=fmt(s["refund_cents"]),
                request_id=sid,
                custom_id=sid,
                note="Blindspot refund: labels from an annotator who failed quality checks",
            )
        except PayPalUncertain as exc:
            self._set(sid, refund_state="UNCERTAIN", error=str(exc))
            return
        except PayPalDefinite as exc:
            self._set(sid, refund_state="FAILED", error=json.dumps({"name": exc.name, "issues": exc.issues}))
            self.db.event("refund_refused", s["job_id"], settlement_id=sid, name=exc.name, issues=exc.issues)
            return
        self._record_refund(self._s(sid), res)

    def _record_refund(self, s: Any, res: dict[str, Any]) -> None:
        status = res.get("status")
        amount_ok = (
            res.get("amount", {}).get("value") == fmt(s["refund_cents"])
            and res.get("amount", {}).get("currency_code", "USD") == "USD"
        )
        if not amount_ok:
            self._set(s["id"], refund_state="REVIEW", error="refund amount differs from what was asked")
            return
        if status == "PENDING":
            if s["refund_state"] != "PENDING":
                self._set(s["id"], refund_state="PENDING", refund_id=res.get("id"))
                self.db.event("refund_pending", s["job_id"], settlement_id=s["id"], refund_id=res.get("id"))
            return
        if status in ("FAILED", "CANCELLED"):
            self._set(s["id"], refund_state="FAILED", refund_id=res.get("id"), error=f"refund {status}")
            self.db.event("refund_refused", s["job_id"], settlement_id=s["id"], name=status, issues=[])
            return
        if status != "COMPLETED":
            self._set(s["id"], refund_state="UNCERTAIN", error=f"unrecognised refund status {status}")
            return
        with self.db.tx() as conn:
            cur = conn.execute(
                "UPDATE settlements SET refund_state='DONE', refund_id=?, error=NULL, updated_at=? "
                "WHERE id=? AND refund_state!='DONE'",
                (res.get("id"), self.clock(), s["id"]),
            )
            if cur.rowcount:
                conn.execute(
                    "UPDATE jobs SET refunded_cents=refunded_cents+?, updated_at=? WHERE id=?",
                    (s["refund_cents"], self.clock(), s["job_id"]),
                )
                self.db.event(
                    "refunded",
                    s["job_id"],
                    conn=conn,
                    settlement_id=s["id"],
                    refund_id=res.get("id"),
                    amount=fmt(s["refund_cents"]),
                    status=status,
                )

    # --- recovery -------------------------------------------------------------------------------
    def unresolved(self, job_id: str) -> bool:
        """A charge whose outcome is not final: no new purchase and no close until it is resolved."""
        marks = ",".join("?" * len(UNRESOLVED_CAPTURE))
        return bool(
            self.db.one(
                f"SELECT 1 FROM settlements WHERE job_id=? AND capture_state IN ({marks})", job_id, *UNRESOLVED_CAPTURE
            )
        )

    async def reconcile(self, job_id: str | None = None) -> None:
        q = (
            "SELECT id FROM settlements WHERE (capture_state IN ('SENDING','UNCERTAIN','PENDING') OR payout_state IN "
            "('SENDING','UNCERTAIN','SUBMITTED','ITEMS_PENDING') OR refund_state IN ('SENDING','UNCERTAIN','PENDING') "
            "OR (kind='round' AND capture_state='DONE' AND payout_state='NONE'))"
        )
        rows = self.db.all(q + (" AND job_id=?" if job_id else ""), *([job_id] if job_id else []))
        for r in rows:
            s = self._s(r["id"])
            if s["capture_state"] in ("SENDING", "UNCERTAIN", "PENDING"):
                await self.charge(s["id"])
            if self._s(s["id"])["payout_state"] in ("NONE", "SENDING", "UNCERTAIN") and s["kind"] == "round":
                await self.pay(s["id"])  # only once the charge is DONE; a replayed sender_batch_id is refused
            if self._s(s["id"])["payout_state"] in ("SUBMITTED", "ITEMS_PENDING"):
                await self.refresh_payout(s["id"])
            if s["refund_state"] in ("SENDING", "UNCERTAIN", "PENDING"):
                await self.refund(s["id"])

    # --- internals ------------------------------------------------------------------------------
    async def _find_capture(self, job: Any, sid: str) -> str:
        """'found' (and recorded), 'absent' (order read, no such capture) or 'unknown' (PayPal could not be read)."""
        try:
            order = await self.paypal.get_order(job["paypal_order_id"])
        except (PayPalUncertain, PayPalDefinite):
            return "unknown"
        for c in captures_of(order):
            if c.get("custom_id") == sid:
                self._record_capture(job, self._s(sid), c)
                return "found"
        return "absent"

    async def _poll_capture(self, s: Any) -> None:
        try:
            cap = await self.paypal.get_capture(s["capture_id"])
        except (PayPalUncertain, PayPalDefinite):
            return
        job = self.db.one("SELECT * FROM jobs WHERE id=?", s["job_id"])
        self._record_capture(job, s, cap)

    def _record_capture(self, job: Any, s: Any, cap: dict[str, Any]) -> None:
        """COMPLETED -> DONE (labels start training, payouts may go); PENDING -> PENDING (hold reserved, nobody paid);
        DECLINED/FAILED -> charge failed; anything else, or a wrong amount or currency -> REVIEW."""
        status = cap.get("status")
        amount = cap.get("amount", {})
        if amount.get("value") != fmt(s["charge_cents"]) or amount.get("currency_code", "USD") != "USD":
            self._set(s["id"], capture_state="REVIEW", capture_id=cap.get("id"), error="capture amount differs")
            self.db.event("charge_needs_review", s["job_id"], settlement_id=s["id"])
            return
        if status in ("DECLINED", "FAILED"):
            self._fail_charge(s, status, [], capture_id=cap.get("id"))
            return
        if status not in ("COMPLETED", "PENDING"):
            self._set(s["id"], capture_state="REVIEW", capture_id=cap.get("id"), error=f"capture status {status}")
            return
        new_state = "DONE" if status == "COMPLETED" else "PENDING"
        with self.db.tx() as conn:
            prev = conn.execute("SELECT capture_state FROM settlements WHERE id=?", (s["id"],)).fetchone()[
                "capture_state"
            ]
            if prev == new_state or prev in ("DONE", "FAILED"):
                return
            conn.execute(
                "UPDATE settlements SET capture_state=?, capture_id=?, error=NULL, updated_at=? WHERE id=?",
                (new_state, cap["id"], self.clock(), s["id"]),
            )
            if prev != "PENDING":  # the hold is used from the moment PayPal takes the capture, pending or not
                conn.execute(
                    "UPDATE jobs SET captured_cents=captured_cents+?, updated_at=? WHERE id=?",
                    (s["charge_cents"], self.clock(), s["job_id"]),
                )
            if new_state == "DONE":
                conn.execute(
                    "UPDATE labels SET active=1 WHERE job_id=? AND task_id IN "
                    "(SELECT id FROM tasks WHERE round_id=? AND kind='learn' AND status='ACCEPTED')",
                    (s["job_id"], s["round_id"]),
                )
            self.db.event(
                "charged" if new_state == "DONE" else "charge_pending",
                s["job_id"],
                conn=conn,
                settlement_id=s["id"],
                capture_id=cap["id"],
                amount=fmt(s["charge_cents"]),
                status=status,
            )

    def _fail_charge(self, s: Any, name: str, issues: list[str], capture_id: str | None = None) -> None:
        with self.db.tx() as conn:
            prev = conn.execute("SELECT capture_state FROM settlements WHERE id=?", (s["id"],)).fetchone()[
                "capture_state"
            ]
            conn.execute(
                "UPDATE settlements SET capture_state='FAILED', capture_id=COALESCE(?, capture_id), error=?, "
                "updated_at=? WHERE id=?",
                (capture_id, json.dumps({"name": name, "issues": issues}), self.clock(), s["id"]),
            )
            if prev == "PENDING":  # the pending capture had reserved part of the hold
                conn.execute(
                    "UPDATE jobs SET captured_cents=captured_cents-? WHERE id=?", (s["charge_cents"], s["job_id"])
                )
            # The customer did not pay for these labels: they must not train the model.
            conn.execute(
                "UPDATE labels SET active=0 WHERE job_id=? AND task_id IN "
                "(SELECT id FROM tasks WHERE round_id=? AND kind='learn')",
                (s["job_id"], s["round_id"]),
            )
            conn.execute(
                "UPDATE jobs SET status='PAYMENT_PROBLEM', updated_at=? WHERE id=?", (self.clock(), s["job_id"])
            )
            self.db.event("charge_refused", s["job_id"], conn=conn, settlement_id=s["id"], name=name, issues=issues)

    def _s(self, sid: str) -> Any:
        row = self.db.one("SELECT * FROM settlements WHERE id=?", sid)
        if row is None:
            raise KeyError(sid)
        return row

    def _set(self, sid: str, **fields: Any) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.db.tx() as conn:
            conn.execute(
                f"UPDATE settlements SET {cols}, updated_at=? WHERE id=?", (*fields.values(), self.clock(), sid)
            )
