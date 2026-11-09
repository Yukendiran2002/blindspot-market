# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liceron
"""Blindspot service: a learning job from funding to close.

job (budget, price per label) -> PayPal budget hold -> rounds:
  agent decides (rules) -> blind-spot tasks + hidden gold questions -> annotators answer
  -> audit -> charge accepted labels (partial capture) -> pay annotators (Payouts) -> retrain -> measure
-> stop rule -> void the rest of the hold.
Simulated annotators are labelled as simulated everywhere; their answers come from a fixed accuracy, not from people.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from typing import Any

import numpy as np

from blindspot import agent as agent_mod
from blindspot.db import Database, new_id
from blindspot.learner import Corpus, Model, pretty
from blindspot.money import fmt, show
from blindspot.paypal import PayPal, PayPalDefinite, PayPalUncertain, authorizations_of
from blindspot.settlement import MIN_GOLD, TRUST_MIN, Settlement, gold_record, trust

GOLD_EVERY = 8  # one hidden gold question per this many learning tasks, per annotator (at least MIN_GOLD)
OPTIONS = 5  # model suggestions shown to an annotator (they can also search all intents)
# PayPal's honour period: captures are most likely to succeed in the first 3 days of an authorization
# (https://developer.paypal.com/payment-methods/auth-honor/). A job buys labels only inside this window.
HOLD_WINDOW_S = 3 * 24 * 3600
SANDBOX_CARD = {
    "number": "4111111111111111",
    "expiry": "2030-12",
    "security_code": "123",
    "name": "Demo Customer",
    "billing_address": {
        "address_line_1": "1 Main St",
        "admin_area_2": "San Jose",
        "admin_area_1": "CA",
        "postal_code": "95131",
        "country_code": "US",
    },
}


class Blindspot:
    def __init__(
        self, db: Database, paypal: PayPal, corpus: Corpus, *, gemini_key: str | None = None, clock=time.time
    ) -> None:
        self.db, self.paypal, self.c, self.clock = db, paypal, corpus, clock
        self.settle = Settlement(db, paypal, clock)
        self.gemini_key = gemini_key
        self._models: dict[str, Model] = {}

    # --- annotators ---------------------------------------------------------------------------
    def add_annotator(
        self,
        name: str,
        paypal_email: str,
        *,
        simulated_accuracy: float | None = None,
        simulated_late_accuracy: float | None = None,
    ) -> str:
        aid = new_id("ann")
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO annotators(id, name, paypal_email, kind, sim_accuracy, sim_late, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    aid,
                    name,
                    paypal_email.strip().lower(),
                    "simulated" if simulated_accuracy is not None else "human",
                    simulated_accuracy,
                    simulated_late_accuracy,
                    self.clock(),
                ),
            )
        return aid

    def add_human_annotator(self, name: str, paypal_email: str) -> tuple[str, str]:
        """A real person. Returns (annotator id, access token); only the token's hash is stored."""
        name, email = name.strip()[:60], paypal_email.strip().lower()
        if not name or "@" not in email or len(email) > 254 or any(c.isspace() for c in email):
            raise ValueError("a name and a PayPal email are required")
        aid = self.add_annotator(name, email)
        token = secrets.token_urlsafe(24)
        with self.db.tx() as conn:
            conn.execute("UPDATE annotators SET token_hash=? WHERE id=?", (_hash(token), aid))
            self.db.event("annotator_joined", None, conn=conn, annotator_id=aid, name=name)
        return aid, token

    def annotator_for_token(self, token: str) -> str | None:
        if not token:
            return None
        row = self.db.one("SELECT id FROM annotators WHERE token_hash=? AND kind='human'", _hash(token))
        return row["id"] if row else None

    def annotator_view(self, annotator_id: str) -> dict[str, Any]:
        a = self.db.one("SELECT * FROM annotators WHERE id=?", annotator_id)
        if a is None:
            raise KeyError(annotator_id)
        counts = {
            r["status"]: r["n"]
            for r in self.db.all(
                "SELECT t.status, COUNT(*) n FROM tasks t WHERE t.annotator_id=? GROUP BY t.status", annotator_id
            )
        }
        earned = self.db.one(
            "SELECT COALESCE(SUM(j.worker_cents),0) c FROM tasks t JOIN jobs j ON j.id=t.job_id "
            "WHERE t.annotator_id=? AND t.status IN ('ACCEPTED','REVOKED')",
            annotator_id,
        )["c"]
        return {
            "id": a["id"],
            "name": a["name"],
            "kind": a["kind"],
            "banned": bool(a["banned"]),
            "open": counts.get("OPEN", 0),
            "answered": sum(counts.get(k, 0) for k in ("ANSWERED", "ACCEPTED", "REJECTED", "REVOKED")),
            "accepted": counts.get("ACCEPTED", 0),
            "earned": fmt(earned),
        }

    # --- jobs ---------------------------------------------------------------------------------
    def create_job(
        self,
        name: str,
        *,
        budget: str,
        price: str = "0.50",
        worker_pay: str = "0.40",
        round_size: int = 100,
        target_accuracy: float = 0.90,
        min_gain_per_10: float = 0.5,
        seed_count: int = 200,
        strategy: str = "blindspot",
        seed: int = 0,
    ) -> str:
        from blindspot.money import parse_user_amount

        budget_c, price_c, worker_c = (parse_user_amount(x) for x in (budget, price, worker_pay))
        if not (0 < worker_c <= price_c) or budget_c < price_c or strategy not in ("blindspot", "random"):
            raise ValueError("invalid job terms")
        jid = new_id("job")
        rng = np.random.default_rng(seed)
        seed_idx = rng.choice(len(self.c.texts), seed_count, replace=False)
        now = self.clock()
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO jobs(id, name, status, price_cents, worker_cents, budget_cents, round_size, "
                "target_accuracy, min_gain_per_10, strategy, seed_count, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    jid,
                    name,
                    "DRAFT",
                    price_c,
                    worker_c,
                    budget_c,
                    round_size,
                    target_accuracy,
                    min_gain_per_10,
                    strategy,
                    seed_count,
                    now,
                    now,
                ),
            )
            for i in seed_idx:  # the customer's existing labelled examples
                conn.execute(
                    "INSERT INTO labels(job_id, pool_index, label, source) VALUES (?,?,?,?)",
                    (jid, int(i), int(self.c.labels[i]), "seed"),
                )
            self.db.event(
                "job_created",
                jid,
                conn=conn,
                name=name,
                budget=fmt(budget_c),
                price=fmt(price_c),
                worker_pay=fmt(worker_c),
                seed_labels=seed_count,
                strategy=strategy,
            )
        m = self._model(jid)
        self.db.event(
            "model_measured",
            jid,
            labels=seed_count,
            validation=m.accuracy("validation"),
            sealed=m.accuracy("sealed"),
            spent="0.00",
            charged="0.00",
        )
        return jid

    async def fund_with_card(self, job_id: str) -> dict[str, Any]:
        """Sandbox test card: PayPal authorizes the hold immediately."""
        job = self._job(job_id)
        rid = f"fund{job_id}"
        order = await self.paypal.create_budget_order(
            fmt(job["budget_cents"]),
            job_id=job_id,
            request_id=rid,
            description=f"Blindspot learning budget: {job['name']}",
            card=SANDBOX_CARD,
        )
        return self._record_hold(job_id, order)

    async def start_wallet_funding(self, job_id: str, return_url: str, cancel_url: str) -> dict[str, Any]:
        job = self._job(job_id)
        order = await self.paypal.create_budget_order(
            fmt(job["budget_cents"]),
            job_id=job_id,
            request_id=f"fundw{job_id}",
            description=f"Blindspot learning budget: {job['name']}",
            return_url=return_url,
            cancel_url=cancel_url,
        )
        with self.db.tx() as conn:
            conn.execute(
                "UPDATE jobs SET status='FUNDING', paypal_order_id=?, updated_at=? WHERE id=?",
                (order["id"], self.clock(), job_id),
            )
        approve = next((l["href"] for l in order.get("links", []) if l.get("rel") in ("payer-action", "approve")), None)
        return {"order_id": order["id"], "approve_url": approve}

    async def finish_wallet_funding(self, job_id: str) -> dict[str, Any]:
        job = self._job(job_id)
        order = await self.paypal.authorize_order(job["paypal_order_id"], request_id=f"auth{job_id}")
        return self._record_hold(job_id, order)

    def _record_hold(self, job_id: str, order: dict[str, Any]) -> dict[str, Any]:
        auths = authorizations_of(order)
        if not auths or auths[0].get("status") not in ("CREATED", "PENDING"):
            raise RuntimeError(f"PayPal did not create a budget hold (order status {order.get('status')})")
        a = auths[0]
        job = self._job(job_id)
        if a["amount"]["value"] != fmt(job["budget_cents"]) or a["amount"]["currency_code"] != "USD":
            raise RuntimeError("hold amount differs from the job budget")
        with self.db.tx() as conn:
            conn.execute(
                "UPDATE jobs SET status='ACTIVE', paypal_order_id=?, authorization_id=?, hold_status=?, "
                "funded_at=COALESCE(funded_at, ?), updated_at=? WHERE id=?",
                (order["id"], a["id"], a["status"], self.clock(), self.clock(), job_id),
            )
            self.db.event(
                "budget_held",
                job_id,
                conn=conn,
                order_id=order["id"],
                authorization_id=a["id"],
                amount=a["amount"]["value"],
                expires=a.get("expiration_time"),
            )
        return {"order_id": order["id"], "authorization_id": a["id"], "amount": a["amount"]["value"]}

    # --- rounds -------------------------------------------------------------------------------
    async def plan_round(self, job_id: str, annotator_ids: list[str]) -> dict[str, Any]:
        job = self._job(job_id)
        if job["status"] != "ACTIVE":
            raise RuntimeError(f"job is {job['status']}")
        if self.db.one("SELECT 1 FROM rounds WHERE job_id=? AND status IN ('LABELLING','SETTLING')", job_id):
            raise RuntimeError("a round is already open")
        if job["funded_at"] is not None and self.clock() - job["funded_at"] > HOLD_WINDOW_S:
            detail = "The budget hold is past PayPal's 3-day honour period; no more labels are bought on it."
            self.db.event("agent_stopped", job_id, reason="hold_window", detail=detail, gain_per_10=None)
            return {"buy": False, "decision": {"buy": False, "n": 0, "reason": "hold_window", "detail": detail}}
        active = [a for a in annotator_ids if not self.db.one("SELECT banned FROM annotators WHERE id=?", a)["banned"]]
        if not active:
            raise RuntimeError("no annotators available")
        m = self._model(job_id)
        val = m.accuracy("validation")
        history = self._history(job_id)
        labelled = {r["pool_index"] for r in self.db.all("SELECT pool_index FROM labels WHERE job_id=?", job_id)}
        pending = {
            r["pool_index"]
            for r in self.db.all(
                "SELECT pool_index FROM tasks WHERE job_id=? AND kind='learn' AND status!='EXPIRED'", job_id
            )
        }
        available = np.array(sorted(set(range(len(self.c.texts))) - labelled - pending))
        open_cents = job["captured_cents"]
        d = agent_mod.decide(
            val_acc=val,
            target=job["target_accuracy"],
            history=history,
            price_cents=job["price_cents"],
            money_left_cents=job["budget_cents"] - open_cents,
            round_size=job["round_size"],
            min_gain_per_10=job["min_gain_per_10"],
            pool_left=len(available),
        )
        number = (self.db.one("SELECT COALESCE(MAX(number),0) n FROM rounds WHERE job_id=?", job_id)["n"]) + 1
        if not d.buy:
            d.explanation = d.detail
            self.db.event(
                "agent_stopped", job_id, round=number, reason=d.reason, detail=d.detail, gain_per_10=d.gain_per_10
            )
            return {"buy": False, "decision": json.loads(d.to_json())}
        if job["strategy"] == "random":
            rng = np.random.default_rng(number)
            d.picks = [
                {"pool_index": i, "margin": None, "confused": None, "cluster": None}
                for i in m.random_picks(available, d.n, rng)
            ]
            confusions: list[dict[str, Any]] = []
        else:
            d.picks = m.blind_spots(available, d.n, seed=number)
            confusions = agent_mod.summarize_picks(d.picks, self.c.intents)
        d.explanation = agent_mod.template_explanation(d, confusions)
        if self.gemini_key:
            try:
                d.explanation = await agent_mod.llm_explanation(self.gemini_key, d, confusions)
            except Exception as exc:  # noqa: BLE001 - the template explanation stands
                self.db.event("explanation_fallback", job_id, error=type(exc).__name__)
        guide: list[dict[str, str]] = []
        if self.gemini_key and confusions:
            try:
                guide = await agent_mod.llm_guidance(self.gemini_key, confusions)
            except Exception:  # noqa: BLE001 - tips are optional help for annotators
                guide = []
        rid = new_id("rnd")
        now = self.clock()
        preds = m.predict(np.array([p["pool_index"] for p in d.picks]), top=OPTIONS)
        seeds = [
            r["pool_index"]
            for r in self.db.all("SELECT pool_index FROM labels WHERE job_id=? AND source='seed'", job_id)
        ]
        rng = np.random.default_rng(number + 7)
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO rounds(id, job_id, number, status, plan, val_before, sealed_before, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    rid,
                    job_id,
                    number,
                    "LABELLING",
                    json.dumps({**json.loads(d.to_json()), "confusions": confusions, "guide": guide}),
                    val,
                    m.accuracy("sealed"),
                    now,
                ),
            )
            for k, (pick, pred) in enumerate(zip(d.picks, preds, strict=True)):
                ann = active[k % len(active)]
                conn.execute(
                    "INSERT INTO tasks(id, job_id, round_id, pool_index, kind, options, annotator_id, status) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        new_id("tsk"),
                        job_id,
                        rid,
                        pick["pool_index"],
                        "learn",
                        json.dumps([o for o, _ in pred]),
                        ann,
                        "OPEN",
                    ),
                )
            for ann in active:  # hidden gold questions: the customer's own labelled examples
                share = sum(1 for k in range(len(d.picks)) if active[k % len(active)] == ann)
                for gi in rng.choice(seeds, max(MIN_GOLD, share // GOLD_EVERY), replace=False):
                    gpred = m.predict(np.array([gi]), top=OPTIONS)[0]
                    conn.execute(
                        "INSERT INTO tasks(id, job_id, round_id, pool_index, kind, gold_label, options, annotator_id, "
                        "status) VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            new_id("tsk"),
                            job_id,
                            rid,
                            int(gi),
                            "gold",
                            int(self.c.labels[gi]),
                            json.dumps([o for o, _ in gpred]),
                            ann,
                            "OPEN",
                        ),
                    )
            self.db.event(
                "round_planned",
                job_id,
                conn=conn,
                round=number,
                round_id=rid,
                labels=d.n,
                cost=show(d.est_cost_cents),
                validation=val,
                explanation=d.explanation,
                confusions=confusions,
            )
        return {
            "buy": True,
            "round_id": rid,
            "number": number,
            "decision": json.loads(d.to_json()),
            "confusions": confusions,
        }

    def next_task(self, annotator_id: str) -> dict[str, Any] | None:
        t = self.db.one("SELECT * FROM tasks WHERE annotator_id=? AND status='OPEN' ORDER BY id LIMIT 1", annotator_id)
        if t is None:
            return None
        opts = json.loads(t["options"])
        plan = json.loads(self.db.one("SELECT plan FROM rounds WHERE id=?", t["round_id"])["plan"])
        labels = {pretty(self.c.intents[o]) for o in opts}
        tips = [g for g in plan.get("guide", []) if g["a"] in labels or g["b"] in labels]
        return {
            "task_id": t["id"],
            "tips": tips[:2],
            "text": self.c.texts[t["pool_index"]],
            "options": [{"id": o, "label": pretty(self.c.intents[o])} for o in opts],
            "all_intents": [{"id": i, "label": pretty(n)} for i, n in enumerate(self.c.intents)],
        }

    def answer(self, task_id: str, annotator_id: str, intent: int) -> None:
        if not (-1 <= int(intent) < self.c.n_classes):
            raise ValueError("unknown intent")
        with self.db.tx() as conn:
            cur = conn.execute(
                "UPDATE tasks SET answer=?, status='ANSWERED', answered_at=? WHERE id=? AND annotator_id=? "
                "AND status='OPEN'",
                (int(intent), self.clock(), task_id, annotator_id),
            )
            if cur.rowcount != 1:
                raise ValueError("task is not open for this annotator")

    def open_tasks(self, round_id: str) -> int:
        return self.db.one("SELECT COUNT(*) n FROM tasks WHERE round_id=? AND status='OPEN'", round_id)["n"]

    def expire_open_tasks(self, round_id: str) -> int:
        """Tasks nobody answered in time: never charged, never paid; their messages can be bought again later."""
        with self.db.tx() as conn:
            cur = conn.execute("UPDATE tasks SET status='EXPIRED' WHERE round_id=? AND status='OPEN'", (round_id,))
            if cur.rowcount:
                job_id = conn.execute("SELECT job_id FROM rounds WHERE id=?", (round_id,)).fetchone()["job_id"]
                self.db.event("tasks_expired", job_id, conn=conn, round_id=round_id, count=cur.rowcount)
        return cur.rowcount

    def simulate_answers(self, round_id: str, seed: int = 0) -> int:
        """SIMULATED annotators only: answer from their fixed accuracy (wrong answers are random intents)."""
        rng = np.random.default_rng(seed)
        n = 0
        number = self.db.one("SELECT number FROM rounds WHERE id=?", round_id)["number"]
        for t in self.db.all(
            "SELECT t.id, t.pool_index, t.annotator_id, a.sim_accuracy, a.sim_late FROM tasks t JOIN "
            "annotators a ON a.id=t.annotator_id WHERE t.round_id=? AND t.status='OPEN' "
            "AND a.kind='simulated'",
            round_id,
        ):
            truth = int(self.c.labels[t["pool_index"]])
            acc = t["sim_late"] if (t["sim_late"] is not None and number >= 2) else t["sim_accuracy"]
            ans = truth if rng.random() < acc else int(rng.integers(self.c.n_classes))
            self.answer(t["id"], t["annotator_id"], ans)
            n += 1
        return n

    async def settle_round(self, round_id: str) -> dict[str, Any]:
        rnd = self.db.one("SELECT * FROM rounds WHERE id=?", round_id)
        job = self._job(rnd["job_id"])
        if rnd["status"] != "LABELLING":
            raise RuntimeError(f"round is {rnd['status']}")
        if self.db.one("SELECT 1 FROM tasks WHERE round_id=? AND status='OPEN'", round_id):
            raise RuntimeError("tasks still open")
        sid = self.settle.prepare_round(job, round_id)
        await self.settle.charge(sid)
        await self.settle.pay(sid)
        # late evidence: anyone whose overall gold record fell below the bar loses earlier accepted labels
        for a in self.db.all("SELECT DISTINCT annotator_id FROM tasks WHERE job_id=?", job["id"]):
            correct, total = gold_record(self.db, job["id"], a["annotator_id"])
            banned = self.db.one("SELECT banned FROM annotators WHERE id=?", a["annotator_id"])["banned"]
            if not banned and total >= MIN_GOLD and trust(correct, total) < TRUST_MIN:
                await self.settle.revoke_annotator(
                    self._job(job["id"]), a["annotator_id"], f"gold accuracy {correct}/{total} below the trust bar"
                )
        m = self._model(job["id"], refit=True)
        val, sealed = m.accuracy("validation"), m.accuracy("sealed")
        job = self._job(job["id"])
        with self.db.tx() as conn:
            conn.execute(
                "UPDATE rounds SET status='SETTLED', val_after=?, sealed_after=?, settled_at=? WHERE id=?",
                (val, sealed, self.clock(), round_id),
            )
            n_labels = conn.execute(
                "SELECT COUNT(*) n FROM labels WHERE job_id=? AND active=1", (job["id"],)
            ).fetchone()["n"]
            self.db.event(
                "model_measured",
                job["id"],
                conn=conn,
                round=rnd["number"],
                labels=n_labels,
                validation=val,
                sealed=sealed,
                spent=fmt(job["captured_cents"] - job["refunded_cents"]),
                charged=fmt(job["captured_cents"]),
            )
        s = self.db.one("SELECT * FROM settlements WHERE id=?", sid)
        return {
            "settlement_id": sid,
            "accepted": s["accepted"],
            "rejected": s["rejected"],
            "charged": fmt(s["charge_cents"]),
            "capture": s["capture_state"],
            "paid": fmt(s["payout_cents"]),
            "payout": s["payout_state"],
            "validation": val,
            "sealed": sealed,
        }

    async def close_job(self, job_id: str) -> dict[str, Any]:
        job = self._job(job_id)
        if job["status"] == "CLOSED":
            return {"status": "CLOSED"}
        await self.settle.reconcile(job_id)
        rid = job["void_request_id"] or new_id("void")
        with self.db.tx() as conn:
            conn.execute(
                "UPDATE jobs SET status='CLOSING', void_request_id=?, updated_at=? WHERE id=?",
                (rid, self.clock(), job_id),
            )
        left = job["budget_cents"] - job["captured_cents"]
        if not job["authorization_id"]:
            state = "NEVER_FUNDED"
        elif left <= 0:
            state = "FULLY_USED"  # the whole hold was charged; nothing to release
        else:
            try:
                await self.paypal.void_authorization(job["authorization_id"], request_id=rid)
                state = "VOIDED"
            except PayPalDefinite as exc:
                state = "ALREADY_CLOSED" if exc.status == 422 else f"VOID_REFUSED {exc.name}"
            except PayPalUncertain:
                return {"status": "CLOSING", "note": "void outcome unknown; retry close"}
        with self.db.tx() as conn:
            conn.execute(
                "UPDATE jobs SET status='CLOSED', hold_status=?, updated_at=? WHERE id=?", (state, self.clock(), job_id)
            )
            self.db.event(
                "budget_released",
                job_id,
                conn=conn,
                released=fmt(left),
                hold=state,
                spent=fmt(job["captured_cents"] - job["refunded_cents"]),
            )
        return {"status": "CLOSED", "released": fmt(left), "hold": state}

    # --- views --------------------------------------------------------------------------------
    def summary(self, job_id: str) -> dict[str, Any]:
        job = self._job(job_id)
        curve = [
            json.loads(e["data"])
            for e in self.db.all(
                "SELECT data FROM events WHERE job_id=? AND kind='model_measured' ORDER BY seq", job_id
            )
        ]
        anns = []
        for a in self.db.all("SELECT * FROM annotators"):
            c, t = gold_record(self.db, job_id, a["id"])
            if t:
                earned = (
                    self.db.one(
                        "SELECT COUNT(*) n FROM tasks WHERE job_id=? AND annotator_id=? AND status IN "
                        "('ACCEPTED','REVOKED')",
                        job_id,
                        a["id"],
                    )["n"]
                    * job["worker_cents"]
                )  # paid work
                anns.append(
                    {
                        "id": a["id"],
                        "name": a["name"],
                        "kind": a["kind"],
                        "banned": bool(a["banned"]),
                        "gold": f"{c}/{t}",
                        "trust": round(trust(c, t), 2),
                        "earned": fmt(earned),
                    }
                )
        return {
            "job": dict(job),
            "curve": curve,
            "annotators": anns,
            "spent": fmt(job["captured_cents"] - job["refunded_cents"]),
            "left": fmt(job["budget_cents"] - job["captured_cents"]),
        }

    def rounds(self, job_id: str) -> list[dict[str, Any]]:
        out = []
        for r in self.db.all("SELECT * FROM rounds WHERE job_id=? ORDER BY number", job_id):
            plan = json.loads(r["plan"])
            s = self.db.one("SELECT * FROM settlements WHERE round_id=? AND kind='round'", r["id"])
            out.append(
                {
                    "id": r["id"],
                    "number": r["number"],
                    "status": r["status"],
                    "labels_planned": plan.get("n"),
                    "explanation": plan.get("explanation"),
                    "confusions": plan.get("confusions", []),
                    "guide": plan.get("guide", []),
                    "val_before": r["val_before"],
                    "val_after": r["val_after"],
                    "sealed_after": r["sealed_after"],
                    "open_tasks": self.open_tasks(r["id"]),
                    "settlement": None if s is None else _settlement_view(s),
                }
            )
        return out

    def settlements(self, job_id: str) -> list[dict[str, Any]]:
        return [
            _settlement_view(s)
            for s in self.db.all("SELECT * FROM settlements WHERE job_id=? ORDER BY created_at", job_id)
        ]

    def events(self, job_id: str, after: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        return [
            {"seq": e["seq"], "at": e["at"], "kind": e["kind"], "data": json.loads(e["data"])}
            for e in self.db.all(
                "SELECT * FROM events WHERE (job_id=? OR job_id IS NULL) AND seq>? ORDER BY seq LIMIT ?",
                job_id,
                after,
                limit,
            )
        ]

    # --- internals ----------------------------------------------------------------------------
    def _job(self, job_id: str) -> Any:
        row = self.db.one("SELECT * FROM jobs WHERE id=?", job_id)
        if row is None:
            raise KeyError(job_id)
        return row

    def _model(self, job_id: str, refit: bool = False) -> Model:
        if refit or job_id not in self._models:
            rows = self.db.all("SELECT pool_index, label FROM labels WHERE job_id=? AND active=1 AND label>=0", job_id)
            m = Model(self.c)
            m.fit(np.array([r["pool_index"] for r in rows]), np.array([r["label"] for r in rows]))
            self._models[job_id] = m
        return self._models[job_id]

    def _history(self, job_id: str) -> list[tuple[int, float]]:
        out = []
        for e in self.db.all("SELECT data FROM events WHERE job_id=? AND kind='model_measured' ORDER BY seq", job_id):
            d = json.loads(e["data"])
            # gross money committed, not net of refunds: refunds must not make a round look free or negative
            whole, frac = d.get("charged", d["spent"]).split(".")
            out.append((int(whole) * 100 + int(frac), d["validation"]))
        return out


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _settlement_view(s: Any) -> dict[str, Any]:
    return {
        "id": s["id"],
        "kind": s["kind"],
        "round_id": s["round_id"],
        "accepted": s["accepted"],
        "rejected": s["rejected"],
        "charge": fmt(s["charge_cents"]),
        "capture_id": s["capture_id"],
        "capture_state": s["capture_state"],
        "refund": fmt(s["refund_cents"]),
        "refund_id": s["refund_id"],
        "refund_state": s["refund_state"],
        "payout": fmt(s["payout_cents"]),
        "payout_batch_id": s["payout_batch_id"],
        "payout_state": s["payout_state"],
        "error": s["error"],
    }
