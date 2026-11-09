# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liceron
"""HTTP API for Blindspot (FastAPI).

Two audiences:
- the customer console (operator key): create and fund a learning job, run the buyer agent, watch money and accuracy;
- annotators (personal token from /api/annotators/join): get the next task, answer it, see their earnings.

The autopilot is the agent loop: plan a round (rules decide whether to buy) -> wait for answers (simulated annotators
answer at once; people answer on the annotator page) -> settle (charge, pay, revoke) -> repeat until a stop rule -> void.
"""

from __future__ import annotations

import asyncio
import hmac
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from blindspot.db import Database
from blindspot.learner import load_corpus
from blindspot.money import MoneyError
from blindspot.paypal import PayPal, PayPalError
from blindspot.records import ToolkitReader, ask_about_money, audit_job
from blindspot.service import Blindspot


class JobIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    budget: str = Field(pattern=r"^\d{1,4}(\.\d{1,2})?$")
    price: str = Field(default="0.50", pattern=r"^\d{1,3}(\.\d{1,2})?$")
    worker_pay: str = Field(default="0.40", pattern=r"^\d{1,3}(\.\d{1,2})?$")
    round_size: int = Field(default=40, ge=1, le=200)
    target_accuracy: float = Field(default=0.90, gt=0, le=1)
    min_gain_per_10: float = Field(default=0.5, ge=-100, le=100)
    seed_count: int = Field(default=200, ge=20, le=2000)
    strategy: str = Field(default="blindspot", pattern="^(blindspot|random)$")


class SimAnnotatorIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    paypal_email: str = Field(min_length=3, max_length=254)
    accuracy: float = Field(ge=0, le=1)
    late_accuracy: float | None = Field(default=None, ge=0, le=1)


class JoinIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    paypal_email: str = Field(min_length=3, max_length=254)
    invite: str = ""


class AnswerIn(BaseModel):
    task_id: str = Field(min_length=1, max_length=64)
    intent: int = Field(ge=-1, le=10_000)


class AskIn(BaseModel):
    question: str = Field(min_length=2, max_length=500)


class AutopilotIn(BaseModel):
    rounds: int = Field(default=10, ge=1, le=50)
    wait_seconds: float = Field(default=0, ge=0, le=1800)  # how long people get to answer a round
    close_when_done: bool = True


class Autopilot:
    """One background agent loop per job. State lives in the database; this only drives it."""

    def __init__(self, app: Blindspot, locks: dict[str, asyncio.Lock]) -> None:
        self.app, self.locks = app, locks
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.stop_reason: dict[str, str] = {}

    def running(self, job_id: str) -> bool:
        t = self.tasks.get(job_id)
        return t is not None and not t.done()

    def start(self, job_id: str, opts: AutopilotIn) -> None:
        if self.running(job_id):
            raise RuntimeError("autopilot already running for this job")
        self.tasks[job_id] = asyncio.create_task(self._run(job_id, opts), name=f"autopilot-{job_id}")

    async def stop(self, job_id: str, reason: str = "stopped by operator") -> None:
        self.stop_reason[job_id] = reason
        t = self.tasks.get(job_id)
        if t and not t.done():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    async def _run(self, job_id: str, opts: AutopilotIn) -> None:
        app = self.app
        app.db.event("autopilot_started", job_id, **opts.model_dump())
        try:
            for _ in range(opts.rounds):
                async with self.locks.setdefault(job_id, asyncio.Lock()):
                    open_round = app.db.one(
                        "SELECT id FROM rounds WHERE job_id=? AND status='LABELLING' ORDER BY number DESC", job_id
                    )
                    if open_round is None:
                        # with no waiting time only simulated annotators take part (people would get no time to answer)
                        kinds = "('human','simulated')" if opts.wait_seconds > 0 else "('simulated')"
                        anns = [
                            r["id"] for r in app.db.all(f"SELECT id FROM annotators WHERE banned=0 AND kind IN {kinds}")
                        ]
                        plan = await app.plan_round(job_id, anns)
                        if not plan["buy"]:
                            break
                        round_id = plan["round_id"]
                        app.simulate_answers(round_id, seed=plan["number"])
                    else:
                        round_id = open_round["id"]
                waited = 0.0
                while app.open_tasks(round_id) and waited < opts.wait_seconds:
                    await asyncio.sleep(1.0)
                    waited += 1.0
                async with self.locks.setdefault(job_id, asyncio.Lock()):
                    app.expire_open_tasks(round_id)
                    res = await app.settle_round(round_id)
                # a charge PayPal has not confirmed yet: wait for it (bounded), never start another purchase meanwhile
                tries = 0
                while res.get("pending") and tries < 30:
                    await asyncio.sleep(2.0)
                    tries += 1
                    async with self.locks.setdefault(job_id, asyncio.Lock()):
                        await app.settle.reconcile(job_id)
                        res = await app.finalize_round(round_id)
                if res.get("pending"):
                    app.db.event("autopilot_waiting", job_id, detail="PayPal has not confirmed a charge; paused")
                    return
                if app._job(job_id)["status"] != "ACTIVE":
                    break
            if opts.close_when_done:
                async with self.locks.setdefault(job_id, asyncio.Lock()):
                    await app.close_job(job_id)
            app.db.event("autopilot_finished", job_id)
        except asyncio.CancelledError:
            app.db.event("autopilot_stopped", job_id, reason=self.stop_reason.pop(job_id, "stopped by operator"))
            raise
        except Exception as exc:  # noqa: BLE001 - recorded for the operator; money state is in settlements
            app.db.event("autopilot_failed", job_id, error=f"{type(exc).__name__}: {exc}"[:300])


def create_app(
    service: Blindspot,
    *,
    operator_key: str | None,
    invite_code: str | None = None,
    paypal_client_id: str | None = None,
    static_dir: Path | None = None,
    experiment_path: Path | None = None,
    records: ToolkitReader | None = None,
) -> FastAPI:
    """operator_key=None means an open console (local development only)."""
    locks: dict[str, asyncio.Lock] = {}
    pilot = Autopilot(service, locks)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await service.resume()  # finish money calls and rounds a previous process left open
        # restart agent runs that a previous process was in the middle of (deploy or restart, not an operator stop)
        for r in service.db.all(
            "SELECT e.job_id, e.kind, e.data FROM events e JOIN (SELECT job_id, MAX(seq) seq FROM events "
            "WHERE kind LIKE 'autopilot_%' GROUP BY job_id) last ON last.seq = e.seq"
        ):
            data = json.loads(r["data"])
            interrupted = r["kind"] in ("autopilot_started", "autopilot_resumed") or (
                r["kind"] == "autopilot_stopped" and data.get("reason") == "server restart"
            )
            if interrupted and service._job(r["job_id"])["status"] == "ACTIVE":
                started = service.db.one(
                    "SELECT data FROM events WHERE job_id=? AND kind='autopilot_started' ORDER BY seq DESC LIMIT 1",
                    r["job_id"],
                )
                opts = AutopilotIn(
                    **{k: v for k, v in json.loads(started["data"]).items() if k in AutopilotIn.model_fields}
                )
                service.db.event("autopilot_resumed", r["job_id"], reason="server restarted")
                pilot.start(r["job_id"], opts)
        yield
        for job_id in list(pilot.tasks):
            await pilot.stop(job_id, reason="server restart")

    api = FastAPI(title="Blindspot Market", version="0.1.0", lifespan=lifespan)

    def operator(x_operator_key: str = Header(default="")) -> None:
        if operator_key is None:
            return
        given = x_operator_key  # header only: keys never appear in URLs or access logs
        if not given or not hmac.compare_digest(given.encode(), operator_key.encode()):
            raise HTTPException(401, "operator key required")

    def annotator(authorization: str = Header(default="")) -> str:
        token = authorization.removeprefix("Bearer ").strip()
        aid = service.annotator_for_token(token)
        if aid is None:
            raise HTTPException(401, "annotator token required")
        return aid

    def lock(job_id: str) -> asyncio.Lock:
        return locks.setdefault(job_id, asyncio.Lock())

    def job_or_404(job_id: str) -> Any:
        try:
            return service._job(job_id)
        except KeyError:
            raise HTTPException(404, "job not found") from None

    def guard(exc: Exception) -> HTTPException:
        if isinstance(exc, PayPalError):
            return HTTPException(502, f"PayPal: {exc.name} {exc.issues}")
        return HTTPException(409, str(exc))

    # --- public -------------------------------------------------------------------------------
    @api.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "paypal": "sandbox", "intents": service.c.n_classes, "pool": len(service.c.texts)}

    @api.get("/api/config")
    def config() -> dict[str, Any]:
        return {"paypal_client_id": paypal_client_id, "invite_required": bool(invite_code), "environment": "sandbox"}

    @api.get("/api/experiment")
    def experiment() -> dict[str, Any]:
        """The offline benchmark (Banking77, sealed official test split, 5 seeds): accuracy per dollar by strategy."""
        if experiment_path is None or not experiment_path.exists():
            raise HTTPException(404, "no experiment results")
        d = json.loads(experiment_path.read_text())
        seed_labels = d["labels"][0]
        return {
            "dataset": "Banking77 (PolyAI, CC-BY-4.0), 77 intents",
            "split": "official test split (3,080 messages), never used for any decision",
            "seeds": d["seeds"],
            "price": d["price"],
            "seed_labels": seed_labels,
            "points": [
                {
                    "labels": n,
                    "dollars": round((n - seed_labels) * d["price"], 2),
                    **{k: round(v[i], 4) for k, v in d["mean"].items()},
                }
                for i, n in enumerate(d["labels"])
            ],
            "per_seed": d["curves"],
        }

    # --- annotators ---------------------------------------------------------------------------
    @api.post("/api/annotators/join")
    def join(body: JoinIn) -> dict[str, Any]:
        if invite_code and not hmac.compare_digest(body.invite.encode(), invite_code.encode()):
            raise HTTPException(403, "invite code required")
        try:
            aid, token = service.add_human_annotator(body.name, body.paypal_email)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return {"annotator_id": aid, "token": token}

    @api.get("/api/me")
    def me(aid: str = Depends(annotator)) -> dict[str, Any]:
        return service.annotator_view(aid)

    @api.get("/api/me/task")
    def my_task(aid: str = Depends(annotator)) -> dict[str, Any]:
        return {"task": service.next_task(aid), "me": service.annotator_view(aid)}

    @api.post("/api/me/answer")
    def my_answer(body: AnswerIn, aid: str = Depends(annotator)) -> dict[str, Any]:
        try:
            service.answer(body.task_id, aid, body.intent)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        return {"task": service.next_task(aid), "me": service.annotator_view(aid)}

    # --- operator console ---------------------------------------------------------------------
    @api.post("/api/annotators/simulated", dependencies=[Depends(operator)])
    def add_simulated(body: SimAnnotatorIn) -> dict[str, Any]:
        aid = service.add_annotator(
            body.name, body.paypal_email, simulated_accuracy=body.accuracy, simulated_late_accuracy=body.late_accuracy
        )
        return {"annotator_id": aid}

    @api.get("/api/annotators", dependencies=[Depends(operator)])
    def list_annotators() -> list[dict[str, Any]]:
        return [
            service.annotator_view(r["id"]) for r in service.db.all("SELECT id FROM annotators ORDER BY created_at")
        ]

    @api.post("/api/jobs", dependencies=[Depends(operator)])
    def create_job(body: JobIn) -> dict[str, Any]:
        try:
            jid = service.create_job(
                body.name,
                budget=body.budget,
                price=body.price,
                worker_pay=body.worker_pay,
                round_size=body.round_size,
                target_accuracy=body.target_accuracy,
                min_gain_per_10=body.min_gain_per_10,
                seed_count=body.seed_count,
                strategy=body.strategy,
            )
        except (ValueError, MoneyError) as exc:
            raise HTTPException(422, str(exc)) from None
        return {"job_id": jid}

    @api.get("/api/jobs", dependencies=[Depends(operator)])
    def list_jobs() -> list[dict[str, Any]]:
        return [dict(r) for r in service.db.all("SELECT * FROM jobs ORDER BY created_at DESC")]

    @api.get("/api/jobs/{job_id}", dependencies=[Depends(operator)])
    def get_job(job_id: str) -> dict[str, Any]:
        job_or_404(job_id)
        return {
            **service.summary(job_id),
            "rounds": service.rounds(job_id),
            "settlements": service.settlements(job_id),
            "autopilot": pilot.running(job_id),
        }

    @api.post("/api/jobs/{job_id}/fund/card", dependencies=[Depends(operator)])
    async def fund_card(job_id: str) -> dict[str, Any]:
        job_or_404(job_id)
        async with lock(job_id):
            try:
                return await service.fund_with_card(job_id)
            except (PayPalError, RuntimeError) as exc:
                raise guard(exc) from None

    @api.post("/api/jobs/{job_id}/fund/wallet/order", dependencies=[Depends(operator)])
    async def wallet_order(job_id: str, request: Request) -> dict[str, Any]:
        """For the PayPal JS SDK createOrder callback: returns the order id the buyer approves in the popup."""
        job_or_404(job_id)
        base = str(request.base_url).rstrip("/")
        async with lock(job_id):
            try:
                return await service.start_wallet_funding(
                    job_id, f"{base}/?funded={job_id}", f"{base}/?cancel={job_id}"
                )
            except PayPalError as exc:
                raise guard(exc) from None

    @api.post("/api/jobs/{job_id}/fund/wallet/authorize", dependencies=[Depends(operator)])
    async def wallet_authorize(job_id: str) -> dict[str, Any]:
        """For the JS SDK onApprove callback: places the hold after the buyer approved."""
        job_or_404(job_id)
        async with lock(job_id):
            try:
                return await service.finish_wallet_funding(job_id)
            except (PayPalError, RuntimeError) as exc:
                raise guard(exc) from None

    @api.post("/api/jobs/{job_id}/autopilot", dependencies=[Depends(operator)])
    async def start_autopilot(job_id: str, body: AutopilotIn) -> dict[str, Any]:
        if job_or_404(job_id)["status"] != "ACTIVE":
            raise HTTPException(409, "fund the job first")
        try:
            pilot.start(job_id, body)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None
        return {"autopilot": True}

    @api.delete("/api/jobs/{job_id}/autopilot", dependencies=[Depends(operator)])
    async def stop_autopilot(job_id: str) -> dict[str, Any]:
        await pilot.stop(job_id)
        return {"autopilot": False}

    @api.post("/api/jobs/{job_id}/close", dependencies=[Depends(operator)])
    async def close(job_id: str) -> dict[str, Any]:
        job_or_404(job_id)
        await pilot.stop(job_id)
        async with lock(job_id):
            try:
                return await service.close_job(job_id)
            except (PayPalError, RuntimeError) as exc:
                raise guard(exc) from None

    @api.post("/api/jobs/{job_id}/reconcile", dependencies=[Depends(operator)])
    async def reconcile(job_id: str) -> dict[str, Any]:
        job_or_404(job_id)
        async with lock(job_id):
            await service.settle.reconcile(job_id)
            for s in service.settlements(job_id):
                if s["payout_state"] in ("SUBMITTED", "ITEMS_PENDING"):
                    await service.settle.refresh_payout(s["id"])
        return {"settlements": service.settlements(job_id)}

    @api.post("/api/jobs/{job_id}/audit", dependencies=[Depends(operator)])
    async def audit(job_id: str) -> dict[str, Any]:
        """Check the ledger against PayPal's own records, read through the PayPal Agent Toolkit."""
        job_or_404(job_id)
        if records is None:
            raise HTTPException(503, "PayPal Agent Toolkit reader not configured")
        try:
            await service.settle.reconcile(job_id)
            return await audit_job(service.db, records, job_id, paypal=service.paypal)
        except (RuntimeError, TimeoutError) as exc:
            raise HTTPException(409, str(exc) or "PayPal did not answer in time") from None
        except Exception as exc:  # noqa: BLE001 - the toolkit raises requests errors
            raise HTTPException(502, f"PayPal Agent Toolkit: {type(exc).__name__}") from None

    @api.post("/api/jobs/{job_id}/ask", dependencies=[Depends(operator)])
    async def ask(job_id: str, body: AskIn) -> dict[str, Any]:
        """Gemini answers questions about this job's money with read-only tools (PayPal Agent Toolkit + ledger)."""
        job_or_404(job_id)
        if records is None or not service.gemini_allowed():
            raise HTTPException(503, "money questions need the toolkit reader and a Gemini key")
        try:
            return await ask_about_money(service.db, records, job_id, body.question, api_key=service.gemini_key or "")
        except Exception as exc:  # noqa: BLE001 - shown to the operator, never moves money
            raise HTTPException(502, f"could not answer: {type(exc).__name__}") from None

    @api.get("/api/jobs/{job_id}/events", dependencies=[Depends(operator)])
    async def stream(job_id: str, request: Request, after: int = 0) -> StreamingResponse:
        """Server-sent events: every recorded step of the job, in order, then live as it happens."""
        job_or_404(job_id)

        async def gen() -> AsyncIterator[str]:
            last = after
            idle = 0
            while not await request.is_disconnected():
                batch = service.events(job_id, after=last)
                for e in batch:
                    last = e["seq"]
                    yield f"id: {e['seq']}\nevent: {e['kind']}\ndata: {json.dumps(e)}\n\n"
                idle = 0 if batch else idle + 1
                if idle and idle % 30 == 0:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})

    if static_dir is not None and (static_dir / "index.html").exists():
        api.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @api.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(404, "not found")
            return FileResponse(static_dir / "index.html")

    return api


def main() -> None:  # pragma: no cover - process entry point
    import uvicorn

    from blindspot.settings import gemini_key, load_env

    env = load_env()
    gemini = gemini_key(env)
    service = Blindspot(
        Database(env.get("BLINDSPOT_DB", "blindspot.db")),
        PayPal(env["PAYPAL_CLIENT_ID"], env["PAYPAL_CLIENT_SECRET"]),
        load_corpus(Path(env.get("BLINDSPOT_DATA", "data"))),
        gemini_key=gemini,
    )
    service.gemini_daily_limit = int(env.get("GEMINI_DAILY_LIMIT", "300"))
    operator_key = env.get("BLINDSPOT_OPERATOR_KEY")
    if not operator_key and env.get("BLINDSPOT_OPEN_CONSOLE") != "1":
        raise SystemExit("set BLINDSPOT_OPERATOR_KEY (or BLINDSPOT_OPEN_CONSOLE=1 for local development)")
    app = create_app(
        service,
        operator_key=operator_key or None,
        invite_code=env.get("BLINDSPOT_INVITE") or None,
        paypal_client_id=env["PAYPAL_CLIENT_ID"],
        static_dir=Path(env.get("BLINDSPOT_WEB", "web/dist")),
        experiment_path=Path(env.get("BLINDSPOT_EXPERIMENT", "experiment/results.json")),
        records=ToolkitReader(env["PAYPAL_CLIENT_ID"], env["PAYPAL_CLIENT_SECRET"]),
    )
    uvicorn.run(app, host=env.get("HOST", "127.0.0.1"), port=int(env.get("PORT", "8077")))
