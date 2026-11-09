# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liceron
"""HTTP API and autopilot tests against the LOCAL FAKE PayPal and the synthetic corpus."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from test_blindspot import Env, run

from blindspot.api import create_app

KEY = {"X-Operator-Key": "op-secret"}


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


@pytest.fixture
def client(env):
    with TestClient(create_app(env.app, operator_key="op-secret", invite_code="join-me")) as c:
        yield c


def wait_for(client, job_id, cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}", headers=KEY).json()
        if cond(j):
            return j
        time.sleep(0.2)
    raise AssertionError(f"timed out; job is {j['job']['status']}, autopilot={j['autopilot']}")


def new_job(client, **kw):
    body = {
        "name": "api test",
        "budget": "30.00",
        "round_size": 20,
        "seed_count": 60,
        "target_accuracy": 0.99,
        "min_gain_per_10": -100,
        **kw,
    }
    r = client.post("/api/jobs", json=body, headers=KEY)
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    assert client.post(f"/api/jobs/{jid}/fund/card", headers=KEY).status_code == 200
    return jid


def join(client, name="Priya", email="priya@x.test"):
    r = client.post("/api/annotators/join", json={"name": name, "paypal_email": email, "invite": "join-me"})
    assert r.status_code == 200, r.text
    return r.json()["annotator_id"], {"Authorization": f"Bearer {r.json()['token']}"}


def test_console_needs_the_operator_key(client):
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/api/jobs", headers={"X-Operator-Key": "wrong"}).status_code == 401
    assert client.get("/api/jobs", headers=KEY).status_code == 200
    assert client.get("/api/health").json()["ok"]


def test_join_needs_invite_and_token_scopes_the_annotator(client):
    r = client.post("/api/annotators/join", json={"name": "X", "paypal_email": "x@x.test", "invite": "nope"})
    assert r.status_code == 403
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/me", headers={"Authorization": "Bearer forged"}).status_code == 401
    _, h = join(client)
    me = client.get("/api/me", headers=h).json()
    assert me["name"] == "Priya" and me["kind"] == "human" and me["earned"] == "0.00"


def test_autopilot_runs_a_whole_job_with_simulated_annotators(client, env):
    env.annotators(("Ana", "ana@x.test", 1.0, None), ("Spam", "spam@x.test", 0.0, None))
    jid = new_job(client)
    r = client.post(f"/api/jobs/{jid}/autopilot", json={"rounds": 2, "wait_seconds": 0}, headers=KEY)
    assert r.status_code == 200
    j = wait_for(client, jid, lambda j: j["job"]["status"] == "CLOSED" and not j["autopilot"])
    assert len(j["rounds"]) == 2 and all(r["status"] == "SETTLED" for r in j["rounds"])
    assert env.fake.paid_to("ana@x.test") > 0 and env.fake.paid_to("spam@x.test") == 0
    kinds = [e["kind"] for e in env.app.events(jid)]
    for k in ("budget_held", "round_planned", "charged", "payout_submitted", "budget_released", "autopilot_finished"):
        assert k in kinds, k
    aid = env.app.db.one("SELECT authorization_id FROM jobs WHERE id=?", jid)["authorization_id"]
    assert env.fake.auths[aid]["status"] == "VOIDED"


def test_a_person_labels_on_the_annotator_page_and_is_paid(client, env):
    _aid, h = join(client)
    jid = new_job(client)
    client.post(f"/api/jobs/{jid}/autopilot", json={"rounds": 1, "wait_seconds": 60}, headers=KEY)
    deadline = time.time() + 20
    task = None
    while task is None and time.time() < deadline:  # the agent is planning the round
        task = client.get("/api/me/task", headers=h).json()["task"]
        time.sleep(0.1)
    answered = 0
    while task is not None:
        truth = int(env.c.labels[int(task["text"].split()[-1])])  # synthetic texts are "message <pool index>"
        r = client.post("/api/me/answer", json={"task_id": task["task_id"], "intent": truth}, headers=h)
        assert r.status_code == 200
        task = r.json()["task"]
        answered += 1
    j = wait_for(client, jid, lambda j: j["job"]["status"] == "CLOSED")
    assert answered >= 20
    s = j["rounds"][0]["settlement"]
    assert s["accepted"] == 20 and s["capture_state"] == "DONE" and s["payout_state"] == "DONE"
    assert env.fake.paid_to("priya@x.test") > 0
    assert client.get("/api/me", headers=h).json()["accepted"] >= 20


def test_tasks_nobody_answers_expire_and_cost_nothing(client, env):
    join(client)
    jid = new_job(client)
    client.post(f"/api/jobs/{jid}/autopilot", json={"rounds": 1, "wait_seconds": 1}, headers=KEY)
    j = wait_for(client, jid, lambda j: j["job"]["status"] == "CLOSED")
    assert j["rounds"][0]["settlement"]["charge"] == "0.00"
    assert not env.fake.captures and not env.fake.payouts
    expired = env.app.db.one("SELECT COUNT(*) n FROM tasks WHERE job_id=? AND status='EXPIRED'", jid)["n"]
    assert expired > 0


def test_answering_someone_elses_task_is_refused(client, env):
    _, h1 = join(client, "A", "a@x.test")
    _, h2 = join(client, "B", "b@x.test")
    jid = new_job(client)
    client.post(f"/api/jobs/{jid}/autopilot", json={"rounds": 1, "wait_seconds": 30}, headers=KEY)
    deadline = time.time() + 20
    task = None
    while task is None and time.time() < deadline:
        task = client.get("/api/me/task", headers=h1).json()["task"]
        time.sleep(0.1)
    r = client.post("/api/me/answer", json={"task_id": task["task_id"], "intent": 0}, headers=h2)
    assert r.status_code == 409
    client.delete(f"/api/jobs/{jid}/autopilot", headers=KEY)


def test_unclaimed_payout_item_is_not_reported_as_paid(client, env):
    env.annotators(("Ana", "ana@x.test", 1.0, None), ("Ben", "ben@x.test", 1.0, None))
    env.fake.item_status["ben@x.test"] = "UNCLAIMED"
    jid = new_job(client)
    client.post(f"/api/jobs/{jid}/autopilot", json={"rounds": 1, "wait_seconds": 0}, headers=KEY)
    wait_for(client, jid, lambda j: j["job"]["status"] == "CLOSED")
    states = [s["payout_state"] for s in client.post(f"/api/jobs/{jid}/reconcile", headers=KEY).json()["settlements"]]
    assert states == ["ITEMS_PENDING"]
    env.fake.item_status.clear()
    states = [s["payout_state"] for s in client.post(f"/api/jobs/{jid}/reconcile", headers=KEY).json()["settlements"]]
    assert states == ["DONE"]


def test_no_labels_bought_after_the_hold_honour_period(client, env):
    env.annotators(("Ana", "ana@x.test", 1.0, None))
    jid = new_job(client)
    with env.app.db.tx() as conn:
        conn.execute("UPDATE jobs SET funded_at=funded_at-4*86400 WHERE id=?", (jid,))
    anns = [r["id"] for r in env.app.db.all("SELECT id FROM annotators")]
    plan = run(env.app.plan_round(jid, anns))
    assert not plan["buy"] and plan["decision"]["reason"] == "hold_window"


def test_autopilot_needs_a_funded_job(client):
    r = client.post("/api/jobs", json={"name": "x", "budget": "10.00", "seed_count": 60}, headers=KEY)
    jid = r.json()["job_id"]
    assert client.post(f"/api/jobs/{jid}/autopilot", json={}, headers=KEY).status_code == 409
    assert client.get("/api/jobs/nope", headers=KEY).status_code == 404


def _kinds(env, jid):
    return [(e["kind"], e.get("data", {})) for e in env.app.events(jid)]


def test_an_autopilot_cut_by_a_restart_resumes_on_startup(env):
    env.annotators(("Ana", "ana@x.test", 1.0, None))
    with TestClient(create_app(env.app, operator_key="op-secret", invite_code="join-me")) as c:
        join(c)  # a person holds open tasks, so the run is still waiting when the server goes down
        jid = new_job(c)
        assert (
            c.post(f"/api/jobs/{jid}/autopilot", json={"rounds": 2, "wait_seconds": 2}, headers=KEY).status_code == 200
        )
        end = time.time() + 30
        while "round_planned" not in [k for k, _ in _kinds(env, jid)] and time.time() < end:
            time.sleep(0.1)
    kinds = [k for k, _ in _kinds(env, jid)]
    assert kinds[-1] == "autopilot_stopped"
    env.restart()
    with TestClient(create_app(env.app, operator_key="op-secret", invite_code="join-me")) as c:
        j = wait_for(c, jid, lambda j: j["job"]["status"] == "CLOSED" and not j["autopilot"])
    kinds = [k for k, _ in _kinds(env, jid)]
    assert "autopilot_resumed" in kinds and kinds.count("autopilot_finished") == 1
    assert all(r["status"] == "SETTLED" for r in j["rounds"])


def test_an_operator_stop_is_not_resumed_after_a_restart(env):
    env.annotators(("Ana", "ana@x.test", 1.0, None))
    with TestClient(create_app(env.app, operator_key="op-secret", invite_code="join-me")) as c:
        join(c)
        jid = new_job(c)
        c.post(f"/api/jobs/{jid}/autopilot", json={"rounds": 2, "wait_seconds": 30}, headers=KEY)
        assert c.delete(f"/api/jobs/{jid}/autopilot", headers=KEY).status_code == 200
    env.restart()
    with TestClient(create_app(env.app, operator_key="op-secret", invite_code="join-me")) as c:
        time.sleep(0.5)
        assert not c.get(f"/api/jobs/{jid}", headers=KEY).json()["autopilot"]
    assert "autopilot_resumed" not in [k for k, _ in _kinds(env, jid)]


def test_the_gemini_daily_limit_survives_a_restart(env):
    env.app.gemini_key, env.app.gemini_daily_limit = "k", 2
    assert env.app.gemini_allowed() and env.app.gemini_allowed()
    assert not env.app.gemini_allowed()
    app = env.restart()
    app.gemini_key, app.gemini_daily_limit = "k", 2
    assert not app.gemini_allowed()
