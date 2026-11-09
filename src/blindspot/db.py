# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liceron
"""SQLite storage (WAL). Every PayPal call is preceded by a committed row holding its idempotency ids."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    status            TEXT NOT NULL,              -- DRAFT, FUNDING, ACTIVE, CLOSING, CLOSED
    price_cents       INTEGER NOT NULL,           -- customer pays per accepted label
    worker_cents      INTEGER NOT NULL,           -- annotator earns per accepted label
    budget_cents      INTEGER NOT NULL,
    round_size        INTEGER NOT NULL,
    target_accuracy   REAL NOT NULL,
    min_gain_per_10   REAL NOT NULL,              -- stop when validation gain per $10 drops below this (points)
    strategy          TEXT NOT NULL DEFAULT 'blindspot',
    paypal_order_id   TEXT,
    authorization_id  TEXT,
    hold_status       TEXT,
    captured_cents    INTEGER NOT NULL DEFAULT 0,
    refunded_cents    INTEGER NOT NULL DEFAULT 0,
    void_request_id   TEXT,
    seed_count        INTEGER NOT NULL,
    funded_at         REAL,                       -- when the hold was placed (capture window starts here)
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS annotators (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    paypal_email  TEXT NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN ('human', 'simulated')),
    sim_accuracy  REAL,
    sim_late      REAL,                          -- simulated accuracy from round 2 on (a 'sleeper')
    banned        INTEGER NOT NULL DEFAULT 0,
    token_hash    TEXT,                          -- sha256 of the annotator's access token (humans only)
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS rounds (
    id            TEXT PRIMARY KEY,
    job_id        TEXT NOT NULL REFERENCES jobs(id),
    number        INTEGER NOT NULL,
    status        TEXT NOT NULL,                  -- LABELLING, SETTLING, SETTLED, CANCELLED
    plan          TEXT NOT NULL,                  -- the agent's decision and explanation (JSON)
    val_before    REAL, val_after REAL, sealed_before REAL, sealed_after REAL,
    created_at    REAL NOT NULL, settled_at REAL,
    UNIQUE(job_id, number)
);
CREATE TABLE IF NOT EXISTS tasks (
    id            TEXT PRIMARY KEY,
    job_id        TEXT NOT NULL,
    round_id      TEXT NOT NULL REFERENCES rounds(id),
    pool_index    INTEGER NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN ('learn', 'gold')),
    gold_label    INTEGER,
    options       TEXT NOT NULL,                  -- JSON list of candidate intent ids shown to the annotator
    annotator_id  TEXT,
    answer        INTEGER,                        -- intent id, or -1 for "none of these"
    status        TEXT NOT NULL,                  -- OPEN, ANSWERED, ACCEPTED, REJECTED, REVOKED, EXPIRED
    answered_at   REAL
);
CREATE INDEX IF NOT EXISTS tasks_by_round ON tasks(round_id, status);
CREATE TABLE IF NOT EXISTS labels (
    job_id        TEXT NOT NULL,
    pool_index    INTEGER NOT NULL,
    label         INTEGER NOT NULL,
    source        TEXT NOT NULL,                  -- seed | bought
    task_id       TEXT,
    active        INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (job_id, pool_index)
);
CREATE TABLE IF NOT EXISTS settlements (
    id               TEXT PRIMARY KEY,            -- PayPal-Request-Id / custom_id / sender_batch_id base
    job_id           TEXT NOT NULL,
    round_id         TEXT,
    kind             TEXT NOT NULL CHECK (kind IN ('round', 'revocation')),
    accepted         INTEGER NOT NULL DEFAULT 0,
    rejected         INTEGER NOT NULL DEFAULT 0,
    charge_cents     INTEGER NOT NULL DEFAULT 0,
    capture_id       TEXT,
    capture_state    TEXT NOT NULL DEFAULT 'NONE', -- NONE, SENDING, DONE, UNCERTAIN, FAILED
    refund_cents     INTEGER NOT NULL DEFAULT 0,
    refund_capture_id TEXT,
    refund_id        TEXT,
    refund_state     TEXT NOT NULL DEFAULT 'NONE',
    payout_cents     INTEGER NOT NULL DEFAULT 0,
    payout_items     TEXT NOT NULL DEFAULT '[]',
    payout_batch_id  TEXT,
    payout_state     TEXT NOT NULL DEFAULT 'NONE', -- NONE, SENDING, SUBMITTED, ITEMS_PENDING, DONE, UNCERTAIN, FAILED
    error            TEXT,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq      INTEGER PRIMARY KEY AUTOINCREMENT,
    at       REAL NOT NULL,
    job_id   TEXT,
    kind     TEXT NOT NULL,
    data     TEXT NOT NULL DEFAULT '{}'
);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:20]}"


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._local = threading.local()
        self._conn().executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after the first live runs (SQLite has no ADD COLUMN IF NOT EXISTS)."""
        for table, column, decl in (("jobs", "funded_at", "REAL"), ("annotators", "token_hash", "TEXT")):
            cols = {r["name"] for r in self._conn().execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                self._conn().execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, isolation_level=None, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    def one(self, sql: str, *args: Any) -> Any:  # sqlite3.Row or None; callers check where a row may be missing
        return self._conn().execute(sql, args).fetchone()

    def all(self, sql: str, *args: Any) -> list[sqlite3.Row]:
        return self._conn().execute(sql, args).fetchall()

    def event(self, kind: str, job_id: str | None = None, conn: sqlite3.Connection | None = None, **data: Any) -> None:
        args = (time.time(), job_id, kind, json.dumps(data, default=str, sort_keys=True))
        sql = "INSERT INTO events(at, job_id, kind, data) VALUES (?,?,?,?)"
        if conn is not None:
            conn.execute(sql, args)
        else:
            with self.tx() as c:
                c.execute(sql, args)
