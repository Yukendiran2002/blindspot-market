# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liceron
"""LOCAL FAKE PayPal for tests (httpx MockTransport). Not PayPal.

Mirrors sandbox behaviour verified on 2026-11-08: AUTHORIZE orders with a card are authorized at once; partial captures
(final_capture=false) up to the authorized amount, then 422 MAX_CAPTURE_AMOUNT_EXCEEDED; a repeated PayPal-Request-Id
returns the original capture/refund; captures appear in GET order with custom_id; a repeated Payouts sender_batch_id is
refused with 400 USER_BUSINESS_ERROR "Batch with given sender_batch_id already exists"; void after partial captures.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import httpx

_ids = itertools.count(1)


def _id(p: str) -> str:
    return f"{p}{next(_ids):010d}"


def _c(v: str) -> int:
    a, b = v.split(".")
    return int(a) * 100 + int(b)


def _v(c: int) -> str:
    return f"{c // 100}.{c % 100:02d}"


class FakePayPal:
    def __init__(self) -> None:
        self.orders: dict[str, dict[str, Any]] = {}
        self.auths: dict[str, dict[str, Any]] = {}
        self.captures: dict[str, dict[str, Any]] = {}
        self.refunds: dict[str, dict[str, Any]] = {}
        self.payouts: dict[str, dict[str, Any]] = {}
        self.idem: dict[str, dict[str, Any]] = {}
        self.faults: dict[str, list[str]] = {}  # endpoint kind -> queue of faults
        self.calls: list[str] = []
        self.hide_captures = 0  # next N order views omit captures (delayed visibility)
        self.item_status: dict[str, str] = {}  # receiver email -> payout item status (default SUCCESS)
        self.transport = httpx.MockTransport(self._handle)

    def fault(self, kind: str, *faults: str) -> None:
        self.faults.setdefault(kind, []).extend(faults)

    def _take(self, kind: str) -> str | None:
        q = self.faults.get(kind) or []
        return q.pop(0) if q else None

    async def _handle(self, r: httpx.Request) -> httpx.Response:
        path, m = r.url.path, r.method
        if path == "/v1/oauth2/token":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 32400})
        body = json.loads(r.content) if r.content else {}
        rid = r.headers.get("PayPal-Request-Id")
        p = path.strip("/").split("/")
        self.calls.append(f"{m} {path}")
        if m == "POST" and path == "/v2/checkout/orders":
            return self._create_order(body)
        if m == "GET" and p[:3] == ["v2", "checkout", "orders"]:
            return httpx.Response(200, json=self._order_view(p[3]))
        if p[:3] == ["v2", "payments", "authorizations"] and p[-1] == "capture":
            return self._capture(p[3], body, rid)
        if p[:3] == ["v2", "payments", "authorizations"] and p[-1] == "void":
            a = self.auths[p[3]]
            if a["status"] in ("VOIDED", "CAPTURED"):
                return httpx.Response(
                    422,
                    json={"name": "UNPROCESSABLE_ENTITY", "details": [{"issue": "AUTHORIZATION_ALREADY_COMPLETED"}]},
                )
            a["status"] = "VOIDED"
            return httpx.Response(204)
        if p[:3] == ["v2", "payments", "captures"] and p[-1] == "refund":
            return self._refund(p[3], body, rid)
        if m == "POST" and path == "/v1/payments/payouts":
            return self._payout(body)
        if m == "GET" and p[:3] == ["v1", "payments", "payouts"]:
            b = self.payouts[p[3]]
            return httpx.Response(
                200,
                json={
                    "batch_header": {"payout_batch_id": p[3], "batch_status": "SUCCESS"},
                    "items": [
                        {"transaction_status": self.item_status.get(i["receiver"], "SUCCESS"), "payout_item": i}
                        for i in b["items"]
                    ],
                },
            )
        return httpx.Response(404, json={"name": "NOT_FOUND"})

    def _create_order(self, body: dict[str, Any]) -> httpx.Response:
        unit = body["purchase_units"][0]
        oid, aid = _id("ORD"), _id("AUT")
        self.auths[aid] = {
            "id": aid,
            "status": "CREATED",
            "amount": unit["amount"],
            "captured": 0,
            "order": oid,
            "expiration_time": "2026-12-07T00:00:00Z",
        }
        self.orders[oid] = {"id": oid, "status": "COMPLETED", "unit": unit, "auth": aid}
        return httpx.Response(201, json=self._order_view(oid))

    def _order_view(self, oid: str) -> dict[str, Any]:
        o = self.orders[oid]
        a = self.auths[o["auth"]]
        caps = [c for c in self.captures.values() if c["_auth"] == a["id"]]
        if self.hide_captures > 0:
            self.hide_captures -= 1
            caps = []
        return {
            "id": oid,
            "status": o["status"],
            "purchase_units": [
                {
                    "reference_id": o["unit"].get("reference_id"),
                    "payments": {
                        "authorizations": [{k: v for k, v in a.items() if k not in ("captured", "order")}],
                        "captures": [{k: v for k, v in c.items() if not k.startswith("_")} for c in caps],
                        "refunds": [
                            {k: v for k, v in r.items() if not k.startswith("_")}
                            for r in self.refunds.values()
                            if r["_cap"] in {c["id"] for c in caps}
                        ],
                    },
                }
            ],
        }

    def _capture(self, aid: str, body: dict[str, Any], rid: str | None) -> httpx.Response:
        if rid and rid in self.idem:
            return httpx.Response(200, json=self.idem[rid])
        f = self._take("capture")
        if f == "timeout_before":
            raise httpx.ReadTimeout("fake: capture timed out before processing")
        if f == "422":
            return httpx.Response(
                422, json={"name": "UNPROCESSABLE_ENTITY", "details": [{"issue": "PAYER_CANNOT_PAY"}]}
            )
        a = self.auths[aid]
        cents = _c(body["amount"]["value"])
        if a["status"] == "VOIDED" or a["captured"] + cents > _c(a["amount"]["value"]):
            return httpx.Response(
                422, json={"name": "UNPROCESSABLE_ENTITY", "details": [{"issue": "MAX_CAPTURE_AMOUNT_EXCEEDED"}]}
            )
        a["captured"] += cents
        a["status"] = "PARTIALLY_CAPTURED"
        cap = {
            "id": _id("CAP"),
            "status": "COMPLETED",
            "amount": {"currency_code": "USD", "value": _v(cents)},
            "custom_id": body.get("custom_id"),
            "invoice_id": body.get("invoice_id"),
            "_auth": aid,
        }
        self.captures[cap["id"]] = cap
        public = {k: v for k, v in cap.items() if not k.startswith("_")}
        if rid:
            self.idem[rid] = public
        if f == "lose_response":
            raise httpx.ReadTimeout("fake: capture processed, response lost")
        return httpx.Response(201, json=public)

    def _refund(self, cid: str, body: dict[str, Any], rid: str | None) -> httpx.Response:
        if rid and rid in self.idem:
            return httpx.Response(200, json=self.idem[rid])
        cap = self.captures[cid]
        done = sum(_c(r["amount"]["value"]) for r in self.refunds.values() if r["_cap"] == cid)
        cents = _c(body["amount"]["value"])
        if done + cents > _c(cap["amount"]["value"]):
            return httpx.Response(
                422, json={"name": "UNPROCESSABLE_ENTITY", "details": [{"issue": "REFUND_AMOUNT_EXCEEDED"}]}
            )
        ref = {
            "id": _id("REF"),
            "status": "COMPLETED",
            "amount": {"currency_code": "USD", "value": _v(cents)},
            "custom_id": body.get("custom_id"),
            "_cap": cid,
        }
        self.refunds[ref["id"]] = ref
        public = {k: v for k, v in ref.items() if not k.startswith("_")}
        if rid:
            self.idem[rid] = public
        return httpx.Response(201, json=public)

    def _payout(self, body: dict[str, Any]) -> httpx.Response:
        sbid = body["sender_batch_header"]["sender_batch_id"]
        if any(b["sender_batch_id"] == sbid for b in self.payouts.values()):
            return httpx.Response(
                400,
                json={
                    "name": "USER_BUSINESS_ERROR",
                    "message": "User business error.",
                    "details": [{"issue": "Batch with given sender_batch_id already exists"}],
                },
            )
        f = self._take("payout")
        if f == "500_before":
            return httpx.Response(500, json={"name": "INTERNAL_SERVER_ERROR"})
        pid = _id("PB")
        self.payouts[pid] = {"sender_batch_id": sbid, "items": body["items"]}
        if f == "lose_response":
            raise httpx.ReadTimeout("fake: payout accepted, response lost")
        return httpx.Response(
            201,
            json={
                "batch_header": {
                    "payout_batch_id": pid,
                    "batch_status": "PENDING",
                    "sender_batch_header": body["sender_batch_header"],
                }
            },
        )

    # helpers
    def captured_total(self) -> int:
        return sum(_c(c["amount"]["value"]) for c in self.captures.values())

    def paid_to(self, email: str) -> int:
        return sum(
            _c(i["amount"]["value"]) for b in self.payouts.values() for i in b["items"] if i["receiver"] == email
        )
