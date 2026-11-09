# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liceron
"""Async PayPal REST client for Blindspot (sandbox only).

Rails used (all verified live in the sandbox on 2026-11-08):
- Orders v2 with intent AUTHORIZE: the customer's learning-budget hold.
- Payments v2 authorizations: partial capture (final_capture=false, custom_id=settlement id), void.
- Payments v2 captures: refund.
- Payouts v1: annotator payments (sender_batch_id + PayPal-Request-Id for idempotency).

Error model: PayPalDefinite = PayPal refused (4xx), nothing happened.
PayPalUncertain = timeout, connection loss or 5xx; reconcile before acting again, and retry only with the same ids.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

SANDBOX = "https://api-m.sandbox.paypal.com"


class PayPalError(Exception):
    def __init__(self, status: int | None, name: str, message: str, issues: list[str], debug_id: str | None):
        super().__init__(f"{status} {name}: {message} {issues}".strip())
        self.status, self.name, self.issues, self.debug_id = status, name, issues, debug_id


class PayPalDefinite(PayPalError):
    """PayPal refused the request; no money moved."""


class PayPalUncertain(PayPalError):
    """Outcome unknown; reconcile before doing anything else."""


def _safe(resource_id: str) -> str:
    if not isinstance(resource_id, str) or not resource_id.isalnum() or len(resource_id) > 64:
        raise PayPalDefinite(None, "INVALID_ID", f"invalid PayPal id {resource_id!r}", [], None)
    return resource_id


class PayPal:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        base_url: str = SANDBOX,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        url = httpx.URL(base_url.rstrip("/"))
        if url.scheme != "https" or url.host != "api-m.sandbox.paypal.com" or url.port not in (None, 443):
            raise ValueError("Blindspot only talks to https://api-m.sandbox.paypal.com")
        if not client_id or not client_secret:
            raise ValueError("PayPal client id and secret are required")
        self._id, self._secret = client_id, client_secret
        self._http = httpx.AsyncClient(base_url=str(url), transport=transport, timeout=timeout)
        self._token: str | None = None
        self._expiry = 0.0
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _access_token(self, force: bool = False) -> str:
        async with self._lock:
            if not force and self._token and time.monotonic() < self._expiry:
                return self._token
            try:
                r = await self._http.post(
                    "/v1/oauth2/token", data={"grant_type": "client_credentials"}, auth=(self._id, self._secret)
                )
            except httpx.HTTPError as exc:
                raise PayPalUncertain(None, "TRANSPORT_ERROR", type(exc).__name__, [], None) from exc
            if r.status_code >= 500 or r.status_code == 429:
                raise PayPalUncertain(r.status_code, "AUTH_UNAVAILABLE", "token endpoint unavailable", [], None)
            if r.status_code != 200:
                raise PayPalDefinite(r.status_code, "AUTH_FAILED", "could not obtain an access token", [], None)
            body = r.json()
            self._token = body["access_token"]
            self._expiry = time.monotonic() + max(60, int(body.get("expires_in", 3600)) - 60)
            return self._token

    async def _request(
        self, method: str, path: str, *, json: Any = None, request_id: str | None = None, prefer_full: bool = True
    ) -> dict[str, Any]:
        for attempt in (1, 2):
            headers = {"Authorization": f"Bearer {await self._access_token(force=attempt == 2)}"}
            if prefer_full:
                headers["Prefer"] = "return=representation"
            if request_id:
                headers["PayPal-Request-Id"] = request_id
            try:
                r = await self._http.request(method, path, json=json, headers=headers)
            except httpx.HTTPError as exc:
                raise PayPalUncertain(None, "TRANSPORT_ERROR", type(exc).__name__, [], None) from exc
            if r.status_code == 401 and attempt == 1:
                continue
            return self._decode(r)
        raise AssertionError("unreachable")

    @staticmethod
    def _decode(r: httpx.Response) -> dict[str, Any]:
        try:
            body = r.json() if r.content else {}
        except ValueError:
            body = {}
        if 200 <= r.status_code < 300:
            if r.content and not isinstance(body, dict):
                raise PayPalUncertain(r.status_code, "UNREADABLE_RESPONSE", "success status, unreadable body", [], None)
            return body
        name = str(body.get("name") or "HTTP_ERROR")
        issues = [str(d.get("issue")) for d in body.get("details", []) if isinstance(d, dict) and d.get("issue")]
        cls = PayPalUncertain if r.status_code >= 500 or r.status_code in (409, 429) else PayPalDefinite
        raise cls(r.status_code, name, str(body.get("message") or ""), issues, body.get("debug_id"))

    # --- budget hold --------------------------------------------------------------------------
    async def create_budget_order(
        self,
        amount: str,
        *,
        job_id: str,
        description: str,
        request_id: str,
        card: Mapping[str, Any] | None = None,
        return_url: str | None = None,
        cancel_url: str | None = None,
    ) -> dict[str, Any]:
        """intent=AUTHORIZE. With a sandbox test card PayPal authorizes immediately; with the PayPal wallet the
        buyer approves first (JS SDK popup or payer-action link) and then :meth:`authorize_order` is called."""
        unit = {
            "reference_id": job_id,
            "custom_id": job_id,
            "description": description[:127],
            "amount": {"currency_code": "USD", "value": amount},
        }
        source: dict[str, Any]
        if card is not None:
            source = {"card": dict(card)}
        else:
            source = {
                "paypal": {
                    "experience_context": {
                        "return_url": return_url or "https://example.com/return",
                        "cancel_url": cancel_url or "https://example.com/cancel",
                        "user_action": "CONTINUE",
                        "shipping_preference": "NO_SHIPPING",
                        "brand_name": "Blindspot Market",
                    }
                }
            }
        return await self._request(
            "POST",
            "/v2/checkout/orders",
            request_id=request_id,
            json={"intent": "AUTHORIZE", "purchase_units": [unit], "payment_source": source},
        )

    async def authorize_order(self, order_id: str, *, request_id: str) -> dict[str, Any]:
        return await self._request(
            "POST", f"/v2/checkout/orders/{_safe(order_id)}/authorize", json={}, request_id=request_id
        )

    async def get_order(self, order_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v2/checkout/orders/{_safe(order_id)}")

    async def get_authorization(self, authorization_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v2/payments/authorizations/{_safe(authorization_id)}")

    async def capture_authorization(
        self,
        authorization_id: str,
        *,
        amount: str,
        request_id: str,
        custom_id: str,
        invoice_id: str,
        final: bool = False,
        note: str = "",
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "amount": {"currency_code": "USD", "value": amount},
            "final_capture": final,
            "custom_id": custom_id[:127],
            "invoice_id": invoice_id[:127],
        }
        if note:
            body["note_to_payer"] = note[:255]
        return await self._request(
            "POST", f"/v2/payments/authorizations/{_safe(authorization_id)}/capture", json=body, request_id=request_id
        )

    async def void_authorization(self, authorization_id: str, *, request_id: str) -> dict[str, Any]:
        return await self._request(
            "POST", f"/v2/payments/authorizations/{_safe(authorization_id)}/void", json={}, request_id=request_id
        )

    async def get_capture(self, capture_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v2/payments/captures/{_safe(capture_id)}")

    async def refund_capture(
        self, capture_id: str, *, amount: str, request_id: str, custom_id: str, note: str = ""
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"amount": {"currency_code": "USD", "value": amount}, "custom_id": custom_id[:127]}
        if note:
            body["note_to_payer"] = note[:255]
        return await self._request(
            "POST", f"/v2/payments/captures/{_safe(capture_id)}/refund", json=body, request_id=request_id
        )

    # --- payouts ----------------------------------------------------------------------------
    async def create_payout(
        self, *, sender_batch_id: str, items: Sequence[Mapping[str, Any]], subject: str, message: str
    ) -> dict[str, Any]:
        body = {
            "sender_batch_header": {
                "sender_batch_id": sender_batch_id[:127],
                "email_subject": subject[:255],
                "email_message": message[:1000],
            },
            "items": [dict(i) for i in items],
        }
        return await self._request(
            "POST", "/v1/payments/payouts", json=body, request_id=sender_batch_id, prefer_full=False
        )

    async def get_payout(self, payout_batch_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/payments/payouts/{_safe(payout_batch_id)}", prefer_full=False)


def authorizations_of(order: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [a for u in order.get("purchase_units", []) for a in (u.get("payments", {}).get("authorizations") or [])]


def captures_of(order: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [c for u in order.get("purchase_units", []) for c in (u.get("payments", {}).get("captures") or [])]
