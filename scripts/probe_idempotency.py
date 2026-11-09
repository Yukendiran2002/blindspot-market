# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liceron
"""Sandbox: (1) Payouts with the same PayPal-Request-Id + sender_batch_id twice; (2) find captures by custom_id via GET order."""

import asyncio
import uuid

from blindspot.paypal import PayPal, PayPalError, authorizations_of, captures_of
from blindspot.settings import load_env

env = load_env()  # PAYPAL_CLIENT_ID / PAYPAL_CLIENT_SECRET (+ PAYPAL_SANDBOX_BUYER_EMAIL for live_job)
CARD = {
    "number": "4111111111111111",
    "expiry": "2030-12",
    "security_code": "123",
    "name": "Test Buyer",
    "billing_address": {
        "address_line_1": "1 Main St",
        "admin_area_2": "San Jose",
        "admin_area_1": "CA",
        "postal_code": "95131",
        "country_code": "US",
    },
}


async def main():
    pp = PayPal(env["PAYPAL_CLIENT_ID"], env["PAYPAL_CLIENT_SECRET"])
    sid = f"probe{uuid.uuid4().hex[:16]}"
    items = [
        {
            "recipient_type": "EMAIL",
            "amount": {"value": "1.20", "currency": "USD"},
            "receiver": env["PAYPAL_SANDBOX_BUYER_EMAIL"],
            "note": "probe",
            "sender_item_id": f"{sid}-a",
        }
    ]
    a = await pp.create_payout(sender_batch_id=sid, items=items, subject="probe", message="probe")
    try:
        b = await pp.create_payout(sender_batch_id=sid, items=items, subject="probe", message="probe")
        print(
            "payout replay -> same batch:",
            a["batch_header"]["payout_batch_id"] == b["batch_header"]["payout_batch_id"],
            a["batch_header"]["payout_batch_id"],
            b["batch_header"]["payout_batch_id"],
        )
    except PayPalError as e:
        print("payout replay refused:", e.status, e.name, e.issues)
    o = await pp.create_budget_order(
        "10.00", job_id="probejob", description="probe", request_id=str(uuid.uuid4()), card=CARD
    )
    aid = authorizations_of(o)[0]["id"]
    sett = f"set{uuid.uuid4().hex[:12]}"
    c = await pp.capture_authorization(aid, amount="2.00", request_id=sett, custom_id=sett, invoice_id=sett)
    c2 = await pp.capture_authorization(aid, amount="2.00", request_id=sett, custom_id=sett, invoice_id=sett)
    print("capture replay -> same capture:", c["id"] == c2["id"])
    got = await pp.get_order(o["id"])
    print(
        "order view captures:",
        [(x["id"], x.get("custom_id"), x.get("invoice_id"), x["amount"]["value"]) for x in captures_of(got)],
    )
    await pp.void_authorization(aid, request_id=str(uuid.uuid4()))
    await pp.aclose()


asyncio.run(main())
