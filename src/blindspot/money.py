# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liceron
"""Money as integer minor units. Floats are never used for amounts.

The MVP supports USD only (two decimal places). PayPal sends and receives amounts as strings
such as ``"49.99"``; :func:`to_cents` accepts only that exact form.
"""

from __future__ import annotations

import re

SUPPORTED_CURRENCIES = {"USD": 2}
_AMOUNT = re.compile(r"^(0|[1-9][0-9]{0,8})\.[0-9]{2}$")
_LOOSE = re.compile(r"^\$?\s*(0|[1-9][0-9]{0,8})(\.[0-9]{1,2})?$")


class MoneyError(ValueError):
    """An amount or currency PayOps refuses to handle."""


def check_currency(currency: str) -> str:
    if currency not in SUPPORTED_CURRENCIES:
        raise MoneyError(f"Unsupported currency {currency!r}; this build handles USD only.")
    return currency


def to_cents(value: str) -> int:
    """Strict PayPal form ``"123.45"`` → 12345. Anything else is refused."""
    if not isinstance(value, str) or _AMOUNT.fullmatch(value) is None:
        raise MoneyError(f"Amount {value!r} must look like '49.99'.")
    whole, frac = value.split(".")
    return int(whole) * 100 + int(frac)


def parse_user_amount(value: str) -> int:
    """Lenient form typed by a person (``"$49.99"``, ``"50"``, ``"7.5"``) → cents."""
    if not isinstance(value, str):
        raise MoneyError("Amount must be text.")
    match = _LOOSE.fullmatch(value.strip().replace(",", ""))
    if match is None:
        raise MoneyError(f"Amount {value!r} is not a valid USD amount.")
    whole = int(match.group(1))
    frac = (match.group(2) or ".00")[1:].ljust(2, "0")
    return whole * 100 + int(frac)


def fmt(cents: int) -> str:
    """12345 → ``"123.45"`` (PayPal form)."""
    if isinstance(cents, bool) or not isinstance(cents, int) or cents < 0:
        raise MoneyError(f"Invalid cents value {cents!r}.")
    return f"{cents // 100}.{cents % 100:02d}"


def show(cents: int, currency: str = "USD") -> str:
    """Human form, e.g. ``$49.99``."""
    return ("$" if currency == "USD" else f"{currency} ") + fmt(cents)
