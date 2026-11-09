# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liceron
"""Settings from the environment, optionally from a KEY=VALUE file (BLINDSPOT_ENV_FILE, default ./.env)."""

from __future__ import annotations

import os
from pathlib import Path


def load_env() -> dict[str, str]:
    env_file = Path(os.environ.get("BLINDSPOT_ENV_FILE", ".env")).expanduser()
    file_env: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                file_env[k.strip()] = v.strip().strip('"').strip("'")
    env = {**file_env, **os.environ}
    if not env.get("PAYPAL_CLIENT_ID") or not env.get("PAYPAL_CLIENT_SECRET"):
        raise SystemExit("set PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET (sandbox REST app) in the environment or .env")
    return env


def gemini_key(env: dict[str, str]) -> str | None:
    if env.get("GEMINI_API_KEY"):
        return env["GEMINI_API_KEY"]
    key_file = env.get("GEMINI_API_KEY_FILE")
    if key_file and Path(key_file).expanduser().exists():
        return Path(key_file).expanduser().read_text().strip() or None
    return None
