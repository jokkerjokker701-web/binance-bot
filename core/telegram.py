# core/telegram.py
from __future__ import annotations

import os
import time
from typing import Optional

import requests


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def tg_debug_env(force: bool = False) -> dict:
    """
    Environment'da TG_BOT_TOKEN va TG_CHAT_ID bor-yo'qligini tekshiradi.
    main.py ichida tg_debug_env(force=True) chaqirilsa ham xato bermaydi.
    """
    token = _env("TG_BOT_TOKEN")
    chat_id = _env("TG_CHAT_ID")

    info = {
        "has_token": bool(token),
        "has_chat": bool(chat_id),
        "token_prefix": (token[:10] + "…") if token else "",
        "chat_id": chat_id,
        "cwd": os.getcwd(),
    }

    if force or (not info["has_token"] or not info["has_chat"]):
        print(info, flush=True)

    return info


def tg_send(text: str, *, timeout: int = 15, retries: int = 2) -> bool:
    """
    Telegramga xabar yuboradi.
    .env localda ishlaydi. Serverda esa Environment Variables orqali ishlaydi.
    """
    token = _env("TG_BOT_TOKEN")
    chat_id = _env("TG_CHAT_ID")

    if not token or not chat_id:
        print("[TG] Missing TG_BOT_TOKEN or TG_CHAT_ID in environment (.env).", flush=True)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }

    last_err: Optional[str] = None
    for _ in range(max(1, retries + 1)):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if r.ok:
                return True
            last_err = f"{r.status_code} {r.text}"
        except Exception as e:
            last_err = str(e)

        time.sleep(1)

    print(f"[TG] Failed to send: {last_err}", flush=True)
    return False