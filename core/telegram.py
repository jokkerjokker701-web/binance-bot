# core/telegram.py
from __future__ import annotations

import os
import time
from typing import Optional

import requests

# .env ni lokalda o‘qish uchun (serverda bo‘lmasligi mumkin — normal)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


_last_env_log_ts = 0.0


def _get_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip()
    return v if v else None


def tg_env_ok() -> tuple[bool, bool]:
    token_ok = bool(_get_env("TG_BOT_TOKEN"))
    chat_ok = bool(_get_env("TG_CHAT_ID"))
    return token_ok, chat_ok


def tg_debug_env(force: bool = False) -> None:
    """
    Tokenni chiqarmaydi. Faqat bor/yo‘qligini ko‘rsatadi.
    """
    global _last_env_log_ts
    now = time.time()
    if (not force) and (now - _last_env_log_ts) < 120:
        return
    _last_env_log_ts = now

    token_ok, chat_ok = tg_env_ok()
    print(f"[TG] env check: token_ok={token_ok} chat_ok={chat_ok}", flush=True)


def tg_send(text: str, *, silent_fail: bool = True) -> bool:
    """
    Telegramga xabar yuboradi.
    - silent_fail=True bo‘lsa: token/chat bo‘lmasa crash qilmaydi, log yozib False qaytaradi.
    """
    # har safar env reload (lokalda .env o‘zgarsa ham)
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass

    token = _get_env("TG_BOT_TOKEN")
    chat_id = _get_env("TG_CHAT_ID")

    if not token or not chat_id:
        tg_debug_env(force=True)
        msg = "[TG] TG_BOT_TOKEN yoki TG_CHAT_ID yo'q (serverda Variables qo'yilmagan bo'lishi mumkin)."
        print(msg, flush=True)
        return False if silent_fail else (_raise(msg))

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}

    try:
        r = requests.post(url, json=payload, timeout=15)
        if not r.ok:
            # tokenni chiqarmaymiz, faqat status + response text
            print(f"[TG] send failed: {r.status_code} {r.text}", flush=True)
            return False if silent_fail else (_raise(f"TG send failed: {r.status_code}"))
        return True
    except Exception as e:
        print(f"[TG] send exception: {e}", flush=True)
        return False if silent_fail else (_raise(str(e)))


def _raise(msg: str) -> bool:
    raise RuntimeError(msg)