# core/telegram.py

import os
import requests
from dotenv import load_dotenv
from pathlib import Path

# Force load .env from project root
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)


def tg_send(text: str) -> bool:
    token = os.getenv("TG_BOT_TOKEN")
    chat_id = os.getenv("TG_CHAT_ID")

    if not token or not chat_id:
        print("[TG] Missing TG_BOT_TOKEN or TG_CHAT_ID in environment (.env).")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    try:
        r = requests.post(
            url,
            data={
                "chat_id": chat_id,
                "text": text,
            },
            timeout=10,
        )

        if r.status_code != 200:
            print(f"[TG] HTTP {r.status_code}: {r.text}")
            return False

        return True

    except Exception as e:
        print(f"[TG] Exception: {e}")
        return False