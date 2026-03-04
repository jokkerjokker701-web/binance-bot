# core/journal.py
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    record = dict(record)
    record["_ts"] = record.get("_ts") or utc_now_iso()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def classify_outcome(
    *,
    pnl: float,
    regime: str,
    global_bias: str,
    picked_kind: str,
    adx: Optional[float],
    atr: Optional[float],
) -> str:
    """
    Juda sodda 'mistake classifier' (fundlarda keyin kengaytiriladi).
    """
    if pnl >= 0:
        return "OK_WIN"

    # Loss reasons (heuristic)
    if global_bias == "LONG_ONLY" and picked_kind in ("MEAN_REV", "VOL_BRK") and regime == "TREND":
        return "LOSS_MISMATCH_KIND_VS_REGIME"

    if adx is not None and adx < 18:
        return "LOSS_WEAK_TREND_CHOP"

    if atr is not None and atr > 0:
        return "LOSS_VOLATILE_MOVE"

    return "LOSS_OTHER"