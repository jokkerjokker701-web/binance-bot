# core/regime.py
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from .indicators import atr as atr_func
from .indicators import adx as adx_func


def detect_regime(
    df: pd.DataFrame,
    atr_ma_period: Optional[int] = None,
    adx_period: Optional[int] = None,
    atr_vol_mult: Optional[float] = None,
    adx_trend: Optional[float] = None,
    cfg: Any = None,
) -> str:
    """
    Flexible regime detector.

    Accepts BOTH:
      1) detect_regime(df, cfg=cfg)  or detect_regime(df, cfg)
      2) detect_regime(df, atr_ma_period, adx_period, atr_vol_mult, adx_trend)

    Output regimes:
      - "TREND"
      - "HIGH_VOL"
      - "RANGE"
    """

    # ---- allow detect_regime(df, cfg) as 2nd positional ----
    if cfg is None and atr_ma_period is not None and hasattr(atr_ma_period, "atr_ma_period"):
        cfg = atr_ma_period
        atr_ma_period = None

    # ---- pull params from cfg if provided ----
    if cfg is not None:
        atr_ma_period = int(getattr(cfg, "atr_ma_period", 50))
        adx_period = int(getattr(cfg, "adx_period", 14))
        atr_vol_mult = float(getattr(cfg, "atr_vol_mult", 1.8))
        adx_trend = float(getattr(cfg, "adx_trend", 22.0))

    # ---- defaults (if still None) ----
    atr_ma_period = int(atr_ma_period or 50)
    adx_period = int(adx_period or 14)
    atr_vol_mult = float(atr_vol_mult or 1.8)
    adx_trend = float(adx_trend or 22.0)

    d = df.copy()

    # Ensure ATR exists
    if "atr" not in d.columns:
        d["atr"] = atr_func(d, period=14)

    # Ensure ADX exists
    if "adx" not in d.columns:
        d["adx"] = adx_func(d, period=adx_period)

    atr_v = float(d["atr"].iloc[-1]) if pd.notna(d["atr"].iloc[-1]) else 0.0
    adx_v = float(d["adx"].iloc[-1]) if pd.notna(d["adx"].iloc[-1]) else 0.0

    atr_ma = d["atr"].rolling(atr_ma_period).mean()
    atr_ma_v = float(atr_ma.iloc[-1]) if len(atr_ma) and pd.notna(atr_ma.iloc[-1]) else 0.0

    # ATR ratio vs its MA
    if atr_ma_v > 0:
        atr_ratio = atr_v / atr_ma_v
    else:
        atr_ratio = 1.0

    # Priority: TREND first, then HIGH_VOL, else RANGE
    if adx_v >= adx_trend:
        return "TREND"
    if atr_ratio >= atr_vol_mult:
        return "HIGH_VOL"
    return "RANGE"