# core/strategy.py
from __future__ import annotations

import pandas as pd
from typing import Literal

Signal = Literal["LONG", "SHORT", "HOLD"]


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ===============================
# EMA CROSS
# ===============================
def signal_ema_cross(
    df: pd.DataFrame,
    ema_fast: int,
    ema_slow: int,
) -> Signal:
    """
    Simple EMA cross strategy
    """
    if len(df) < max(ema_fast, ema_slow) + 5:
        return "HOLD"

    close = df["close"].astype(float)

    ef = _ema(close, ema_fast)
    es = _ema(close, ema_slow)

    prev_fast = float(ef.iloc[-2])
    prev_slow = float(es.iloc[-2])

    cur_fast = float(ef.iloc[-1])
    cur_slow = float(es.iloc[-1])

    # Cross up
    if prev_fast < prev_slow and cur_fast > cur_slow:
        return "LONG"

    # Cross down
    if prev_fast > prev_slow and cur_fast < cur_slow:
        return "SHORT"

    return "HOLD"


# ===============================
# TREND BREAKOUT
# ===============================
def signal_trend_breakout(
    df: pd.DataFrame,
    lookback: int,
    break_pct: float,
) -> Signal:

    if len(df) < lookback + 5:
        return "HOLD"

    close = df["close"].astype(float)

    prev_range = close.iloc[-lookback - 1 : -1]
    last = float(close.iloc[-1])

    hi = float(prev_range.max())
    lo = float(prev_range.min())

    if last > hi * (1.0 + break_pct):
        return "LONG"

    if last < lo * (1.0 - break_pct):
        return "SHORT"

    return "HOLD"


# ===============================
# MEAN REVERSION
# ===============================
def signal_mean_reversion(
    df: pd.DataFrame,
    lookback: int,
    z_enter: float,
) -> Signal:

    if len(df) < lookback + 5:
        return "HOLD"

    close = df["close"].astype(float)
    ma = close.rolling(lookback).mean()
    std = close.rolling(lookback).std()

    if std.iloc[-1] == 0 or pd.isna(std.iloc[-1]):
        return "HOLD"

    z = (close.iloc[-1] - ma.iloc[-1]) / std.iloc[-1]

    if z <= -z_enter:
        return "LONG"

    if z >= z_enter:
        return "SHORT"

    return "HOLD"


# ===============================
# VOLATILITY BREAKOUT
# ===============================
def signal_vol_breakout(
    df: pd.DataFrame,
    lookback: int,
    vol_mult: float,
) -> Signal:

    if len(df) < lookback + 5:
        return "HOLD"

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_range = high.iloc[-lookback - 1 : -1] - low.iloc[-lookback - 1 : -1]
    avg_range = prev_range.mean()

    cur_range = high.iloc[-1] - low.iloc[-1]

    if avg_range == 0 or pd.isna(avg_range):
        return "HOLD"

    if cur_range > avg_range * vol_mult:
        if close.iloc[-1] > close.iloc[-2]:
            return "LONG"
        else:
            return "SHORT"

    return "HOLD"