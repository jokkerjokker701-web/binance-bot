# binance_bot/core/indicators.py
from __future__ import annotations

import pandas as pd


# ===============================
# Internal helpers
# ===============================

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR (Wilder smoothing using EMA alpha=1/period)
    Requires columns: high, low, close
    """
    tr = _true_range(df["high"], df["low"], df["close"])
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ADX (Wilder):
      +DM, -DM from high/low changes
      TR from true range
      Smooth all with Wilder EMA (alpha=1/period)
    Requires columns: high, low, close
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_high = high.shift(1)
    prev_low = low.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = _true_range(high, low, close)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, 1e-12))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, 1e-12))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-12)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI (Wilder smoothing using EMA alpha=1/period)
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ===============================
# Public API (main.py uchun)
# ===============================

def add_indicators(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """
    Adds indicators used by the bot:
      - atr, adx, rsi
      - ema_50, ema_200
    Expects columns: open, high, low, close
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    # ensure numeric
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col].astype(float)

    atr_p = int(getattr(cfg, "atr_period", 14)) if cfg is not None else 14
    adx_p = int(getattr(cfg, "adx_period", 14)) if cfg is not None else 14
    rsi_p = int(getattr(cfg, "rsi_period", 14)) if cfg is not None else 14

    out["atr"] = _atr(out, atr_p)
    out["adx"] = _adx(out, adx_p)
    out["rsi"] = _rsi(out["close"], rsi_p)

    out["ema_50"] = _ema(out["close"], 50)
    out["ema_200"] = _ema(out["close"], 200)

    return out


# ===============================
# Compatibility wrappers (core/regime.py uchun)
# core/regime.py ichida: from .indicators import atr, adx bo'lishi mumkin
# ===============================

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _atr(df, period)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _adx(df, period)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    return _rsi(close.astype(float), period)
def ema(series: pd.Series, period: int):
    return _ema(series.astype(float), period)