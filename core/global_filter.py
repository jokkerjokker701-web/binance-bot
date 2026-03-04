from __future__ import annotations
import pandas as pd

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def _true_range(df: pd.DataFrame) -> pd.Series:
    h = df["high"]
    l = df["low"]
    c = df["close"].shift(1)
    tr = (h - l).abs()
    tr2 = (h - c).abs()
    tr3 = (l - c).abs()
    return pd.concat([tr, tr2, tr3], axis=1).max(axis=1)

def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    up = high.diff()
    down = -low.diff()

    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    tr = _true_range(df)
    atr = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0.0)
    adx = dx.rolling(period).mean().fillna(0.0)
    return adx

def _market_structure_bias(df: pd.DataFrame, lookback: int = 20, break_pct: float = 0.001) -> str:
    """
    Juda sodda va stabil structure:
    - Oxirgi close, lookback ichidagi previous swing-high dan yuqori => bullish BOS
    - Oxirgi close, lookback ichidagi previous swing-low dan past => bearish BOS
    Aks holda NEUTRAL
    """
    if len(df) < lookback + 5:
        return "NEUTRAL"

    window = df.iloc[-(lookback+1):-1]
    last = df.iloc[-1]
    prev_high = float(window["high"].max())
    prev_low = float(window["low"].min())
    c = float(last["close"])

    if c > prev_high * (1 + break_pct):
        return "BULL"
    if c < prev_low * (1 - break_pct):
        return "BEAR"
    return "NEUTRAL"

def detect_global_bias(
    df_1h: pd.DataFrame,
    ema_fast: int = 50,
    ema_slow: int = 200,
    adx_period: int = 14,
    adx_min: float = 18.0,
    ms_lookback: int = 20,
    ms_break_pct: float = 0.001
) -> dict:
    """
    Returns:
      {
        "bias": "LONG_ONLY" | "SHORT_ONLY" | "RANGE",
        "score": float,
        "ema_fast": float,
        "ema_slow": float,
        "adx": float,
        "ms": "BULL"|"BEAR"|"NEUTRAL"
      }
    """
    df = df_1h.copy()
    close = df["close"].astype(float)

    ef = _ema(close, ema_fast)
    es = _ema(close, ema_slow)
    adx = _adx(df.assign(close=close), adx_period)

    ef_last = float(ef.iloc[-1])
    es_last = float(es.iloc[-1])
    adx_last = float(adx.iloc[-1])
    ms = _market_structure_bias(df, ms_lookback, ms_break_pct)

    score = 0.0

    # Trend direction (EMA regime)
    if ef_last > es_last:
        score += 1.0
    elif ef_last < es_last:
        score -= 1.0

    # Trend strength (ADX)
    if adx_last >= adx_min:
        score += 1.0 if score > 0 else (-1.0 if score < 0 else 0.0)

    # Market structure confirmation
    if ms == "BULL":
        score += 1.0
    elif ms == "BEAR":
        score -= 1.0

    # Final bias
    if score >= 1.5:
        bias = "LONG_ONLY"
    elif score <= -1.5:
        bias = "SHORT_ONLY"
    else:
        bias = "RANGE"

    return {
        "bias": bias,
        "score": float(score),
        "ema_fast": ef_last,
        "ema_slow": es_last,
        "adx": adx_last,
        "ms": ms
    }