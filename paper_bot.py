import time
import json
from dataclasses import dataclass
from typing import Optional, Literal

import httpx
import pandas as pd

# Public (API key kerak emas). Narx/klayn shu yerdan olinadi.
PUBLIC_FAPI = "https://fapi.binance.com"

Side = Literal["LONG", "SHORT"]

@dataclass
class Config:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    limit: int = 300

    start_balance: float = 500.0
    leverage: int = 3

    risk_per_trade: float = 0.005     # 0.5%
    max_daily_loss: float = 0.02      # 2% of start_balance
    max_consecutive_losses: int = 3
    cooldown_seconds: int = 60

    ema_fast: int = 20
    ema_slow: int = 50
    atr_period: int = 14
    adx_period: int = 14
    adx_min: float = 18.0

    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.2

    fee_rate: float = 0.0004          # taxminiy taker fee
    slippage: float = 0.0002          # 0.02%

    loop_seconds: int = 30

@dataclass
class Position:
    side: Side
    qty: float
    entry: float
    sl: float
    tp: float

@dataclass
class State:
    balance: float
    day_start_balance: float
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    last_trade_ts: float = 0.0
    last_bar_close_ts: Optional[int] = None
    pos: Optional[Position] = None

def log(event: dict):
    print(json.dumps(event, ensure_ascii=False))

def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr1 = (h - l).abs()
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    down = -l.diff()

    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    tr1 = (h - l).abs()
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_ = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
    return dx.rolling(period).mean()

def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    with httpx.Client(timeout=20.0) as c:
        r = c.get(f"{PUBLIC_FAPI}/fapi/v1/klines", params={
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        })
        r.raise_for_status()
        data = r.json()

    # [openTime, open, high, low, close, volume, closeTime, ...]
    rows = []
    for k in data:
        rows.append({
            "open_ts": int(k[0]),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "close_ts": int(k[6]),
        })
    return pd.DataFrame(rows)

def compute_signal(df: pd.DataFrame, cfg: Config) -> tuple[str, dict]:
    df = df.copy()
    df["ema_fast"] = ema(df["close"], cfg.ema_fast)
    df["ema_slow"] = ema(df["close"], cfg.ema_slow)
    df["atr"] = atr(df, cfg.atr_period)
    df["adx"] = adx(df, cfg.adx_period)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    info = {
        "close": float(last["close"]),
        "ema_fast": float(last["ema_fast"]),
        "ema_slow": float(last["ema_slow"]),
        "atr": float(last["atr"]) if pd.notna(last["atr"]) else None,
        "adx": float(last["adx"]) if pd.notna(last["adx"]) else None,
        "close_ts": int(last["close_ts"]),
    }

    if info["atr"] is None or info["adx"] is None:
        return "HOLD", info

    if info["adx"] < cfg.adx_min:
        return "HOLD", info

    prev_diff = float(prev["ema_fast"] - prev["ema_slow"])
    curr_diff = float(last["ema_fast"] - last["ema_slow"])

    if prev_diff <= 0 and curr_diff > 0:
        return "LONG", info
    if prev_diff >= 0 and curr_diff < 0:
        return "SHORT", info
    return "HOLD", info

def risk_ok(st: State, cfg: Config) -> bool:
    max_loss = -cfg.start_balance * cfg.max_daily_loss
    return st.daily_pnl > max_loss and st.consecutive_losses < cfg.max_consecutive_losses

def open_position(side: Side, price: float, atr_v: float, st: State, cfg: Config) -> Position:
    # Risk-based qty (soddalashtirilgan)
    risk_usd = cfg.start_balance * cfg.risk_per_trade
    sl_dist = cfg.sl_atr_mult * atr_v
    qty = max(risk_usd / sl_dist, 0.001)

    if side == "LONG":
        entry = price * (1 + cfg.slippage)
        sl = entry - cfg.sl_atr_mult * atr_v
        tp = entry + cfg.tp_atr_mult * atr_v
    else:
        entry = price * (1 - cfg.slippage)
        sl = entry + cfg.sl_atr_mult * atr_v
        tp = entry - cfg.tp_atr_mult * atr_v

    return Position(side=side, qty=qty, entry=entry, sl=sl, tp=tp)

def maybe_close(st: State, price: float, cfg: Config):
    if st.pos is None:
        return

    p = st.pos
    hit = None
    exit_price = None

    if p.side == "LONG":
        if price <= p.sl:
            hit = "SL"
            exit_price = p.sl * (1 - cfg.slippage)
        elif price >= p.tp:
            hit = "TP"
            exit_price = p.tp * (1 - cfg.slippage)
    else:
        if price >= p.sl:
            hit = "SL"
            exit_price = p.sl * (1 + cfg.slippage)
        elif price <= p.tp:
            hit = "TP"
            exit_price = p.tp * (1 + cfg.slippage)

    if hit is None:
        return

    # PnL (soddalashtirilgan, fee bilan)
    if p.side == "LONG":
        gross = (exit_price - p.entry) * p.qty * cfg.leverage
    else:
        gross = (p.entry - exit_price) * p.qty * cfg.leverage

    fee = (p.entry * p.qty + exit_price * p.qty) * cfg.fee_rate
    pnl = gross - fee

    st.balance += pnl
    st.daily_pnl += pnl
    st.consecutive_losses = (st.consecutive_losses + 1) if pnl <= 0 else 0

    log({
        "type": "CLOSE",
        "hit": hit,
        "side": p.side,
        "entry": round(p.entry, 2),
        "exit": round(exit_price, 2),
        "qty": round(p.qty, 6),
        "pnl": round(pnl, 2),
        "balance": round(st.balance, 2),
        "daily_pnl": round(st.daily_pnl, 2),
        "consec_losses": st.consecutive_losses
    })
    st.pos = None
    st.last_trade_ts = time.time()

def main():
    cfg = Config()
    st = State(balance=cfg.start_balance, day_start_balance=cfg.start_balance)

    log({"type": "START", "symbol": cfg.symbol, "interval": cfg.interval, "balance": cfg.start_balance})

    while True:
        try:
            df = fetch_klines(cfg.symbol, cfg.interval, cfg.limit)

            # Faqat yangi bar yopilganda signal
            last_close_ts = int(df.iloc[-1]["close_ts"])
            price = float(df.iloc[-1]["close"])

            # pozitsiya bor bo‘lsa SL/TP tekshir
            maybe_close(st, price, cfg)

            if st.last_bar_close_ts == last_close_ts:
                time.sleep(cfg.loop_seconds)
                continue
            st.last_bar_close_ts = last_close_ts

            sig, info = compute_signal(df, cfg)

            log({
                "type": "BAR",
                "close_ts": info["close_ts"],
                "price": round(info["close"], 2),
                "adx": None if info["adx"] is None else round(info["adx"], 2),
                "atr": None if info["atr"] is None else round(info["atr"], 2),
                "signal": sig,
                "pos": None if st.pos is None else st.pos.side,
                "balance": round(st.balance, 2),
                "daily_pnl": round(st.daily_pnl, 2),
            })

            if st.pos is not None:
                time.sleep(cfg.loop_seconds)
                continue

            if not risk_ok(st, cfg):
                log({"type": "RISK_STOP", "msg": "Daily limit or consecutive losses hit. Bot paused."})
                time.sleep(60)
                continue

            if (time.time() - st.last_trade_ts) < cfg.cooldown_seconds:
                time.sleep(cfg.loop_seconds)
                continue

            if sig in ("LONG", "SHORT"):
                atr_v = info["atr"]
                if atr_v is None or atr_v <= 0:
                    time.sleep(cfg.loop_seconds)
                    continue

                pos = open_position(sig, info["close"], atr_v, st, cfg)
                st.pos = pos
                st.last_trade_ts = time.time()

                log({
                    "type": "OPEN",
                    "side": pos.side,
                    "entry": round(pos.entry, 2),
                    "sl": round(pos.sl, 2),
                    "tp": round(pos.tp, 2),
                    "qty": round(pos.qty, 6),
                })

            time.sleep(cfg.loop_seconds)

        except Exception as e:
            log({"type": "ERROR", "msg": str(e)})
            time.sleep(10)

if __name__ == "__main__":
    main()