# main.py
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List, Set

import pandas as pd

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from config import Config

from core.data import fetch_klines
from core.indicators import add_indicators
from core.regime import detect_regime
from core.learner import RegimeBandit, Arm

from core.strategy import (
    signal_ema_cross,
    signal_trend_breakout,
    signal_mean_reversion,
    signal_vol_breakout,
)

from core.risk import position_size
from core.execution import Position, open_position, check_close, pnl_usd
from core.storage import append_trade_csv
from core.telegram import tg_send

from core.discipline import DisciplineState
from core.journal import append_jsonl, classify_outcome
from core.arm_guard import ArmGuard


# -------------------- helpers --------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(obj: Dict[str, Any]) -> None:
    obj["_ts"] = utc_now_iso()
    print(obj, flush=True)


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and pd.isna(x):
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _call_compat(func, *args):
    try:
        n = len(inspect.signature(func).parameters)
        return func(*args[:n])
    except Exception:
        return func(*args)


def _signal_ema_cross_compat(df: pd.DataFrame, fast: int, slow: int, cfg: Config) -> str:
    try:
        n = len(inspect.signature(signal_ema_cross).parameters)
    except Exception:
        n = 3

    if n <= 3:
        return signal_ema_cross(df, fast, slow)

    adx_min = float(getattr(cfg, "adx_min", getattr(cfg, "adx_trend", 22.0)))
    atr_period = int(getattr(cfg, "atr_period", 14))
    adx_period = int(getattr(cfg, "adx_period", 14))
    return signal_ema_cross(df, fast, slow, adx_min, atr_period, adx_period)


def call_position_size_compat(cfg: Config, *, balance: float, price: float, risk_pct: float, atr: float, sl_mult: float) -> float:
    lev = int(getattr(cfg, "leverage", 3))
    try:
        params = list(inspect.signature(position_size).parameters.keys())
    except Exception:
        try:
            return float(position_size(balance, price, risk_pct, atr, sl_mult, lev))
        except TypeError:
            return float(position_size(balance, price, risk_pct, atr, sl_mult))

    if "balance" in params:
        kwargs = {}
        if "balance" in params: kwargs["balance"] = balance
        if "price" in params: kwargs["price"] = price
        if "risk_pct" in params: kwargs["risk_pct"] = risk_pct
        if "atr" in params: kwargs["atr"] = atr
        if "sl_mult" in params: kwargs["sl_mult"] = sl_mult
        if "leverage" in params: kwargs["leverage"] = lev
        return float(position_size(**kwargs))

    try:
        return float(position_size(balance, price, risk_pct, atr, sl_mult, lev))
    except TypeError:
        return float(position_size(balance, price, risk_pct, atr, sl_mult))


# -------------------- global bias (HTF) --------------------

def compute_global_bias(df_ht: pd.DataFrame, cfg: Config) -> Tuple[str, float, float, str]:
    df_ht = df_ht.copy()
    df_ht = add_indicators(df_ht, cfg)
    close = df_ht["close"].astype(float)

    ht_ema_fast = int(getattr(cfg, "ht_ema_fast", 50))
    ht_ema_slow = int(getattr(cfg, "ht_ema_slow", 200))
    ht_adx_min = float(getattr(cfg, "ht_adx_min", 18.0))

    ef = _ema(close, ht_ema_fast)
    es = _ema(close, ht_ema_slow)

    ht_adx = safe_float(df_ht["adx"].iloc[-1]) if "adx" in df_ht.columns else None
    ht_adx_v = float(ht_adx) if ht_adx is not None else 0.0

    ms_lookback = int(getattr(cfg, "ms_lookback", 20))
    ms_break_pct = float(getattr(cfg, "ms_break_pct", 0.001))

    ht_ms = "NEUTRAL"
    if len(close) >= ms_lookback + 2:
        look = close.iloc[-ms_lookback - 1:-1]
        last = float(close.iloc[-1])
        hi = float(look.max())
        lo = float(look.min())
        if last > hi * (1.0 + ms_break_pct):
            ht_ms = "BOS_UP"
        elif last < lo * (1.0 - ms_break_pct):
            ht_ms = "BOS_DOWN"

    score = 0.0
    if float(ef.iloc[-1]) > float(es.iloc[-1]):
        score += 1.0
    elif float(ef.iloc[-1]) < float(es.iloc[-1]):
        score -= 1.0

    if ht_adx_v >= ht_adx_min:
        if score > 0:
            score += 1.0
        elif score < 0:
            score -= 1.0

    if ht_ms == "BOS_UP":
        score += 1.0
    elif ht_ms == "BOS_DOWN":
        score -= 1.0

    if score >= 1.5:
        return "LONG_ONLY", score, ht_adx_v, ht_ms
    if score <= -1.5:
        return "SHORT_ONLY", score, ht_adx_v, ht_ms
    return "RANGE", score, ht_adx_v, ht_ms


# -------------------- pullback logic --------------------

def is_pullback_long(df: pd.DataFrame, fast: int, slow: int) -> bool:
    if len(df) < max(fast, slow) + 5:
        return False
    close = df["close"].astype(float)
    ef = _ema(close, fast)
    es = _ema(close, slow)
    if float(ef.iloc[-1]) <= float(es.iloc[-1]):
        return False
    return float(close.iloc[-2]) < float(ef.iloc[-2]) and float(close.iloc[-1]) > float(ef.iloc[-1])


def is_pullback_short(df: pd.DataFrame, fast: int, slow: int) -> bool:
    if len(df) < max(fast, slow) + 5:
        return False
    close = df["close"].astype(float)
    ef = _ema(close, fast)
    es = _ema(close, slow)
    if float(ef.iloc[-1]) >= float(es.iloc[-1]):
        return False
    return float(close.iloc[-2]) > float(ef.iloc[-2]) and float(close.iloc[-1]) < float(ef.iloc[-1])


# -------------------- arms & signals --------------------

def build_arms(_: Config) -> List[Arm]:
    return [
        Arm("EMA_A", "EMA", {"ema_fast": 9, "ema_slow": 21, "sl": 1.25, "tp": 1.8}),
        Arm("EMA_C", "EMA", {"ema_fast": 20, "ema_slow": 50, "sl": 1.25, "tp": 1.8}),
        Arm("TREND_20", "TREND_BRK", {"lookback": 20, "break_pct": 0.001, "sl": 1.3, "tp": 2.0}),
        Arm("MR_20", "MEAN_REV", {"lookback": 20, "z_enter": 1.2, "sl": 1.2, "tp": 1.6}),
    ]


def choose_signal(df: pd.DataFrame, arm: Arm, cfg: Config) -> str:
    k = arm.kind
    p = arm.params
    if k == "EMA":
        return _signal_ema_cross_compat(df, int(p["ema_fast"]), int(p["ema_slow"]), cfg)
    if k == "TREND_BRK":
        return signal_trend_breakout(df, int(p["lookback"]), float(p["break_pct"]))
    if k == "MEAN_REV":
        return signal_mean_reversion(df, int(p["lookback"]), float(p["z_enter"]))
    if k == "VOL_BRK":
        return signal_vol_breakout(df, int(p["lookback"]), float(p["vol_mult"]))
    return "HOLD"


# -------------------- fund policy --------------------

def allowed_kinds_for(regime: str, global_bias: str) -> Set[str]:
    if regime == "TREND":
        return {"EMA", "TREND_BRK"}
    if regime == "RANGE":
        if global_bias == "RANGE":
            return {"MEAN_REV"}
        if global_bias in ("LONG_ONLY", "SHORT_ONLY"):
            return {"EMA"}
        return set()
    if regime == "HIGH_VOL":
        return set()
    return {"EMA", "TREND_BRK", "MEAN_REV"}


def policy_fallback_arm(regime: str, global_bias: str, arms: List[Arm], guard: ArmGuard) -> Optional[Arm]:
    name_map = {a.name: a for a in arms}

    def pick_first(names: Tuple[str, ...]) -> Optional[Arm]:
        for nm in names:
            a = name_map.get(nm)
            if a is not None and not guard.is_disabled(a.name):
                return a
        return None

    if regime == "TREND":
        return pick_first(("TREND_20", "EMA_A", "EMA_C"))
    if regime == "RANGE":
        if global_bias == "RANGE":
            a = name_map.get("MR_20")
            if a is not None and not guard.is_disabled(a.name):
                return a
            return None
        if global_bias in ("LONG_ONLY", "SHORT_ONLY"):
            return pick_first(("EMA_A", "EMA_C"))
    return None


def pick_arm_fund_style(learner: RegimeBandit, regime: str, global_bias: str, arms: List[Arm], guard: ArmGuard) -> Tuple[Arm, bool, Optional[str]]:
    allow = allowed_kinds_for(regime, global_bias)
    arm = learner.pick(regime)

    if guard.is_disabled(arm.name):
        fb = policy_fallback_arm(regime, global_bias, arms, guard)
        if fb is not None:
            return fb, True, "ARM_DISABLED"
        for a in arms:
            if a.kind in allow and not guard.is_disabled(a.name):
                return a, True, "ARM_DISABLED"
        return arm, False, None

    if allow and arm.kind not in allow:
        fb = policy_fallback_arm(regime, global_bias, arms, guard)
        if fb is not None and fb.kind in allow:
            return fb, True, "KIND_NOT_ALLOWED"
        for a in arms:
            if a.kind in allow and not guard.is_disabled(a.name):
                return a, True, "KIND_NOT_ALLOWED"

    return arm, False, None


# -------------------- confidence + dynamic slippage --------------------

def compute_confidence(sig: str, regime: str, global_bias: str, global_score: float, adx: Optional[float], atr: Optional[float], atr_ma: float) -> float:
    if sig not in ("LONG", "SHORT"):
        return 0.0

    conf = 0.45

    if sig == "LONG" and global_bias == "LONG_ONLY":
        conf += 0.20
    if sig == "SHORT" and global_bias == "SHORT_ONLY":
        conf += 0.20
    if global_bias == "RANGE":
        conf -= 0.05

    conf += clamp(global_score / 6.0, -0.10, 0.15)

    if adx is not None:
        conf += clamp((adx - 18.0) / 60.0, -0.10, 0.25)

    if regime == "TREND":
        conf += 0.05
    elif regime == "HIGH_VOL":
        conf -= 0.30
    elif regime == "RANGE":
        conf -= 0.05

    if atr is not None and atr_ma > 0:
        ratio = float(atr) / float(atr_ma)
        if ratio > 2.5:
            conf -= 0.25
        elif ratio > 1.7:
            conf -= 0.12
        elif ratio < 0.7:
            conf -= 0.06

    return clamp(conf, 0.05, 0.95)


def confidence_risk_multiplier(conf: float, min_mult: float, max_mult: float) -> float:
    conf = clamp(conf, 0.0, 1.0)
    return float(min_mult + (max_mult - min_mult) * conf)


def dynamic_slippage_bps(base_bps: float, atr: Optional[float], atr_ma: float, vol_kill: bool) -> float:
    """
    ATR/ATR_MA ratio bo‘yicha slippage oshadi.
    Vol juda yuqori bo‘lsa slippage ko‘proq.
    """
    if atr is None or atr_ma <= 0:
        return float(base_bps)

    ratio = float(atr) / float(atr_ma)
    extra = 0.0
    if ratio > 2.5:
        extra = 6.0
    elif ratio > 1.8:
        extra = 3.0
    elif ratio > 1.4:
        extra = 1.5

    if vol_kill:
        extra += 4.0

    return float(clamp(base_bps + extra, base_bps, base_bps + 12.0))


def fee_mode_for(regime: str, arm_kind: str) -> str:
    """
    Paperda:
      - TREND breakout = taker (tez kirish)
      - RANGE mean reversion = maker (limit)
      - EMA = taker default
    """
    if arm_kind == "MEAN_REV":
        return "maker"
    if arm_kind == "TREND_BRK":
        return "taker"
    return "taker"


# -------------------- runtime --------------------

@dataclass
class RuntimeState:
    balance: float
    last_trade_ts: float = 0.0
    last_close_ts: int = 0


# -------------------- main --------------------

def main() -> None:
    cfg = Config()
    arms = build_arms(cfg)

    # paths
    journal_path = str(getattr(cfg, "journal_path", "state/journal.jsonl"))
    trades_path = str(getattr(cfg, "trades_path", "state/trades.csv"))
    arm_guard_path = str(getattr(cfg, "arm_guard_path", "state/arm_guard.json"))

    st = RuntimeState(balance=float(getattr(cfg, "start_balance", 500.0)))

    disc = DisciplineState()
    disc.roll_day_if_needed(st.balance)

    guard = ArmGuard(path=arm_guard_path)
    guard.load()

    learner = RegimeBandit(
        ["TREND", "RANGE", "HIGH_VOL"],
        arms,
        getattr(cfg, "learner_path", "state/learner_state.json"),
        getattr(cfg, "arm_status_path", "state/arm_status.json"),
    )

    pos: Optional[Position] = None
    last_status_ts = 0.0

    # discipline params
    max_trades_per_day = int(getattr(cfg, "max_trades_per_day", 8))
    max_stopouts_per_day = int(getattr(cfg, "max_stopouts_per_day", 3))
    max_daily_loss_usd = float(getattr(cfg, "max_daily_loss_usd", 15.0))
    max_daily_drawdown = float(getattr(cfg, "max_daily_drawdown", 0.05))
    vol_kill_atr_mult = float(getattr(cfg, "vol_kill_atr_mult", 2.8))

    # arm guard params
    arm_disable_loss_streak = int(getattr(cfg, "arm_disable_loss_streak", 4))
    arm_disable_hours = float(getattr(cfg, "arm_disable_hours", 24.0))

    # fees/slippage
    maker_fee = float(getattr(cfg, "maker_fee", 0.0002))
    taker_fee = float(getattr(cfg, "taker_fee", 0.0004))
    base_slip_bps = float(getattr(cfg, "slippage_bps", 2.0))

    # confidence risk shaping
    conf_risk_min_mult = float(getattr(cfg, "conf_risk_min_mult", 0.6))
    conf_risk_max_mult = float(getattr(cfg, "conf_risk_max_mult", 1.35))

    loop_seconds = float(getattr(cfg, "loop_seconds", 30))
    cooldown_seconds = float(getattr(cfg, "cooldown_seconds", 60))

    tg_send(
        "🟢 BOT ONLINE\n"
        f"Symbol: {getattr(cfg, 'symbol', 'BTCUSDT')}\n"
        f"TF: {getattr(cfg, 'interval', '15m')}\n"
        f"Balance: {st.balance:.2f}\n"
        f"Fees(m/t): {maker_fee:.5f}/{taker_fee:.5f}\n"
        f"Slip base: {base_slip_bps} bps\n"
    )

    log(
        {
            "type": "START",
            "symbol": getattr(cfg, "symbol", "BTCUSDT"),
            "interval": getattr(cfg, "interval", "15m"),
            "balance": st.balance,
            "learner": getattr(cfg, "learner_path", "state/learner_state.json"),
            "arm_status": getattr(cfg, "arm_status_path", "state/arm_status.json"),
            "trades": trades_path,
            "ht_interval": getattr(cfg, "ht_interval", "1h"),
            "discipline": {
                "max_trades_per_day": max_trades_per_day,
                "max_stopouts_per_day": max_stopouts_per_day,
                "max_daily_loss_usd": max_daily_loss_usd,
                "max_daily_drawdown": max_daily_drawdown,
                "vol_kill_atr_mult": vol_kill_atr_mult,
            },
            "arm_guard": {
                "path": arm_guard_path,
                "loss_streak_disable": arm_disable_loss_streak,
                "disable_hours": arm_disable_hours,
            },
            "fees": {"maker": maker_fee, "taker": taker_fee},
            "conf_risk": {"min": conf_risk_min_mult, "max": conf_risk_max_mult},
        }
    )

    while True:
        try:
            disc.roll_day_if_needed(st.balance)

            symbol = getattr(cfg, "symbol", "BTCUSDT")
            interval = getattr(cfg, "interval", "15m")
            limit = int(getattr(cfg, "limit", 400))

            df = fetch_klines(symbol, interval, limit)
            if df is None or len(df) < 60:
                log({"type": "WARN", "msg": "Not enough klines"})
                time.sleep(loop_seconds)
                continue

            df = add_indicators(df, cfg)

            last_close_ts = int(df["close_ts"].iloc[-1])
            if last_close_ts == st.last_close_ts:
                time.sleep(loop_seconds)
                continue
            st.last_close_ts = last_close_ts

            price = float(df["close"].iloc[-1])
            atr_v = safe_float(df["atr"].iloc[-1]) if "atr" in df.columns else None
            adx_v = safe_float(df["adx"].iloc[-1]) if "adx" in df.columns else None

            atr_ma_period = int(getattr(cfg, "atr_ma_period", 50))
            atr_ma = 0.0
            if "atr" in df.columns and len(df) >= atr_ma_period:
                atr_ma_val = df["atr"].rolling(atr_ma_period).mean().iloc[-1]
                atr_ma = float(atr_ma_val) if not pd.isna(atr_ma_val) else 0.0

            vol_kill = False
            if atr_v is not None and atr_ma > 0:
                vol_kill = float(atr_v) > atr_ma * vol_kill_atr_mult

            regime = _call_compat(
                detect_regime,
                df,
                int(getattr(cfg, "atr_ma_period", 50)),
                int(getattr(cfg, "adx_period", 14)),
                float(getattr(cfg, "atr_vol_mult", 1.8)),
                float(getattr(cfg, "adx_trend", 22.0)),
            )
            regime = str(regime)

            ht_interval = getattr(cfg, "ht_interval", "1h")
            ms_lookback = int(getattr(cfg, "ms_lookback", 20))
            df_ht = fetch_klines(symbol, ht_interval, max(400, ms_lookback + 60))
            if df_ht is None or len(df_ht) < 200:
                global_bias, global_score, ht_adx, ht_ms = "RANGE", 0.0, 0.0, "NEUTRAL"
            else:
                global_bias, global_score, ht_adx, ht_ms = compute_global_bias(df_ht, cfg)

            arm, overridden, override_reason = pick_arm_fund_style(learner, regime, global_bias, arms, guard)
            if overridden:
                log({"type": "POLICY_OVERRIDE", "regime": regime, "global_bias": global_bias, "arm": arm.name, "kind": arm.kind, "override_reason": override_reason})

            sig = choose_signal(df, arm, cfg)

            # ✅ TREND secondary entry (AUTOMATIC)
            if regime == "TREND" and sig == "HOLD":
                if global_bias == "LONG_ONLY":
                    s2 = _signal_ema_cross_compat(df, 9, 21, cfg)
                    if s2 == "LONG":
                        for a in arms:
                            if a.name == "EMA_A" and not guard.is_disabled(a.name):
                                arm = a
                                sig = "LONG"
                                log({"type": "SECONDARY_ENTRY", "reason": "TREND_HOLD->EMA", "arm": arm.name})
                                break
                elif global_bias == "SHORT_ONLY":
                    s2 = _signal_ema_cross_compat(df, 9, 21, cfg)
                    if s2 == "SHORT":
                        for a in arms:
                            if a.name == "EMA_A" and not guard.is_disabled(a.name):
                                arm = a
                                sig = "SHORT"
                                log({"type": "SECONDARY_ENTRY", "reason": "TREND_HOLD->EMA", "arm": arm.name})
                                break

            # global direction filter
            if sig == "LONG" and global_bias == "SHORT_ONLY":
                sig = "HOLD"
            if sig == "SHORT" and global_bias == "LONG_ONLY":
                sig = "HOLD"

            # RANGE pullback policy
            pullback_ok = None
            if regime == "RANGE":
                if global_bias == "RANGE":
                    if arm.kind != "MEAN_REV":
                        sig = "HOLD"
                else:
                    if arm.kind != "EMA":
                        sig = "HOLD"
                        pullback_ok = False
                    else:
                        if global_bias == "LONG_ONLY":
                            pullback_ok = (sig == "LONG") and is_pullback_long(df, int(arm.params["ema_fast"]), int(arm.params["ema_slow"]))
                            if not pullback_ok:
                                sig = "HOLD"
                        elif global_bias == "SHORT_ONLY":
                            pullback_ok = (sig == "SHORT") and is_pullback_short(df, int(arm.params["ema_fast"]), int(arm.params["ema_slow"]))
                            if not pullback_ok:
                                sig = "HOLD"
                        else:
                            pullback_ok = False
                            sig = "HOLD"

            # discipline
            cooldown_ok = (time.time() - st.last_trade_ts) >= cooldown_seconds
            disc.update_equity(st.balance)
            allowed, block_reason = disc.can_trade(
                max_trades_per_day=max_trades_per_day,
                max_stopouts_per_day=max_stopouts_per_day,
                max_daily_loss_usd=max_daily_loss_usd,
                max_daily_drawdown=max_daily_drawdown,
                cooldown_ok=cooldown_ok,
                vol_kill=vol_kill,
            )
            if not allowed:
                sig = "HOLD"
                log({"type": "BLOCK", "reason": block_reason, "stop_today": disc.stop_today})

            # confidence + risk
            conf = compute_confidence(sig, regime, global_bias, float(global_score), adx_v, atr_v, float(atr_ma))
            conf_mult = confidence_risk_multiplier(conf, conf_risk_min_mult, conf_risk_max_mult)

            # dynamic slippage (NEW)
            slip_bps = dynamic_slippage_bps(base_slip_bps, atr_v, float(atr_ma), vol_kill)

            log(
                {
                    "type": "BAR",
                    "close_ts": last_close_ts,
                    "price": round(price, 1),
                    "regime": regime,
                    "picked_arm": arm.name,
                    "picked_kind": arm.kind,
                    "signal": sig,
                    "confidence": round(conf, 3),
                    "conf_mult": round(conf_mult, 3),
                    "slip_bps": round(slip_bps, 2),
                    "balance": round(st.balance, 2),
                    "daily_pnl": round(disc.daily_pnl, 2),
                    "trades_today": disc.trades_today,
                    "stopouts_today": disc.stopouts_today,
                    "dd_day": round(disc.max_dd_day, 4),
                    "adx": None if adx_v is None else round(float(adx_v), 2),
                    "atr": None if atr_v is None else round(float(atr_v), 2),
                    "atr_ma": round(float(atr_ma), 2),
                    "vol_kill": vol_kill,
                    "global_bias": global_bias,
                    "global_score": round(global_score, 2),
                    "ht_adx": round(ht_adx, 2),
                    "ht_ms": ht_ms,
                    "pullback_ok": pullback_ok,
                    "stop_today": disc.stop_today,
                }
            )

            # CLOSE
            if pos is not None and check_close(pos, price):
                pnl = pnl_usd(pos, price)
                st.balance += pnl
                disc.update_equity(st.balance)

                was_stopout = pnl < 0
                disc.on_close(pnl=pnl, was_stopout=was_stopout)

                risk_proxy = 1.0
                if atr_v is not None and pos.qty is not None:
                    risk_proxy = max(1.0, float(atr_v) * float(pos.qty))
                reward = float(pnl) / float(risk_proxy)
                learner.update(regime=pos.regime, arm_name=pos.arm, reward=reward)

                disabled_now, disable_reason = guard.on_close(
                    arm_name=pos.arm,
                    pnl=pnl,
                    loss_streak_disable=arm_disable_loss_streak,
                    disable_hours=arm_disable_hours,
                )
                if disabled_now:
                    sec = guard.disabled_for_seconds(pos.arm)
                    log({"type": "ARM_DISABLED", "arm": pos.arm, "reason": disable_reason, "disabled_hours": round(sec/3600, 2)})
                    tg_send(f"⛔️ ARM DISABLED: {pos.arm}\nReason: {disable_reason}\nFor: {sec/3600:.1f} hours")
                    append_jsonl(journal_path, {"event": "ARM_DISABLED", "arm": pos.arm, "reason": disable_reason, "disabled_for_sec": sec})

                append_trade_csv(
                    trades_path,
                    {
                        "ts": utc_now_iso(),
                        "symbol": symbol,
                        "tf": interval,
                        "side": pos.side,
                        "regime": pos.regime,
                        "arm": pos.arm,
                        "kind": pos.kind,
                        "entry": pos.entry,
                        "exit": pos.exit_exec or price,
                        "qty": pos.qty,
                        "sl": pos.sl,
                        "tp": pos.tp,
                        "pnl": pnl,
                        "balance": st.balance,
                        "fees": getattr(pos, "fees_paid", 0.0),
                        "maker_fee": getattr(pos, "maker_fee", maker_fee),
                        "taker_fee": getattr(pos, "taker_fee", taker_fee),
                        "fee_mode": getattr(pos, "fee_mode", "taker"),
                        "slip_bps": getattr(pos, "slippage_bps", slip_bps),
                    },
                )

                outcome_tag = classify_outcome(pnl=pnl, regime=pos.regime, global_bias=global_bias, picked_kind=pos.kind, adx=adx_v, atr=atr_v)
                append_jsonl(journal_path, {"event": "CLOSE", "pnl": pnl, "reward": reward, "tag": outcome_tag, "balance": st.balance})

                tg_send(
                    "✅ CLOSE\n"
                    f"Side: {pos.side}\nEntry(exec): {pos.entry:.2f}\nExit(exec): {(pos.exit_exec or price):.2f}\n"
                    f"PnL(net): {pnl:.2f}\nBal: {st.balance:.2f}\n"
                    f"FeeMode: {getattr(pos,'fee_mode','taker')} | Slip(bps): {getattr(pos,'slippage_bps',slip_bps)}\n"
                    f"Tag: {outcome_tag}"
                )
                log({"type": "CLOSE", "pnl": round(pnl, 2), "balance": round(st.balance, 2), "reward": round(reward, 6), "tag": outcome_tag})

                pos = None
                st.last_trade_ts = time.time()

            # OPEN
            if pos is None and allowed and sig in ("LONG", "SHORT") and atr_v is not None and float(atr_v) > 0:
                base_risk = float(getattr(cfg, "risk_per_trade", 0.005))
                risk_used = clamp(base_risk * conf_mult, 0.0001, 0.02)

                qty = call_position_size_compat(
                    cfg,
                    balance=float(st.balance),
                    price=float(price),
                    risk_pct=float(risk_used),
                    atr=float(atr_v),
                    sl_mult=float(arm.params["sl"]),
                )

                if qty > 0:
                    fee_mode = fee_mode_for(regime, arm.kind)

                    pos = open_position(
                        side=sig,
                        entry=float(price),
                        atr=float(atr_v),
                        qty=float(qty),
                        sl_mult=float(arm.params["sl"]),
                        tp_mult=float(arm.params["tp"]),
                        regime=regime,
                        arm=arm.name,
                        kind=arm.kind,
                        maker_fee=maker_fee,
                        taker_fee=taker_fee,
                        fee_mode=fee_mode,
                        slippage_bps=slip_bps,
                    )

                    st.last_trade_ts = time.time()
                    disc.on_open()

                    append_jsonl(journal_path, {"event": "OPEN", "side": pos.side, "arm": pos.arm, "kind": pos.kind, "risk_used": risk_used, "conf": conf, "fee_mode": fee_mode, "slip_bps": slip_bps})

                    tg_send(
                        "🚀 OPEN\n"
                        f"Side: {pos.side}\nRegime: {regime} | Global: {global_bias}\n"
                        f"Arm: {arm.name} ({arm.kind})\n"
                        f"Entry(exec): {pos.entry:.2f}\nSL: {pos.sl:.2f}\nTP: {pos.tp:.2f}\n"
                        f"Qty: {pos.qty:.6f}\nRiskUsed: {risk_used:.4f} | Conf: {conf:.2f}\n"
                        f"FeeMode: {fee_mode} | Slip(bps): {slip_bps:.2f}\n"
                        f"Bal: {st.balance:.2f}"
                    )
                    log({"type": "OPEN", "side": pos.side, "arm": pos.arm, "fee_mode": fee_mode, "slip_bps": round(slip_bps, 2), "risk_used": round(risk_used, 5), "qty": round(pos.qty, 6)})

            # HEARTBEAT
            now = time.time()
            if now - last_status_ts > 600:
                last_status_ts = now
                tg_send(
                    "✅ BOT RUNNING\n"
                    f"{symbol} {interval} | Regime={regime} | Global={global_bias}({global_score:.1f})\n"
                    f"Bal={st.balance:.2f} | TodayPnL={disc.daily_pnl:.2f} | Trades={disc.trades_today} | Stopouts={disc.stopouts_today}\n"
                    f"DD={disc.max_dd_day:.2%} | StopToday={disc.stop_today}\n"
                    f"Pos={'NONE' if pos is None else pos.side}"
                )

            time.sleep(loop_seconds)

        except KeyboardInterrupt:
            log({"type": "STOP", "msg": "KeyboardInterrupt"})
            tg_send("🟡 BOT STOPPED (KeyboardInterrupt)")
            break
        except Exception as e:
            log({"type": "ERROR", "msg": str(e)})
            try:
                tg_send(f"⚠️ BOT ERROR: {e}")
            except Exception:
                pass
            time.sleep(10)


if __name__ == "__main__":
    main()