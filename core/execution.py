# core/execution.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Position:
    side: str  # "LONG" or "SHORT"
    entry: float  # executed entry
    qty: float
    sl: float
    tp: float

    regime: str
    arm: str
    kind: str

    # realism fields
    entry_raw: float = 0.0
    exit_exec: Optional[float] = None

    # fees & slippage
    maker_fee: float = 0.0002    # 0.02%
    taker_fee: float = 0.0004    # 0.04%
    fee_mode: str = "taker"      # "maker" or "taker"
    slippage_bps: float = 2.0
    fees_paid: float = 0.0


def _slip_mult(bps: float) -> float:
    return float(bps) / 10000.0


def _apply_entry_slippage(side: str, price: float, slippage_bps: float) -> float:
    s = _slip_mult(slippage_bps)
    if side == "LONG":
        return price * (1.0 + s)
    return price * (1.0 - s)


def _apply_exit_slippage(side: str, price: float, slippage_bps: float) -> float:
    s = _slip_mult(slippage_bps)
    if side == "LONG":
        return price * (1.0 - s)
    return price * (1.0 + s)


def _fee_rate(pos: Position) -> float:
    return float(pos.maker_fee if pos.fee_mode == "maker" else pos.taker_fee)


def _fees(notional: float, fee_rate: float) -> float:
    return abs(float(notional)) * float(fee_rate)


def open_position(
    *,
    side: str,
    entry: float,
    atr: float,
    qty: float,
    sl_mult: float,
    tp_mult: float,
    regime: str,
    arm: str,
    kind: str,
    maker_fee: float = 0.0002,
    taker_fee: float = 0.0004,
    fee_mode: str = "taker",      # maker/taker
    slippage_bps: float = 2.0,
) -> Position:
    entry_raw = float(entry)
    entry_exec = _apply_entry_slippage(side, entry_raw, slippage_bps)

    atr = float(atr)
    qty = float(qty)

    if side == "LONG":
        sl = entry_exec - atr * float(sl_mult)
        tp = entry_exec + atr * float(tp_mult)
    else:
        sl = entry_exec + atr * float(sl_mult)
        tp = entry_exec - atr * float(tp_mult)

    pos = Position(
        side=side,
        entry=float(entry_exec),
        qty=qty,
        sl=float(sl),
        tp=float(tp),
        regime=str(regime),
        arm=str(arm),
        kind=str(kind),
        entry_raw=float(entry_raw),
        exit_exec=None,
        maker_fee=float(maker_fee),
        taker_fee=float(taker_fee),
        fee_mode=str(fee_mode),
        slippage_bps=float(slippage_bps),
        fees_paid=0.0,
    )

    # entry fee
    entry_notional = pos.entry * pos.qty
    pos.fees_paid += _fees(entry_notional, _fee_rate(pos))
    return pos


def check_close(pos: Position, price: float) -> bool:
    p = float(price)
    if pos.side == "LONG":
        return p <= pos.sl or p >= pos.tp
    return p >= pos.sl or p <= pos.tp


def pnl_usd(pos: Position, price: float) -> float:
    exit_mid = float(price)
    exit_exec = _apply_exit_slippage(pos.side, exit_mid, pos.slippage_bps)
    pos.exit_exec = float(exit_exec)

    if pos.side == "LONG":
        gross = (exit_exec - pos.entry) * pos.qty
    else:
        gross = (pos.entry - exit_exec) * pos.qty

    exit_notional = exit_exec * pos.qty
    exit_fee = _fees(exit_notional, _fee_rate(pos))

    net = gross - pos.fees_paid - exit_fee
    return float(net)