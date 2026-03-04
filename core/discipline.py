# core/discipline.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple


def utc_day_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@dataclass
class DisciplineState:
    # Daily stats
    day_key: str = ""
    trades_today: int = 0
    stopouts_today: int = 0  # SL or losing trades count (you decide what counts as stopout)
    daily_pnl: float = 0.0

    # Equity / drawdown
    start_balance_day: float = 0.0
    peak_balance_day: float = 0.0
    max_dd_day: float = 0.0  # as fraction, e.g. 0.03 = -3%

    # Mode
    stop_today: bool = False
    stop_reason: str = ""

    def roll_day_if_needed(self, balance: float) -> None:
        dk = utc_day_key()
        if self.day_key != dk:
            self.day_key = dk
            self.trades_today = 0
            self.stopouts_today = 0
            self.daily_pnl = 0.0
            self.start_balance_day = float(balance)
            self.peak_balance_day = float(balance)
            self.max_dd_day = 0.0
            self.stop_today = False
            self.stop_reason = ""

    def update_equity(self, balance: float) -> None:
        """Update peak & max drawdown for the day."""
        balance = float(balance)
        if balance > self.peak_balance_day:
            self.peak_balance_day = balance
        if self.peak_balance_day > 0:
            dd = (self.peak_balance_day - balance) / self.peak_balance_day
            if dd > self.max_dd_day:
                self.max_dd_day = dd

    def set_stop_today(self, reason: str) -> None:
        self.stop_today = True
        self.stop_reason = reason

    def can_trade(
        self,
        *,
        max_trades_per_day: int,
        max_stopouts_per_day: int,
        max_daily_loss_usd: float,
        max_daily_drawdown: float,
        cooldown_ok: bool,
        vol_kill: bool,
    ) -> Tuple[bool, Optional[str]]:
        """
        Returns (allowed, reason_if_blocked)
        """
        if self.stop_today:
            return False, f"STOP_TODAY: {self.stop_reason}"

        if vol_kill:
            return False, "VOL_KILL: volatility too high"

        if not cooldown_ok:
            return False, "COOLDOWN"

        if max_trades_per_day > 0 and self.trades_today >= max_trades_per_day:
            return False, "MAX_TRADES_PER_DAY"

        if max_stopouts_per_day > 0 and self.stopouts_today >= max_stopouts_per_day:
            return False, "MAX_STOPOUTS_PER_DAY"

        # daily pnl loss limit in USD (negative)
        if max_daily_loss_usd > 0 and self.daily_pnl <= -abs(max_daily_loss_usd):
            return False, "MAX_DAILY_LOSS"

        # daily max drawdown fraction
        if max_daily_drawdown > 0 and self.max_dd_day >= max_daily_drawdown:
            return False, "MAX_DAILY_DRAWDOWN"

        return True, None

    def on_open(self) -> None:
        self.trades_today += 1

    def on_close(self, pnl: float, was_stopout: bool) -> None:
        self.daily_pnl += float(pnl)
        if was_stopout:
            self.stopouts_today += 1