# core/arm_guard.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple


def _now() -> float:
    return time.time()


@dataclass
class ArmStats:
    total_trades: int = 0
    total_pnl: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    disabled_until: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "total_pnl": self.total_pnl,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "disabled_until": self.disabled_until,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ArmStats":
        s = ArmStats()
        s.total_trades = int(d.get("total_trades", 0))
        s.total_pnl = float(d.get("total_pnl", 0.0))
        s.consecutive_losses = int(d.get("consecutive_losses", 0))
        s.consecutive_wins = int(d.get("consecutive_wins", 0))
        s.disabled_until = float(d.get("disabled_until", 0.0))
        return s


@dataclass
class ArmGuard:
    path: str
    stats: Dict[str, ArmStats] = field(default_factory=dict)

    def load(self) -> None:
        if not self.path:
            return
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            st = {}
            for k, v in data.get("stats", {}).items():
                st[k] = ArmStats.from_dict(v)
            self.stats = st
        except Exception:
            # if file corrupted, ignore (bot will recreate)
            self.stats = {}

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        data = {"stats": {k: v.to_dict() for k, v in self.stats.items()}}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get(self, arm_name: str) -> ArmStats:
        if arm_name not in self.stats:
            self.stats[arm_name] = ArmStats()
        return self.stats[arm_name]

    def is_disabled(self, arm_name: str) -> bool:
        s = self.get(arm_name)
        return s.disabled_until > _now()

    def disabled_for_seconds(self, arm_name: str) -> float:
        s = self.get(arm_name)
        return max(0.0, s.disabled_until - _now())

    def on_close(
        self,
        *,
        arm_name: str,
        pnl: float,
        loss_streak_disable: int,
        disable_hours: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Returns: (disabled_now, reason)
        """
        s = self.get(arm_name)
        s.total_trades += 1
        s.total_pnl += float(pnl)

        if pnl > 0:
            s.consecutive_wins += 1
            s.consecutive_losses = 0
        elif pnl < 0:
            s.consecutive_losses += 1
            s.consecutive_wins = 0
        else:
            # pnl == 0: do not change streaks
            pass

        disabled_now = False
        reason = None

        if loss_streak_disable > 0 and s.consecutive_losses >= loss_streak_disable:
            s.disabled_until = _now() + float(disable_hours) * 3600.0
            # reset losses so it doesn't instantly re-disable after expiry
            s.consecutive_losses = 0
            disabled_now = True
            reason = f"LOSS_STREAK_{loss_streak_disable}_DISABLE_{disable_hours}H"

        self.save()
        return disabled_now, reason