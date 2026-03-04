import json, os, random, time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Literal, Any, Optional

StrategyKind = Literal["EMA", "TREND_BRK", "MEAN_REV", "VOL_BRK"]

@dataclass
class Arm:
    name: str
    kind: StrategyKind
    params: Dict[str, Any]

class RegimeBandit:
    """
    1) Thompson Sampling (alpha/beta) bilan arm tanlaydi
    2) Har arm uchun so'nggi N trade natijasi bo'yicha yomonlarini 6 soat BAN qiladi
    """
    def __init__(
        self,
        regimes: List[str],
        arms: List[Arm],
        stats_path: str,
        arm_status_path: str,
        ban_hours: int = 6,
        window: int = 30,
        min_winrate: float = 0.35,
        min_avg_pnl: float = 0.0,
        min_trades_to_judge: int = 20,
    ):
        self.regimes = regimes
        self.arms = arms
        self.stats_path = stats_path
        self.arm_status_path = arm_status_path

        self.ban_seconds = ban_hours * 3600
        self.window = window
        self.min_winrate = min_winrate
        self.min_avg_pnl = min_avg_pnl
        self.min_trades_to_judge = min_trades_to_judge

        # Thompson stats: stats[regime][arm] = (alpha,beta)
        self.stats: Dict[str, Dict[str, Tuple[float, float]]] = {
            r: {a.name: (1.0, 1.0) for a in arms} for r in regimes
        }

        # Status: rolling performance + bans (regime-based)
        # status[regime][arm] = {"pnl": [..], "win": [0/1], "ban_until": epoch}
        self.status: Dict[str, Dict[str, Dict[str, Any]]] = {
            r: {a.name: {"pnl": [], "win": [], "ban_until": 0.0} for a in arms} for r in regimes
        }

        self._load_stats()
        self._load_status()

    # ---------- public ----------
    def pick(self, regime: str) -> Arm:
        now = time.time()

        # 1) unbanned arms
        available = []
        for a in self.arms:
            ban_until = float(self.status[regime][a.name].get("ban_until", 0.0))
            if now >= ban_until:
                available.append(a)

        # agar hammasi banned bo'lsa: xavfsizlik uchun banlarni vaqtincha ignore qilamiz
        # (aks holda bot "o'ladi"). Bu kam uchraydi.
        if not available:
            available = list(self.arms)

        best, best_score = None, -1.0
        for a in available:
            alpha, beta = self.stats[regime][a.name]
            score = random.betavariate(alpha, beta)
            if score > best_score:
                best_score = score
                best = a
        return best  # type: ignore

    def update(self, regime: str, arm_name: str, pnl: float):
        # 1) Thompson update
        alpha, beta = self.stats[regime].get(arm_name, (1.0, 1.0))
        if pnl > 0:
            alpha += 1.0
        else:
            beta += 1.0
        self.stats[regime][arm_name] = (alpha, beta)
        self._save_stats()

        # 2) Rolling performance update
        st = self.status[regime][arm_name]
        st["pnl"].append(float(pnl))
        st["win"].append(1 if pnl > 0 else 0)

        # keep last window
        if len(st["pnl"]) > self.window:
            st["pnl"] = st["pnl"][-self.window:]
            st["win"] = st["win"][-self.window:]

        # 3) Retirement / BAN logic
        self._maybe_ban(regime, arm_name)

        self._save_status()

    def is_banned(self, regime: str, arm_name: str) -> bool:
        return time.time() < float(self.status[regime][arm_name].get("ban_until", 0.0))

    def ban_left_seconds(self, regime: str, arm_name: str) -> float:
        return max(0.0, float(self.status[regime][arm_name].get("ban_until", 0.0)) - time.time())

    # ---------- internal ----------
    def _maybe_ban(self, regime: str, arm_name: str):
        st = self.status[regime][arm_name]
        pnl_list = st.get("pnl", [])
        win_list = st.get("win", [])

        n = len(pnl_list)
        if n < self.min_trades_to_judge:
            return

        winrate = sum(win_list) / n if n else 0.0
        avg_pnl = sum(pnl_list) / n if n else 0.0

        # yomon ishlasa -> BAN
        if (winrate < self.min_winrate) and (avg_pnl < self.min_avg_pnl):
            st["ban_until"] = time.time() + self.ban_seconds

    def _save_stats(self):
        os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

    def _load_stats(self):
        if not os.path.exists(self.stats_path):
            return
        try:
            with open(self.stats_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in self.regimes:
                if r in data and isinstance(data[r], dict):
                    for a in self.stats[r].keys():
                        v = data[r].get(a)
                        if isinstance(v, list) and len(v) == 2:
                            self.stats[r][a] = (float(v[0]), float(v[1]))
        except Exception:
            pass

    def _save_status(self):
        os.makedirs(os.path.dirname(self.arm_status_path), exist_ok=True)
        with open(self.arm_status_path, "w", encoding="utf-8") as f:
            json.dump(self.status, f, ensure_ascii=False, indent=2)

    def _load_status(self):
        if not os.path.exists(self.arm_status_path):
            return
        try:
            with open(self.arm_status_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in self.regimes:
                if r in data and isinstance(data[r], dict):
                    for a in self.status[r].keys():
                        if a in data[r] and isinstance(data[r][a], dict):
                            # minimal merge
                            self.status[r][a]["pnl"] = data[r][a].get("pnl", [])[-self.window:]
                            self.status[r][a]["win"] = data[r][a].get("win", [])[-self.window:]
                            self.status[r][a]["ban_until"] = float(data[r][a].get("ban_until", 0.0))
        except Exception:
            pass