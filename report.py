import csv
import os
from collections import defaultdict

TRADES_PATH = "state/trades.csv"

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def load_trades(path: str):
    if not os.path.exists(path):
        print(f"Not found: {path}")
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def summarize(trades):
    # overall
    total = len(trades)
    wins = sum(1 for t in trades if safe_float(t.get("pnl")) > 0)
    losses = total - wins
    pnl_sum = sum(safe_float(t.get("pnl")) for t in trades)
    avg_pnl = pnl_sum / total if total else 0.0
    winrate = wins / total if total else 0.0

    print("\n=== OVERALL ===")
    print(f"Trades: {total}")
    print(f"Winrate: {winrate:.2%} (W:{wins} / L:{losses})")
    print(f"Total PnL: {pnl_sum:.2f}")
    print(f"Avg PnL: {avg_pnl:.2f}")

    # by regime,arm
    by = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0})
    regimes = set()
    arms = set()

    for t in trades:
        r = (t.get("regime") or "NA").strip()
        a = (t.get("arm") or "NA").strip()
        regimes.add(r); arms.add(a)

        key = (r, a)
        by[key]["n"] += 1
        p = safe_float(t.get("pnl"))
        by[key]["pnl"] += p
        if p > 0:
            by[key]["w"] += 1

    regimes = sorted(regimes)
    arms = sorted(arms)

    print("\n=== TOP ARMS (by total PnL) ===")
    ranked = []
    for (r, a), v in by.items():
        n = v["n"]
        w = v["w"]
        pnl = v["pnl"]
        wr = (w / n) if n else 0.0
        ranked.append((pnl, wr, n, r, a))
    ranked.sort(reverse=True)

    for pnl, wr, n, r, a in ranked[:10]:
        print(f"{r:9s} | {a:10s} | n={n:3d} | wr={wr:.1%} | pnl={pnl:.2f}")

    # heatmap table text
    print("\n=== HEATMAP (Winrate %) Regime x Arm ===")
    header = "Regime \\ Arm | " + " | ".join(f"{a:10s}" for a in arms)
    print(header)
    print("-" * len(header))

    for r in regimes:
        row = [f"{r:12s} |"]
        for a in arms:
            v = by.get((r, a))
            if not v or v["n"] == 0:
                cell = "   -   "
            else:
                wr = v["w"] / v["n"]
                cell = f"{wr*100:6.1f}%"
            row.append(f"{cell:>10s}")
        print(" ".join(row))

    print("\n=== HEATMAP (Avg PnL) Regime x Arm ===")
    header = "Regime \\ Arm | " + " | ".join(f"{a:10s}" for a in arms)
    print(header)
    print("-" * len(header))

    for r in regimes:
        row = [f"{r:12s} |"]
        for a in arms:
            v = by.get((r, a))
            if not v or v["n"] == 0:
                cell = "   -   "
            else:
                avg = v["pnl"] / v["n"]
                cell = f"{avg:8.2f}"
            row.append(f"{cell:>10s}")
        print(" ".join(row))

def main():
    trades = load_trades(TRADES_PATH)
    summarize(trades)

if __name__ == "__main__":
    main()