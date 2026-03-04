import os
import csv

def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def append_trade_csv(path: str, row: dict):
    ensure_parent(path)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)