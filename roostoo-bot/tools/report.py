"""Summarise the bot's own journal: equity curve, drawdown, trades per UTC day, fees, holdings.

    python -m tools.report            # reads logs/*.jsonl
This is also the evidence for 'consistent autonomous execution': every order row carries its strategy reason."""
from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

LOGS = Path(__file__).resolve().parent.parent / "logs"


def read(kind: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(LOGS / f"{kind}-*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return pd.DataFrame(rows)


def main() -> None:
    eq, orders, dec = read("equity"), read("orders"), read("decisions")
    if len(eq):
        eq["ts"] = pd.to_datetime(eq["ts"])
        e = eq.set_index("ts")["equity"]
        peak = e.cummax()
        print(f"equity now {e.iloc[-1]:,.2f}  return {e.iloc[-1] / e.iloc[0] - 1:+.2%}  max drawdown {(1 - e / peak).max():.2%}")
        daily = e.resample("1D").last().pct_change().dropna()
        if len(daily):
            print("daily returns:\n" + (daily * 100).round(2).to_string())
    if len(orders):
        o = orders[orders["status"].isin(["FILLED"])].copy()
        o["day"] = pd.to_datetime(o["ts"]).dt.strftime("%Y-%m-%d")
        print(f"\nfilled orders: {len(o)}  fees paid: {o['fee'].sum():.2f} USD")
        print("fills per UTC day:\n" + o.groupby("day").size().to_string())
        print("\nby reason:\n" + o["reason"].value_counts().to_string())
    if len(dec):
        print(f"\ndecisions logged: {len(dec)}")


if __name__ == "__main__":
    main()
