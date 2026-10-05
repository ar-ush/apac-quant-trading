"""Run a strategy version through the backtester and print competition-style statistics.

    python -m backtest.run_backtest --strategy v1_momentum_rotation --start 2023-04-01 --cost 0.0012
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

import strategies
from backtest.data import load_panel
from backtest.engine import Costs, run_backtest
from backtest.metrics import max_drawdown, ratios, rolling_windows, summarize
from bot.config import load_config
from bot.data.universe import daily_pools

# spread proxy for the backtest only (the live bot filters on the real Roostoo bid/ask spread)
WIDE_SPREAD = {"PEPE", "SHIB", "BONK", "WLFI", "OPEN", "LISTA", "STO", "1000CHEEMS", "HEMI", "SOMI", "TUT", "BIO", "BMT", "EDEN",
               "MIRA", "AVNT", "LINEA", "FORM", "CFX", "S", "TON", "DOT", "EIGEN", "PLUME", "ARB"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--start", default="2023-04-01")
    ap.add_argument("--cost", type=float, default=0.0012)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    cfg = load_config(a.config)
    name = a.strategy or cfg.strategy.name
    strat = strategies.make(name, cfg.strategy.params if name == cfg.strategy.name else {})
    closes, qv = load_panel()
    drop = [c for c in closes.columns if c in WIDE_SPREAD or c in cfg.universe.exclude]
    closes, qv = closes.drop(columns=drop), qv.drop(columns=drop)
    pools = daily_pools(qv, cfg.universe.size, cfg.universe.history_days)
    eq, trades, _ = run_backtest(strat, closes, pools, Costs(per_side=a.cost), start=a.start)
    w = rolling_windows(eq)
    s = summarize(w)
    yrs = {y: float(eq[eq.index.year == y].iloc[-1] / eq[eq.index.year == y].iloc[0] - 1) for y in sorted(set(eq.index.year))}
    out = dict(strategy=name, params=strat.params, start=str(eq.index[0]), end=str(eq.index[-1]), cost_per_side=a.cost,
               total_return=float(eq.iloc[-1] - 1), max_drawdown=max_drawdown(eq.values), trades=len(trades),
               by_year=yrs, rolling14d=s, full_period=ratios(eq))
    print(json.dumps(out, indent=2, default=float))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
