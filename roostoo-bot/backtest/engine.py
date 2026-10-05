"""Hourly backtest that replays the live strategy class bar by bar.

At every closed hourly bar the strategy sees exactly what the live bot sees (a trailing window of closed bars, the
point-in-time liquid pool, current weights) and its target weights are executed at that bar's close with a per-side cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.base import Snapshot, Strategy


@dataclass
class Costs:
    per_side: float = 0.0012      # taker fee 0.10% + 2 bps slippage; use ~0.0007 for a mostly-maker execution
    min_trade_frac: float = 0.004  # ignore rebalance trades smaller than 0.4% of equity (exits always trade)


def run_backtest(strategy: Strategy, closes: pd.DataFrame, pools: Dict[pd.Timestamp, List[str]], costs: Costs = Costs(),
                 start: Optional[str] = None, window: Optional[int] = None):
    window = window or max(strategy.btc_history_hours(), strategy.required_history_hours()) + 10
    idx = closes.index
    cols = list(closes.columns)
    col_ix = {c: i for i, c in enumerate(cols)}
    px = closes.values
    n = len(idx)
    first = max(window, 0)
    if start:
        first = max(first, int(idx.searchsorted(pd.Timestamp(start, tz="UTC"))))
    units = np.zeros(len(cols))
    cash = 1.0
    eq = np.full(n, np.nan)
    trades = []
    decisions = []
    for i in range(first, n):
        price = px[i]
        pv = np.where(np.isnan(price), 0.0, units * np.nan_to_num(price))
        equity = cash + pv.sum()
        eq[i] = equity
        held = {cols[j]: pv[j] / equity for j in np.nonzero(units)[0] if pv[j] > 0}
        day = idx[i].normalize()
        pool = pools.get(day, [])
        snap = Snapshot(now=idx[i] + pd.Timedelta(hours=1), closes=closes.iloc[i - window + 1: i + 1], pool=pool, held=held)
        dec = strategy.decide(snap)
        if dec.targets is None:
            continue
        decisions.append((idx[i], dict(dec.targets)))
        tgt = np.zeros(len(cols))
        for c, w in dec.targets.items():
            if c in col_ix and not np.isnan(price[col_ix[c]]):
                tgt[col_ix[c]] = w * equity
        cur = pv
        # sells first
        for j in np.nonzero((cur - tgt) > 0)[0]:
            delta = cur[j] - tgt[j]
            full_exit = tgt[j] <= 0
            if not full_exit and delta < costs.min_trade_frac * equity:
                continue
            sell_units = units[j] if full_exit else delta / price[j]
            proceeds = sell_units * price[j] * (1 - costs.per_side)
            units[j] -= sell_units
            cash += proceeds
            trades.append((idx[i], cols[j], "SELL", sell_units * price[j]))
        for j in np.nonzero((tgt - cur) > 0)[0]:
            delta = tgt[j] - cur[j]
            if delta < costs.min_trade_frac * equity:
                continue
            spend = min(delta, cash / (1 + costs.per_side))
            if spend <= 0:
                continue
            units[j] += spend / price[j]
            cash -= spend * (1 + costs.per_side)
            trades.append((idx[i], cols[j], "BUY", spend))
    eqs = pd.Series(eq, index=idx).dropna()
    return eqs / eqs.iloc[0], pd.DataFrame(trades, columns=["time", "coin", "side", "usd"]), decisions
