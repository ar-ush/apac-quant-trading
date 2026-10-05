"""v1 — BTC-gated 14-day momentum rotation.

Idea (see V1_MOMENTUM_ROTATION.md for the evidence): in crypto, the strongest liquid coins over the last two weeks keep
outperforming for days (cross-sectional momentum, Liu-Tsyvinski-Wu 2022; trend factor, Fieberg et al. 2024), but only
while the market is in an up-trend. Two ingredients:

  * Regime gate  : BTC EMA(168h) > EMA(672h)  =>  risk-on, else 100% cash.  (checked every hour)
  * Rotation     : at 00:00 UTC rank the liquid pool by 336h return; hold the top-k positive-momentum coins, equal weight.
                   A holding is kept (no churn) while its rank <= k + hysteresis and its return is still > 0.
                   Winners are never topped up; one that grows beyond max_weight_multiple/k is trimmed.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .base import Decision, Snapshot, Strategy


def ema_last(series: pd.Series, span: int) -> float:
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


class MomentumRotation(Strategy):
    name = "v1_momentum_rotation"
    version = "1.0"

    DEFAULTS = dict(lookback_hours=336, top_k=3, hysteresis=2, gate_fast_hours=168, gate_slow_hours=672,
                    rebalance_hour_utc=0, max_weight_multiple=1.6, gross_cap=0.98)

    def __init__(self, **params):
        p = dict(self.DEFAULTS)
        unknown = set(params) - set(p)
        if unknown:
            raise ValueError(f"unknown v1 parameters: {sorted(unknown)}")
        p.update(params)
        super().__init__(**p)
        self.p = p

    def required_history_hours(self) -> int:
        return self.p["lookback_hours"] + 24

    def btc_history_hours(self) -> int:
        return max(2000, 3 * self.p["gate_slow_hours"])

    # ------------------------------------------------------------------ gate
    def gate_on(self, btc: pd.Series) -> bool:
        p = self.p
        if len(btc) < p["gate_slow_hours"] + 50:
            return False  # not enough history to trust the slow EMA: stay out
        return ema_last(btc, p["gate_fast_hours"]) > ema_last(btc, p["gate_slow_hours"])

    # ------------------------------------------------------------------ ranking
    def momentum(self, closes: pd.DataFrame, pool: List[str]) -> pd.Series:
        lb = self.p["lookback_hours"]
        cols = [c for c in pool if c in closes.columns]
        w = closes[cols].iloc[-(lb + 1):]
        if len(w) < lb + 1:
            return pd.Series(dtype=float)
        ret = w.iloc[-1] / w.iloc[0] - 1.0
        return ret.dropna()

    def decide(self, snap: Snapshot) -> Decision:
        p = self.p
        closes = snap.closes
        if "BTC" not in closes.columns:
            return Decision(None, info={"error": 1.0})
        gate = self.gate_on(closes["BTC"].dropna())
        info: Dict[str, float] = {"gate_on": float(gate)}

        if not gate:
            if snap.held:
                return Decision({}, {c: "gate_off_exit" for c in snap.held}, info)
            return Decision(None, info=info)

        is_rebalance_bar = snap.now.hour == p["rebalance_hour_utc"]
        if not (is_rebalance_bar or snap.force_rebalance):
            return Decision(None, info=info)

        ret = self.momentum(closes, snap.pool)
        if ret.empty:
            return Decision(None, info={**info, "no_momentum_data": 1.0})
        rank = ret.rank(ascending=False, method="first")
        k, h = p["top_k"], p["hysteresis"]

        keep = [c for c in snap.held
                if c in ret.index and rank[c] <= k + h and ret[c] > 0]
        keep = sorted(keep, key=lambda c: rank[c])[:k]
        candidates = [c for c in ret.sort_values(ascending=False).index
                      if rank[c] <= k and ret[c] > 0 and c not in keep]
        add = candidates[: k - len(keep)]

        targets: Dict[str, float] = {}
        reasons: Dict[str, str] = {}
        cap = p["max_weight_multiple"] / k
        for c in keep:
            cur = snap.held.get(c, 0.0)
            targets[c] = min(cur, cap) if cur > 0 else 1.0 / k
            reasons[c] = "hold" if cur <= cap else "trim_winner"
        for c in add:
            targets[c] = 1.0 / k
            reasons[c] = "entry_top_momentum"
        for c in snap.held:
            if c not in targets:
                reasons[c] = "exit_rank_or_negative_momentum"

        gross = sum(targets.values())
        if gross > p["gross_cap"]:
            f = p["gross_cap"] / gross
            targets = {c: w * f for c, w in targets.items()}
        info.update(n_positions=float(len(targets)), gross=float(sum(targets.values())),
                    best_ret=float(ret.max()), pool_size=float(len(ret)))
        for c in targets:
            info[f"ret336_{c}"] = float(ret[c])
        return Decision(targets, reasons, info)
