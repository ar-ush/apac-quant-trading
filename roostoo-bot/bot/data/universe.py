"""Liquid-universe selection shared by the live bot and the backtester.

Rule: among Roostoo crypto pairs that are tradable, not excluded and not too wide, take the top-N by 30-day median
daily Binance dollar volume (point-in-time: the ranking only uses days that are complete when it is made)."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import pandas as pd


def select_pool(median_daily_qv: pd.Series, history_days: pd.Series, spread_bps: Optional[pd.Series], size: int,
                min_history_days: int, max_spread_bps: float, exclude: Iterable[str] = ()) -> List[str]:
    s = median_daily_qv.dropna()
    s = s[[c for c in s.index if c not in set(exclude)]]
    s = s[history_days.reindex(s.index).fillna(0) >= min_history_days]
    if spread_bps is not None:
        sp = spread_bps.reindex(s.index)
        s = s[~(sp > max_spread_bps)]  # unknown spread (NaN) is not penalised
    return list(s.sort_values(ascending=False).head(size).index)


def daily_pools(qv_hourly: pd.DataFrame, size: int, min_history_days: int, exclude: Iterable[str] = ()) -> Dict[pd.Timestamp, List[str]]:
    """Backtest helper: pool in force on each UTC day, computed from data up to the end of the previous day."""
    dq = qv_hourly.resample("1D").sum(min_count=1)
    med = dq.rolling(30, min_periods=20).median()
    hist = dq.notna().cumsum()
    pools = {}
    for day in dq.index[1:]:
        prev = dq.index[dq.index.get_loc(day) - 1]
        pools[day] = select_pool(med.loc[prev], hist.loc[prev], None, size, min_history_days, 1e9, exclude)
    return pools
