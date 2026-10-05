"""Strategy interface. A strategy is a pure function of a market snapshot: no I/O, no API calls.

The same class is called by the live engine and by the backtester, so research and production cannot diverge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Snapshot:
    now: pd.Timestamp                 # decision time = close time of the last closed hourly bar (UTC)
    closes: pd.DataFrame              # hourly closes; index = bar OPEN time (UTC); last row = last CLOSED bar
    pool: List[str]                   # tradable liquid universe (coin symbols, e.g. "ETH")
    held: Dict[str, float]            # coin -> current portfolio weight (fraction of equity)
    force_rebalance: bool = False     # engine sets this on (re)start so an idle account deploys immediately


@dataclass
class Decision:
    targets: Optional[Dict[str, float]]          # target weights per coin; coins absent are sold; None = leave book untouched
    reasons: Dict[str, str] = field(default_factory=dict)
    info: Dict[str, float] = field(default_factory=dict)   # diagnostics logged with the decision


class Strategy:
    name = "base"
    version = "0"

    def __init__(self, **params):
        self.params = params

    def required_history_hours(self) -> int:
        """Hourly bars of history the engine must load for every coin."""
        return 400

    def btc_history_hours(self) -> int:
        """Hourly bars of BTC history (long EMAs need a longer warm-up)."""
        return 2000

    def decide(self, snap: Snapshot) -> Decision:
        raise NotImplementedError
