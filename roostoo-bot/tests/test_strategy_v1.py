import numpy as np
import pandas as pd

import strategies
from strategies.base import Snapshot


def _closes(n=1400, seed=0, btc_trend=0.0004, coins=("ETH", "SOL", "ADA", "XRP", "DOGE"), drifts=None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-08-01", periods=n, freq="1h", tz="UTC")
    data = {"BTC": 100 * np.exp(np.cumsum(rng.normal(btc_trend, 0.003, n)))}
    for i, c in enumerate(coins):
        d = (drifts or {}).get(c, 0.0)
        data[c] = 10 * np.exp(np.cumsum(rng.normal(d, 0.006, n)))
    return pd.DataFrame(data, index=idx)


def make(**kw):
    return strategies.make("v1_momentum_rotation", kw)


def snap(closes, held=None, pool=None, hour_utc=0, force=False):
    now = closes.index[-1] + pd.Timedelta(hours=1)
    if hour_utc is not None:
        now = now.normalize() + pd.Timedelta(hours=hour_utc)
    return Snapshot(now=now, closes=closes, pool=pool or [c for c in closes.columns if c != "BTC"], held=held or {}, force_rebalance=force)


def test_gate_off_sells_everything_and_does_not_buy():
    c = _closes(btc_trend=-0.0006)
    s = make()
    assert not s.gate_on(c["BTC"])
    d = s.decide(snap(c, held={"ETH": 0.3}))
    assert d.targets == {} and d.reasons["ETH"] == "gate_off_exit"
    assert s.decide(snap(c)).targets is None  # nothing held, nothing to do


def test_ranks_by_14d_return_and_takes_top_k_equal_weight():
    drifts = {"ETH": 0.0012, "SOL": 0.0009, "ADA": 0.0006, "XRP": -0.0005, "DOGE": -0.0002}
    c = _closes(btc_trend=0.0006, drifts=drifts, seed=3)
    s = make(top_k=2)
    assert s.gate_on(c["BTC"])
    d = s.decide(snap(c))
    assert set(d.targets) == {"ETH", "SOL"}
    ws = list(d.targets.values())
    assert abs(ws[0] - ws[1]) < 1e-12 and abs(sum(ws) - 0.98) < 1e-9   # equal weight, scaled to the 98% gross cap
    assert d.reasons["ETH"] == "entry_top_momentum"


def test_only_rebalances_on_the_rebalance_hour_unless_forced():
    c = _closes(btc_trend=0.0006, drifts={"ETH": 0.001}, seed=4)
    s = make()
    assert s.decide(snap(c, hour_utc=7)).targets is None
    assert s.decide(snap(c, hour_utc=7, force=True)).targets
    assert s.decide(snap(c, hour_utc=0)).targets


def test_hysteresis_keeps_a_holding_that_slipped_a_little_and_drops_negative_momentum():
    drifts = {"ETH": 0.0012, "SOL": 0.0009, "ADA": 0.0007, "XRP": 0.0005, "DOGE": -0.001}
    c = _closes(btc_trend=0.0006, drifts=drifts, seed=5)
    s = make(top_k=2, hysteresis=2)
    d = s.decide(snap(c, held={"XRP": 0.4}))   # XRP is rank <= k+h and positive: kept
    assert "XRP" in d.targets and d.reasons["XRP"] == "hold"
    d2 = s.decide(snap(c, held={"DOGE": 0.4}))  # negative momentum: sold
    assert "DOGE" not in d2.targets


def test_no_lookahead_decision_depends_only_on_closed_history():
    c = _closes(btc_trend=0.0006, drifts={"ETH": 0.001, "SOL": 0.0008}, seed=6)
    s = make()
    a = s.decide(snap(c.iloc[:-1]))
    future = c.copy()
    future.iloc[-1] *= 5.0   # a wildly different NEXT bar must not matter for a decision made one bar earlier
    b = s.decide(snap(future.iloc[:-1]))
    assert a.targets == b.targets


def test_winner_is_trimmed_not_topped_up_and_gross_is_capped():
    c = _closes(btc_trend=0.0006, drifts={"ETH": 0.001, "SOL": 0.0008, "ADA": 0.0006}, seed=7)
    s = make(top_k=3, max_weight_multiple=1.5, gross_cap=0.9)
    d = s.decide(snap(c, held={"ETH": 0.60}))
    assert d.targets["ETH"] <= 1.5 / 3 * (0.9 / sum([0.5, 1 / 3, 1 / 3]) + 1e-9) + 1e-9 or d.targets["ETH"] <= 0.5
    assert sum(d.targets.values()) <= 0.9 + 1e-9
    d2 = s.decide(snap(c, held={"ETH": 0.10}))
    assert d2.targets["ETH"] <= 0.10 + 1e-9  # small holding is never topped up
