"""Engine-level tests with a fake exchange and fake Binance: bar cadence, deployment, gate exit, kill switch, activity rule."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bot.config import load_config
from bot.engine import Bot
from bot.execution.client import CallBudget
from bot.journal import Journal
from bot.state import StateStore

H = 3_600_000


def ms(y, mo, d, h, mi=0):
    return int(datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp() * 1000)


class FakeRoostoo:
    def __init__(self, now_ms):
        self.now = now_ms
        self.recent = []
        self.budget = CallBudget(1000)
        self.wallet = {"USD": {"Free": 100_000.0, "Lock": 0.0}}
        self.prices = {"ETH": 2000.0, "SOL": 100.0, "ADA": 0.5, "BTC": 80_000.0, "XRP": 1.5}
        self.placed = []
        self.oid = 1

    def now_ms(self): return self.now
    def sync_clock(self): return 0
    def error_rate(self): return 0.0

    def exchange_info(self):
        tp = {f"{c}/USD": dict(Coin=c, PricePrecision=4, AmountPrecision=4, MiniOrder=1, CanTrade=True, AssetType="crypto")
              for c in self.prices}
        return {"TradePairs": tp}

    def ticker(self, pair=None):
        d = {f"{c}/USD": dict(MaxBid=p * 0.9999, MinAsk=p * 1.0001, LastPrice=p) for c, p in self.prices.items()}
        return {pair: d[pair]} if pair else d

    def balance(self): return self.wallet

    def place_order(self, pair, side, quantity, order_type="MARKET", price=None):
        coin = pair.split("/")[0]
        q = float(quantity)
        px = self.prices[coin] * (1.0001 if side == "BUY" else 0.9999)
        self.placed.append((pair, side, q, order_type))
        self.wallet.setdefault(coin, {"Free": 0.0, "Lock": 0.0})
        sgn = 1 if side == "BUY" else -1
        self.wallet["USD"]["Free"] -= sgn * q * px
        self.wallet[coin]["Free"] += sgn * q
        oid, self.oid = self.oid, self.oid + 1
        return dict(OrderID=oid, Status="FILLED", Role="TAKER", FilledQuantity=q, CoinChange=q, FilledAverPrice=px,
                    CommissionCoin="USD", CommissionChargeValue=q * px * 0.001)

    def query_order(self, **kw): return []
    def cancel_order(self, **kw): return []


class FakeBinance:
    def __init__(self, bot_clock, trend=0.0005, price_fn=None):
        self.clock, self.trend, self.price_fn = bot_clock, trend, price_fn

    def daily_stats(self, coin, days=60):
        return 5e7, 200

    def hourly_closes(self, coins, n_by_coin, now_ms=None):
        now_ms = now_ms or self.clock()
        last_open = (now_ms // H) * H - H
        n = max(n_by_coin.values()) + 5
        idx = pd.to_datetime([last_open - (n - 1 - i) * H for i in range(n)], unit="ms", utc=True)
        rng = np.random.default_rng(1)
        data = {}
        for c in coins:
            drift = {"BTC": self.trend, "ETH": 0.0012, "SOL": 0.0009, "ADA": 0.0006}.get(c, -0.0003)
            path = np.exp(np.cumsum(rng.normal(drift, 0.002, n)))
            data[c] = path / path[-1] * (self.price_fn(c) if self.price_fn else 50.0)   # end at the exchange price
        return pd.DataFrame(data, index=idx)


def make_bot(tmp_path, now_ms, trend=0.0005):
    cfg = load_config(account="test")
    cfg.api_key = cfg.secret_key = "x"
    cfg.universe.size = 5
    cfg.execution.use_limit_orders = False      # market orders: fills are instantaneous in the fake
    cfg.execution.min_trade_usd = 50.0
    client = FakeRoostoo(now_ms)
    bot = Bot(cfg, client, FakeBinance(lambda: client.now, trend, lambda c: client.prices[c]), Journal(tmp_path / "logs"), StateStore(tmp_path / "state", "t"),
              dry_run=False, sleep=lambda s: None)
    bot.startup()
    return bot, client


def test_deploys_once_per_bar_and_not_between_rebalance_hours(tmp_path):
    bot, cl = make_bot(tmp_path, ms(2026, 10, 6, 0, 2))          # 00:02 UTC: rebalance bar
    bot.tick()
    buys = [p for p in cl.placed if p[1] == "BUY"]
    assert len(buys) == 3 and {p[0] for p in buys} == {"ETH/USD", "SOL/USD", "ADA/USD"}
    n = len(cl.placed)
    cl.now += 60_000
    bot.tick()                                                    # same bar: no second decision
    assert len(cl.placed) == n
    cl.now = ms(2026, 10, 6, 5, 2)
    bot.tick()                                                    # 05:02: new bar but not the rebalance hour, gate on -> no trades
    assert len(cl.placed) == n


def test_gate_off_liquidates_within_the_hour(tmp_path):
    bot, cl = make_bot(tmp_path, ms(2026, 10, 6, 0, 2))
    bot.tick()
    assert any(p[1] == "BUY" for p in cl.placed)
    bot.binance.trend = -0.0008                                   # BTC now trends down -> gate off
    cl.now = ms(2026, 10, 6, 3, 2)
    bot.tick()
    sells = [p for p in cl.placed if p[1] == "SELL"]
    assert {p[0] for p in sells} == {"ETH/USD", "SOL/USD", "ADA/USD"}
    assert all(cl.wallet[c]["Free"] < 1e-6 for c in ("ETH", "SOL", "ADA"))


def test_kill_switch_sells_everything_and_pauses(tmp_path):
    bot, cl = make_bot(tmp_path, ms(2026, 10, 6, 0, 2))
    bot.tick()
    cl.prices = {c: p * 0.5 for c, p in cl.prices.items()}       # crash: equity down ~49%
    cl.now += 5 * 60_000
    bot.tick()
    assert bot.state.halted_until_ms > cl.now
    assert all(cl.wallet[c]["Free"] < 1e-6 for c in ("ETH", "SOL", "ADA"))
    before = len(cl.placed)
    cl.now += 3_600_000
    bot.tick()
    assert len(cl.placed) == before                               # still paused: no new entries


def test_activity_rule_makes_one_small_trade_when_nothing_else_filled(tmp_path):
    bot, cl = make_bot(tmp_path, ms(2026, 10, 6, 21, 30), trend=-0.0008)  # gate off: strategy stays in cash
    bot.tick()
    probe = [p for p in cl.placed if p[0] == "BTC/USD"]
    assert len(probe) == 1 and probe[0][1] == "BUY" and probe[0][2] * 80_000 < 0.005 * 100_000 * 1.2
    assert bot.state.probe_qty > 0
    cl.now += 60_000
    bot.tick()
    assert len([p for p in cl.placed if p[0] == "BTC/USD"]) == 1  # only once per UTC day
    # next day, still nothing to do: the probe is reversed
    cl.now = ms(2026, 10, 7, 21, 30)
    bot.tick()
    assert [p[1] for p in cl.placed if p[0] == "BTC/USD"] == ["BUY", "SELL"]


# ---------------------------------------------------------------- regression tests from the pre-deploy audit (2026-10-06)
def test_probe_restarts_after_a_rebalance_sold_the_probe_btc(tmp_path):
    bot, cl = make_bot(tmp_path, ms(2026, 10, 6, 21, 30), trend=-0.0008)
    bot.tick()
    assert bot.state.probe_qty > 0
    cl.wallet["BTC"]["Free"] = 0.0                                # a normal rebalance sold it
    cl.now = ms(2026, 10, 7, 21, 30)
    bot.tick()
    assert [p[1] for p in cl.placed if p[0] == "BTC/USD"] == ["BUY", "BUY"]   # the probe is alive again
    assert bot.state.probe_qty > 0


def test_missing_quote_does_not_trip_the_kill_switch(tmp_path):
    bot, cl = make_bot(tmp_path, ms(2026, 10, 6, 0, 2))
    bot.tick()
    n = len(cl.placed)
    del cl.prices["ETH"]                                          # ticker silently omits a held coin
    cl.now += 5 * 60_000
    bot.tick()
    assert bot.state.halted_until_ms == 0 and len(cl.placed) == n


def test_empty_ticker_skips_the_tick(tmp_path):
    import pytest
    from bot.execution.client import RoostooError
    bot, cl = make_bot(tmp_path, ms(2026, 10, 6, 0, 2))
    cl.ticker = lambda pair=None: {}
    with pytest.raises(RoostooError):
        bot.tick()
    assert cl.placed == []


def test_missed_midnight_rebalance_is_caught_up(tmp_path):
    bot, cl = make_bot(tmp_path, ms(2026, 10, 6, 5, 2))
    bot.state.started, bot.state.last_rebal_day = True, "2026-10-05"   # the 00:00 bar of Oct 6 never ran
    bot.state.last_bar_id = (ms(2026, 10, 6, 4, 2) // H)
    bot.tick()
    assert {p[0] for p in cl.placed if p[1] == "BUY"} == {"ETH/USD", "SOL/USD", "ADA/USD"}
    assert bot.state.last_rebal_day == "2026-10-06"
    n = len(cl.placed)
    cl.now = ms(2026, 10, 6, 9, 2)
    bot.tick()
    assert len(cl.placed) == n                                    # caught up once, not every hour


def test_held_coin_with_no_data_is_not_sold_as_a_ranking_exit(tmp_path):
    bot, cl = make_bot(tmp_path, ms(2026, 10, 6, 0, 2))
    bot.tick()
    orig = bot.binance.hourly_closes
    bot.binance.hourly_closes = lambda coins, need, now_ms=None: orig(coins, need, now_ms).drop(columns=["ETH"])
    cl.now = ms(2026, 10, 7, 0, 2)
    bot.tick()
    assert not [p for p in cl.placed if p[0] == "ETH/USD" and p[1] == "SELL"]


def test_activity_guard_uses_a_rolling_18h_window(tmp_path):
    bot, cl = make_bot(tmp_path, ms(2026, 10, 6, 0, 2))
    bot.tick()                                                    # strategy fills at 00:02
    cl.now = ms(2026, 10, 6, 20, 5)                               # 20h later, no fill since: probe is due even though same UTC day
    bot.tick()
    assert [p for p in cl.placed if p[0] == "BTC/USD"]
