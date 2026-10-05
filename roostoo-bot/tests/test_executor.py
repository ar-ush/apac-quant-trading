from decimal import Decimal

from bot.config import ExecutionConfig
from bot.execution.executor import Executor
from bot.execution.rules import PairRules
from bot.journal import Journal


class FakeClient:
    """Minimal in-memory exchange: limit orders rest until `fill_limits` is flipped, market orders fill at once."""

    def __init__(self, wallet, quotes, fill_limits=False):
        self.wallet, self.quotes, self.fill_limits = wallet, quotes, fill_limits
        self.orders, self.next_id, self.calls = {}, 1, []

    def balance(self):
        return self.wallet

    def ticker(self, pair=None):
        return {pair: self.quotes[pair]} if pair else self.quotes

    def place_order(self, pair, side, quantity, order_type="MARKET", price=None):
        self.calls.append(("place", pair, side, quantity, order_type, price))
        oid, self.next_id = self.next_id, self.next_id + 1
        q = float(quantity)
        px = float(price) if price else float(self.quotes[pair]["MinAsk" if side == "BUY" else "MaxBid"])
        status = "FILLED" if (order_type == "MARKET" or self.fill_limits) else "PENDING"
        o = dict(OrderID=oid, Pair=pair, Side=side, Status=status, FilledQuantity=q if status == "FILLED" else 0.0,
                 FilledAverPrice=px if status == "FILLED" else 0.0, CommissionChargeValue=0.0, Role="TAKER" if order_type == "MARKET" else "MAKER")
        self.orders[oid] = o
        if status == "FILLED":
            self._settle(pair, side, q, px)
        return o

    def _settle(self, pair, side, q, px):
        coin = pair.split("/")[0]
        w = self.wallet
        w.setdefault(coin, {"Free": 0.0, "Lock": 0.0})
        if side == "BUY":
            w["USD"]["Free"] -= q * px
            w[coin]["Free"] += q
        else:
            w["USD"]["Free"] += q * px
            w[coin]["Free"] -= q

    def query_order(self, order_id=None, pair=None, pending_only=None, limit=None):
        if order_id is not None:
            return [self.orders[order_id]] if order_id in self.orders else []
        if pending_only:
            return [o for o in self.orders.values() if o["Status"] == "PENDING"]
        return list(self.orders.values())

    def cancel_order(self, order_id=None, pair=None):
        self.calls.append(("cancel", order_id))
        for oid, o in self.orders.items():
            if (order_id is None or oid == order_id) and o["Status"] == "PENDING":
                o["Status"] = "CANCELED"
        return []


def rules():
    return {"ETH/USD": PairRules("ETH/USD", 2, 4, 1.0, True), "SOL/USD": PairRules("SOL/USD", 2, 3, 1.0, True),
            "ADA/USD": PairRules("ADA/USD", 4, 1, 1.0, True)}


def make_exec(tmp_path, client, **cfg):
    c = ExecutionConfig(limit_timeout_sec=30, poll_interval_sec=10, min_trade_usd=50.0, **cfg)
    t = [0.0]

    def sleep(s):
        t[0] += s

    return Executor(client, rules(), c, Journal(tmp_path), dry_run=False, sleep=sleep, clock=lambda: t[0])


QUOTES = {"ETH/USD": {"MaxBid": 2000.0, "MinAsk": 2000.5}, "SOL/USD": {"MaxBid": 100.0, "MinAsk": 100.1},
          "ADA/USD": {"MaxBid": 0.50, "MinAsk": 0.5005}}


def test_sells_first_then_buys_with_cash_actually_free(tmp_path):
    wallet = {"USD": {"Free": 1000.0, "Lock": 0.0}, "ETH": {"Free": 10.0, "Lock": 0.0}}  # equity 21000
    cl = FakeClient(wallet, QUOTES, fill_limits=True)
    ex = make_exec(tmp_path, cl)
    equity = 1000 + 10 * 2000
    orders = ex.rebalance({"SOL": 0.5, "ADA": 0.3}, {"SOL": "entry", "ADA": "entry"}, equity, wallet, QUOTES)
    sides = [c[2] for c in cl.calls if c[0] == "place"]
    assert sides[0] == "SELL" and set(sides[1:]) == {"BUY"}          # ETH sold first to fund the buys
    assert wallet["ETH"]["Free"] < 1e-9 and wallet["USD"]["Free"] >= -1e-6   # never overspends
    assert all(o.filled > 0 for o in orders)


def test_unfilled_limit_is_cancelled_then_market_fallback(tmp_path):
    wallet = {"USD": {"Free": 10000.0, "Lock": 0.0}}
    cl = FakeClient(wallet, QUOTES, fill_limits=False)
    ex = make_exec(tmp_path, cl)
    ex.rebalance({"SOL": 0.5}, {"SOL": "entry"}, 10000.0, wallet, QUOTES)
    kinds = [(c[0], c[4] if c[0] == "place" else None) for c in cl.calls]
    assert ("place", "LIMIT") in kinds and ("cancel", None) in kinds and ("place", "MARKET") in kinds
    assert wallet["SOL"]["Free"] > 0


def test_entry_is_skipped_when_price_ran_away(tmp_path):
    wallet = {"USD": {"Free": 10000.0, "Lock": 0.0}}
    cl = FakeClient(wallet, QUOTES, fill_limits=False)
    ex = make_exec(tmp_path, cl, entry_chase_cap=0.015)
    orig_place = cl.place_order

    def place(pair, side, quantity, order_type="MARKET", price=None):
        if order_type == "MARKET":
            raise AssertionError("must not chase")
        return orig_place(pair, side, quantity, order_type, price)

    cl.place_order = place
    cl.quotes = {**QUOTES, "SOL/USD": {"MaxBid": 103.0, "MinAsk": 103.1}}  # +3% after the decision
    ex.rebalance({"SOL": 0.5}, {"SOL": "entry"}, 10000.0, wallet, {**QUOTES})
    assert "SOL" not in wallet or wallet["SOL"]["Free"] == 0


def test_dust_and_small_drift_do_not_trade(tmp_path):
    wallet = {"USD": {"Free": 5000.0, "Lock": 0.0}, "SOL": {"Free": 50.0, "Lock": 0.0}}  # 5000 usd in SOL, equity 10000
    cl = FakeClient(wallet, QUOTES, fill_limits=True)
    ex = make_exec(tmp_path, cl)
    ex.rebalance({"SOL": 0.51}, {"SOL": "hold"}, 10000.0, wallet, QUOTES)  # delta = $100 > min_trade 50 -> buys
    n1 = len([c for c in cl.calls if c[0] == "place"])
    ex2 = make_exec(tmp_path, FakeClient({"USD": {"Free": 5000.0, "Lock": 0.0}, "SOL": {"Free": 50.0, "Lock": 0.0}}, QUOTES, True))
    ex2.rebalance({"SOL": 0.502}, {"SOL": "hold"}, 10000.0, ex2.client.wallet, QUOTES)  # delta $20 < min_trade
    assert n1 == 1 and not [c for c in ex2.client.calls if c[0] == "place"]


def test_dry_run_places_nothing(tmp_path):
    wallet = {"USD": {"Free": 10000.0, "Lock": 0.0}}
    cl = FakeClient(wallet, QUOTES)
    ex = make_exec(tmp_path, cl)
    ex.dry_run = True
    orders = ex.rebalance({"SOL": 0.5}, {"SOL": "entry"}, 10000.0, wallet, QUOTES)
    assert not cl.calls and orders and orders[0].status == "DRY_RUN"


def test_resting_limit_order_with_bogus_filled_quantity_is_not_treated_as_filled():
    from bot.execution.executor import Order
    o = Order("AAVE/USD", "BUY", "entry", Decimal("2.747"), "LIMIT", Decimal("182.00"), 182.0)
    # real Roostoo payload for a resting order: FilledQuantity equals Quantity, CoinChange is 0, Status PENDING
    o.apply(dict(OrderID=1, Status="PENDING", Role="MAKER", Quantity=2.747, FilledQuantity=2.747, FilledAverPrice=0, CoinChange=0))
    assert not o.done and o.filled == 0.0
    o.apply(dict(OrderID=1, Status="CANCELED", Quantity=2.747, FilledQuantity=2.747, FilledAverPrice=0, CoinChange=0))
    assert o.done and o.filled == 0.0                      # cancelled with nothing exchanged
    o.apply(dict(OrderID=1, Status="FILLED", Quantity=2.747, FilledQuantity=2.747, FilledAverPrice=181.9, CoinChange=2.747,
                 CommissionCoin="USD", CommissionChargeValue=0.25))
    assert o.done and o.filled == 2.747 and o.avg_price == 181.9 and o.fee == 0.25
