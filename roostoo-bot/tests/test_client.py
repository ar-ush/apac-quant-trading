from decimal import Decimal

from bot.execution.client import CallBudget, build_query, sign
from bot.execution.rules import fmt, parse_exchange_info, round_price, round_qty


def test_query_is_sorted_and_signature_is_hmac_sha256_hex():
    q = build_query({"timestamp": "1580000000000", "pair": "BTC/USD", "side": "BUY"})
    assert q == "pair=BTC/USD&side=BUY&timestamp=1580000000000"
    sig = sign(q, "secret")
    assert len(sig) == 64 and sig == sign(q, "secret") and sig != sign(q, "other")
    # reference value computed independently with openssl: echo -n <q> | openssl dgst -sha256 -hmac secret
    import hashlib, hmac
    assert sig == hmac.new(b"secret", q.encode(), hashlib.sha256).hexdigest()


def test_call_budget_blocks_when_exhausted():
    t = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        t[0] += s

    b = CallBudget(3, 60.0, clock=lambda: t[0], sleep=sleep)
    for _ in range(3):
        b.acquire()
    assert not slept
    b.acquire()  # the 4th must wait for the first call to leave the window
    assert slept and sum(slept) >= 59.9


def test_rounding_never_exceeds_and_never_crosses_the_spread():
    assert round_qty(1.239999, 2) == Decimal("1.23")
    assert round_qty(0.00009999, 4) == Decimal("0.0000")
    assert fmt(round_qty(0.000012, 6)) == "0.000012"          # never exponent notation
    assert round_price(100.019, 2, "BUY") == Decimal("100.01")   # buy rounds down
    assert round_price(100.011, 2, "SELL") == Decimal("100.02")  # sell rounds up


def test_exchange_info_parsing():
    info = {"TradePairs": {"BTC/USD": {"PricePrecision": 2, "AmountPrecision": 5, "MiniOrder": 1, "CanTrade": True, "AssetType": "crypto"},
                           "NVDAB/USD": {"PricePrecision": 2, "AmountPrecision": 3, "MiniOrder": 1, "CanTrade": False, "AssetType": "stock"}}}
    r = parse_exchange_info(info)
    assert r["BTC/USD"].amount_decimals == 5 and r["BTC/USD"].can_trade
    assert not r["NVDAB/USD"].can_trade and r["NVDAB/USD"].asset_type == "stock"
