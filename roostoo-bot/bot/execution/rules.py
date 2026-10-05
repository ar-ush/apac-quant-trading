"""Exchange trading rules and decimal-safe rounding (quantities down, prices toward our side of the book)."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Any, Dict


@dataclass(frozen=True)
class PairRules:
    pair: str
    price_decimals: int
    amount_decimals: int
    min_order_value: float
    can_trade: bool
    asset_type: str = "crypto"


def parse_exchange_info(info: Dict[str, Any]) -> Dict[str, PairRules]:
    rules = {}
    for pair, p in info.get("TradePairs", {}).items():
        rules[pair] = PairRules(pair, int(p["PricePrecision"]), int(p["AmountPrecision"]), float(p.get("MiniOrder", 1.0)),
                                bool(p.get("CanTrade", True)), str(p.get("AssetType", "crypto")))
    return rules


def _q(decimals: int) -> Decimal:
    return Decimal(1).scaleb(-decimals)


def round_qty(value: float, decimals: int) -> Decimal:
    """Round a quantity DOWN so we never ask for more than we have/can afford."""
    return Decimal(repr(float(value))).quantize(_q(decimals), rounding=ROUND_DOWN)


def round_price(value: float, decimals: int, side: str) -> Decimal:
    """Buys round down, sells round up: a passive limit order can then never cross the spread."""
    mode = ROUND_DOWN if side == "BUY" else ROUND_UP
    return Decimal(repr(float(value))).quantize(_q(decimals), rounding=mode)


def fmt(d: Decimal) -> str:
    """Plain decimal text; the API must never see exponent notation like 1E-5."""
    return format(d, "f")
