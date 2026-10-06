"""Turns target weights into Roostoo orders.

Flow for one rebalance:
  1. cancel any stale pending orders (a crashed run may have left some)
  2. plan: compare target dollar value with the wallet; exits trade whole free balance, other trades must clear min_trade_usd
  3. SELLS first: passive limit at the best ask (maker 0.05%); unfilled after limit_timeout_sec -> cancel -> market (taker 0.1%)
  4. re-read wallet, scale BUYS to the cash actually free, passive limit at the best bid; unfilled -> cancel -> market,
     unless the price ran away from the decision price by more than entry_chase_cap (then the entry is skipped)

API budget: one batched poll (`query_order(pending_only=True)`) per poll interval, not one call per order.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from bot.config import ExecutionConfig
from bot.execution.client import RoostooClient, RoostooError
from bot.execution.rules import PairRules, fmt, round_price, round_qty
from bot.journal import Journal

log = logging.getLogger("executor")
FINAL = {"FILLED", "CANCELED", "CANCELLED", "REJECTED", "EXPIRED"}


@dataclass
class Order:
    pair: str
    side: str
    reason: str
    qty: Decimal
    otype: str = "LIMIT"
    price: Optional[Decimal] = None
    ref_price: float = 0.0          # price at decision time (for the chase cap and slippage reporting)
    order_id: Optional[int] = None
    status: str = ""
    role: str = ""
    filled: float = 0.0
    avg_price: float = 0.0
    fee: float = 0.0
    fee_coin: str = ""
    error: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)   # last raw API response, journaled for trade-log integrity

    def apply(self, d: Dict[str, Any]) -> None:
        """Absorb an OrderDetail. Live quirk (seen 2026-10-05): a RESTING limit order reports FilledQuantity == Quantity while
        Status is still PENDING (CoinChange 0), and a cancelled one keeps that bogus value. So the fill size is trusted only
        when Status is FILLED, otherwise CoinChange (the coin amount actually exchanged) is used."""
        self.raw = d
        self.order_id = d.get("OrderID", self.order_id)
        self.status = str(d.get("Status") or self.status).upper()
        self.role = str(d.get("Role") or self.role)
        coin_change = abs(float(d.get("CoinChange") or 0.0))
        if self.status == "FILLED":
            self.filled = float(d.get("FilledQuantity") or coin_change or float(self.qty))
        else:
            self.filled = coin_change
        self.avg_price = float(d.get("FilledAverPrice") or 0.0) or self.avg_price
        self.fee = float(d.get("CommissionChargeValue") or self.fee or 0.0)
        self.fee_coin = str(d.get("CommissionCoin") or self.fee_coin)

    @property
    def done(self) -> bool:
        return self.status in FINAL

    def row(self) -> Dict[str, Any]:
        return dict(pair=self.pair, side=self.side, type=self.otype, order_id=self.order_id, status=self.status,
                    role=self.role, qty=fmt(self.qty), limit_price=fmt(self.price) if self.price is not None else "",
                    ref_price=self.ref_price, filled=self.filled, avg_price=self.avg_price, fee=self.fee,
                    fee_coin=self.fee_coin, reason=self.reason, error=self.error, response=self.raw)


class Executor:
    def __init__(self, client: RoostooClient, rules: Dict[str, PairRules], cfg: ExecutionConfig, journal: Journal,
                 dry_run: bool = True, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.client, self.rules, self.cfg, self.journal = client, rules, cfg, journal
        self.dry_run, self.sleep, self.clock = dry_run, sleep, clock

    # ------------------------------------------------------------------ public
    def rebalance(self, targets: Dict[str, float], reasons: Dict[str, str], equity: float,
                  wallet: Dict[str, Dict[str, float]], quotes: Dict[str, Dict[str, float]]) -> List[Order]:
        self._cancel_stale()
        sells, buys = self._plan(targets, reasons, equity, wallet, quotes)
        done: List[Order] = []
        if sells:
            done += self._execute(sells, quotes, self.cfg.exit_timeout_sec)
        if buys:
            wallet = self.client.balance() if not self.dry_run else wallet
            self._scale_to_cash(buys, wallet, quotes)
            done += self._execute([o for o in buys if o.qty > 0], quotes, self.cfg.limit_timeout_sec)
        return done

    def market_probe(self, coin: str, side: str, usd: float, quotes: Dict[str, Dict[str, float]], reason: str,
                     qty_override: Optional[float] = None) -> Optional[Order]:
        """One small market order (used only by the daily-activity maintenance rule)."""
        pair = f"{coin}/USD"
        r, q = self.rules.get(pair), quotes.get(pair)
        if r is None or q is None or not r.can_trade:
            return None
        px = float(q["MinAsk"] if side == "BUY" else q["MaxBid"])
        qty = round_qty(qty_override if qty_override is not None else max(usd, r.min_order_value * 1.5) / px, r.amount_decimals)
        if qty <= 0 or float(qty) * px < r.min_order_value:
            return None
        order = Order(pair, side, reason, qty, "MARKET", None, px)
        self._place(order)
        self._log(order)
        return order

    # ------------------------------------------------------------------ planning
    def _plan(self, targets, reasons, equity, wallet, quotes):
        sells: List[Order] = []
        buys: List[Order] = []
        held = {c: v for c, v in wallet.items() if c != "USD"}
        coins = set(targets) | {c for c, v in held.items() if float(v.get("Free", 0)) + float(v.get("Lock", 0)) > 0}
        for coin in sorted(coins):
            pair = f"{coin}/USD"
            r, q = self.rules.get(pair), quotes.get(pair)
            if r is None or q is None or not r.can_trade:
                continue
            bid, ask = float(q["MaxBid"]), float(q["MinAsk"])
            free = float(held.get(coin, {}).get("Free", 0.0))
            cur_val = (free + float(held.get(coin, {}).get("Lock", 0.0))) * bid
            tgt_val = targets.get(coin, 0.0) * equity
            delta = tgt_val - cur_val
            why = reasons.get(coin, "rebalance")
            if tgt_val <= 0:
                qty = round_qty(free, r.amount_decimals)
                if qty > 0 and float(qty) * bid >= r.min_order_value:  # dust below the exchange minimum is ignored
                    sells.append(self._new(pair, "SELL", why, qty, ask))
            elif delta < 0 and -delta >= max(self.cfg.min_trade_usd, r.min_order_value):
                qty = round_qty(min(free, -delta / bid), r.amount_decimals)
                if qty > 0 and float(qty) * bid >= r.min_order_value:
                    sells.append(self._new(pair, "SELL", why, qty, ask))
            elif delta > 0 and delta >= max(self.cfg.min_trade_usd, r.min_order_value):
                buys.append(self._new(pair, "BUY", why, Decimal(0), bid, usd=delta))
        return sells, buys

    def _new(self, pair, side, why, qty, touch_px, usd: float = 0.0) -> Order:
        r = self.rules[pair]
        o = Order(pair, side, why, qty, "LIMIT" if self.cfg.use_limit_orders else "MARKET", None, touch_px)
        o._usd = usd  # type: ignore[attr-defined]
        if o.otype == "LIMIT":
            o.price = round_price(touch_px, r.price_decimals, side)
        return o

    def _scale_to_cash(self, buys: List[Order], wallet, quotes) -> None:
        cash = float(wallet.get("USD", {}).get("Free", 0.0)) * (1.0 - self.cfg.cash_buffer)
        want = sum(getattr(o, "_usd", 0.0) for o in buys)
        scale = min(1.0, cash / want) if want > 0 else 0.0
        for o in buys:
            r = self.rules[o.pair]
            px = float(o.price) if o.price is not None else float(quotes[o.pair]["MinAsk"])
            o.qty = round_qty(getattr(o, "_usd", 0.0) * scale / px, r.amount_decimals)
            if float(o.qty) * px < max(r.min_order_value, 1.0):
                o.qty = Decimal(0)

    # ------------------------------------------------------------------ order handling
    def _execute(self, orders: List[Order], quotes, timeout: Optional[int] = None) -> List[Order]:
        out: List[Order] = []
        resting: List[Order] = []
        for o in orders:
            self._place(o)
            (resting if (o.otype == "LIMIT" and not o.done and o.status != "ERROR") else out).append(o)
            if o.otype != "LIMIT" or o.done or o.status == "ERROR":
                self._log(o)
        if resting:
            self._wait(resting, timeout if timeout is not None else self.cfg.limit_timeout_sec)
            for o in resting:
                if not o.done:
                    self._cancel(o)
                self._log(o)
                out.append(o)
                remainder = round_qty(float(o.qty) - o.filled, self.rules[o.pair].amount_decimals)
                if remainder <= 0 or self.dry_run:
                    continue
                fresh = self._fresh_quote(o.pair, quotes)
                px = float(fresh["MinAsk"] if o.side == "BUY" else fresh["MaxBid"])
                if o.side == "BUY" and o.ref_price > 0 and px > o.ref_price * (1 + self.cfg.entry_chase_cap):
                    self.journal.event(dict(type="entry_skipped_chase_cap", pair=o.pair, ref=o.ref_price, now=px))
                    continue
                if float(remainder) * px < self.rules[o.pair].min_order_value:
                    continue
                m = Order(o.pair, o.side, o.reason + "|market_fallback", remainder, "MARKET", None, o.ref_price)
                self._place(m)
                self._log(m)
                out.append(m)
        return out

    def _place(self, o: Order) -> None:
        if self.dry_run:
            o.status = "DRY_RUN"
            log.info("DRY %s %s %s %s @ %s (%s)", o.otype, o.side, fmt(o.qty), o.pair, o.price, o.reason)
            return
        try:
            d = self.client.place_order(o.pair, o.side, fmt(o.qty), o.otype, fmt(o.price) if o.price is not None else None)
            o.apply(d)
            if not o.status:
                o.status = "PENDING"
            log.info("%s %s %s %s -> %s id=%s", o.otype, o.side, fmt(o.qty), o.pair, o.status, o.order_id)
        except RoostooError as exc:
            o.status, o.error = "ERROR", str(exc)
            log.error("order failed %s %s %s: %s", o.side, o.pair, fmt(o.qty), exc)

    def _wait(self, orders: List[Order], timeout: int) -> None:
        deadline = self.clock() + timeout
        pending = [o for o in orders if not o.done]
        while pending and self.clock() < deadline:
            self.sleep(self.cfg.poll_interval_sec)
            if self.dry_run:
                break
            try:
                still = {int(m.get("OrderID")) for m in self.client.query_order(pending_only=True)}
            except RoostooError as exc:
                log.warning("pending poll failed: %s", exc)
                continue
            for o in list(pending):
                if o.order_id is not None and int(o.order_id) not in still:
                    self._refresh(o)
                    pending.remove(o)

    def _refresh(self, o: Order) -> None:
        try:
            rows = self.client.query_order(order_id=o.order_id)
            if rows:
                o.apply(rows[0])
        except RoostooError as exc:
            log.warning("query_order %s failed: %s", o.order_id, exc)

    def _cancel(self, o: Order) -> None:
        if self.dry_run or o.order_id is None:
            return
        try:
            self.client.cancel_order(order_id=o.order_id)
        except RoostooError as exc:
            log.warning("cancel %s failed: %s", o.order_id, exc)
        self._refresh(o)  # a partial fill may have happened before the cancel

    def _cancel_stale(self) -> None:
        if self.dry_run:
            return
        try:
            if self.client.query_order(pending_only=True):
                self.client.cancel_order()
        except RoostooError as exc:
            log.warning("stale-order sweep failed: %s", exc)

    def _fresh_quote(self, pair: str, fallback) -> Dict[str, float]:
        try:
            return self.client.ticker(pair)[pair]
        except Exception:
            return fallback[pair]

    def _log(self, o: Order) -> None:
        self.journal.order(o.row())
