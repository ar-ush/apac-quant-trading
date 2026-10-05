"""The trading loop: account snapshot -> risk guards -> (new hourly bar?) strategy decision -> execution -> journal."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

import strategies
from bot.config import Config
from bot.data.binance import BinanceData, BinanceUnavailable
from bot.data.universe import select_pool
from bot.execution.client import RoostooClient, RoostooError
from bot.execution.executor import Executor
from bot.execution.rules import PairRules, parse_exchange_info
from bot.journal import Journal
from bot.risk.guards import KillSwitch, price_deviation_bps, spread_bps
from bot.state import State, StateStore
from strategies.base import Snapshot

log = logging.getLogger("engine")
H_MS = 3_600_000


class Bot:
    def __init__(self, cfg: Config, client: RoostooClient, binance: BinanceData, journal: Journal, store: StateStore,
                 dry_run: bool = True, sleep=time.sleep):
        self.cfg, self.client, self.binance, self.journal, self.store = cfg, client, binance, journal, store
        self.dry_run, self.sleep = dry_run, sleep
        self.strategy = strategies.make(cfg.strategy.name, cfg.strategy.params)
        self.kill = KillSwitch(cfg.risk)
        self.state: State = store.load()
        self.rules: Dict[str, PairRules] = {}
        self.executor: Optional[Executor] = None
        self.last_equity_log = 0.0
        self.pause_orders_until = 0.0

    # ------------------------------------------------------------------ setup
    def startup(self) -> None:
        self.client.sync_clock()
        self.rules = parse_exchange_info(self.client.exchange_info())
        self.executor = Executor(self.client, self.rules, self.cfg.execution, self.journal, dry_run=self.dry_run)
        self.journal.event(dict(type="startup", strategy=self.strategy.name, version=self.strategy.version,
                                params=self.strategy.params, account=self.cfg.account, dry_run=self.dry_run))
        log.info("started: strategy=%s account=%s dry_run=%s pairs=%d", self.strategy.name, self.cfg.account,
                 self.dry_run, len(self.rules))

    # ------------------------------------------------------------------ account
    def snapshot_account(self) -> Tuple[Dict, Dict, float, Dict[str, float]]:
        wallet = self.client.balance()
        quotes = self.client.ticker()
        usd = float(wallet.get("USD", {}).get("Free", 0)) + float(wallet.get("USD", {}).get("Lock", 0))
        equity, values = usd, {}
        for coin, w in wallet.items():
            if coin == "USD":
                continue
            qty = float(w.get("Free", 0)) + float(w.get("Lock", 0))
            q = quotes.get(f"{coin}/USD")
            if qty > 0 and q:
                v = qty * float(q["MaxBid"])
                values[coin] = v
                equity += v
        # weights used by the strategy exclude the daily-activity probe position and dust
        held = {}
        for coin, v in values.items():
            r = self.rules.get(f"{coin}/USD")
            probe_v = self.state.probe_qty * float(quotes[f"{coin}/USD"]["MaxBid"]) if coin == "BTC" else 0.0
            net = v - probe_v
            if r and net >= max(r.min_order_value, 0.002 * equity):
                held[coin] = net / equity
        return wallet, quotes, equity, held

    # ------------------------------------------------------------------ universe
    def refresh_pool(self, quotes: Dict) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.pool_day == today and self.state.pool:
            return
        cands = [r.pair.split("/")[0] for r in self.rules.values()
                 if r.asset_type == "crypto" and r.can_trade and r.pair in quotes
                 and r.pair.split("/")[0] not in self.cfg.universe.exclude]
        med, hist, sp = {}, {}, {}
        for c in cands:
            try:
                m, d = self.binance.daily_stats(c, self.cfg.universe.history_days)
            except BinanceUnavailable:
                continue
            med[c], hist[c] = m, d
            s = spread_bps(quotes[f"{c}/USD"])
            if s is not None:
                sp[c] = s
        if not med:
            log.warning("pool refresh failed, keeping previous pool of %d coins", len(self.state.pool))
            return
        pool = select_pool(pd.Series(med), pd.Series(hist), pd.Series(sp), self.cfg.universe.size,
                           self.cfg.universe.history_days, self.cfg.universe.max_spread_bps, self.cfg.universe.exclude)
        self.state.pool, self.state.pool_day = pool, today
        self.store.save(self.state)
        self.journal.event(dict(type="pool", day=today, pool=pool))
        log.info("pool refreshed: %s", pool)

    # ------------------------------------------------------------------ one tick
    def tick(self) -> None:
        now_ms = self.client.now_ms()
        wallet, quotes, equity, held = self.snapshot_account()
        self._log_equity(now_ms, equity, wallet, quotes)

        peak, halted_until, triggered = self.kill.update(equity, self.state.peak_equity or equity, now_ms,
                                                         self.state.halted_until_ms)
        self.state.peak_equity, self.state.halted_until_ms = peak, halted_until
        if triggered:
            self.journal.event(dict(type="kill_switch", equity=equity, peak=peak, until_ms=halted_until))
            log.error("KILL SWITCH: drawdown limit hit, going to cash until %s", halted_until)
        halted = now_ms < self.state.halted_until_ms
        if halted:
            if held:
                self._execute({}, {c: "kill_switch_exit" for c in held}, equity, wallet, quotes)
            self.store.save(self.state)
            return

        if time.time() < self.pause_orders_until:
            return
        if self.client.error_rate() > 0.5 and len(self.client.recent) >= 10:
            self.pause_orders_until = time.time() + self.cfg.risk.api_error_pause_sec
            self.journal.event(dict(type="api_error_pause", rate=self.client.error_rate()))
            return

        bar_id = now_ms // H_MS
        into_hour = (now_ms % H_MS) / 1000.0
        new_bar = self.state.last_bar_id != bar_id and (into_hour >= self.cfg.runtime.bar_close_delay_sec or not self.state.started)
        if new_bar:
            self.refresh_pool(quotes)
            if self._decide(now_ms, bar_id, wallet, quotes, equity, held):
                self.state.last_bar_id = bar_id
                self.state.started = True
        self._activity_guard(now_ms, wallet, quotes, equity)
        self.store.save(self.state)

    # ------------------------------------------------------------------ decision
    def _decide(self, now_ms: int, bar_id: int, wallet, quotes, equity: float, held: Dict[str, float]) -> bool:
        if not self.state.pool:
            log.warning("empty pool; skipping decision")
            return False
        coins = sorted(set(self.state.pool) | {"BTC"})
        need = {c: self.strategy.required_history_hours() for c in coins}
        need["BTC"] = self.strategy.btc_history_hours()
        try:
            closes = self.binance.hourly_closes(coins, need, now_ms)
        except BinanceUnavailable as exc:
            log.error("Binance unavailable: %s", exc)
            return False
        last_open = bar_id * H_MS - H_MS
        if closes.empty or "BTC" not in closes.columns or int(closes.index[-1].timestamp() * 1000) < last_open:
            age_h = (now_ms - int(closes.index[-1].timestamp() * 1000)) / H_MS if len(closes) else 99
            log.warning("latest closed bar missing or stale (age %.1fh)", age_h)
            if age_h > self.cfg.risk.max_data_age_hours:
                self.journal.event(dict(type="stale_data", age_hours=age_h))
            return False
        snap = Snapshot(now=pd.Timestamp(bar_id * H_MS, unit="ms", tz="UTC"), closes=closes, pool=list(self.state.pool),
                        held=held, force_rebalance=(not self.state.started and not held))
        dec = self.strategy.decide(snap)
        self.journal.decision(dict(bar=str(snap.now), targets=dec.targets, reasons=dec.reasons, info=dec.info,
                                   held=held, equity=equity, force=snap.force_rebalance))
        if dec.targets is None:
            return True
        targets = self._sanity_filter(dec.targets, held, closes, quotes)
        self._execute(targets, dec.reasons, equity, wallet, quotes)
        return True

    def _sanity_filter(self, targets: Dict[str, float], held: Dict[str, float], closes: pd.DataFrame, quotes) -> Dict[str, float]:
        """Drop NEW entries whose Roostoo price disagrees with Binance (bad feed / stale quote)."""
        out = {}
        for c, w in targets.items():
            q = quotes.get(f"{c}/USD")
            if c not in held and q and c in closes.columns:
                mid = (float(q["MaxBid"]) + float(q["MinAsk"])) / 2
                dev = price_deviation_bps(mid, float(closes[c].dropna().iloc[-1]))
                if dev > self.cfg.risk.max_price_deviation_bps:
                    self.journal.event(dict(type="entry_dropped_price_deviation", coin=c, bps=dev))
                    continue
            out[c] = w
        return out

    def _execute(self, targets, reasons, equity, wallet, quotes) -> None:
        orders = self.executor.rebalance(targets, reasons, equity, wallet, quotes)
        if any(o.filled > 0 for o in orders):
            self.state.last_fill_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.store.save(self.state)

    # ------------------------------------------------------------------ compliance: >=1 filled order per UTC day
    def _activity_guard(self, now_ms: int, wallet, quotes, equity: float) -> None:
        a = self.cfg.activity
        if not a.enabled:
            return
        now = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
        today = now.strftime("%Y-%m-%d")
        if now.hour < a.deadline_hour_utc or self.state.last_fill_day == today:
            return
        usd = equity * a.probe_usd_fraction
        if self.state.probe_qty > 0:
            free = float(wallet.get("BTC", {}).get("Free", 0))
            order = self.executor.market_probe("BTC", "SELL", usd, quotes, "activity_probe_sell",
                                               qty_override=min(self.state.probe_qty, free))
        else:
            order = self.executor.market_probe("BTC", "BUY", usd, quotes, "activity_probe_buy")
        if order is not None and order.filled > 0:
            self.state.probe_qty = max(0.0, self.state.probe_qty + (order.filled if order.side == "BUY" else -order.filled))
            self.state.last_fill_day = today
            self.journal.event(dict(type="activity_probe", side=order.side, filled=order.filled, probe_qty=self.state.probe_qty))

    # ------------------------------------------------------------------ logging
    def _log_equity(self, now_ms: int, equity: float, wallet, quotes) -> None:
        if time.time() - self.last_equity_log < 55:
            return
        self.last_equity_log = time.time()
        pos = {c: float(w.get("Free", 0)) + float(w.get("Lock", 0)) for c, w in wallet.items()
               if c != "USD" and float(w.get("Free", 0)) + float(w.get("Lock", 0)) > 0}
        self.journal.equity(dict(equity=round(equity, 2), usd=float(wallet.get("USD", {}).get("Free", 0)), positions=pos,
                                 api_calls_last_min=self.client.budget.used()))

    # ------------------------------------------------------------------ main loop
    def run_forever(self) -> None:
        self.startup()
        while True:
            t0 = time.time()
            try:
                self.tick()
            except RoostooError as exc:
                log.error("roostoo error: %s", exc)
                self.journal.event(dict(type="roostoo_error", error=str(exc)))
            except Exception:  # never let the loop die: log and carry on
                log.exception("tick failed")
                self.journal.event(dict(type="tick_exception"))
            self.sleep(max(1.0, self.cfg.runtime.loop_seconds - (time.time() - t0)))
