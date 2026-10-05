"""Event-driven multi-coin long-only simulator on OHLC bars (intrabar stops/targets).

Signals are computed at bar close t (using bars <= t). A new position is opened at close[t] (+slippage, taker fee or maker fee),
i.e. it is exposed from bar t+1 onwards. Exits are checked on bars t+1.. using high/low; if stop and target are both inside one
bar the stop is assumed to hit first (conservative). Gap-through stops fill at the open.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Params:
    slots: int = 3
    size: float = 0.0  # fraction of equity per position; 0 -> 1/slots
    tp: float = 0.0  # take-profit as fraction of entry price (0 = none)  (array per coin allowed through tp_arr)
    sl: float = 0.0  # fixed stop fraction below entry (0 = none)
    trail: float = 0.0  # trailing stop fraction below highest high since entry (0 = none)
    max_hold: int = 96  # bars
    cooldown: int = 4  # bars after exit before re-entry in same coin
    fee_entry: float = 0.0010  # taker
    fee_exit_tp: float = 0.0005  # maker on take profit
    fee_exit_other: float = 0.0010
    slip: float = 0.0002
    min_gap_bars: int = 0
    # profit lock overlay (portfolio): once equity gain since window start >= lock_gain stop opening new positions
    lock_gain: float = 0.0
    lock_window_bars: int = 96 * 14
    halt_dd: float = 0.0  # stop new entries when drawdown from peak > halt_dd (resets at new peak)


def run(O, H, L, C, entry, score, p: Params, tp_arr=None, sl_arr=None, start=0, allowed=None, exit_sig=None, trail_arr=None,
        size_arr=None):
    """O,H,L,C: float arrays (T,N). entry: bool (T,N) signals at close of bar t. score: float (T,N) ranking (higher first).
    tp_arr / sl_arr: optional (T,N) arrays of per-entry tp / sl fractions evaluated at signal time.
    allowed: optional bool (T,N) universe mask. Returns equity array (T,), trades list of dicts."""
    T, N = C.shape
    Cz = np.nan_to_num(C, nan=0.0)
    size = p.size if p.size > 0 else 1.0 / p.slots
    cash = 1.0
    units = np.zeros(N)
    entry_px = np.zeros(N)
    entry_bar = np.full(N, -10 ** 9)
    tp_lvl = np.full(N, np.inf)
    sl_lvl = np.full(N, -np.inf)
    hh = np.zeros(N)
    trail_frac = np.zeros(N)
    last_exit = np.full(N, -10 ** 9)
    held = np.zeros(N, dtype=bool)
    eq = np.ones(T)
    trades = []
    peak = 1.0
    halted = False
    for t in range(start, T):
        # ---------- exits on bar t for positions opened at bars < t
        if held.any():
            idx = np.where(held)[0]
            for i in idx:
                o, h, l, c = O[t, i], H[t, i], L[t, i], C[t, i]
                exit_px = None
                kind = None
                stop = sl_lvl[i]
                if trail_frac[i] > 0:
                    stop = max(stop, hh[i] * (1 - trail_frac[i]))
                if l <= stop and stop > -np.inf:
                    exit_px = min(stop, o) if o < stop else stop
                    kind = "sl"
                elif h >= tp_lvl[i]:
                    exit_px = max(tp_lvl[i], o) if o > tp_lvl[i] else tp_lvl[i]
                    kind = "tp"
                elif t - entry_bar[i] >= p.max_hold:
                    exit_px = c
                    kind = "time"
                elif exit_sig is not None and exit_sig[t, i]:
                    exit_px = c
                    kind = "sig"
                hh[i] = max(hh[i], h)
                if exit_px is not None:
                    fee = p.fee_exit_tp if kind == "tp" else p.fee_exit_other
                    slip = 0.0 if kind == "tp" else p.slip
                    px = exit_px * (1 - slip)
                    proceeds = units[i] * px * (1 - fee)
                    cash += proceeds
                    trades.append(dict(coin=i, entry_bar=int(entry_bar[i]), exit_bar=t, entry=entry_px[i], exit=px,
                                       ret=px / entry_px[i] - 1 - p.fee_entry - fee, kind=kind))
                    units[i] = 0.0
                    held[i] = False
                    last_exit[i] = t
        # ---------- mark equity at close
        e = cash + float(np.dot(units, Cz[t]))
        eq[t] = e
        peak = max(peak, e)
        if p.halt_dd > 0:
            if e >= peak:
                halted = False
            elif 1 - e / peak > p.halt_dd:
                halted = True
        locked = False
        if p.lock_gain > 0:
            lo = max(start, t - p.lock_window_bars)
            if e / eq[lo] - 1 >= p.lock_gain:
                locked = True
        # ---------- entries at close of bar t
        if not halted and not locked:
            free = p.slots - int(held.sum())
            if free > 0:
                cand = np.where(entry[t] & ~held & ((t - last_exit) > p.cooldown))[0]
                if allowed is not None:
                    cand = cand[allowed[t, cand]]
                if len(cand):
                    order = cand[np.argsort(-score[t, cand])][:free]
                    for i in order:
                        sz = size_arr[t, i] if size_arr is not None else size
                        alloc = min(sz * e, cash / (1 + p.fee_entry))
                        if alloc < 1e-4:
                            break
                        px = C[t, i] * (1 + p.slip)
                        u = alloc / px
                        cash -= alloc * (1 + p.fee_entry)
                        units[i] = u
                        entry_px[i] = px
                        entry_bar[i] = t
                        held[i] = True
                        hh[i] = C[t, i]
                        trail_frac[i] = trail_arr[t, i] if trail_arr is not None else p.trail
                        tpv = tp_arr[t, i] if tp_arr is not None else p.tp
                        slv = sl_arr[t, i] if sl_arr is not None else p.sl
                        tp_lvl[i] = px * (1 + tpv) if tpv > 0 else np.inf
                        sl_lvl[i] = px * (1 - slv) if slv > 0 else -np.inf
    return eq, trades
