"""research10_lit_lib: shared helpers for the literature-scan tests (wraps research9_alt_lib engine).
Baseline v1 = BTC EMA168/672 gate, 336h return top-3 eq-weight, hysteresis 2, daily 00:00 UTC rebalance, pool top-25 liquid,
cost 0.12%/side, trades at close of signal bar. Windows = rolling 14d, step 1d."""
import numpy as np
import pandas as pd
from research9_alt_lib import *   # C,H,L,QV,O,TBQ,POOL,R1,IDX,COLS,T,N,GATE,REBAL,BTC,HOUR,membership,eqw,run,evaluate,v1_inputs
from lib import rolling_windows, summarize_windows, max_drawdown

YEARS = [2023, 2024, 2025, 2026]


def stats(eq, label="", extra=None):
    """Full evaluation incl. P(r14>20%), per-year totals and per-year mean-14d."""
    w = rolling_windows(eq)
    s = summarize_windows(w)
    out = dict(label=label, tot=eq.iloc[-1] - 1, mdd=max_drawdown(eq.values), mean14=s["mean_ret"], med14=s["med_ret"],
               p10=s["p10"], p90=s["p90"], pos=s["pos"], p20=(w.ret > 0.20).mean(), mcomp=s["mean_comp"], medcomp=s["med_comp"],
               mdd14=s["med_mdd"])
    for y in YEARS:
        e = eq[eq.index.year == y]
        out[f"y{y}"] = e.iloc[-1] / e.iloc[0] - 1
    for y in YEARS:
        out[f"m14_{y}"] = w.ret[w.index.year == y].mean()
    out["nyr_ok"] = None
    if extra:
        out.update(extra)
    return out


def v1_eq(k=3, lb=336, hyst=2, gate=None, cost=0.0012):
    ML, WL, ret = v1_inputs(lb, k, hyst, gate)
    reg = None if gate is None else np.where(gate, 1, 0)
    eq, info = run(ML, WL, regime=reg, cost_l=cost)
    return eq, info


def run_sel(score, k=3, hyst=2, gate=None, thr=0.0, W=None, cost=0.0012, extra_gate=None):
    """Top-k by `score` (DataFrame, NaN outside pool handled by membership), gate = bool array (default BTC EMA gate)."""
    g = GATE if gate is None else gate
    if extra_gate is not None:
        g = g & extra_gate
    ML = membership(score, k, hyst, thr, +1, reset=~g)
    WLw = eqw(ML, k) if W is None else W
    eq, info = run(ML, WLw, regime=np.where(g, 1, 0), cost_l=cost)
    return eq, info
