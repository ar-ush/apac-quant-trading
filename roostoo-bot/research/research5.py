"""Research batch 5: momentum rotation with real position-level risk control (event sim on 1h bars)."""
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

import research1 as r1
from evsim import Params, run
from lib import *

C, H, L = r1.C, r1.H, r1.L
O = r1.P["open"][C.columns]
POOL = r1.POOL
VOLD = r1.hvol().values  # daily vol estimate
IDX = C.index
_G = {}


def gate(kind):
    b = C["BTC"]
    if kind == "none":
        return np.ones(len(C), bool)
    if kind == "e168_672":
        return (b.ewm(span=168, adjust=False).mean() > b.ewm(span=672, adjust=False).mean()).values
    if kind == "e96_384":
        return (b.ewm(span=96, adjust=False).mean() > b.ewm(span=384, adjust=False).mean()).values
    if kind == "e48_336":
        return (b.ewm(span=48, adjust=False).mean() > b.ewm(span=336, adjust=False).mean()).values


def job(a):
    lb, k, hyst, trail, size_mode, gk, rebal_h, rebal_phase = a
    ret = (C / C.shift(lb) - 1)
    score = ret.where(POOL)
    rk = score.rank(axis=1, ascending=False).values
    retv = ret.values
    g = gate(gk)[:, None]
    is_rebal = ((np.arange(len(C)) % rebal_h) == rebal_phase)[:, None]
    entry = (rk <= k) & (retv > 0) & g & is_rebal
    exit_sig = (~g) | (is_rebal & ((rk > k + hyst) | (retv <= 0) | np.isnan(rk)))
    exit_sig = np.broadcast_to(exit_sig, C.shape)
    if isinstance(trail, str):  # vol multiple
        m = float(trail)
        trail_arr = np.clip(m * VOLD, 0.04, 0.30)
    else:
        trail_arr = np.full(C.shape, trail)
    if size_mode == "eq":
        size_arr = np.full(C.shape, 1.0 / k)
    else:
        tv = float(size_mode)  # target daily vol contribution per position
        size_arr = np.clip(tv / np.where(np.isnan(VOLD), 1, VOLD), 0.02, 1.0 / k)
    p = Params(slots=k, max_hold=10 ** 9, cooldown=24 if rebal_h >= 24 else 6, fee_entry=0.0010, fee_exit_other=0.0010,
               fee_exit_tp=0.0005, slip=0.0002)
    eq, tr = run(O.values, H.values, L.values, C.values, entry, np.nan_to_num(retv, nan=-9), p, start=700,
                 exit_sig=exit_sig, trail_arr=trail_arr, size_arr=size_arr)
    eqs = pd.Series(eq, index=IDX)
    eqs = eqs[eqs.index >= pd.Timestamp("2023-03-01", tz="UTC")]
    eqs = eqs / eqs.iloc[0]
    win = rolling_windows(eqs)
    s = summarize_windows(win)
    yrs = [eqs[eqs.index.year == y].iloc[-1] / eqs[eqs.index.year == y].iloc[0] - 1 for y in [2023, 2024, 2025, 2026]]
    # last 45 days
    e45 = eqs.loc[eqs.index[-1] - pd.Timedelta(days=45):]
    t = pd.DataFrame(tr)
    return dict(lb=lb, k=k, hyst=hyst, trail=trail, size=size_mode, gate=gk, rb=rebal_h, n=len(t),
                avg_tr=t.ret.mean() if len(t) else 0, tot=eqs.iloc[-1] - 1, mdd=max_drawdown(eqs.values), mean14=s["mean_ret"],
                med14=s["med_ret"], p10=s["p10"], p90=s["p90"], pos=s["pos"], gt5=s["gt5"], mdd14=s["med_mdd"],
                p90mdd=s["p90_mdd"], mcomp=s["mean_comp"], y23=yrs[0], y24=yrs[1], y25=yrs[2], y26=yrs[3], minyr=min(yrs),
                l45=e45.iloc[-1] / e45.iloc[0] - 1)


if __name__ == "__main__":
    grid = []
    for k, hyst, trail, size, gk in itertools.product([3, 4, 5], [0, 2], [0.0, 0.10, 0.15, "2.0", "3.0"], ["eq", "0.008"],
                                                      ["e168_672", "e48_336"]):
        grid.append((336, k, hyst, trail, size, gk, 24, 23))
    print(len(grid), "configs", flush=True)
    with Pool(12) as pool:
        res = pool.map(job, grid, chunksize=2)
    df = pd.DataFrame(res)
    df.to_csv("out_research5.csv", index=False)
    pd.set_option("display.width", 280)
    cols = ["k", "hyst", "trail", "size", "gate", "n", "tot", "mdd", "mean14", "p10", "p90", "pos", "gt5", "mdd14", "p90mdd", "mcomp", "y23", "y24", "y25", "y26", "l45"]
    print(df.sort_values("mean14", ascending=False)[cols].head(25).round(3).to_string())
    print()
    for col in ["k", "hyst", "trail", "size", "gate"]:
        print(df.groupby(col)[["mean14", "p10", "mdd", "mdd14", "p90mdd", "mcomp", "pos", "tot"]].mean().round(3).to_string(), "\n")
