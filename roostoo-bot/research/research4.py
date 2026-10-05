"""Research batch 4: parameter plateau scan of cross-sectional momentum rotation (long-only, BTC regime gate)."""
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

import research1 as r1
from lib import *

C, POOL = r1.C, r1.POOL
VOL = r1.hvol()
_cache = {}


def regime(kind):
    if kind in _cache:
        return _cache[kind]
    b = C["BTC"]
    if kind == "none":
        on = pd.Series(1.0, index=C.index)
    elif kind == "ema168_672":
        on = (b.ewm(span=168, adjust=False).mean() > b.ewm(span=672, adjust=False).mean()).astype(float)
    elif kind == "ema96_384":
        on = (b.ewm(span=96, adjust=False).mean() > b.ewm(span=384, adjust=False).mean()).astype(float)
    elif kind == "ema48_200":
        on = (b.ewm(span=48, adjust=False).mean() > b.ewm(span=200, adjust=False).mean()).astype(float)
    elif kind == "px_ema336":
        on = (b > b.ewm(span=336, adjust=False).mean()).astype(float)
    _cache[kind] = on
    return on


def build(lb, k, rebal, reg, rank, thr=0.0):
    ret = C / C.shift(lb) - 1
    score = ret / (VOL * np.sqrt(lb / 24)) if rank == "z" else ret
    score = score.where(POOL)
    rk = score.rank(axis=1, ascending=False)
    sel = ((rk <= k) & (ret > thr)).astype(float)
    sel = sel.mul(regime(reg), axis=0)
    w = sel.div(sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    # per-name cap 1/k so partially filled books stay partially in cash
    w = w.clip(upper=1.0 / k) if True else w
    return r1.hold_every(w, rebal)


def job(a):
    lb, k, rebal, reg, rank = a
    w = build(lb, k, rebal, reg, rank)
    eq, info = simulate(C, w, cost=0.0012)
    win = rolling_windows(eq)
    s = summarize_windows(win)
    yrs = [eq[eq.index.year == y].iloc[-1] / eq[eq.index.year == y].iloc[0] - 1 for y in [2023, 2024, 2025, 2026]]
    return dict(lb=lb, k=k, rebal=rebal, reg=reg, rank=rank, tot=eq.iloc[-1] - 1, mdd=max_drawdown(eq.values),
                turn=info["turnover_per_day"], gross=info["avg_gross"], mean14=s["mean_ret"], med14=s["med_ret"], p10=s["p10"],
                p90=s["p90"], pos=s["pos"], gt5=s["gt5"], mdd14=s["med_mdd"], mcomp=s["mean_comp"],
                y23=yrs[0], y24=yrs[1], y25=yrs[2], y26=yrs[3], minyr=min(yrs))


if __name__ == "__main__":
    grid = list(itertools.product([72, 120, 168, 240, 336, 504, 720], [2, 3, 4, 5], [12, 24, 48, 96],
                                  ["none", "ema168_672", "ema96_384", "px_ema336"], ["raw", "z"]))
    print(len(grid), "configs")
    with Pool(12) as pool:
        res = pool.map(job, grid, chunksize=8)
    df = pd.DataFrame(res)
    df.to_csv("out_research4.csv", index=False)
    pd.set_option("display.width", 260)
    print("TOP by mean14 among minyr>-0.35:")
    print(df[df.minyr > -0.35].sort_values("mean14", ascending=False).head(20).round(3).to_string())
    print("\nTOP by mcomp:")
    print(df.sort_values("mcomp", ascending=False).head(15).round(3).to_string())
    print("\nmarginals (mean14 / mcomp / mdd):")
    for col in ["lb", "k", "rebal", "reg", "rank"]:
        print(df.groupby(col)[["mean14", "mcomp", "mdd", "tot", "pos", "p10"]].mean().round(3).to_string(), "\n")
