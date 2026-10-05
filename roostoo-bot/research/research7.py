"""Research batch 7: trend-quality rankings (regression t-stat, Clenow slope*R2), breadth gates, vs raw 14d momentum."""
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

import research1 as r1
import research4 as r4
from lib import *

C, POOL = r1.C, r1.POOL
LOGC = np.log(C)


def rolling_trend_stats(win):
    """Rolling OLS of log price on time over `win` bars: returns slope (per bar), t-stat, R2."""
    x = np.arange(win, dtype=float)
    xm = x.mean()
    xx = ((x - xm) ** 2).sum()
    y = LOGC
    ybar = y.rolling(win).mean()
    # cov(x,y) via rolling sum of x*y
    # sum_i (x_i - xm)*y_i : use convolution via rolling apply on array (vectorised with stride tricks)
    arr = y.values
    n = len(arr)
    out_slope = np.full(arr.shape, np.nan)
    out_t = np.full(arr.shape, np.nan)
    out_r2 = np.full(arr.shape, np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    for j in range(arr.shape[1]):
        col = arr[:, j]
        ok = ~np.isnan(col)
        if ok.sum() < win + 10:
            continue
        sw = sliding_window_view(col, win)  # (n-win+1, win)
        valid = ~np.isnan(sw).any(axis=1)
        swz = np.where(valid[:, None], sw, 0.0)
        ym = swz.mean(axis=1)
        sxy = ((x - xm)[None, :] * (swz - ym[:, None])).sum(axis=1)
        syy = ((swz - ym[:, None]) ** 2).sum(axis=1)
        slope = sxy / xx
        r2 = np.where(syy > 0, sxy ** 2 / (xx * syy), 0)
        resid_var = np.where(syy > 0, (syy - slope * sxy) / (win - 2), np.nan)
        se = np.sqrt(resid_var / xx)
        t = np.where(se > 0, slope / se, np.nan)
        idx = np.arange(win - 1, n)
        out_slope[idx, j] = np.where(valid, slope, np.nan)
        out_t[idx, j] = np.where(valid, t, np.nan)
        out_r2[idx, j] = np.where(valid, r2, np.nan)
    mk = lambda a: pd.DataFrame(a, index=C.index, columns=C.columns)
    return mk(out_slope), mk(out_t), mk(out_r2)


_cache = {}


def score_frame(kind, lb):
    key = (kind, lb)
    if key in _cache:
        return _cache[key]
    ret = C / C.shift(lb) - 1
    if kind == "raw":
        s = ret
    else:
        slope, t, r2 = rolling_trend_stats(lb)
        if kind == "tstat":
            s = t
        elif kind == "clenow":
            s = (np.exp(slope * 24 * 365) - 1) * r2
        elif kind == "slope":
            s = slope
        elif kind == "ret_x_r2":
            s = ret * r2
    _cache[key] = (s, ret)
    return s, ret


def breadth_gate(th):
    r14 = C / C.shift(336) - 1
    br = (r14.where(POOL) > 0).sum(axis=1) / POOL.sum(axis=1).replace(0, np.nan)
    return (br > th).astype(float)


def build(kind, lb, k, rebal, gate_kind):
    s, ret = score_frame(kind, lb)
    s = s.where(POOL)
    rk = s.rank(axis=1, ascending=False)
    sel = ((rk <= k) & (ret > 0)).astype(float)
    if gate_kind == "btc":
        on = r4.regime("ema168_672")
    elif gate_kind == "btc_and_breadth":
        on = r4.regime("ema168_672") * breadth_gate(0.4)
    elif gate_kind == "breadth":
        on = breadth_gate(0.5)
    else:
        on = r4.regime("none")
    sel = sel.mul(on, axis=0)
    w = sel.div(sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).clip(upper=1.0 / k)
    return r1.hold_every(w, rebal)


def job(a):
    kind, lb, k, rebal, gk = a
    w = build(kind, lb, k, rebal, gk)
    eq, info = simulate(C, w, cost=0.0012)
    s = summarize_windows(rolling_windows(eq))
    yrs = [eq[eq.index.year == y].iloc[-1] / eq[eq.index.year == y].iloc[0] - 1 for y in [2023, 2024, 2025, 2026]]
    return dict(kind=kind, lb=lb, k=k, rebal=rebal, gate=gk, tot=eq.iloc[-1] - 1, mdd=max_drawdown(eq.values), turn=info["turnover_per_day"],
                mean14=s["mean_ret"], p10=s["p10"], p90=s["p90"], pos=s["pos"], gt5=s["gt5"], mdd14=s["med_mdd"], mcomp=s["mean_comp"],
                y23=yrs[0], y24=yrs[1], y25=yrs[2], y26=yrs[3])


if __name__ == "__main__":
    grid = list(itertools.product(["raw", "tstat", "clenow", "ret_x_r2", "slope"], [168, 336, 504], [3], [24],
                                  ["btc", "btc_and_breadth", "breadth", "none"]))
    for kind in ["raw"]:
        pass
    # warm caches in the parent so workers (fork not available on Windows) recompute -> run sequentially instead
    res = []
    for g in grid:
        res.append(job(g))
        print(g, round(res[-1]["mean14"], 4), round(res[-1]["tot"], 2), flush=True)
    df = pd.DataFrame(res)
    df.to_csv("out_research7.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.sort_values("mean14", ascending=False).round(3).to_string())
