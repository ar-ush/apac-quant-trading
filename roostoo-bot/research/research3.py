"""Research batch 3: 15m burst-continuation family with event-driven sim (own implementation of the idea)."""
import itertools
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from evsim import Params, run
from lib import *

COINS = None


def load15():
    P = load_panel("15m", start="2023-06-01")
    return P


def prep(P):
    C = P["close"].copy()
    # drop coins with big holes
    C = C.loc[:, C.notna().mean() > 0.2]
    idx = C.index
    O, H, L, QV = P["open"][C.columns], P["high"][C.columns], P["low"][C.columns], P["qv"][C.columns]
    return C, O, H, L, QV


_G = {}


def init():
    P = load15()
    C, O, H, L, QV = prep(P)
    # liquidity pool: top-25 by trailing 30d median daily dollar volume, known at day start
    dq = QV.resample("1D").sum()
    med = dq.rolling(30, min_periods=20).median()
    hist = C.resample("1D").last().notna().cumsum()
    med = med.where(hist >= 60)
    mask = (med.rank(axis=1, ascending=False) <= 25).shift(1)
    allowed = mask.reindex(C.index, method="ffill").fillna(False).values.astype(bool)
    _G.update(C=C, O=O.values, H=H.values, L=L.values, Cv=C.values, allowed=allowed, idx=C.index)
    R1 = C.pct_change()
    _G["R1"] = R1
    _G["dv"] = (R1.rolling(96, min_periods=48).std() * np.sqrt(96)).values  # daily vol
    _G["feat"] = {}


def zfeat(k):
    f = _G["feat"]
    if k not in f:
        C = _G["C"]
        rk = C / C.shift(k) - 1
        sd = rk.rolling(96, min_periods=48).std()
        f[k] = (rk / sd).values
    return f[k]


def job(args):
    if not _G:
        init()
    k, Z, tp_m, sl_m, trail_m, slots, hold = args
    z = zfeat(k)
    dv = _G["dv"]
    ent = np.nan_to_num(z, nan=-9) >= Z
    ent &= np.nan_to_num(z, nan=-9) < 99
    ent &= ~np.isnan(dv)
    tp_arr = tp_m * dv if tp_m > 0 else None
    sl_arr = sl_m * dv if sl_m > 0 else None
    # trailing as fixed fraction of 1 daily vol * trail_m  -> use per-trade constant approx via median dv
    p = Params(slots=slots, tp=0.0, sl=0.0, trail=0.0, max_hold=hold, cooldown=4)
    eq, trades = run(_G["O"], _G["H"], _G["L"], _G["Cv"], ent, z, p, tp_arr=tp_arr, sl_arr=sl_arr, allowed=_G["allowed"], start=100)
    eqs = pd.Series(eq, index=_G["idx"])
    eqs = eqs[eqs.index >= pd.Timestamp("2023-09-01", tz="UTC")]
    eqs = eqs / eqs.iloc[0]
    eqh = eqs.resample("1h").last().dropna()
    win = rolling_windows(eqh)
    s = summarize_windows(win)
    tr = pd.DataFrame(trades)
    yrs = {}
    for y in [2023, 2024, 2025, 2026]:
        e = eqh[eqh.index.year == y]
        yrs[y] = e.iloc[-1] / e.iloc[0] - 1 if len(e) > 10 else np.nan
    return dict(k=k, Z=Z, tp=tp_m, sl=sl_m, slots=slots, hold=hold, n=len(tr),
                avg_tr=tr.ret.mean() if len(tr) else 0, win=(tr.ret > 0).mean() if len(tr) else 0,
                tot=eqh.iloc[-1] - 1, mdd=max_drawdown(eqh.values), med14=s["med_ret"], mean14=s["mean_ret"], pos=s["pos"],
                p10=s["p10"], p90=s["p90"], gt5=s["gt5"], mdd14=s["med_mdd"], comp=s["med_comp"],
                y23=yrs[2023], y24=yrs[2024], y25=yrs[2025], y26=yrs[2026])


if __name__ == "__main__":
    grid = []
    for k, Z in itertools.product([2, 3, 4], [2.5, 3.0]):
        for tp_m, sl_m in [(2, 0), (1, 0), (1, 1), (2, 1), (1.5, 1.5)]:
            grid.append((k, Z, tp_m, sl_m, 0, 3, 96))
    with Pool(8) as pool:
        res = pool.map(job, grid)
    df = pd.DataFrame(res)
    pd.set_option("display.width", 260)
    print(df.round(3).to_string())
    df.to_csv("out_research3.csv", index=False)
