"""research9_alt_A: short sleeve in the BTC-gate-off regime, combined with v1 (long sleeve in gate-on)."""
import itertools
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from research9_alt_lib import *

VOLD = (R1.rolling(168, min_periods=84).std() * np.sqrt(24))
RALLY = ((C / C.shift(24) - 1 > 0.10) | (C / C.shift(72) - 1 > 0.20))
_G = {}


def base():
    if not _G:
        ML, WL, _ = v1_inputs()
        _G["ML"], _G["WL"] = ML, WL
        _G["REG_COMB"] = np.where(GATE, 1, -1)
        _G["REG_ONLY"] = np.where(GATE, 0, -1)
    return _G


def short_inputs(lb, k, gross, size, filt, hyst=0):
    ret = C / C.shift(lb) - 1
    pool = POOL & ~RALLY if filt == "rally" else POOL
    MS = membership(ret, k, hyst, 0.0, -1, pool=pool, reset=GATE)
    if size == "eq":
        WS = np.where(MS, gross / k, 0.0)
    else:
        inv = np.where(MS, 1.0 / VOLD.values, 0.0)
        inv = np.nan_to_num(inv, nan=0.0, posinf=0.0)
        s = inv.sum(axis=1, keepdims=True)
        WS = np.where(s > 0, inv / np.maximum(s, 1e-12), 0.0) * gross
        WS = np.minimum(WS, 1.5 * gross / k)
    return MS, WS


def job(a):
    kind = a[0]
    G = base()
    if kind == "sel":
        _, lb, k, gross, stop, size, filt, borrow = a
        MS, WS = short_inputs(lb, k, gross, size, filt)
        label = f"sel lb{lb} k{k} g{gross} stop{stop} {size} {filt} b{borrow}"
    elif kind == "btc":
        _, gross, stop, borrow, name = a
        MS = np.zeros((T, N), bool); WS = np.zeros((T, N))
        cols = ["BTC"] if name == "BTC" else (["BTC", "ETH"] if name == "BTCETH" else None)
        if cols is None:  # whole pool equal weight
            MS = POOL.values.copy()
            WS = np.where(MS, gross / np.maximum(MS.sum(axis=1, keepdims=True), 1), 0.0)
        else:
            for c in cols:
                j = COLS.index(c)
                MS[:, j] = True; WS[:, j] = gross / len(cols)
        label = f"{name} short g{gross} stop{stop} b{borrow}"
    comb, _ = run(ML=G["ML"], WL=G["WL"], MS=MS, WS=WS, regime=G["REG_COMB"], stop_s=stop, borrow_day=borrow)
    only, inf = run(MS=MS, WS=WS, regime=G["REG_ONLY"], stop_s=stop, borrow_day=borrow)
    r1_ = evaluate(comb, "COMB " + label, dict(kind=kind, nstop=inf["nstop"], gross_s=inf["gross"]))
    r2_ = evaluate(only, "ONLY " + label, dict(kind=kind, nstop=inf["nstop"], gross_s=inf["gross"]))
    return [r1_, r2_]


if __name__ == "__main__":
    grid = []
    for lb, k, gross, stop, size, filt in itertools.product([168, 336, 672], [3, 5, 8], [0.5, 1.0], [0, 0.15, 0.30],
                                                            ["eq", "vol"], ["none", "rally"]):
        grid.append(("sel", lb, k, gross, stop, size, filt, 0.0002))
    for gross, stop, name in itertools.product([0.5, 1.0], [0, 0.15, 0.30], ["BTC", "BTCETH", "POOL"]):
        grid.append(("btc", gross, stop, 0.0002, name))
    print(len(grid), "configs", flush=True)
    with Pool(8) as p:
        res = p.map(job, grid, chunksize=2)
    rows = [r for rr in res for r in rr]
    df = pd.DataFrame(rows)
    df.to_csv("out_research9_alt_A.csv", index=False)
    print("saved", len(df))
