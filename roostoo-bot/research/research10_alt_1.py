"""research10_alt_1: TEST 1 - funding / perp-spot basis as cross-sectional filter / tilt on v1 top-3 momentum selection.
Configs are one add-on at a time on the unchanged v1 (BTC EMA gate, 336h momentum, k=3, hyst 2, daily 00:00, 0.12%/side)."""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from research10_alt_lib import *

_G = {}


def S():
    if _G:
        return _G
    F = {"F3": trail(FUND, 72), "F7": trail(FUND, 168), "B3": trail(PREM, 72), "B7": trail(PREM, 168)}
    _G["F"] = F
    _G["pr"] = {k: PR(v) for k, v in F.items()}  # NaN where no perp history / not in pool
    return _G


def build(spec):
    g = S()
    fam = spec[0]
    if fam == "v1":
        return v1_build()
    sig = spec[1]
    p = g["pr"][sig]
    if fam == "skip_hi":      # drop crowded (high funding/basis) names
        return v1_build(elig=~(p > spec[2]).fillna(False))
    if fam == "skip_lo":      # drop the LOWEST-funding names (keeps trend-confirmed crowding)
        return v1_build(elig=~(p < 1 - spec[2]).fillna(False))
    if fam == "tilt":         # blend momentum rank with w * (low-funding rank); w<0 favours HIGH funding
        w = spec[2]
        sc = PR(MOM) + w * (1 - p).fillna(0.5)
        return v1_build(score=sc)
    if fam == "topm":         # among top-m momentum choose by funding: dir=-1 lowest, +1 highest
        _, _, m, d = spec
        topm = MOM.where(POOL).rank(axis=1, ascending=False) <= m
        sc = (-d) * p.fillna(0.5 if d < 0 else 0.5)
        return v1_build(score=(-d) * p.fillna(0.5) + 1.0, elig=topm)
    if fam == "downw":        # halve the entry weight of crowded names (cash remainder)
        ML, WL = v1_build()
        crowd = (p > spec[2]).fillna(False).values
        return ML, np.where(crowd, WL * 0.5, WL)
    if fam == "abs":          # absolute (a priori) thresholds on trailing level
        th = spec[2]
        f = g["F"][sig]
        return v1_build(elig=~(f > th).fillna(False))
    raise ValueError(spec)


def job(spec):
    ML, WL = build(spec)
    label = " ".join(str(x) for x in spec)
    r = score_cfg(label, ML, WL, extra=dict(fam=spec[0], changed=frac_changed(ML) if spec[0] != "v1" else 0.0))
    r.pop("_eq")
    return r


def grid():
    G = [("v1",)]
    sigs = ["F3", "F7", "B3", "B7"]
    G += [("skip_hi", s, q) for s in sigs for q in [0.7, 0.8, 0.9]]
    G += [("skip_lo", s, q) for s in sigs for q in [0.7, 0.8, 0.9]]
    G += [("tilt", s, w) for s in sigs for w in [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]]
    G += [("topm", s, m, d) for s in sigs for m in [5, 8] for d in [-1, 1]]
    G += [("downw", s, q) for s in sigs for q in [0.8, 0.9]]
    G += [("abs", "F3", th) for th in [0.0002, 0.0003, 0.0005]] + [("abs", "F7", th) for th in [0.0002, 0.0003]]
    G += [("abs", "B3", th) for th in [0.001, 0.002, 0.004]]
    return G


if __name__ == "__main__":
    G = grid()
    print(len(G), "configs", flush=True)
    with Pool(14) as p:
        res = p.map(job, G, chunksize=1)
    df = pd.DataFrame(res)
    df.to_csv("out_research10_alt_1.csv", index=False)
    cols = ["label", "changed", "mean14", "med14", "p10", "p90", "pos", "gt20", "mcomp", "tot", "mdd", "y2023", "y2024", "y2025", "y2026", "t_vs_v1"]
    print(fmt(df, cols))
