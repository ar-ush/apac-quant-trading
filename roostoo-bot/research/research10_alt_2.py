"""research10_alt_2: TEST 2 - market-level regime features (funding, basis, F&G, DVOL, stablecoin growth) as overlays on the v1 BTC gate.
Walk-forward: every threshold is an EXPANDING-window percentile of the feature (min 90 days history; before that the overlay is inactive).
  veto_hi / veto_lo : if gate on and feature percentile > q (or < q) -> flat for that UTC day
  cap               : if gate OFF and 'capitulation' feature extreme -> hold the v1 selection at 50% exposure that day
Also writes a forward-return table: 14d forward alt-basket / BTC / v1 returns by walk-forward percentile bucket and gate state."""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from research10_alt_lib import *

_G = {}
RB = np.where(REBAL)[0]


def S():
    if _G:
        return _G
    fng, st, dv = macro_hourly()
    F7, B3 = trail(FUND, 168).where(POOL), trail(PREM, 72).where(POOL)
    feats = {
        "mf_med7": F7.median(axis=1),
        "mf_btc7": trail(FUND, 168)["BTC"],
        "mb_med3": B3.median(axis=1),
        "mb_btc3": trail(PREM, 72)["BTC"],
        "fng": fng,
        "dvol": dv,
        "dvol_chg7": dv / dv.shift(168) - 1,
        "stab_g30": st / st.shift(720) - 1,
    }
    pct = {}
    for k, s in feats.items():
        d = s.iloc[RB]
        p = day_expanding_pct(d)
        pct[k] = p.reindex(IDX)  # only at rebal rows
    _G["feats"], _G["pct"] = feats, pct
    # alt-basket forward 14d from each rebal bar (no costs): equal weight of pool alts
    alts = [c for c in COLS if c != "BTC"]
    fwd = (C.shift(-336) / C - 1)[alts].where(POOL[alts])
    _G["fwd_alt"] = fwd.mean(axis=1)
    _G["fwd_btc"] = C["BTC"].shift(-336) / C["BTC"] - 1
    return _G


def day_flag(cond_at_rebal):
    """cond (bool Series on IDX, only meaningful at rebal rows) -> bool array valid for the whole UTC day."""
    x = pd.Series(np.where(REBAL, cond_at_rebal.fillna(False).values, np.nan), index=IDX).ffill().fillna(0).astype(bool)
    return x.values


def build(spec):
    g = S()
    fam = spec[0]
    if fam == "v1":
        ML, WL = v1_build()
        return ML, WL, np.where(GATE, 1, 0), None
    f, q = spec[1], spec[2]
    pc = g["pct"][f]
    if fam == "veto_hi":
        veto = day_flag(pc > q)
        gate = GATE & ~veto
        ML, WL = v1_build(gate=gate)
        return ML, WL, np.where(gate, 1, 0), None
    if fam == "veto_lo":
        veto = day_flag(pc < q)
        gate = GATE & ~veto
        ML, WL = v1_build(gate=gate)
        return ML, WL, np.where(gate, 1, 0), None
    if fam == "veto_abs":   # a-priori absolute F&G thresholds (no fitting)
        veto = day_flag(g["feats"]["fng"] > q)
        gate = GATE & ~veto
        ML, WL = v1_build(gate=gate)
        return ML, WL, np.where(gate, 1, 0), None
    if fam in ("cap", "cap_abs"):
        if fam == "cap_abs":
            cond = g["feats"]["fng"] < q
        elif f in ("dvol", "dvol_chg7"):
            cond = pc > 1 - q
        else:
            cond = pc < q
        allow = day_flag(cond) & ~GATE
        eff = GATE | allow
        ML, WL = v1_build(gate=eff)
        scale = np.where(allow, 0.5, 1.0)
        return ML, WL, np.where(eff, 1, 0), scale
    raise ValueError(spec)


def job(spec):
    ML, WL, reg, scale = build(spec)
    label = " ".join(str(x) for x in spec)
    gate = np.where(reg == 1, 1, 0)
    r = score_cfg(label, ML, WL, regime=reg, scale=scale, extra=dict(fam=spec[0], days_off=float(((reg == 0) & GATE & REBAL & (IDX >= START)).sum()),
                                                                     days_on_extra=float(((reg == 1) & ~GATE & REBAL & (IDX >= START)).sum())))
    r.pop("_eq")
    return r


def grid():
    G = [("v1",)]
    feats = ["mf_med7", "mf_btc7", "mb_med3", "mb_btc3", "fng", "dvol", "dvol_chg7", "stab_g30"]
    G += [("veto_hi", f, q) for f in feats for q in [0.8, 0.9, 0.95]]
    G += [("veto_lo", f, q) for f in feats for q in [0.05, 0.1, 0.2]]
    G += [("veto_abs", "fng", q) for q in [75, 85]]
    G += [("cap", f, q) for f in ["mf_med7", "mf_btc7", "mb_med3", "mb_btc3", "fng", "dvol", "dvol_chg7"] for q in [0.05, 0.1, 0.2]]
    G += [("cap_abs", "fng", q) for q in [15, 25]]
    return G


def nw_t(x, L=13):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan
    d = x - x.mean()
    v = d @ d / n
    for l in range(1, L + 1):
        v += 2 * (1 - l / (L + 1)) * (d[l:] @ d[:-l]) / n
    return x.mean() / np.sqrt(v / n)


def fwd_table():
    g = S()
    ref = v1_ref()
    v1f = (ref.shift(-336) / ref - 1).reindex(IDX)
    rows = []
    sel = REBAL & (IDX >= START) & (IDX <= IDX[-1] - pd.Timedelta(hours=336))
    for f, pc in g["pct"].items():
        for gs, gm in [("gate_on", GATE), ("gate_off", ~GATE), ("all", np.ones(T, bool))]:
            for bn, lo, hi in [("lo<.2", 0, .2), ("mid", .2, .8), ("hi>.8", .8, 1.01)]:
                m = sel & gm & (pc.values >= lo) & (pc.values < hi)
                n = int(m.sum())
                if n < 20:
                    continue
                a = g["fwd_alt"].values[m]; b = g["fwd_btc"].values[m]; v = v1f.values[m]
                rows.append(dict(feat=f, gate=gs, bucket=bn, n_days=n, alt14=np.nanmean(a), alt14_t=nw_t(a), btc14=np.nanmean(b),
                                 v1_14=np.nanmean(v), v1_t=nw_t(v)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    G = grid()
    print(len(G), "configs", flush=True)
    with Pool(14) as p:
        res = p.map(job, G, chunksize=1)
    df = pd.DataFrame(res)
    df.to_csv("out_research10_alt_2.csv", index=False)
    cols = ["label", "days_off", "days_on_extra", "mean14", "med14", "p10", "p90", "pos", "gt20", "mcomp", "tot", "mdd", "y2023", "y2024", "y2025", "y2026", "t_vs_v1"]
    print(fmt(df, cols))
    ft = fwd_table()
    ft.to_csv("out_research10_alt_2_fwd.csv", index=False)
    print(fmt(ft))
