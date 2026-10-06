"""research10_alt_4: TEST 4 - sector / theme momentum on top of v1 (BTC gate, 336h momentum>0 eligibility, k=3, hyst 2, daily).
Themes assigned by hand (below). Sector score = mean or median trailing return of the sector's liquid-pool members (>=2 members required).
 (a) 'sec_best': pick top-S sectors by sector momentum, then best positive-momentum coin(s) within (S=1: top-3 of best sector; S=2: top-2 of
     best + top-1 of 2nd; S=3: top-1 of each of the 3 best sectors), hysteresis as in v1.
 (b) 'tilt': coin score = rank(mom) + w*rank(sector momentum)  (w<0 favours the coldest sectors);  'resid': blend with rank(mom - sector mean)."""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from research10_alt_lib import *

THEMES = {
    "L1": ["ETH", "SOL", "ADA", "AVAX", "NEAR", "SUI", "APT", "SEI", "TRX", "ICP", "HBAR", "BNB", "XPL"],
    "LEGACY": ["BTC", "LTC", "XRP", "XLM", "ZEC", "ZEN"],
    "DEFI": ["AAVE", "UNI", "CRV", "PENDLE", "CAKE", "ENA", "ONDO", "ASTER"],
    "AI": ["FET", "TAO", "WLD", "VIRTUAL"],
    "MEME": ["DOGE", "WIF", "FLOKI", "TRUMP", "PENGU", "PUMP"],
    "INFRA": ["LINK", "FIL", "POL", "OMNI"],
}
COIN_SEC = {c: s for s, cs in THEMES.items() for c in cs}
assert set(COIN_SEC) == set(COLS), set(COLS) ^ set(COIN_SEC)
_G = {}


def sector_scores(lb, stat):
    """DataFrame T x N: sector score of each coin's sector (NaN if <2 pool members with data)."""
    mom = C / C.shift(lb) - 1
    out = pd.DataFrame(np.nan, index=IDX, columns=COLS)
    secs = {}
    for s, cs in THEMES.items():
        m = mom[cs].where(POOL[cs])
        v = m.mean(axis=1) if stat == "mean" else m.median(axis=1)
        v = v.where(m.notna().sum(axis=1) >= 2)
        secs[s] = v
        for c in cs:
            out[c] = v
    return out, pd.DataFrame(secs)


def sec_best_score(S_, lb, stat):
    """priority score (higher better) for rebal rows; others NaN."""
    _, secs = sector_scores(lb, stat)
    mom = MOM.where(POOL & (MOM > 0))
    score = np.full((T, N), np.nan)
    sec_names = list(THEMES)
    members = {s: [COLS.index(c) for c in cs] for s, cs in THEMES.items()}
    mv, sv = mom.values, secs.values
    for t in np.where(REBAL & (IDX >= START))[0]:
        row = sv[t]
        # sectors with >=1 eligible coin only
        avail = [i for i, s in enumerate(sec_names) if not np.isnan(row[i]) and np.isfinite(mv[t, members[s]]).any()]
        avail.sort(key=lambda i: -row[i])
        pr = {}
        for rank, i in enumerate(avail, 1):
            js = [j for j in members[sec_names[i]] if np.isfinite(mv[t, j])]
            js.sort(key=lambda j: -mv[t, j])
            for r_in, j in enumerate(js, 1):
                if S_ == 1:
                    pr[j] = rank * 100 + r_in
                elif S_ == 3:
                    pr[j] = r_in * 100 + rank
                else:
                    pr[j] = {(1, 1): 1, (1, 2): 2, (2, 1): 3}.get((rank, r_in), 1000 + rank * 10 + r_in)
        for j, p in pr.items():
            score[t, j] = -p
    return pd.DataFrame(score, index=IDX, columns=COLS)


def build(spec):
    fam = spec[0]
    if fam == "v1":
        return v1_build()
    if fam == "sec_best":
        _, S_, stat, lb = spec
        sc = sec_best_score(S_, lb, stat)
        return v1_build(score=sc)
    if fam == "tilt":
        _, w, lb = spec
        ss, _ = sector_scores(lb, "mean")
        ps = ss.where(POOL).rank(axis=1, pct=True).fillna(0.5)
        return v1_build(score=PR(MOM) + w * ps)
    if fam == "resid":
        _, w = spec
        ss, _ = sector_scores(336, "mean")
        res = MOM - ss
        return v1_build(score=(1 - w) * PR(MOM) + w * PR(res).fillna(0.5))
    raise ValueError(spec)


def job(spec):
    ML, WL = build(spec)
    label = " ".join(str(x) for x in spec)
    r = score_cfg(label, ML, WL, extra=dict(fam=spec[0], changed=frac_changed(ML) if spec[0] != "v1" else 0.0))
    r.pop("_eq")
    return r


def grid():
    G = [("v1",)]
    G += [("sec_best", S_, st, lb) for S_ in [1, 2, 3] for st in ["mean", "median"] for lb in [168, 336, 672]]
    G += [("tilt", w, lb) for w in [-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0] for lb in [168, 336]]
    G += [("resid", w) for w in [0.5, 1.0]]
    return G


if __name__ == "__main__":
    G = grid()
    print(len(G), "configs", flush=True)
    with Pool(14) as p:
        res = p.map(job, G, chunksize=1)
    df = pd.DataFrame(res)
    df.to_csv("out_research10_alt_4.csv", index=False)
    cols = ["label", "changed", "mean14", "med14", "p10", "p90", "pos", "gt20", "mcomp", "tot", "mdd", "y2023", "y2024", "y2025", "y2026", "t_vs_v1"]
    print(fmt(df, cols))
