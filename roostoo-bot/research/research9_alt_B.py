"""research9_alt_B: alternative market-derived long-sleeve signals inside the v1 BTC gate (long-only, top-3 equal weight,
hysteresis 2, daily rebalance at 00:00 UTC, 0.12%/side). Families B1..B7 + information-coefficient table for tbq flow."""
import itertools
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from research9_alt_lib import *

K = 3
_G = {}
NEG = -1e9


def S():
    if _G:
        return _G
    g = _G
    g["ret"] = {h: C / C.shift(h) - 1 for h in [24, 48, 72, 168, 336, 672]}
    g["pr"] = lambda df: df.where(POOL).rank(axis=1, pct=True)
    g["vol168"] = R1.rolling(168, min_periods=84).std()
    imb = 2 * TBQ - QV
    g["ofi"] = {h: imb.rolling(h, min_periods=h // 2).sum() / QV.rolling(h, min_periods=h // 2).sum() for h in [24, 72, 168]}
    # beta vs BTC (trailing 720h) and residual return
    rb = R1["BTC"]
    cov = R1.rolling(720, min_periods=360).cov(rb)
    g["beta"] = cov.div(rb.rolling(720, min_periods=360).var(), axis=0)
    g["ret_btc336"] = g["ret"][336]["BTC"]
    g["rollmax"] = C.rolling(336, min_periods=168).max()
    g["gate"] = GATE
    return g


def long_only(score, elig=None, k=K, hyst=2, thr=NEG, wmode="eq", gate=None):
    g = S()
    gate = g["gate"] if gate is None else gate
    pool = POOL if elig is None else (POOL & elig)
    ML = membership(score, k, hyst, thr, +1, pool=pool, reset=~gate)
    WL = eqw(ML, k)
    return ML, WL


def finish(ML, WL, label, family, params, ref=None, **kw):
    g = S()
    reg = np.where(g["gate"], 1, 0)
    eq, info = run(ML=ML, WL=WL, regime=reg, **kw)
    ex = dict(family=family, params=params, turn=info["turn"], gross=info["gross"])
    r = evaluate(eq, label, ex)
    r["_eq"] = eq
    return r


def build(spec):
    g = S()
    fam = spec[0]
    ret = g["ret"]
    mom = ret[336]
    pr = g["pr"]
    if fam == "v1":
        return long_only(mom, elig=mom > 0, thr=0.0)
    if fam == "rev":  # B1: momentum rank + w * reversal rank over h hours
        _, h, w = spec
        sc = pr(mom) + w * pr(-ret[h])
        return long_only(sc, elig=mom > 0)
    if fam == "dip":  # B1b: top-m momentum, buy the weakest recent h-hour
        _, h, m = spec
        topm = mom.where(POOL).rank(axis=1, ascending=False) <= m
        return long_only(-ret[h], elig=(mom > 0) & topm)
    if fam == "lowvol":  # B2a: among top-m momentum choose lowest vol
        _, m = spec
        topm = mom.where(POOL).rank(axis=1, ascending=False) <= m
        return long_only(-g["vol168"], elig=(mom > 0) & topm)
    if fam == "lowvol_pure":
        return long_only(-g["vol168"], elig=mom > 0)
    if fam == "near_high":  # B2b: proximity to 336h high (breakout), momentum positive
        _, win = spec
        rm = C.rolling(win, min_periods=win // 2).max()
        return long_only(C / rm, elig=mom > 0)
    if fam == "highvol":
        _, m = spec
        topm = mom.where(POOL).rank(axis=1, ascending=False) <= m
        return long_only(g["vol168"], elig=(mom > 0) & topm)
    if fam == "ofi":  # B3: blend momentum rank with taker-flow imbalance rank
        _, h, w = spec
        sc = pr(mom) + w * pr(g["ofi"][h])
        return long_only(sc, elig=mom > 0)
    if fam == "ofi_sel":
        _, h, m = spec
        topm = mom.where(POOL).rank(axis=1, ascending=False) <= m
        return long_only(g["ofi"][h], elig=(mom > 0) & topm)
    if fam == "ofi_pure":
        _, h = spec
        return long_only(g["ofi"][h], elig=mom > 0)
    if fam == "resid":  # B4: residual (beta-neutral) momentum
        _, w = spec
        res = mom - g["beta"].mul(g["ret_btc336"], axis=0)
        sc = (1 - w) * pr(mom) + w * pr(res)
        return long_only(sc, elig=mom > 0)
    if fam == "multi":  # B5: multi-horizon blend
        _, hs = spec
        sc = sum(pr(ret[h]) for h in hs) / len(hs)
        return long_only(sc, elig=mom > 0)
    if fam == "tsbasket":  # B5b: hold every pool alt with positive mom (equal weight, cap 1/4)
        _, hmin, cap = spec
        el = (mom > 0) & (ret[hmin] > 0) if hmin != 336 else mom > 0
        pool = POOL & el
        M = pool.values & ~np.isnan(mom.values)
        M = M & REBAL[:, None]
        # persistence: membership only evaluated at rebal bars; non-rebal rows unused
        cnt = np.maximum(M.sum(axis=1, keepdims=True), 1)
        W = np.minimum(0.98 / cnt, cap) * M
        return M, W
    if fam == "dom":  # B7: BTC dominance rotation
        _, mode, th = spec
        alts = mom.drop(columns=["BTC"]).where(POOL.drop(columns=["BTC"]))
        dom = (mom["BTC"] - alts.median(axis=1)).values  # >0: BTC outperforming alts
        ML0, WL0 = long_only(mom, elig=mom > 0, thr=0.0)
        Mb = np.zeros((T, N), bool); Mb[:, COLS.index("BTC")] = True
        Wb = np.where(Mb, 0.98, 0.0)
        up = (dom > th)[:, None]
        if mode == "btc_when_up":
            return np.where(up, Mb, ML0), np.where(up, Wb, WL0)
        if mode == "cash_when_up":
            return np.where(up, False, ML0), np.where(up, 0.0, WL0)
        if mode == "alts_only_up":
            return np.where(up, ML0, False), np.where(up, WL0, 0.0)
        if mode == "half_btc":
            return np.where(up, Mb | ML0, ML0), np.where(up, 0.5 * Wb + 0.5 * WL0, WL0)
    if fam == "hedge":  # B4b: beta-hedged winners (short BTC), total gross <= 1 (1x collateral)
        _, h, base_sig, lb = spec
        mom_l = ret[lb]
        noBTC = pd.DataFrame(True, index=C.index, columns=COLS); noBTC["BTC"] = False
        if base_sig == "raw":
            sc = mom_l
        else:
            sc = mom_l - g["beta"].mul(C["BTC"] / C["BTC"].shift(lb) - 1, axis=0)
        ML0 = membership(sc, K, 2, NEG, +1, pool=POOL & noBTC & (mom_l > 0), reset=~g["gate"])
        beta = np.nan_to_num(g["beta"].values, nan=1.0)
        bp = (np.where(ML0, 1.0 / K, 0.0) * beta).sum(axis=1)
        bp = np.clip(bp, 0.0, 2.0)
        gl = 0.98 / (1 + h * bp)  # long gross so that long+short <= 0.98
        WL = np.where(ML0, (gl / K)[:, None], 0.0)
        MS = np.zeros((T, N), bool); jb = COLS.index("BTC")
        MS[:, jb] = (ML0.sum(axis=1) > 0) & (h > 0)
        WS = np.zeros((T, N)); WS[:, jb] = h * bp * gl
        return ML0, WL, MS, WS
    raise ValueError(spec)


def paired_t(eq, ref):
    a = daily_ret(eq); b = daily_ret(ref)
    d = (a - b).dropna()
    n = len(d)
    x = d.values - d.values.mean()
    L = 14
    v = (x @ x) / n
    for l in range(1, L + 1):
        v += 2 * (1 - l / (L + 1)) * (x[l:] @ x[:-l]) / n
    se = np.sqrt(max(v, 1e-18) / n)
    return d.mean() / se


def daily_ret(eq):
    e = eq.resample("1D").last().dropna()
    return e.pct_change().dropna()


def job(spec):
    out = build(spec)
    label = " ".join(str(x) for x in spec)
    if spec[0] == "hedge":
        ML, WL, MS, WS = out
        g = S()
        reg = np.where(g["gate"], 2, 0)
        eq, info = run(ML=ML, WL=WL, MS=MS, WS=WS, regime=reg, resize_s=True)
        r = evaluate(eq, label, dict(family="hedge", params=label, turn=info["turn"], gross=info["gross"]))
    else:
        ML, WL = out
        r = finish(ML, WL, label, spec[0], label)
        eq = r.pop("_eq")
    ref = REF()
    r["tstat_vs_v1"] = paired_t(eq, ref) if spec[0] != "v1" else np.nan
    return r


def REF():
    if "ref_eq" not in _G:
        ML, WL = build(("v1",))
        eq, _ = run(ML=ML, WL=WL, regime=np.where(S()["gate"], 1, 0))
        _G["ref_eq"] = eq
    return _G["ref_eq"]


def ic_table():
    """Pooled cross-sectional Spearman IC of signals vs forward returns, evaluated daily at 00:00, by year."""
    g = S()
    sigs = {"mom336": g["ret"][336], "mom168": g["ret"][168], "mom72": g["ret"][72], "rev24": -g["ret"][24], "rev72": -g["ret"][72],
            "ofi24": g["ofi"][24], "ofi72": g["ofi"][72], "ofi168": g["ofi"][168], "lowvol": -g["vol168"],
            "resid336": g["ret"][336] - g["beta"].mul(g["ret_btc336"], axis=0), "nearhigh": C / g["rollmax"]}
    days = np.where(REBAL & (IDX >= START))[0]
    rows = []
    for fh in [24, 72, 168]:
        fwd = C.shift(-fh) / C - 1
        for name, sg in sigs.items():
            sv = sg.where(POOL).values; fv = fwd.values
            ics = []; yrs = []
            for t in days:
                if t + fh >= T:
                    continue
                a, b = sv[t], fv[t]
                ok = ~np.isnan(a) & ~np.isnan(b)
                if ok.sum() < 10:
                    continue
                ra = pd.Series(a[ok]).rank().values; rb_ = pd.Series(b[ok]).rank().values
                ics.append(np.corrcoef(ra, rb_)[0, 1]); yrs.append(IDX[t].year)
            ics = np.array(ics); yrs = np.array(yrs)
            row = dict(sig=name, fwd_h=fh, ic=ics.mean(), t=ics.mean() / ics.std() * np.sqrt(len(ics) / max(fh / 24, 1)), n=len(ics))
            for y in [2023, 2024, 2025, 2026]:
                row[f"ic{y}"] = ics[yrs == y].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def grid():
    G = [("v1",)]
    G += [("rev", h, w) for h in [24, 48, 72] for w in [0.25, 0.5, 1.0, 2.0]]
    G += [("dip", h, m) for h in [24, 72] for m in [5, 8, 12]]
    G += [("lowvol", m) for m in [5, 8, 12]] + [("lowvol_pure",)] + [("highvol", m) for m in [5, 8, 12]]
    G += [("near_high", w) for w in [168, 336, 672]]
    G += [("ofi", h, w) for h in [24, 72, 168] for w in [0.25, 0.5, 1.0, 2.0]]
    G += [("ofi_sel", h, m) for h in [24, 72, 168] for m in [5, 8]] + [("ofi_pure", h) for h in [24, 72, 168]]
    G += [("resid", w) for w in [0.25, 0.5, 0.75, 1.0]]
    G += [("multi", hs) for hs in [(168, 336), (336, 672), (168, 336, 672), (72, 168, 336), (168,), (672,)]]
    G += [("tsbasket", hm, cap) for hm in [336, 168] for cap in [0.2, 0.33]]
    G += [("dom", mode, th) for mode in ["btc_when_up", "cash_when_up", "alts_only_up", "half_btc"] for th in [-0.05, 0.0, 0.05]]
    G += [("hedge", h, bs, lb) for h in [0.5, 1.0] for bs in ["raw", "resid"] for lb in [336]]
    return G


if __name__ == "__main__":
    G = grid()
    print(len(G), "configs", flush=True)
    with Pool(12) as p:
        res = p.map(job, G, chunksize=1)
    df = pd.DataFrame(res)
    df.to_csv("out_research9_alt_B.csv", index=False)
    ic = ic_table()
    ic.to_csv("out_research9_alt_B_ic.csv", index=False)
    print(fmt(ic))
    print("saved", len(df))
