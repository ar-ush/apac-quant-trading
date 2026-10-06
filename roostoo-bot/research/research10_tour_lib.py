"""research10_tour shared lib: fast vectorised rolling-14d window metrics + variant builders on top of research9_alt_lib (numba engine).
Run from F:\\apac-trading-hackathon\\work. Tournament metrics: P(r14>x), upper tail, per-year consistency.
"""
import numpy as np
import pandas as pd

import research9_alt_lib as L
import research1 as r1
from lib import max_drawdown

C, IDX, T, N, COLS = L.C, L.IDX, L.T, L.N, L.COLS
HOUR = L.HOUR
GATE = L.GATE
XS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
YEARS = [2023, 2024, 2025, 2026]
WL_ = 336  # bars per window
ANN = 365.0
_START_IDX = int(np.searchsorted(IDX.values, np.datetime64(L.START.tz_localize(None))))
EQ_IDX = IDX[_START_IDX:]


def window_starts(n_eq):
    """positions (in eq array indexed from START) of 00:00 bars from day 1 on, with full 14d after."""
    eh = EQ_IDX.hour.values
    s = np.where(eh == 0)[0]
    s = s[(s >= 24) & (s + WL_ <= n_eq - 1)]
    return s


def pool_mask(n=25, lookback_days=30, min_days=60):
    return r1.pool_mask(n, lookback_days, min_days).values.astype(bool)


# ---------------------------------------------------------------- engine wrappers
def rebal_mask(off=0, freq=24):
    if freq == 24:
        return HOUR == off
    if freq == 12:
        return (HOUR % 12) == (off % 12)
    if freq == 48:
        day = (IDX.values.astype("datetime64[D]").astype(np.int64))
        return (HOUR == off) & (day % 2 == 0)
    if freq == 6:
        return (HOUR % 6) == (off % 6)
    raise ValueError(freq)


def run_v1(lb=336, k=3, hyst=2, pool=None, rebal=None, gate=None, W=None, score=None, cap_mult=1.6, thr=0.0, ML=None):
    """v1-style gated momentum, returns equity (np array from START)."""
    pool = L.POOL if pool is None else pool
    rebal = L.REBAL if rebal is None else rebal
    gate = GATE if gate is None else gate
    if score is None:
        score = C / C.shift(lb) - 1
    if ML is None:
        ML = L.membership(score, k, hyst, thr, +1, pool=pool, reset=~gate, rebal=rebal)
    WLw = L.eqw(ML, k) if W is None else W
    regime = np.where(gate, 1, 0)
    eq, info = L.run(ML=ML, WL=WLw, regime=regime, rebal=rebal, cap_mult=cap_mult)
    return eq.values, info


# ---------------------------------------------------------------- window metrics
def _cap(x, c):
    return np.clip(x, -c, c)


def window_matrix(eq, s):
    return eq[s[:, None] + np.arange(WL_ + 1)[None, :]] / eq[s][:, None]


def metrics_from_E(E):
    """E: [nw, 337] equity paths normalised to 1. Returns dict of arrays ret, mdd, comp (same as lib.ratios)."""
    nw = E.shape[0]
    ret = E[:, -1] - 1
    mdd = np.max(1 - E / np.maximum.accumulate(E, axis=1), axis=1)
    pts = E[:, [0] + [23 + 24 * j for j in range(14)] + [WL_]]
    r = pts[:, 1:] / pts[:, :-1] - 1
    mu = r.mean(1)
    sd = r.std(1, ddof=1)
    dd = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(sd > 0, mu / np.where(sd > 0, sd, 1) * np.sqrt(ANN), np.where(mu > 0, np.inf, 0.0))
        sortino = np.where(dd > 0, mu / np.where(dd > 0, dd, 1) * np.sqrt(ANN), np.where(mu > 0, np.inf, 0.0))
        ann = np.where(ret > -1, (1 + ret) ** (ANN / 14.0) - 1, -1.0)
        calmar = np.where(mdd > 1e-9, ann / np.where(mdd > 1e-9, mdd, 1), np.where(ann > 0, np.inf, 0.0))
    comp = 0.4 * _cap(sortino, 60) + 0.3 * _cap(sharpe, 40) + 0.3 * _cap(calmar, 200)
    return dict(ret=ret, mdd=mdd, comp=comp)


def summarize(m, starts_idx, gate_on=None, label="", eq=None, extra=None):
    """m: dict of arrays; starts_idx: positions into EQ_IDX; gate_on: bool array per window (gate on at window start)."""
    ret, comp = m["ret"], m["comp"]
    ts = EQ_IDX[starts_idx]
    yr = ts.year.values
    out = dict(label=label, n=len(ret))
    for x in XS:
        out[f"P{int(x*100)}"] = float((ret > x).mean())
    out.update(mean=ret.mean(), med=np.median(ret), p10=np.quantile(ret, .1), p90=np.quantile(ret, .9),
               p99=np.quantile(ret, .99), pos=(ret > 0).mean(), comp=comp.mean(), mdd14=np.median(m["mdd"]))
    out["tail"] = (out["P10"] + out["P20"] + out["P30"]) / 3
    if eq is not None:
        out["tot"] = eq[-1] / eq[0] - 1
        out["mdd"] = max_drawdown(eq)
    for y in YEARS:
        k = yr == y
        out[f"tail{y}"] = float(np.mean([(ret[k] > .10).mean(), (ret[k] > .20).mean(), (ret[k] > .30).mean()])) if k.any() else np.nan
        out[f"mean{y}"] = ret[k].mean() if k.any() else np.nan
        out[f"P20_{y}"] = (ret[k] > .20).mean() if k.any() else np.nan
    if gate_on is not None:
        g = gate_on
        for x in (.10, .20, .30):
            out[f"gP{int(x*100)}"] = float((ret[g] > x).mean()) if g.any() else np.nan
        out["gmean"] = ret[g].mean()
        out["gmed"] = np.median(ret[g])
        out["gp10"] = np.quantile(ret[g], .1)
    if extra:
        out.update(extra)
    return out


class Ctx:
    """window positions + gate-on flag, built once for a given equity length."""
    def __init__(self, n_eq):
        self.s = window_starts(n_eq)
        self.gate_on = GATE[_START_IDX:][self.s]


_ctx = None


def ctx(n_eq):
    global _ctx
    if _ctx is None or len(_ctx.s) == 0:
        _ctx = Ctx(n_eq)
    return _ctx


def evaluate(eq, label="", extra=None):
    c = ctx(len(eq))
    E = window_matrix(eq, c.s)
    m = metrics_from_E(E)
    return summarize(m, c.s, c.gate_on, label, eq, extra)


def eval_from_E(E, label="", extra=None):
    c = ctx(E.shape[0] and (E.shape[0]))  # ctx already built by a prior evaluate()
    m = metrics_from_E(E)
    return summarize(m, c.s, c.gate_on, label, None, extra)


def fmt(df, cols):
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60)
    return df[cols].round(3).to_string()
