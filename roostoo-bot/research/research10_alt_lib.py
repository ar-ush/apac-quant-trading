"""research10_alt_lib: derivative/sentiment data aligned to the 1h panel + a sim with exposure scale + evaluation.
Alignment: panel bar t (labelled by OPEN time) is acted on at its close (as in research9_alt_lib). Every derived series
below is shifted so the value at row t is known by the close of bar t (funding settled at/before t; premium-index close of bar t;
daily macro series lagged by 1-2 days)."""
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from research9_alt_lib import *  # C,H,L,QV,O,POOL,R1,IDX,COLS,T,N,GATE,REBAL,HOUR,START,membership,eqw,evaluate,fmt
from lib import rolling_windows, summarize_windows, max_drawdown

D = Path(__file__).parent / "data/deriv"
MIN_HIST_DAYS = 90


# ------------------------------------------------------------------ derivative panels
def _load_funding():
    out = {}
    for c in COLS:
        f = D / f"fund_{c}.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
        s = s[~s.index.duplicated()].sort_index()
        s.index = s.index.round("1h")
        iv = s.index.to_series().diff().dt.total_seconds().div(3600).bfill().clip(1, 8)
        s8 = s * (8.0 / iv.values)  # per-8h equivalent (some perps move to 4h/1h funding)
        out[c] = s8.reindex(IDX.union(s8.index)).ffill(limit=12).reindex(IDX)
    return pd.DataFrame(out).reindex(columns=COLS)


def _load_prem():
    out = {}
    for c in COLS:
        f = D / f"prem_{c}.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f, index_col=0, parse_dates=True)["c"]
        out[c] = s[~s.index.duplicated()].reindex(IDX)
    return pd.DataFrame(out).reindex(columns=COLS)


FUND = _load_funding()  # per-8h-equivalent funding rate, known from settlement time on
PREM = _load_prem()     # premium index (perp - index)/index, close of bar t


def trail(df, h):
    return df.rolling(h, min_periods=max(h // 2, 6)).mean()


def daily_macro(name, lag_days):
    s = pd.read_csv(D / f"{name}.csv", index_col=0, parse_dates=True).iloc[:, -1 if name == "dvol" else 0]
    return s


def macro_hourly():
    """fng (lag1d), stablecoin supply (lag2d), dvol (value of previous hour bar)."""
    fng = pd.read_csv(D / "fng.csv", index_col=0, parse_dates=True).iloc[:, 0]
    fng = fng[~fng.index.duplicated()].shift(1, freq="1D")  # value of day D usable from D+1 00:00
    st = pd.read_csv(D / "stable.csv", index_col=0, parse_dates=True).iloc[:, 0]
    st = st[~st.index.duplicated()].shift(2, freq="1D")
    dv = pd.read_csv(D / "dvol.csv", index_col=0, parse_dates=True)["c"]
    dv = dv[~dv.index.duplicated()].shift(1, freq="1h")  # close of bar opened at u known at u+1h -> row t uses bar t-1h
    f = lambda s: s.reindex(IDX.union(s.index)).ffill().reindex(IDX)
    return f(fng), f(st), f(dv)


# ------------------------------------------------------------------ sim with exposure scale
@njit(cache=True)
def _sim2(Ca, rebal, regime, ML, WL, scale, cost, cap_mult):
    """Long-only unit sim (same mechanics as research9_alt_lib._sim): at rebal bars with regime==1 enter/keep members at
    scale[t]*WL; kept names are only trimmed (cap) unless scale rises vs the previous applied scale (then topped up)."""
    T, N = Ca.shape
    u = np.zeros(N)
    cash = 1.0
    book = 0
    prev_s = 1.0
    eq = np.empty(T)
    gross_acc = 0.0
    turn_acc = 0.0
    for t in range(T):
        e = cash
        gs = 0.0
        for j in range(N):
            e += u[j] * Ca[t, j]
            gs += abs(u[j]) * Ca[t, j]
        eq[t] = e
        gross_acc += gs / e if e > 0 else 0.0
        r = regime[t]
        if rebal[t]:
            book = r
            sc = scale[t]
            up = sc > prev_s + 1e-9
            if r == 1:
                prev_s = sc
            for j in range(N):
                p = Ca[t, j]
                if p <= 0:
                    continue
                uj = u[j]
                if r == 1 and ML[t, j]:
                    w = WL[t, j] * sc
                    nn = uj * p
                    if uj > 0:
                        capn = cap_mult * w * e
                        if up and nn < w * e:
                            dv = w * e - nn
                            u[j] += dv / p; cash -= dv + dv * cost; turn_acc += dv
                        elif nn > capn:
                            dv = nn - capn
                            u[j] -= dv / p; cash += dv - dv * cost; turn_acc += dv
                    else:
                        v = w * e
                        if v > 0:
                            u[j] = v / p; cash -= v + v * cost; turn_acc += v
                else:
                    if uj != 0:
                        v = uj * p
                        cash += v - v * cost; turn_acc += v
                        u[j] = 0.0
        else:
            if r != book:
                for j in range(N):
                    if u[j] != 0:
                        v = u[j] * Ca[t, j]
                        cash += v - v * cost; turn_acc += v
                        u[j] = 0.0
                book = 0
    return eq, gross_acc / T, turn_acc / (T / 24.0)


def run2(ML, WL, regime, scale=None, cost=0.0012, cap_mult=1.6):
    from research9_alt_lib import _Cf
    scale = np.ones(T) if scale is None else scale.astype(float)
    eq, gross, turn = _sim2(_Cf, REBAL, np.asarray(regime).astype(np.int64), ML, WL.astype(float), scale, cost, cap_mult)
    s = pd.Series(eq, index=IDX)
    s = s[s.index >= START]
    return s / s.iloc[0], dict(gross=gross, turn=turn)


def evaluate2(eq, label="", extra=None):
    w = rolling_windows(eq)
    s = summarize_windows(w)
    out = dict(label=label, tot=eq.iloc[-1] - 1, mdd=max_drawdown(eq.values), mean14=s["mean_ret"], med14=s["med_ret"],
               p10=s["p10"], p90=s["p90"], pos=s["pos"], gt20=(w.ret > 0.2).mean(), mcomp=s["mean_comp"], medcomp=s["med_comp"])
    for y in [2023, 2024, 2025, 2026]:
        e = eq[eq.index.year == y]
        out[f"y{y}"] = e.iloc[-1] / e.iloc[0] - 1
    for y in [2023, 2024, 2025, 2026]:
        out[f"m14_{y}"] = w.ret[w.index.year == y].mean()
    if extra:
        out.update(extra)
    return out


def daily_ret(eq):
    e = eq.resample("1D").last().dropna()
    return e.pct_change().dropna()


def paired_t(eq, ref, L=14):
    a = daily_ret(eq); b = daily_ret(ref)
    d = (a - b).dropna()
    n = len(d)
    x = d.values - d.values.mean()
    v = (x @ x) / n
    for l in range(1, L + 1):
        v += 2 * (1 - l / (L + 1)) * (x[l:] @ x[:-l]) / n
    return d.mean() / np.sqrt(max(v, 1e-18) / n)


# ------------------------------------------------------------------ v1 building blocks
MOM = C / C.shift(336) - 1
PR = lambda df: df.where(POOL).rank(axis=1, pct=True)
_REF = {}


def v1_build(score=None, elig=None, k=3, hyst=2, gate=None, thr=0.0):
    """v1 selection; `score` defaults to 336h momentum. elig is an additional boolean eligibility mask."""
    gate = GATE if gate is None else gate
    sc = MOM if score is None else score
    el = (MOM > 0)
    if elig is not None:
        el = el & elig
    pool = POOL & el
    ML = membership(sc, k, hyst, -1e9, +1, pool=pool, reset=~gate)
    return ML, eqw(ML, k)


def v1_ref():
    if "eq" not in _REF:
        ML, WL = v1_build()
        eq, _ = run2(ML, WL, np.where(GATE, 1, 0))
        _REF["eq"] = eq
    return _REF["eq"]


def day_expanding_pct(x: pd.Series, min_days=MIN_HIST_DAYS):
    """Walk-forward percentile of today's value vs ALL values up to and including today (daily series)."""
    x = x.dropna()
    v = x.values
    out = np.full(len(v), np.nan)
    for i in range(min_days, len(v)):
        out[i] = (v[: i + 1] <= v[i]).mean()
    return pd.Series(out, index=x.index)


# ------------------------------------------------------------------ grid runner helper
def score_cfg(label, ML, WL, regime=None, scale=None, extra=None, ref=None):
    regime = np.where(GATE, 1, 0) if regime is None else regime
    eq, info = run2(ML, WL, regime, scale)
    ex = dict(turn=info["turn"], gross=info["gross"])
    if extra:
        ex.update(extra)
    r = evaluate2(eq, label, ex)
    ref = v1_ref() if ref is None else ref
    r["t_vs_v1"] = paired_t(eq, ref) if label != "v1" else np.nan
    r["_eq"] = eq
    return r


def frac_changed(ML):
    ML0, _ = v1_build()
    rb = REBAL & (IDX >= START) & GATE
    return float((ML[rb] != ML0[rb]).any(axis=1).mean())
