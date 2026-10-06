"""Shared engine for research9_alt*: unit-based long/short sim (numba), membership prepass, scoring.
All signals use info up to the close of bar t and trade at close of bar t (same convention as lib.simulate / bot).
Short model: 1x collateral (gross short <= equity), fee per side, borrow accrual per bar on short notional,
adverse-move stop checked on bar highs (fill at stop or at open if gapped).
"""
import numpy as np
import pandas as pd
from numba import njit

import research1 as r1
from lib import rolling_windows, summarize_windows, max_drawdown, ratios

C, H, L, QV = r1.C, r1.H, r1.L, r1.QV
O = r1.P["open"][C.columns]
TBQ = r1.P["tbq"][C.columns]
POOL = r1.POOL.astype(bool)
R1 = C.pct_change()
IDX = C.index
COLS = list(C.columns)
T, N = C.shape
START = pd.Timestamp("2023-03-01", tz="UTC")
BTC = C["BTC"]
HOUR = IDX.hour.values


def gate_series(fast=168, slow=672):
    return (BTC.ewm(span=fast, adjust=False).mean() > BTC.ewm(span=slow, adjust=False).mean()).values


GATE = gate_series()
REBAL = (HOUR == 0)

_Cf = C.ffill().fillna(0.0).values
_Of = O.ffill().fillna(0.0).values
_Hf = H.ffill().fillna(0.0).values
_Lf = L.ffill().fillna(0.0).values


@njit(cache=True)
def _sim(Oa, Ha, La, Ca, rebal, regime, ML, WL, MS, WS, cost_l, cost_s, borrow_h, stop_s, stop_l, cap_mult, sign_alt, resize_s):
    T, N = Ca.shape
    u = np.zeros(N)
    ent = np.zeros(N)
    cash = 1.0
    book = 0
    eq = np.empty(T)
    gross_acc = 0.0
    turn_acc = 0.0
    nstop = 0
    for t in range(T):
        # 1) stops during bar t
        if t > 0:
            for j in range(N):
                if u[j] < 0 and stop_s > 0:
                    sp = ent[j] * (1.0 + stop_s)
                    if Ha[t, j] >= sp:
                        px = max(Oa[t, j], sp)
                        v = -u[j] * px
                        cash -= v + v * cost_s  # buy back
                        turn_acc += v
                        u[j] = 0.0
                        nstop += 1
                elif u[j] > 0 and stop_l > 0:
                    sp = ent[j] * (1.0 - stop_l)
                    if La[t, j] <= sp:
                        px = min(Oa[t, j], sp)
                        v = u[j] * px
                        cash += v - v * cost_l
                        turn_acc += v
                        u[j] = 0.0
                        nstop += 1
            # 2) borrow accrual on short notional
            if borrow_h > 0:
                sn = 0.0
                for j in range(N):
                    if u[j] < 0:
                        sn += -u[j] * Ca[t, j]
                cash -= sn * borrow_h
        # 3) mark to market
        e = cash
        gs = 0.0
        for j in range(N):
            e += u[j] * Ca[t, j]
            gs += abs(u[j]) * Ca[t, j]
        eq[t] = e
        gross_acc += gs / e if e > 0 else 0.0
        # 4) decisions at close t
        r = regime[t]
        if rebal[t]:
            book = r
            for j in range(N):
                p = Ca[t, j]
                if p <= 0:
                    continue
                if r == 1:
                    m = ML[t, j]; w = WL[t, j]; s = 1.0
                elif r == -1:
                    m = MS[t, j]; w = WS[t, j]; s = sign_alt
                elif r == 2:
                    if ML[t, j]:
                        m = True; w = WL[t, j]; s = 1.0
                    elif MS[t, j]:
                        m = True; w = WS[t, j]; s = -1.0
                    else:
                        m = False; w = 0.0; s = 0.0
                else:
                    m = False; w = 0.0; s = 0.0
                uj = u[j]
                if m and uj * s > 0:
                    # keep; trim if above cap
                    capn = cap_mult * w * e
                    nn = abs(uj) * p
                    if resize_s and s < 0:
                        dv = w * e - nn  # >0: add short, <0: cover
                        if dv > 0:
                            u[j] -= dv / p; cash += dv - dv * cost_s
                        else:
                            u[j] += (-dv) / p; cash -= (-dv) + (-dv) * cost_s
                        turn_acc += abs(dv)
                    elif nn > capn:
                        dv = nn - capn
                        du = dv / p
                        if s > 0:
                            u[j] -= du; cash += dv - dv * cost_l
                        else:
                            u[j] += du; cash -= dv + dv * cost_s
                        turn_acc += dv
                elif m:
                    # close opposite if any (shouldn't happen) then open
                    if uj != 0:
                        v = abs(uj) * p
                        if uj > 0:
                            cash += v - v * cost_l
                        else:
                            cash -= v + v * cost_s
                        turn_acc += v
                        u[j] = 0.0
                    v = w * e
                    if v > 0:
                        u[j] = s * v / p
                        ent[j] = p
                        if s > 0:
                            cash -= v + v * cost_l
                        else:
                            cash += v - v * cost_s
                        turn_acc += v
                else:
                    if uj != 0:
                        v = abs(uj) * p
                        if uj > 0:
                            cash += v - v * cost_l
                        else:
                            cash -= v + v * cost_s
                        turn_acc += v
                        u[j] = 0.0
        else:
            if r != book:
                # regime flip between rebalances: flatten
                for j in range(N):
                    uj = u[j]
                    if uj != 0:
                        v = abs(uj) * Ca[t, j]
                        if uj > 0:
                            cash += v - v * cost_l
                        else:
                            cash -= v + v * cost_s
                        turn_acc += v
                        u[j] = 0.0
                book = 0
    return eq, gross_acc / T, turn_acc / (T / 24.0), nstop


def membership(score, k, hyst=0, thr=0.0, side=1, pool=POOL, reset=None, rebal=REBAL, max_k_cap=None):
    """Top-k membership with rank hysteresis, evaluated at rebal bars.
    score: DataFrame (higher = better for `side`=+1; for shorts pass side=-1 and score is the long-style
    score, i.e. we pick the LOWEST scores). Entry requires side*score > thr (thr on raw score sign).
    reset: bool array (T) True where held set is cleared (regime off)."""
    s = (score.where(pool) * side)
    rk = s.rank(axis=1, ascending=False).values
    sv = s.values
    M = np.zeros((T, N), bool)
    held = np.zeros(N, bool)
    for t in np.where(rebal)[0]:
        if reset is not None and reset[t]:
            held[:] = False
            continue
        ok = ~np.isnan(rk[t])
        keep = held & ok & (rk[t] <= k + hyst) & (sv[t] > thr)
        if keep.sum() > k:
            idx = np.where(keep)[0]
            idx = idx[np.argsort(rk[t, idx])][:k]
            keep = np.zeros(N, bool); keep[idx] = True
        cand = ok & (rk[t] <= k) & (sv[t] > thr) & ~keep
        idx = np.where(cand)[0]
        idx = idx[np.argsort(rk[t, idx])][: k - keep.sum()]
        new = keep.copy(); new[idx] = True
        M[t] = new
        held = new
    return M


def eqw(M, k, gross=0.98):
    W = np.where(M, 1.0 / k, 0.0)
    g = W.sum(axis=1, keepdims=True)
    f = np.where(g > gross, gross / np.maximum(g, 1e-9), 1.0)
    return W * f


def run(ML=None, WL=None, MS=None, WS=None, regime=None, cost_l=0.0012, cost_s=0.0012, borrow_day=0.0002,
        stop_s=0.0, stop_l=0.0, cap_mult=1.6, rebal=REBAL, sign_alt=-1.0, resize_s=False):
    z = np.zeros((T, N), bool)
    zf = np.zeros((T, N))
    ML = z if ML is None else ML; WL = zf if WL is None else WL
    MS = z if MS is None else MS; WS = zf if WS is None else WS
    if regime is None:
        regime = np.where(GATE, 1, 0)
    eq, gross, turn, nstop = _sim(_Of, _Hf, _Lf, _Cf, rebal, regime.astype(np.int64), ML, WL.astype(float), MS, WS.astype(float),
                                  cost_l, cost_s, borrow_day / 24.0, stop_s, stop_l, cap_mult, sign_alt, resize_s)
    s = pd.Series(eq, index=IDX)
    s = s[s.index >= START]
    s = s / s.iloc[0]
    return s, dict(gross=gross, turn=turn, nstop=nstop)


def evaluate(eq, label="", extra=None):
    w = rolling_windows(eq)
    s = summarize_windows(w)
    yrs = {}
    for y in [2023, 2024, 2025, 2026]:
        e = eq[eq.index.year == y]
        yrs[f"y{y}"] = e.iloc[-1] / e.iloc[0] - 1
    # also per-year mean 14d windows
    wy = {f"m14_{y}": w.ret[w.index.year == y].mean() for y in [2023, 2024, 2025, 2026]}
    out = dict(label=label, tot=eq.iloc[-1] - 1, mdd=max_drawdown(eq.values), mean14=s["mean_ret"], med14=s["med_ret"],
               p10=s["p10"], p90=s["p90"], pos=s["pos"], pos1=(w.ret > 0.01).mean(), neg5=(w.ret < -0.05).mean(), mcomp=s["mean_comp"], medcomp=s["med_comp"], mdd14=s["med_mdd"], **yrs, **wy)
    if extra:
        out.update(extra)
    return out


def v1_inputs(lb=336, k=3, hyst=2, gate=None):
    gate = GATE if gate is None else gate
    ret = C / C.shift(lb) - 1
    ML = membership(ret, k, hyst, 0.0, +1, reset=~gate)
    return ML, eqw(ML, k), ret


def fmt(df, cols=None):
    pd.set_option("display.width", 280); pd.set_option("display.max_columns", 50)
    return df.round(3).to_string() if cols is None else df[cols].round(3).to_string()
