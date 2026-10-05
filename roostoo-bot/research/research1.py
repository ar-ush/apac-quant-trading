"""Research batch 1: classic families on 1h panel, scored on rolling 14-day windows.
All signals use info up to close of bar t; simulator applies them from bar t+1 (lag=1).
"""
import sys
import numpy as np
import pandas as pd
from lib import *

WIDE = {"PEPE", "SHIB", "BONK", "WLFI", "OPEN", "LISTA", "STO", "1000CHEEMS", "HEMI", "SOMI", "TUT", "BIO", "BMT",
        "EDEN", "MIRA", "AVNT", "LINEA", "FORM", "CFX", "S", "PAXG", "TON", "DOT", "EIGEN", "PLUME", "ARB"}

P = load_panel("1h")
C, H, L, QV = P["close"], P["high"], P["low"], P["qv"]
C = C.drop(columns=[c for c in C.columns if c in WIDE], errors="ignore")
QV = QV[C.columns]
H = H[C.columns]; L = L[C.columns]
R = C.pct_change()


def pool_mask(n=25, lookback_days=30, min_days=60):
    dq = QV.resample("1D").sum()
    med = dq.rolling(lookback_days, min_periods=20).median()
    hist = C.resample("1D").last().notna().cumsum()
    med = med.where(hist >= min_days)
    rank = med.rank(axis=1, ascending=False)
    m = (rank <= n)
    return m.shift(1).reindex(C.index, method="ffill").fillna(False)  # known at start of day


POOL = pool_mask(25)


def hvol(win=168):
    return R.rolling(win, min_periods=win // 2).std() * np.sqrt(24)  # daily vol


def finish(w, gross_cap=1.0):
    g = w.abs().sum(axis=1)
    scale = np.where(g > gross_cap, gross_cap / g.replace(0, np.nan), 1.0)
    return w.mul(pd.Series(scale, index=w.index).fillna(1.0), axis=0)


def hold_every(w, hours):
    """Only update weights every `hours` bars (sample-and-hold)."""
    idx = np.arange(len(w)) // hours
    keep = pd.Series(idx, index=w.index).drop_duplicates().index
    mask = w.index.isin(keep)
    out = w.copy()
    out[~mask] = np.nan
    return out.ffill().fillna(0.0)


# ----------------------------------------------------------- strategies
def tsmom_long(lookback_h=24 * 14, rebal_h=24, target_vol=0.02, gross=1.0, thr=0.0, pool=True):
    ret = C / C.shift(lookback_h) - 1
    vol = hvol()
    z = ret / (vol * np.sqrt(lookback_h / 24))
    sig = (z > thr).astype(float)
    if pool:
        sig = sig * POOL
    w = sig * (target_vol / vol).clip(upper=1.0)  # per-coin risk budget
    w = w / max(1, 1) * 1.0
    n = sig.sum(axis=1).replace(0, np.nan)
    w = w.div(n, axis=0).fillna(0.0) * 6  # ~equal-risk across active names, scaled
    return hold_every(finish(w, gross), rebal_h)


def xsmom_topk(lookback_h=24 * 14, k=3, rebal_h=24, btc_filter=True, filt=(168, 672), weight="eq", gross=1.0):
    ret = (C / C.shift(lookback_h) - 1).where(POOL)
    rk = ret.rank(axis=1, ascending=False)
    sel = ((rk <= k) & (ret > 0)).astype(float)
    if btc_filter:
        ema_f = C["BTC"].ewm(span=filt[0], adjust=False).mean()
        ema_s = C["BTC"].ewm(span=filt[1], adjust=False).mean()
        on = (ema_f > ema_s).astype(float)
        sel = sel.mul(on, axis=0)
    w = sel.div(sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * gross
    return hold_every(w, rebal_h)


def donchian(entry=20, exit_=10, bar_h=4, gross=1.0, per_name=1 / 12):
    Cb = C.resample(f"{bar_h}h").last()
    up = Cb.rolling(entry).max().shift(1)
    dn = Cb.rolling(exit_).min().shift(1)
    state = pd.DataFrame(0.0, index=Cb.index, columns=Cb.columns)
    s = np.zeros(Cb.shape[1])
    cv, uv, dv = Cb.values, up.values, dn.values
    out = np.zeros_like(cv)
    for i in range(len(Cb)):
        enter = (cv[i] > uv[i]) & (s == 0)
        ex = (cv[i] < dv[i]) & (s == 1)
        s = np.where(enter, 1.0, np.where(ex, 0.0, s))
        out[i] = s
    state = pd.DataFrame(out, index=Cb.index, columns=Cb.columns).shift(1)  # bar i closes at label i+bar_h: usable only after
    w = state.reindex(C.index, method="ffill").fillna(0.0) * POOL
    w = w * per_name
    return finish(w, gross)


def btc_trend(fast=168, slow=672, lev=1.0):
    on = (C["BTC"].ewm(span=fast, adjust=False).mean() > C["BTC"].ewm(span=slow, adjust=False).mean()).astype(float)
    w = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    w["BTC"] = on * lev
    return w


def lowvol_trend(topk=8, rebal_h=24):
    vol = hvol().where(POOL)
    trend = ((C.ewm(span=50, adjust=False).mean() > C.ewm(span=200, adjust=False).mean())).astype(float)
    cand = vol.where(trend > 0)
    rk = cand.rank(axis=1, ascending=True)
    sel = (rk <= topk).astype(float)
    on = (C["BTC"] > C["BTC"].ewm(span=200, adjust=False).mean()).astype(float)
    sel = sel.mul(on, axis=0)
    w = sel.div(sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 0.75
    return hold_every(w, rebal_h)


def ew_hold():
    sel = POOL.astype(float)
    w = sel.div(sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return hold_every(w, 24)


STRATS = {
    "ew_hold": ew_hold,
    "btc_hold": lambda: pd.DataFrame({c: (1.0 if c == "BTC" else 0.0) for c in C.columns}, index=C.index),
    "btc_trend": btc_trend,
    "tsmom_14d": lambda: tsmom_long(24 * 14),
    "tsmom_7d": lambda: tsmom_long(24 * 7),
    "tsmom_3d_6h": lambda: tsmom_long(24 * 3, rebal_h=6),
    "xsmom_top3_14d": lambda: xsmom_topk(24 * 14, 3),
    "xsmom_top3_7d": lambda: xsmom_topk(24 * 7, 3),
    "xsmom_top5_7d": lambda: xsmom_topk(24 * 7, 5),
    "xsmom_top3_3d_12h": lambda: xsmom_topk(24 * 3, 3, rebal_h=12),
    "xsmom_top3_7d_nofilt": lambda: xsmom_topk(24 * 7, 3, btc_filter=False),
    "donchian_4h_20_10": donchian,
    "donchian_4h_10_5": lambda: donchian(10, 5),
    "lowvol_trend": lowvol_trend,
}

if __name__ == "__main__":
    cost = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0012
    rows = []
    for name, fn in STRATS.items():
        w = fn()
        eq, info = simulate(C, w, cost=cost)
        win = rolling_windows(eq)
        s = summarize_windows(win)
        tot = eq.iloc[-1] - 1
        yrs = {y: (eq[eq.index.year == y].iloc[-1] / eq[eq.index.year == y].iloc[0] - 1) for y in [2023, 2024, 2025, 2026]}
        rows.append(dict(strat=name, tot=tot, mdd=max_drawdown(eq.values), turn=info["turnover_per_day"], gross=info["avg_gross"],
                         med14=s["med_ret"], pos=s["pos"], p10=s["p10"], p90=s["p90"], gt5=s["gt5"], mdd14=s["med_mdd"],
                         comp=s["med_comp"], y23=yrs[2023], y24=yrs[2024], y25=yrs[2025], y26=yrs[2026]))
    df = pd.DataFrame(rows).set_index("strat")
    pd.set_option("display.width", 250)
    print(df.round(3).to_string())
