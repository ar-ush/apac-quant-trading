"""Research 9 eval: replace v1's hard EMA gate with probabilistic regime layers (same momentum top-3 selection,
same point-in-time top-25 pool, cost 0.12%/side, daily 00:00 UTC rebalance, rolling 14d windows from 2023-04-01)."""
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import research1 as r1
from lib import *

C, POOL = r1.C, r1.POOL
START = pd.Timestamp("2023-04-01", tz="UTC")
K = 3
BTC = C["BTC"]
EMA_H = (BTC.ewm(span=168, adjust=False).mean() > BTC.ewm(span=672, adjust=False).mean()).astype(float)


def selection():
    ret = C / C.shift(336) - 1
    rk = ret.where(POOL).rank(axis=1, ascending=False)
    sel = ((rk <= K) & (ret > 0)).astype(float)
    w = sel.div(sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).clip(upper=1.0 / K)
    return w


WSEL = selection()
DEC = WSEL[WSEL.index.hour == 23]  # decision at close of 23:00 bar -> executed 00:00 UTC


def to_hourly_exposure(e_daily):
    """e_daily: Series indexed by UTC day d (known at close of d). Returns exposure at decision timestamps d+23h."""
    e = e_daily.copy()
    e.index = e.index + pd.Timedelta(hours=23)
    return e.reindex(DEC.index).fillna(0.0)


def run_variant(e_dec, hourly_gate=None):
    w = DEC.mul(e_dec.values, axis=0).reindex(C.index).ffill().fillna(0.0)
    if hourly_gate is not None:
        w = w.mul(hourly_gate.reindex(C.index).values, axis=0)
    eq, info = simulate(C, w, cost=0.0012)
    wl = w.shift(1).fillna(0.0)
    entries = int(((wl > 1e-9) & (wl.shift(1).fillna(0.0) <= 1e-9)).values.sum())
    eq = eq[eq.index >= START]
    eq = eq / eq.iloc[0]
    win = rolling_windows(eq)
    s = summarize_windows(win)
    out = dict(tot=eq.iloc[-1] - 1, mdd=max_drawdown(eq.values), mean14=s["mean_ret"], med14=s["med_ret"], p10=s["p10"],
               p90=s["p90"], pos=s["pos"], mcomp=s["mean_comp"], gross=float(w.loc[START:].abs().sum(axis=1).mean()),
               trades=entries, turn=info["turnover_per_day"])
    for y in (2023, 2024, 2025, 2026):
        ey = eq[eq.index.year == y]
        out[f"cal{y}"] = ey.iloc[-1] / ey.iloc[0] - 1
        wy = win[win.index.year == y]
        out[f"w{y}"] = wy.ret.mean()
    out["minw"] = min(out[f"w{y}"] for y in (2023, 2024, 2025, 2026))
    return out


def job(a):
    name, form, e_daily = a
    if form == "hourly_ema":
        r = run_variant(to_hourly_exposure(pd.Series(1.0, index=DEC.index.normalize())), hourly_gate=EMA_H)
    else:
        r = run_variant(to_hourly_exposure(e_daily))
    r.update(model=name, form=form)
    return r


def forms(P, ema_d, hard):
    P = P.fillna(0.0)
    d = {}
    for t in ([0.5] if hard else [0.3, 0.4, 0.5, 0.6, 0.7]):
        d[f"thr{t}"] = (P > t).astype(float)
    if not hard:
        d["ramp.3-.7"] = ((P - 0.3) / 0.4).clip(0, 1)
        d["pw"] = P
        d["pw*ema"] = P * ema_d
    d["thr.5&ema"] = (P > 0.5).astype(float) * ema_d
    return d


def bear_report(post, ema_d):
    """How well does P(bear) predict forward 14d BTC / EW-alt-basket returns? (signal at close of day d)"""
    tsd = DEC.index
    iloc = {t: i for i, t in enumerate(C.index)}
    pos = np.array([iloc[t] for t in tsd])
    fwd_ok = pos + 336 < len(C)
    Rew = C.pct_change().where(POOL.shift(1).fillna(False)).drop(columns=["BTC"]).mean(axis=1).fillna(0.0)
    alt_eq = (1 + Rew).cumprod().values
    btc = BTC.values
    f_btc = np.where(fwd_ok, btc[np.minimum(pos + 336, len(C) - 1)] / btc[pos] - 1, np.nan)
    f_alt = np.where(fwd_ok, alt_eq[np.minimum(pos + 336, len(C) - 1)] / alt_eq[pos] - 1, np.nan)
    F = pd.DataFrame({"btc": f_btc, "alt": f_alt}, index=tsd.normalize())
    F = F[F.index >= START].dropna()
    rows = []
    cands = {"ema_off": 1 - ema_d}
    for c in post.columns:
        if c.endswith("_bear"):
            cands[c] = post[c]
        elif not (c.endswith("h") and c.startswith("jm")):
            cands["1-" + c] = 1 - post[c]
        else:
            cands["1-" + c] = 1 - post[c]
    for n, s in cands.items():
        s = s.reindex(F.index).dropna()
        f = F.loc[s.index]
        for thr in ([0.5] if n == "ema_off" or n.startswith("1-jm") else [0.5, 0.7]):
            flag = s > thr
            if flag.sum() < 5 or (~flag).sum() < 5:
                continue
            row = dict(sig=n, thr=thr, freq=flag.mean())
            for a in ("btc", "alt"):
                row[f"{a}_flag"] = f[a][flag].mean(); row[f"{a}_unflag"] = f[a][~flag].mean()
                row[f"{a}_negrate_flag"] = (f[a][flag] < 0).mean(); row[f"{a}_negrate_unflag"] = (f[a][~flag] < 0).mean()
                y = (f[a] < 0).astype(int).values
                rk = rankdata(s.values)
                n1 = y.sum(); n0 = len(y) - n1
                row[f"{a}_auc_neg"] = (rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    post = pd.read_csv(r1.HERE / "out_research9_regime_post.csv", index_col=0, parse_dates=True)
    post.index = post.index.tz_convert("UTC") if post.index.tz is not None else post.index.tz_localize("UTC")
    post["logit_mom"] = post["logit_mom"].fillna(post["logit_btc"])  # warm-up fallback (documented)
    Cd = BTC[BTC.index.hour == 23]
    ema_d = EMA_H[EMA_H.index.hour == 23]
    ema_d.index = ema_d.index.normalize()
    jobs = [("EMA_gate(v1)", "daily", ema_d), ("EMA_gate(v1)", "hourly_ema", None),
            ("always_on", "daily", pd.Series(1.0, index=ema_d.index))]
    for c in post.columns:
        if c.endswith("_bear"):
            continue
        hard = c.startswith("jm") and c.endswith("h")
        for f, e in forms(post[c].reindex(ema_d.index), ema_d, hard).items():
            jobs.append((c, f, e))
    print(len(jobs), "jobs", flush=True)
    with Pool(8) as pool:
        res = pool.map(job, jobs, chunksize=4)
    df = pd.DataFrame(res)
    cols = ["model", "form", "tot", "mdd", "mean14", "med14", "p10", "p90", "pos", "mcomp", "gross", "trades", "turn",
            "cal2023", "cal2024", "cal2025", "cal2026", "w2023", "w2024", "w2025", "w2026", "minw"]
    df = df[cols]
    df.to_csv(r1.HERE / "out_research9_regime.csv", index=False)
    br = bear_report(post, ema_d)
    br.to_csv(r1.HERE / "out_research9_regime_bear.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 500)
    print(df.round(3).to_string())
    print(br.round(3).to_string())
