"""Research 9 (MC): walk-forward probabilistic forecasting of h-day forward return distributions per coin.

Generators (pooled GJR-GARCH(1,1)-t refit monthly, variance targeting per coin; zero simple-return drift unless noted):
  A  GJR-t, no drift             B  FHS block bootstrap (own-coin std. residuals, 365d), no drift
  C  regime FHS (blocks only from days in same BTC-gate regime) + momentum drift
  D  GJR-t + shrunk momentum drift      E  FHS + shrunk momentum drift
  ENS = mixture of C, D, E.  (momentum drift = 0.5 * pooled no-intercept OLS slope of vol-scaled fwd return on z14)
Then: split-conformal + ACI (normalised CQR score on the ensemble 10-90 interval), HistGB quantile regression + CQR.
Everything walk-forward: params/drift/QR fitted on data with outcomes known before the month start; conformal
calibration only on resolved outcomes.  Decision at daily 00:00 UTC (hourly bar labelled 23:00 closes).
Usage: python research9_mc.py [h=7]
"""
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln
from scipy.stats import norm, spearmanr, kstest

import research1 as r1
from lib import *

warnings.filterwarnings("ignore")
SCR = Path(r"C:\Users\arush\AppData\Local\Temp\claude\F--apac-trading-hackathon\236037f4-6cd4-4b18-bd57-11339cda5932\scratchpad")
C, QV, POOLH = r1.C, r1.QV, r1.POOL
dec = C.index[C.index.hour == 23]
DC = C.loc[dec].copy()
DC.index = DC.index.normalize()
POOLD = POOLH.loc[dec].copy()
POOLD.index = DC.index
GATEH = C["BTC"].ewm(span=168, adjust=False).mean() > C["BTC"].ewm(span=672, adjust=False).mean()
GATED = GATEH.loc[dec].copy()
GATED.index = DC.index
LR = np.log(DC).diff()
T, N = DC.shape
lr = LR.values
dcv = DC.values
poolv = POOLD.values
gate_arr = GATED.values.astype(int)
COST_RT = 0.0024
QS19 = np.arange(1, 20) / 20.0
FGRID = np.linspace(0, 1, 11)


# ------------------------------------------------------------------ GJR-GARCH
def gjr_filter(lrw, v, a, g, b):
    e2 = np.where(np.isnan(lrw), v[None, :], lrw ** 2)
    neg = (lrw < 0).astype(float)
    u = np.empty_like(e2)
    u[0] = v
    om = v * (1 - a - g / 2 - b)
    u[1:] = om + (a + g * neg[:-1]) * e2[:-1]
    return lfilter([1.0], [1.0, -b], u, axis=0), om


def nll(theta, lrw, v, cnt_ok):
    a, g, b, nu = theta
    pen = 1e3 * max(0.0, a + g / 2 + b - 0.995) ** 2
    s2, _ = gjr_filter(lrw, v, a, g, b)
    ok = np.isfinite(lrw) & np.isfinite(s2) & cnt_ok
    ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
          - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log1p(lrw ** 2 / (s2 * (nu - 2))))
    return -np.mean(ll[ok]) + pen


def fit_gjr(m, x0):
    lrw = lr[:m]
    cnt = np.cumsum(np.isfinite(lrw), axis=0)
    v = np.nanmean(lrw ** 2, axis=0)
    v[cnt[-1] < 30] = np.nan
    v = np.where(np.isfinite(v), v, np.nanmedian(v))
    res = minimize(nll, x0, args=(lrw, v, cnt >= 20), method="L-BFGS-B",
                   bounds=[(1e-3, 0.3), (0.0, 0.3), (0.5, 0.995), (2.5, 40.0)])
    return res.x, v


# ------------------------------------------------------------------ simulation
def draw_tstd(rng, nu, shape):
    return rng.standard_t(nu, shape) / np.sqrt(nu / (nu - 2))


def sim_generator(kind, rows, par, S, rng, E, Em, Es, first_valid, h, drift, blk=3):
    a, g, b, nu = par
    ti, ci, s1, om = rows["ti"], rows["ci"], rows["s1"], rows["om"]
    R = len(ti)
    s2 = np.repeat(s1[:, None], S, 1)
    tot = np.zeros((R, S))
    starts = None
    if kind in ("fhs", "reg"):
        nb = int(np.ceil(h / blk))
        smin = np.maximum(first_valid[ci] + 10, ti - 364)
        nst = np.maximum(ti - blk + 1 - smin + 1, 1)
        starts = []
        for j in range(nb):
            s = smin[:, None] + np.floor(rng.random((R, S)) * nst[:, None]).astype(int)
            if kind == "reg":
                reg_t = gate_arr[ti][:, None]
                for _ in range(8):
                    bad = gate_arr[s] != reg_t
                    if not bad.any():
                        break
                    s2_ = smin[:, None] + np.floor(rng.random((R, S)) * nst[:, None]).astype(int)
                    s = np.where(bad, s2_, s)
            starts.append(s)
        mu_c = Em[ti, ci][:, None]
        sd_c = Es[ti, ci][:, None]
    for k in range(h):
        if kind == "t":
            z = draw_tstd(rng, nu, (R, S))
        else:
            idx = np.minimum(starts[k // blk] + (k % blk), T - 1)
            z = np.clip((E[idx, ci[:, None]] - mu_c) / np.maximum(sd_c, 0.3), -8.0, 8.0)
        sig = np.sqrt(s2)
        tot += sig * z
        s2 = om[:, None] + (a + g * (z < 0)) * s2 * z ** 2 + b * s2
    tot = np.clip(tot, -3.0, 2.0)
    var = tot.var(axis=1, keepdims=True)
    tot = tot - 0.5 * var + (drift * np.sqrt(s1 * h))[:, None]
    return np.expm1(tot)


def dist_stats(X, y):
    R, S = X.shape
    Xs = np.sort(X, axis=1)
    q19 = np.quantile(Xs, QS19, axis=1, method="linear").T
    out = dict(mean=X.mean(1), sd=X.std(1), pup=(X > 0).mean(1), pcost=(X > COST_RT).mean(1),
               es5=Xs[:, :int(0.05 * S)].mean(1))
    for q, nm in [(0.05, "q05"), (0.10, "q10"), (0.25, "q25"), (0.5, "q50"), (0.75, "q75"), (0.9, "q90"), (0.95, "q95")]:
        out[nm] = np.quantile(Xs, q, axis=1)
    G = np.stack([np.log(np.maximum(1 + f * (X - COST_RT), 1e-6)).mean(1) for f in FGRID], 1)
    out["gstar"] = G.max(1)
    out["fstar"] = FGRID[G.argmax(1)]
    out["g_half"] = G[:, 5]
    out["pit"] = np.where(np.isfinite(y), ((X < y[:, None]).sum(1) + 0.5 * (X == y[:, None]).sum(1)) / S, np.nan)
    return out, q19


# ------------------------------------------------------------------ conformal
def conformal(di, lo, hi, y, h, alpha=0.2, win=180, gamma=0.02, min_cal=300, cal_min_di=None):
    w = np.maximum(hi - lo, 1e-6)
    Esc = np.maximum(lo - y, y - hi) / w
    fin = np.isfinite(Esc)
    dates = np.unique(di)
    Qs = {}
    Qa = {}
    al = alpha
    alphas = {}
    for t in dates:
        cm = fin & (di <= t - h) & (di > t - h - win)
        if cal_min_di is not None:
            cm &= di >= cal_min_di[t]
        sc = Esc[cm]
        n = len(sc)
        if n < min_cal:
            Qs[t] = Qa[t] = np.nan
            alphas[t] = al
            continue
        Qs[t] = np.quantile(sc, min(1.0, np.ceil((1 - alpha) * (n + 1)) / n), method="higher")
        Qa[t] = np.quantile(sc, float(np.clip(1 - al, 0.02, 0.995)), method="higher")
        alphas[t] = al
        res = Esc[fin & (di == t - h)]
        if len(res):
            err = float(np.mean(res > Qa[t]))
            al = float(np.clip(al + gamma * (alpha - err), 0.01, 0.6))
    return (np.array([Qs[t] for t in di]), np.array([Qa[t] for t in di]), w, Esc)


# ------------------------------------------------------------------ main
def run(h=7, S=400, start="2023-03-01", seed=11):
    rng = np.random.default_rng(seed)
    month_starts = pd.date_range(pd.Timestamp(start, tz="UTC"), DC.index[-1], freq="MS")
    if len(month_starts) == 0 or month_starts[0] != pd.Timestamp(start, tz="UTC"):
        month_starts = month_starts.insert(0, pd.Timestamp(start, tz="UTC"))
    gens = {"A": "t", "B": "fhs", "C": "reg", "D": "t", "E": "fhs"}
    use_drift = {"A": False, "B": False, "C": True, "D": True, "E": True}
    store = {k: [] for k in list(gens) + ["ENS"]}
    q19s = {k: [] for k in store}
    rows_all = []
    x0 = np.array([0.05, 0.03, 0.9, 5.0])
    hist_x, hist_y, hist_ti = [], [], []
    params_log = []
    beta = 0.0
    x14_mat = None
    for mi, m0 in enumerate(month_starts):
        m1 = month_starts[mi + 1] if mi + 1 < len(month_starts) else DC.index[-1] + pd.Timedelta(days=1)
        i0 = DC.index.searchsorted(m0)
        i1 = DC.index.searchsorted(m1)
        if i0 >= T:
            break
        t0 = time.time()
        par, v = fit_gjr(i0, x0)
        x0 = par
        a, g, b, nu = par
        s2, om_vec = gjr_filter(lr[:i1], v, a, g, b)
        # next-day variance forecast sigma^2_{t+1} from info at close of t
        e2 = np.where(np.isnan(lr[:i1]), v[None, :], lr[:i1] ** 2)
        neg = (lr[:i1] < 0).astype(float)
        s2n = om_vec + (a + g * neg) * e2 + b * s2
        E = lr[:i1] / np.sqrt(s2)
        Edf = pd.DataFrame(E)
        Em = Edf.rolling(365, min_periods=30).mean().values
        Es = Edf.rolling(365, min_periods=30).std().values
        first_valid = np.argmax(np.isfinite(lr), axis=0)
        E_f = np.nan_to_num(E)
        # rows in this month
        tt, cc = np.where(poolv[i0:i1] & np.isfinite(lr[i0:i1]) & np.isfinite(s2n[i0:i1]))
        ti = tt + i0
        keep = (ti - first_valid[cc]) >= 40
        ti, cc = ti[keep], cc[keep]
        if len(ti) == 0:
            continue
        s1 = s2n[ti, cc]
        y = np.where(ti + h < T, dcv[np.minimum(ti + h, T - 1), cc] / dcv[ti, cc] - 1, np.nan)
        lr14 = np.log(dcv[ti, cc] / dcv[np.maximum(ti - 14, 0), cc])
        x14 = np.clip(lr14 / np.sqrt(s1 * 14), -5, 5)
        # walk-forward momentum slope (outcomes known strictly before month start)
        if hist_x:
            hx, hy, hti = np.concatenate(hist_x), np.concatenate(hist_y), np.concatenate(hist_ti)
            ok = (hti + h <= i0 - 1) & np.isfinite(hy)
            if ok.sum() > 500:
                beta = 0.5 * float((hx[ok] * hy[ok]).sum() / (hx[ok] ** 2).sum())
        drift = np.clip(beta * x14, -0.5, 0.5)
        yn = np.clip(np.log1p(np.where(np.isfinite(y), y, 0)) / np.sqrt(s1 * h), -6, 6)
        yn[~np.isfinite(y)] = np.nan
        hist_x.append(x14)
        hist_y.append(yn)
        hist_ti.append(ti)
        rows = dict(ti=ti, ci=cc, s1=s1, om=v[cc] * (1 - a - g / 2 - b))
        samples = {}
        for gk, kind in gens.items():
            d = drift if use_drift[gk] else np.zeros_like(drift)
            samples[gk] = sim_generator(kind, rows, par, S, rng, E_f, Em, Es, first_valid, h, d)
        samples["ENS"] = np.concatenate([samples["C"], samples["D"], samples["E"]], axis=1)
        for k, X in samples.items():
            st, q19 = dist_stats(X, y)
            store[k].append(pd.DataFrame(st))
            q19s[k].append(q19)
        rows_all.append(pd.DataFrame(dict(ti=ti, ci=cc, s1=s1, y=y, x14=x14, drift=drift, beta=beta)))
        params_log.append(dict(month=str(m0.date()), a=a, g=g, b=b, nu=nu, beta=beta, rows=len(ti), sec=round(time.time() - t0, 1)))
        print(params_log[-1], flush=True)
    rows_df = pd.concat(rows_all, ignore_index=True)
    stats = {k: pd.concat(v, ignore_index=True) for k, v in store.items()}
    q19 = {k: np.concatenate(v) for k, v in q19s.items()}
    return rows_df, stats, q19, pd.DataFrame(params_log)


# ------------------------------------------------------------------ quantile regression + CQR
def qr_features(rows_df):
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa
    ti, ci, s1 = rows_df["ti"].values, rows_df["ci"].values, rows_df["s1"].values
    sg = np.sqrt(s1)
    lvl = np.log(dcv)
    btc = list(DC.columns).index("BTC")
    f = {}
    for k in [3, 7, 14, 30]:
        f[f"x{k}"] = np.clip((lvl[ti, ci] - lvl[np.maximum(ti - k, 0), ci]) / (sg * np.sqrt(k)), -6, 6)
    f["lvol"] = np.log(sg)
    d1 = pd.DataFrame(lr)
    vr = (d1.rolling(5, min_periods=3).std() / d1.rolling(60, min_periods=20).std()).values
    f["vratio"] = vr[ti, ci]
    hi30 = pd.DataFrame(dcv).rolling(30, min_periods=10).max().values
    f["dist_hi"] = dcv[ti, ci] / hi30[ti, ci] - 1
    f["gate"] = gate_arr[ti].astype(float)
    f["btc14"] = lvl[ti, btc] - lvl[np.maximum(ti - 14, 0), btc]
    f["rel14"] = (lvl[ti, ci] - lvl[np.maximum(ti - 14, 0), ci]) - f["btc14"]
    return pd.DataFrame(f)


def run_qr(rows_df, h, first_qr="2023-09-01", cal_days=90):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = qr_features(rows_df)
    ti = rows_df["ti"].values
    y = rows_df["y"].values
    sh = np.sqrt(rows_df["s1"].values * h)
    yn = np.clip(y / sh, -6, 6)
    out = {q: np.full(len(ti), np.nan) for q in (0.1, 0.5, 0.9)}
    cal_min = {}
    month_starts = pd.date_range(pd.Timestamp(first_qr, tz="UTC"), DC.index[-1], freq="MS")
    for mi, m0 in enumerate(month_starts):
        i0 = DC.index.searchsorted(m0)
        i1 = DC.index.searchsorted(month_starts[mi + 1]) if mi + 1 < len(month_starts) else T + 5
        cut_issue = i0 - cal_days - h
        tr = (ti <= cut_issue) & np.isfinite(yn)
        te = (ti >= i0) & (ti < i1)
        if tr.sum() < 3000 or te.sum() == 0:
            continue
        for q in out:
            mdl = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=100, max_depth=3, learning_rate=0.05,
                                                min_samples_leaf=200, l2_regularization=10.0)
            mdl.fit(X[tr].values, yn[tr])
            out[q][te] = mdl.predict(X[te].values)
        for t in range(i0, min(i1, T)):
            cal_min[t] = i0 - cal_days
        print("QR month", m0.date(), tr.sum(), flush=True)
    lo = np.minimum(out[0.1], out[0.9]) * sh
    hi = np.maximum(out[0.1], out[0.9]) * sh
    med = out[0.5] * sh
    return lo, med, hi, cal_min


def calib_report(rows_df, stats, q19, h, eval_start="2023-10-01"):
    ti = rows_df["ti"].values
    dates = DC.index[ti]
    y = rows_df["y"].values
    ev = (dates >= pd.Timestamp(eval_start, tz="UTC")) & np.isfinite(y)
    sg = np.sqrt(rows_df["s1"].values * h)
    # naive benchmark: zero-mean Gaussian with trailing 60d daily vol
    vol60 = pd.DataFrame(lr).rolling(60, min_periods=30).std().values[ti, rows_df["ci"].values]
    qn = norm.ppf(QS19)[None, :] * (vol60 * np.sqrt(h))[:, None]
    def crps(Q):
        d = y[:, None] - Q
        return 2 * np.mean(np.maximum(QS19 * d, (QS19 - 1) * d), axis=1)
    cn = crps(qn)
    recs = []
    for k in stats:
        st = stats[k]
        c = crps(q19[k])
        ic = []
        df = pd.DataFrame(dict(d=ti[ev], m=st["mean"].values[ev], y=y[ev]))
        for d_, g_ in df.groupby("d"):
            if len(g_) > 8:
                ic.append(spearmanr(g_.m, g_.y)[0])
        pit = st["pit"].values[ev]
        ks = kstest(pit, "uniform").statistic
        rec = dict(gen=k, crps=np.nanmean(c[ev]), crps_naive=np.nanmean(cn[ev]), skill_vs_naive=1 - np.nanmean(c[ev]) / np.nanmean(cn[ev]),
                   norm_skill=1 - np.nanmean((c / sg)[ev]) / np.nanmean((cn / sg)[ev]),
                   ks_pit=ks, cov80=np.mean((y >= st["q10"].values) & (y <= st["q90"].values))
                   if False else np.mean(((y >= st["q10"].values) & (y <= st["q90"].values))[ev]),
                   cov90=np.mean(((y >= st["q05"].values) & (y <= st["q95"].values))[ev]),
                   mean_ic=np.nanmean(ic), pit_lo=np.mean(pit < 0.1), pit_hi=np.mean(pit > 0.9),
                   bias=np.mean(y[ev] - st["mean"].values[ev]))
        recs.append(rec)
    return pd.DataFrame(recs).set_index("gen")


if __name__ == "__main__":
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    rows_df, stats, q19, plog = run(h=h)
    ens = stats["ENS"]
    di = rows_df["ti"].values
    y = rows_df["y"].values
    # conformal on ENS 10-90 interval
    Qs, Qa, w, _ = conformal(di, ens["q10"].values, ens["q90"].values, y, h)
    rows_df["ens_lo"], rows_df["ens_hi"] = ens["q10"].values, ens["q90"].values
    rows_df["ens_Qs"], rows_df["ens_Qa"] = Qs, Qa
    # QR + CQR
    lo, med, hi, cal_min = run_qr(rows_df, h)
    rows_df["qr_lo"], rows_df["qr_med"], rows_df["qr_hi"] = lo, med, hi
    ok = np.isfinite(lo)
    Qs2 = np.full(len(di), np.nan)
    Qa2 = np.full(len(di), np.nan)
    if ok.any():
        sel = np.where(ok)[0]
        a_, b_, _, _ = conformal(di[sel], lo[sel], hi[sel], y[sel], h, cal_min_di=cal_min, min_cal=300)
        Qs2[sel], Qa2[sel] = a_, b_
    rows_df["qr_Qs"], rows_df["qr_Qa"] = Qs2, Qa2
    with open(SCR / f"r9mc_cache_h{h}.pkl", "wb") as f:
        pickle.dump(dict(rows=rows_df, stats=stats, q19=q19, plog=plog, h=h), f)
    rep = calib_report(rows_df, stats, q19, h)
    pd.set_option("display.width", 250)
    print(rep.round(4).to_string())
    rep.to_csv(f"F:/apac-trading-hackathon/work/out_research9_mc_calib_dist_h{h}.csv")
    # coverage table for conformal / ACI / QR, by year
    dates = DC.index[di]
    ev = (dates >= pd.Timestamp("2023-10-01", tz="UTC")) & np.isfinite(y)
    recs = []
    for nm, lo_, hi_, Qs_, Qa_ in [("ENS", rows_df.ens_lo.values, rows_df.ens_hi.values, Qs, Qa),
                                   ("QR", lo, hi, Qs2, Qa2)]:
        w_ = np.maximum(hi_ - lo_, 1e-6)
        for meth, Q in [("raw", np.zeros(len(di))), ("split", Qs_), ("aci", Qa_)]:
            m_ = ev & np.isfinite(Q) & np.isfinite(lo_)
            cov = (y >= lo_ - Q * w_) & (y <= hi_ + Q * w_)
            lowcov = y >= lo_ - Q * w_
            for yr in ["all", 2023, 2024, 2025, 2026]:
                mm = m_ if yr == "all" else m_ & (dates.year == yr)
                if mm.sum() > 50:
                    recs.append(dict(model=nm, method=meth, year=yr, n=int(mm.sum()), cov80=cov[mm].mean(),
                                     lower_onesided_cov90=lowcov[mm].mean(), avg_width=(w_ * (1 + 2 * np.nan_to_num(Q)))[mm].mean()))
    cv = pd.DataFrame(recs)
    print(cv.round(3).to_string())
    cv.to_csv(f"F:/apac-trading-hackathon/work/out_research9_mc_coverage_h{h}.csv", index=False)
    plog.to_csv(f"F:/apac-trading-hackathon/work/out_research9_mc_params_h{h}.csv", index=False)
