"""Research 9 (MC) trading layer: rank/size v1-style top-3 BTC-gated rotation by distribution properties.
Reads cache from research9_mc.py (walk-forward distributions + conformal bounds).  Also CTREND-style elastic-net composite.
Usage: python research9_mc_trade.py [h=7]
"""
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import ElasticNet

import research1 as r1
import research9_mc as mc
from lib import *

warnings.filterwarnings("ignore")
DC, POOLD, GATED, C = mc.DC, mc.POOLD, mc.GATED, mc.C
T, N = DC.shape
EVAL0 = pd.Timestamp("2023-10-01", tz="UTC")
K = 3
COST = 0.0012


def piv(rows, vals):
    s = pd.Series(vals, index=pd.MultiIndex.from_arrays([DC.index[rows.ti.values], DC.columns[rows.ci.values]]))
    return s.unstack().reindex(index=DC.index, columns=DC.columns)


def gauss_rank(df):
    r = df.rank(axis=1, pct=True)
    n = df.notna().sum(axis=1).values[:, None]
    x = (r * n - 0.5) / n
    return pd.DataFrame(norm.ppf(x.clip(0.001, 0.999)), index=df.index, columns=df.columns).where(df.notna())


# ------------------------------------------------------------------ CTREND-style composite
def ctrend_features():
    DQ = r1.QV.resample("1D").sum().reindex(DC.index)
    lr = np.log(DC).diff()
    F = {}
    for k in [3, 5, 10, 20, 50, 100]:
        F[f"ma{k}"] = DC / DC.rolling(k).mean() - 1
    for k in [3, 5, 10, 20, 50]:
        F[f"vma{k}"] = np.log(DQ.rolling(k).mean() / DQ.rolling(100).mean())
    for k in [1, 3, 7, 14, 30]:
        F[f"ret{k}"] = np.log(DC / DC.shift(k))
    for k in [7, 14]:
        d = DC.diff()
        up = d.clip(lower=0).rolling(k).mean()
        dn = (-d.clip(upper=0)).rolling(k).mean()
        F[f"rsi{k}"] = up / (up + dn)
    e12, e26 = DC.ewm(span=12, adjust=False).mean(), DC.ewm(span=26, adjust=False).mean()
    macd = (e12 - e26) / DC
    F["macd"] = macd
    F["macd_h"] = macd - macd.ewm(span=9, adjust=False).mean()
    m20, s20 = DC.rolling(20).mean(), DC.rolling(20).std()
    F["pctb"] = (DC - (m20 - 2 * s20)) / (4 * s20)
    F["stoch"] = (DC - DC.rolling(14).min()) / (DC.rolling(14).max() - DC.rolling(14).min())
    for k in [7, 14, 30]:
        F[f"std{k}"] = lr.rolling(k).std()
    F["skew30"] = lr.rolling(30).skew()
    F["max7"] = lr.rolling(7).max()
    br = lr["BTC"]
    F["beta60"] = lr.rolling(60).cov(br).div(br.rolling(60).var(), axis=0)
    return F


def ctrend_scores(h, alpha=0.005, l1=0.5, first="2023-10-01"):
    F = ctrend_features()
    X = {k: gauss_rank(v.where(POOLD)) for k, v in F.items()}
    fwd = (DC.shift(-h) / DC - 1).where(POOLD)
    y = gauss_rank(fwd)
    Xs = pd.concat({k: v.stack() for k, v in X.items()}, axis=1)
    df = Xs.join(y.stack().rename("y"))
    feats = [c for c in df.columns if c != "y"]
    dts = df.index.get_level_values(0)
    out = []
    for m0 in pd.date_range(pd.Timestamp(first, tz="UTC"), DC.index[-1], freq="MS"):
        m1 = m0 + pd.offsets.MonthBegin(1)
        tr = df[(dts + pd.Timedelta(days=h) <= m0 - pd.Timedelta(days=1))].dropna()
        te = df[(dts >= m0) & (dts < m1)].dropna(subset=feats)
        if len(tr) < 2000 or not len(te):
            continue
        mdl = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000).fit(tr[feats].values, tr["y"].values)
        out.append(pd.Series(mdl.predict(te[feats].values), index=te.index))
    return pd.concat(out).unstack().reindex(index=DC.index, columns=DC.columns)


# ------------------------------------------------------------------ portfolio construction
HOURLY_DEC = C.index[C.index.hour == 23]


def to_hourly(Wd):
    """Wd: daily (label d) target weights decided at close of the 23:00 bar; hold to next decision."""
    W = pd.DataFrame(np.nan, index=C.index, columns=C.columns)
    Wd2 = Wd.copy()
    Wd2.index = HOURLY_DEC[:len(Wd2)]
    W.loc[Wd2.index, Wd2.columns] = Wd2.values
    return W.ffill().fillna(0.0)


def build_weights(score, valid=None, size=None, rel_size=None, cap=0.5):
    """Top-K of `score` among pool&valid, BTC gate. size: absolute per-name weight cap in [0,1/K] (cash otherwise).
    rel_size: relative weights, normalised to sum 1 over selected (cap per name)."""
    ok = POOLD & score.notna()
    if valid is not None:
        ok &= valid
    sc = score.where(ok)
    rk = sc.rank(axis=1, ascending=False)
    sel = (rk <= K) & ok
    sel = sel.mul(GATED.astype(float), axis=0).astype(bool)
    if rel_size is not None:
        r = rel_size.where(sel).clip(lower=0)
        w = r.div(r.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).clip(upper=cap)
    elif size is not None:
        w = sel.astype(float) * size.fillna(0.0).clip(0, 1.0 / K)
    else:
        w = sel.astype(float) / K
    return w


def evaluate(name, Wd, extra=None):
    eq, info = simulate(C, to_hourly(Wd), cost=COST)
    eq = eq[eq.index >= EVAL0]
    eq = eq / eq.iloc[0]
    win = rolling_windows(eq)
    s = summarize_windows(win)
    rec = dict(name=name, tot=eq.iloc[-1] - 1, mdd=max_drawdown(eq.values), mean14=s["mean_ret"], med14=s["med_ret"], p10=s["p10"],
               p90=s["p90"], pos=s["pos"], comp=s["mean_comp"], gross=info["avg_gross"], turn=info["turnover_per_day"])
    for yr in [2023, 2024, 2025, 2026]:
        seg = eq[eq.index.year == yr]
        rec[f"tot{yr}"] = seg.iloc[-1] / seg.iloc[0] - 1 if len(seg) > 24 else np.nan
        wy = win[win.index.year == yr]
        rec[f"m14_{yr}"] = wy.ret.mean() if len(wy) else np.nan
    if extra:
        rec.update(extra)
    return rec


def topbook_diag(Wd, fwd):
    """Mean realised h-day fwd return of selected names on gate-on selected days (pre-cost) and hit rate."""
    m = (Wd > 0) & fwd.notna()
    m = m.loc[m.index >= EVAL0]
    v = fwd.loc[m.index].where(m).stack().dropna() if False else fwd.loc[m.index].where(m).values.ravel()
    v = pd.Series(v).dropna()
    return dict(sel_fwd_mean=v.mean(), sel_fwd_med=v.median(), sel_hit=(v > 0).mean(), n_sel=len(v))


if __name__ == "__main__":
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    d = pickle.load(open(mc.SCR / f"r9mc_cache_h{h}.pkl", "rb"))
    rows, stats = d["rows"], d["stats"]
    ens = stats["ENS"]
    fwd = (DC.shift(-h) / DC - 1)
    mom = DC / DC.shift(14) - 1  # == 336h return at 00:00 UTC
    S = {}
    S["ens_mean"] = piv(rows, ens["mean"].values)
    S["ens_gstar"] = piv(rows, ens["gstar"].values)
    S["ens_pcost"] = piv(rows, ens["pcost"].values)
    S["ens_mean_over_es"] = piv(rows, (ens["mean"] / ens["es5"].abs().clip(lower=1e-3)).values)
    S["ens_q10"] = piv(rows, ens["q10"].values)
    w_ = (rows.ens_hi - rows.ens_lo).clip(lower=1e-6).values
    S["ens_lo_split"] = piv(rows, rows.ens_lo.values - rows.ens_Qs.values * w_)
    S["ens_lo_aci"] = piv(rows, rows.ens_lo.values - rows.ens_Qa.values * w_)
    wq = (rows.qr_hi - rows.qr_lo).clip(lower=1e-6).values
    S["qr_med"] = piv(rows, rows.qr_med.values)
    S["qr_lo_aci"] = piv(rows, rows.qr_lo.values - rows.qr_Qa.values * wq)
    S["qr_lo_split"] = piv(rows, rows.qr_lo.values - rows.qr_Qs.values * wq)
    W_aci = piv(rows, w_ * (1 + 2 * rows.ens_Qa.values))  # conformal 80% interval width (ensemble)
    LO_aci = S["ens_lo_aci"]
    CTR = ctrend_scores(h)
    S["ctrend_enet"] = CTR
    fin = lambda x: x.notna()
    # restrict everything to dates where all scores exist (common OOS sample) via EVAL0 slice in evaluate()
    res = []

    def add(name, Wd):
        r = evaluate(name, Wd, topbook_diag(Wd, fwd))
        res.append(r)
        print(f"{name:32s} tot={r['tot']:+.3f} mean14={r['mean14']:+.4f} med={r['med14']:+.3f} p10={r['p10']:+.3f} pos={r['pos']:.2f} "
              f"comp={r['comp']:+.2f} mdd={r['mdd']:.3f} gross={r['gross']:.2f}", flush=True)

    # --- baselines
    add("v1_raw336_top3", build_weights(mom, valid=mom > 0))
    # reproduction check with research1 helper
    eqb, _ = simulate(C, r1.xsmom_topk(336, 3), cost=COST)
    eqb = eqb[eqb.index >= EVAL0]
    eqb = eqb / eqb.iloc[0]
    sb = summarize_windows(rolling_windows(eqb))
    res.append(dict(name="v1_r1_helper_check", tot=eqb.iloc[-1] - 1, mdd=max_drawdown(eqb.values), mean14=sb["mean_ret"], med14=sb["med_ret"],
                    p10=sb["p10"], p90=sb["p90"], pos=sb["pos"], comp=sb["mean_comp"]))
    rng = np.random.default_rng(5)
    rr = []
    for sd in range(12):
        rnd = pd.DataFrame(rng.random((T, N)), index=DC.index, columns=DC.columns)
        rr.append(evaluate("rnd", build_weights(rnd)))
    rrd = pd.DataFrame(rr).mean(numeric_only=True)
    rrd["name"] = "random_top3_gated_mean12"
    res.append(rrd.to_dict())
    # --- ranking by distribution properties
    for nm in ["ens_mean", "ens_gstar", "ens_pcost", "ens_mean_over_es", "ens_q10", "ens_lo_split", "ens_lo_aci", "qr_med",
               "qr_lo_aci", "qr_lo_split", "ctrend_enet"]:
        s = S[nm]
        add(f"rank_{nm}", build_weights(s))
        add(f"rank_{nm}+mom>0", build_weights(s, valid=mom > 0))
    # --- risk layer: v1 selection, skip entries with poor conformal lower bound
    for thr in [-0.15, -0.20, -0.25]:
        add(f"v1_skip_loACI<{thr}", build_weights(mom, valid=(mom > 0) & (LO_aci > thr)))
    add("v1_skip_QRloACI<-0.20", build_weights(mom, valid=(mom > 0) & (S["qr_lo_aci"] > -0.20)))
    # v1 selection but require positive conformal-ish quality: ens mean>0
    add("v1_require_ensmean>0", build_weights(mom, valid=(mom > 0) & (S["ens_mean"] > 0)))
    # --- sizing on v1 selection
    fstar = piv(rows, ens["fstar"].values)
    add("v1_size_kelly0.5", build_weights(mom, valid=mom > 0, size=0.5 * fstar))
    add("v1_size_kelly1.0", build_weights(mom, valid=mom > 0, size=fstar))
    sig_c = W_aci / (2 * 1.2816)
    # conformal Kelly: f = lambda * mu / sigma_c^2, mu = momentum-drift mean forecast; relative and absolute versions
    mu = S["ens_mean"].clip(lower=0)
    add("v1_condKelly_rel", build_weights(mom, valid=mom > 0, rel_size=mu / sig_c ** 2))
    add("v1_condKelly_abs_0.5", build_weights(mom, valid=mom > 0, size=0.5 * mu / sig_c ** 2))
    add("v1_condKelly_abs_2", build_weights(mom, valid=mom > 0, size=2.0 * mu / sig_c ** 2))
    add("v1_invConfWidth_rel", build_weights(mom, valid=mom > 0, rel_size=1.0 / W_aci))
    # momentum-signal Kelly: mu = k * 14d vol-scaled momentum return share (use ens drift is tiny) -> use mom/ (sig_c*sqrt(14/h)) style
    sig_naive = np.sqrt(mc.LR.rolling(60, min_periods=30).std() ** 2 * h)
    add("v1_invVol60_rel", build_weights(mom, valid=mom > 0, rel_size=1.0 / sig_naive))
    # exposure-matched cash-dilution control for kelly0.5
    g = res[-8]
    # --- ensemble generators individually ranked by mean (sanity)
    for gk in ["A", "B", "C", "D", "E"]:
        pass
    out = pd.DataFrame(res).set_index("name")
    # exposure-matched control: v1 scaled by avg gross of the kelly0.5 variant
    for ref in ["v1_size_kelly0.5", "v1_condKelly_abs_0.5"]:
        gx = float(out.loc[ref, "gross"]) / float(out.loc["v1_raw336_top3", "gross"])
        add(f"v1_scaled_to_{ref}_gross", build_weights(mom, valid=mom > 0, size=pd.DataFrame(gx / K, index=DC.index, columns=DC.columns)))
    out = pd.DataFrame(res).set_index("name")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    out.to_csv(f"F:/apac-trading-hackathon/work/out_research9_mc_trade_h{h}.csv")
    print(out.round(4).to_string())
