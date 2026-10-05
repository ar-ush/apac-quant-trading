"""Research batch 6: walk-forward probabilistic cross-sectional model (daily decisions at 00:00 UTC).

Features per (day, coin) -> cross-sectionally rank-gaussianised. Target = forward h-day return, cross-sectionally
rank-gaussianised. Models: Ridge, logistic P(top tercile), HistGradientBoosting. Walk-forward, monthly refit, purge of h days.
Output: OOS rank-IC and a top-k long-only portfolio simulated on the 1h panel with 12bps/side.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

import research1 as r1
from lib import *

warnings.filterwarnings("ignore")
C, QV, POOLH = r1.C, r1.QV, r1.POOL
TBQ = load_panel("1h")["tbq"][C.columns]
# daily decision points: hourly bar labelled 23:00 closes at 00:00 UTC
dec = C.index[C.index.hour == 23]
DC = C.loc[dec]
DC.index = DC.index.normalize()  # label = day that just ended
DQ = QV.resample("1D").sum()
DT = TBQ.resample("1D").sum()
POOLD = POOLH.loc[dec]
POOLD.index = POOLD.index.normalize()
BTC = DC["BTC"]


def gauss_rank(df):
    r = df.rank(axis=1, pct=True)
    n = df.notna().sum(axis=1).values[:, None]
    x = (r * n - 0.5) / n
    return pd.DataFrame(norm.ppf(x.clip(0.001, 0.999)), index=df.index, columns=df.columns).where(df.notna())


def features():
    F = {}
    r1d = DC.pct_change()
    vol14 = r1d.rolling(14).std()
    vol30 = r1d.rolling(30).std()
    for k in [1, 3, 7, 14, 30, 60]:
        rk = DC / DC.shift(k) - 1
        F[f"ret{k}"] = rk
        F[f"z{k}"] = rk / (vol30 * np.sqrt(k))
    for k in [3, 7, 14]:
        F[f"rel{k}"] = (DC / DC.shift(k) - 1).sub(BTC / BTC.shift(k) - 1, axis=0)
    F["vol14"] = vol14
    F["vol_ratio"] = r1d.rolling(5).std() / vol30
    F["dist_hi30"] = DC / DC.rolling(30).max() - 1
    F["dist_lo30"] = DC / DC.rolling(30).min() - 1
    F["volsurge"] = DQ / DQ.rolling(30).median()
    F["volsurge7"] = DQ.rolling(3).mean() / DQ.rolling(30).median()
    F["tb7"] = (DT / DQ).rolling(7).mean() - 0.5
    F["tb1"] = (DT / DQ) - 0.5
    F["skew30"] = r1d.rolling(30).skew()
    F["maxret7"] = r1d.rolling(7).max()
    F["minret7"] = r1d.rolling(7).min()
    F["logdv"] = np.log(DQ.rolling(30).mean())
    # beta to BTC (60d)
    br = BTC.pct_change()
    F["beta60"] = r1d.rolling(60).cov(br).div(br.rolling(60).var(), axis=0)
    return {k: v.reindex(DC.index) for k, v in F.items()}


def build_panel(h):
    F = features()
    X = {k: gauss_rank(v.where(POOLD)) for k, v in F.items()}
    fwd = (DC.shift(-h) / DC - 1).where(POOLD)
    y = gauss_rank(fwd)
    Xs = pd.concat({k: v.stack() for k, v in X.items()}, axis=1)
    ys = y.stack().rename("y")
    df = Xs.join(ys, how="left")
    df["raw"] = fwd.stack()
    return df


def walk_forward(h=3, start="2024-03-01", refit_days=30, model="ridge"):
    df = build_panel(h)
    feats = [c for c in df.columns if c not in ("y", "raw")]
    dates = df.index.get_level_values(0).unique().sort_values()
    preds = []
    cur = pd.Timestamp(start, tz="UTC")
    while cur <= dates[-1]:
        te_end = cur + pd.Timedelta(days=refit_days)
        tr = df[(df.index.get_level_values(0) < cur - pd.Timedelta(days=h + 1))].dropna()
        te = df[(df.index.get_level_values(0) >= cur) & (df.index.get_level_values(0) < te_end)]
        te = te.dropna(subset=feats)
        if len(tr) > 2000 and len(te):
            Xtr, ytr = tr[feats].values, tr["y"].values
            if model == "ridge":
                m = Ridge(alpha=200.0).fit(Xtr, ytr)
                p = m.predict(te[feats].values)
            elif model == "logit":
                m = LogisticRegression(C=0.05, max_iter=300).fit(Xtr, (ytr > 0.43).astype(int))
                p = m.predict_proba(te[feats].values)[:, 1]
            else:
                m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=10.0,
                                                  min_samples_leaf=100).fit(Xtr, ytr)
                p = m.predict(te[feats].values)
            preds.append(pd.Series(p, index=te.index))
        cur = te_end
    P = pd.concat(preds).unstack()
    return P, df


def ic_stats(P, df, h):
    raw = df["raw"].unstack()
    ics = []
    for d in P.index:
        a, b = P.loc[d].dropna(), raw.loc[d].dropna() if d in raw.index else None
        common = a.index.intersection(b.index) if b is not None else []
        if len(common) > 8:
            ics.append(spearmanr(a[common], b[common])[0])
    ics = np.array(ics)
    return ics.mean(), ics.std() / np.sqrt(len(ics) / h), (ics > 0).mean()


def to_weights(P, k=3, gate_on=True, h_hold=24, min_score=None):
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    gate = r1.C["BTC"].ewm(span=168, adjust=False).mean() > r1.C["BTC"].ewm(span=672, adjust=False).mean()
    for d in P.index:
        row = P.loc[d].dropna()
        if min_score is not None:
            row = row[row > min_score]
        top = row.sort_values(ascending=False).head(k)
        t = d + pd.Timedelta(hours=23)  # decision bar (label 23:00 of day d)
        if t not in W.index:
            continue
        if gate_on and not gate.loc[t]:
            continue
        W.loc[t, top.index] = 1.0 / k
    # hold until next decision: forward-fill only across non-decision bars
    dec_mask = W.index.isin([d + pd.Timedelta(hours=23) for d in P.index])
    W2 = W.copy()
    W2[~dec_mask] = np.nan
    return W2.ffill().fillna(0.0)


if __name__ == "__main__":
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    base_start = "2024-03-01"
    rows = []
    # momentum baseline on same OOS period (14d momentum top3 + gate)
    wb = r1.xsmom_topk(24 * 14, 3)
    eqb, _ = simulate(C, wb, cost=0.0012)
    eqb = eqb[eqb.index >= pd.Timestamp(base_start, tz="UTC")]
    eqb = eqb / eqb.iloc[0]
    sb = summarize_windows(rolling_windows(eqb))
    rows.append(dict(name="mom14_top3_gate", ic=np.nan, tot=eqb.iloc[-1] - 1, mdd=max_drawdown(eqb.values), mean14=sb["mean_ret"], p10=sb["p10"], pos=sb["pos"], mcomp=sb["mean_comp"]))
    for model in ["ridge", "logit", "gbm"]:
        P, df = walk_forward(h=h, start=base_start, model=model)
        ic, se, hit = ic_stats(P, df, h)
        for k, gate_on in [(3, True), (3, False), (5, True), (8, True)]:
            W = to_weights(P, k=k, gate_on=gate_on)
            eq, info = simulate(C, W, cost=0.0012)
            eq = eq[eq.index >= pd.Timestamp(base_start, tz="UTC")]
            eq = eq / eq.iloc[0]
            s = summarize_windows(rolling_windows(eq))
            rows.append(dict(name=f"{model}_h{h}_k{k}_gate{int(gate_on)}", ic=ic, ic_t=ic / se, tot=eq.iloc[-1] - 1, mdd=max_drawdown(eq.values),
                             turn=info["turnover_per_day"], mean14=s["mean_ret"], p10=s["p10"], pos=s["pos"], mcomp=s["mean_comp"]))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).set_index("name").round(4).to_string())
