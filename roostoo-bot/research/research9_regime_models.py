"""Research 9: probabilistic regime models, strictly walk-forward / causal.
Output: out_research9_regime_post.csv  -- daily P(bull) per model, index = UTC day d, value known at close of day d
(i.e. usable for the 00:00 UTC execution of day d+1).  Daily BTC history pre-2023 comes from
btc_daily_pre2023_research9.csv (Binance public klines, used only as model warm-up).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import stdtr, ndtr, gammaln
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent


def btc_daily():
    pre = pd.read_csv(HERE / "btc_daily_pre2023_research9.csv")
    pre.index = pd.to_datetime(pre["open_time"], unit="ms", utc=True)
    pre = pre["close"].astype(float)
    pre = pre[pre.index < "2023-01-01"]
    d = pd.read_csv(HERE / "data/klines_1h/BTC.csv.gz")
    d.index = pd.to_datetime(d["open_time"], unit="ms", utc=True)
    h = d["close"]
    day = h.resample("1D").last()
    cnt = h.resample("1D").count()
    day = day[cnt == 24]  # complete days only
    return pd.concat([pre, day]).sort_index()


def breadth_features(C, POOL):
    """daily (sampled at 23:00 bar close) breadth over the point-in-time pool."""
    Cd = C.resample("1D").last()
    ema7 = Cd.ewm(span=7, adjust=False).mean()
    pool_d = POOL.resample("1D").last().astype(bool)
    b1 = ((Cd > ema7) & pool_d).sum(axis=1) / pool_d.sum(axis=1).replace(0, np.nan)
    r14 = Cd / Cd.shift(14) - 1
    b2 = ((r14 > 0) & pool_d).sum(axis=1) / pool_d.sum(axis=1).replace(0, np.nan)
    return pd.DataFrame({"br_ema": b1, "br_r14": b2})


# ------------------------------------------------------------------ (a) HMM
def hmm_features(px):
    r = np.log(px).diff()
    v = r.rolling(10).std()
    sd = r.rolling(250, min_periods=60).std().shift(1)
    rc = r.clip(-4 * sd, 4 * sd)  # winsorised: crude robustness stand-in for Student-t tails
    return pd.DataFrame({"r": rc, "lv": np.log(v)}).dropna()


def forward_filter(model, X):
    logB = model._compute_log_likelihood(X)
    A = model.transmat_
    pi = model.startprob_
    T, K = logB.shape
    out = np.zeros((T, K))
    m = logB.max(axis=1, keepdims=True)
    B = np.exp(logB - m)
    a = pi * B[0]
    a /= a.sum()
    out[0] = a
    for t in range(1, T):
        a = (a @ A) * B[t]
        s = a.sum()
        a = a / s if s > 0 else np.ones(K) / K
        out[t] = a
    return out


def hmm_walk(px, K, refit_dates, n_init=3):
    from hmmlearn.hmm import GaussianHMM
    F = hmm_features(px)
    res_bull = pd.Series(np.nan, index=F.index)
    res_bear = pd.Series(np.nan, index=F.index)
    for i, d0 in enumerate(refit_dates):
        d1 = refit_dates[i + 1] if i + 1 < len(refit_dates) else F.index[-1] + pd.Timedelta(days=1)
        tr = F[F.index < d0]  # data up to day before refit date
        mu, sdv = tr.mean(), tr.std()
        Xtr = ((tr - mu) / sdv).values
        best, bl = None, -1e18
        for s in range(n_init):
            m = GaussianHMM(K, covariance_type="full", n_iter=200, tol=1e-4, random_state=s, min_covar=1e-3)
            try:
                m.fit(Xtr)
                ll = m.score(Xtr)
            except Exception:
                continue
            if ll > bl:
                best, bl = m, ll
        if best is None:
            continue
        order = np.argsort(best.means_[:, 0])  # sort by mean return: last = bull, first = bear
        full = F[F.index < d1]
        post = forward_filter(best, ((full - mu) / sdv).values)
        seg = full.index >= d0
        res_bull.loc[full.index[seg]] = post[seg][:, order[-1]]
        res_bear.loc[full.index[seg]] = post[seg][:, order[0]]
    return res_bull, res_bear


# ------------------------------------------------------------------ (b) Bayesian drift
def bayes_discounted(px, tau):
    r = np.log(px).diff()
    var = (r ** 2).ewm(span=30, min_periods=20).mean().shift(1)  # predictive variance, known before day t
    delta = 1 - 1 / tau
    prec, m = 1e-6, 0.0
    out = []
    for rt, v in zip(r.values, var.values):
        if np.isnan(rt) or np.isnan(v):
            out.append(np.nan)
            continue
        p0 = delta * prec
        prec = p0 + 1 / v
        m = (p0 * m + rt / v) / prec
        out.append(float(ndtr(m * np.sqrt(prec))))
    return pd.Series(out, index=px.index)


def bocpd(px, hazard=1 / 60, rmax=300):
    r = np.log(px).diff()
    sig = r.ewm(span=30, min_periods=20).std().shift(1)
    x = (r / sig).clip(-6, 6)
    mu0, k0, a0, b0 = 0.0, 0.2, 1.0, 1.0
    mu = np.array([mu0]); ka = np.array([k0]); al = np.array([a0]); be = np.array([b0])
    R = np.array([1.0])
    out = []
    for xt in x.values:
        if np.isnan(xt):
            out.append(np.nan)
            continue
        df = 2 * al
        sc2 = be * (ka + 1) / (al * ka)
        z2 = (xt - mu) ** 2 / sc2
        logp = gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(df * np.pi * sc2) - (df + 1) / 2 * np.log1p(z2 / df)
        pred = np.exp(logp)
        g = R * pred * (1 - hazard)
        cp = (R * pred * hazard).sum()
        Rn = np.concatenate([[cp], g])
        s = Rn.sum()
        Rn = Rn / s if s > 0 else np.concatenate([[1.0], np.zeros(len(g))])
        mu_n = (ka * mu + xt) / (ka + 1)
        be_n = be + ka * (xt - mu) ** 2 / (2 * (ka + 1))
        mu = np.concatenate([[mu0], mu_n]); ka = np.concatenate([[k0], ka + 1])
        al = np.concatenate([[a0], al + 0.5]); be = np.concatenate([[b0], be_n])
        R = Rn
        if len(R) > rmax:  # truncate oldest run lengths
            R = R[:rmax] / R[:rmax].sum(); mu, ka, al, be = mu[:rmax], ka[:rmax], al[:rmax], be[:rmax]
        z = mu / np.sqrt(be / (al * ka))
        out.append(float((R * stdtr(2 * al, z)).sum()))
    return pd.Series(out, index=px.index)


# ------------------------------------------------------------------ (c) logistic
def btc_feats(px):
    r = np.log(px).diff()
    e7, e28 = px.ewm(span=7, adjust=False).mean(), px.ewm(span=28, adjust=False).mean()
    rv30 = r.rolling(30).std()
    f = pd.DataFrame({
        "trend": np.log(e7 / e28) / rv30,
        "mom30": np.log(px / px.shift(30)) / (rv30 * np.sqrt(30)),
        "lrv30": np.log(rv30),
        "vr": np.log(r.rolling(7).std() / rv30),
        "dd90": px / px.rolling(90).max() - 1,
    })
    return f


def logit_walk(F, y_fwd, label_lag, refit_dates, min_train=250, Creg=0.3):
    """F features (daily), y_fwd label for day d (known only at d+label_lag days). Expanding monthly refit."""
    out = pd.Series(np.nan, index=F.index)
    Fv = F.dropna()
    for i, d0 in enumerate(refit_dates):
        d1 = refit_dates[i + 1] if i + 1 < len(refit_dates) else F.index[-1] + pd.Timedelta(days=1)
        lab_ok = Fv.index <= d0 - pd.Timedelta(days=label_lag + 1)  # label fully realised before d0
        idx = Fv.index[lab_ok]
        yy = y_fwd.reindex(idx).dropna()
        if len(yy) < min_train or yy.nunique() < 2:
            continue
        Xtr = Fv.loc[yy.index]
        mu, sd = Xtr.mean(), Xtr.std()
        m = LogisticRegression(C=Creg, max_iter=500).fit(((Xtr - mu) / sd).values, (yy > 0).astype(int).values)
        seg = Fv.index[(Fv.index >= d0) & (Fv.index < d1)]
        out.loc[seg] = m.predict_proba(((Fv.loc[seg] - mu) / sd).values)[:, 1]
    return out


# ------------------------------------------------------------------ jump model
def jm_features(px):
    r = np.log(px).diff()
    dn = r.clip(upper=0)
    f = pd.DataFrame({
        "ew5": r.ewm(halflife=5).mean(),
        "ew21": r.ewm(halflife=21).mean(),
        "dd10": np.sqrt((dn ** 2).ewm(halflife=10).mean()),
        "dd21": np.sqrt((dn ** 2).ewm(halflife=21).mean()),
        "ddn": px / px.rolling(60).max() - 1,
    })
    return f.dropna()


def dp_path(X, cent, lam):
    """Viterbi with jump penalty. Returns best final state path and forward costs V[T-1]."""
    T = len(X)
    K = len(cent)
    cost = ((X[:, None, :] - cent[None]) ** 2).sum(-1)  # T x K
    pen = lam * (1 - np.eye(K))
    V = np.zeros((T, K)); bp = np.zeros((T, K), int)
    V[0] = cost[0]
    for t in range(1, T):
        tot = V[t - 1][:, None] + pen  # from i to j
        bp[t] = tot.argmin(0)
        V[t] = cost[t] + tot.min(0)
    s = np.zeros(T, int); s[-1] = V[-1].argmin()
    for t in range(T - 1, 0, -1):
        s[t - 1] = bp[t, s[t]]
    return s, V[-1]


def jm_fit(X, lam, K=2, iters=15):
    # init: split on ew21 sign-ish via quantiles
    q = np.quantile(X[:, 1], np.linspace(0.25, 0.75, K))
    cent = np.array([X[np.abs(X[:, 1] - qq).argmin()] for qq in q])
    for _ in range(iters):
        s, _ = dp_path(X, cent, lam)
        new = np.array([X[s == k].mean(0) if (s == k).sum() > 0 else cent[k] for k in range(K)])
        if np.allclose(new, cent):
            break
        cent = new
    return cent


def jm_walk(px, lam, refit_dates, T_soft=2.0, win=250):
    F = jm_features(px)
    hard = pd.Series(np.nan, index=F.index); soft = hard.copy()
    for i, d0 in enumerate(refit_dates):
        d1 = refit_dates[i + 1] if i + 1 < len(refit_dates) else F.index[-1] + pd.Timedelta(days=1)
        tr = F[F.index < d0]
        mu, sd = tr.mean(), tr.std()
        cent = jm_fit(((tr - mu) / sd).values, lam)
        bull = int(np.argmax(cent[:, 1]))  # state with higher ew21 return
        for d in F.index[(F.index >= d0) & (F.index < d1)]:
            Xw = ((F[F.index <= d].iloc[-win:] - mu) / sd).values  # online: only data <= d
            s, Vl = dp_path(Xw, cent, lam)
            hard.loc[d] = float(s[-1] == bull)
            dv = Vl[bull] - Vl[1 - bull]
            soft.loc[d] = 1 / (1 + np.exp(np.clip(dv / T_soft, -30, 30)))
    return hard, soft


def main():
    px = btc_daily()
    print("BTC daily", px.index[0], px.index[-1], len(px), flush=True)
    refit = list(pd.date_range(pd.Timestamp("2023-04-01", tz="UTC"), px.index[-1], freq="MS"))
    if refit[0] != pd.Timestamp("2023-04-01", tz="UTC"):
        refit = [pd.Timestamp("2023-04-01", tz="UTC")] + refit
    refit = sorted(set(refit))
    P = {}
    for K in (2, 3):
        b, br = hmm_walk(px, K, refit)
        P[f"hmm{K}"] = b; P[f"hmm{K}_bear"] = br
        print("hmm", K, "done", flush=True)
    for tau in (30, 60, 120):
        P[f"bayes{tau}"] = bayes_discounted(px, tau)
    P["bocpd"] = bocpd(px)
    print("bayes/bocpd done", flush=True)
    for lam in (10, 30, 100):
        h, s = jm_walk(px, lam, refit)
        P[f"jm{lam}h"] = h; P[f"jm{lam}s"] = s
        print("jm", lam, "done", flush=True)
    # logistic (c)
    f = btc_feats(px)
    yb = px.shift(-14) / px - 1
    P["logit_btc"] = logit_walk(f, yb, 14, refit)
    # breadth + momentum-basket target (needs 2023+ panel)
    import research1 as r1
    C, POOL = r1.C, r1.POOL
    brd = breadth_features(C, POOL)
    brd.index = brd.index.tz_convert("UTC") if brd.index.tz is not None else brd.index.tz_localize("UTC")
    ret = C / C.shift(336) - 1
    rk = ret.where(POOL).rank(axis=1, ascending=False)
    sel = ((rk <= 3) & (ret > 0)).astype(float)
    w = sel.div(sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 1.0
    wd = w[w.index.hour == 23]
    wd.index = wd.index.normalize()
    Cd23 = C[C.index.hour == 23]
    # basket 14d fwd return: hold daily-rebalanced momentum basket, executed at next 00:00 (= close of 23:00 bar)
    Cn = C.resample("1D").last()
    dr = Cn.pct_change()
    basket_d = (wd.shift(0) * dr.shift(-1)).sum(axis=1)  # weights chosen end of d, earn day d+1
    fb = basket_d[::-1].rolling(14).sum()[::-1].shift(-1)  # sum over d+1..d+14... (approx 14d fwd basket)
    fb = fb.shift(-0)
    # fb at d = sum_{j=0..13} basket_d[d+j]? keep simple & causal-safe: label date = d, realised by d+15
    fb = pd.Series(basket_d[::-1].rolling(14).sum()[::-1].values, index=basket_d.index)
    F2 = f.join(brd, how="left")
    F2.loc[F2.index < "2023-01-01", ["br_ema", "br_r14"]] = np.nan
    m2 = logit_walk(F2[["trend", "mom30", "lrv30", "vr", "dd90", "br_ema", "br_r14"]].dropna(), fb, 15, refit, min_train=250)
    P["logit_mom"] = m2.reindex(px.index)
    df = pd.DataFrame(P)
    df.to_csv(HERE / "out_research9_regime_post.csv")
    print(df.loc["2023-04-01":].describe().round(3).T)


if __name__ == "__main__":
    main()
