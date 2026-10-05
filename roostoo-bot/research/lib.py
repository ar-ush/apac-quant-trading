"""Shared research utilities: data panel, competition metrics, weight-based simulator."""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ANN = 365.0


def load_panel(interval="1h", coins=None, start=None):
    d = HERE / f"data/klines_{interval}"
    fields = {k: {} for k in ["open", "high", "low", "close", "qv", "tbq"]}
    for f in sorted(d.glob("*.csv.gz")):
        c = f.name.split(".")[0]
        if coins is not None and c not in coins:
            continue
        df = pd.read_csv(f)
        idx = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for k, col in [("open", "open"), ("high", "high"), ("low", "low"), ("close", "close"),
                       ("qv", "quote_vol"), ("tbq", "taker_buy_quote")]:
            fields[k][c] = pd.Series(df[col].values, index=idx)
    out = {k: pd.DataFrame(v).sort_index() for k, v in fields.items()}
    if start:
        out = {k: v.loc[start:] for k, v in out.items()}
    return out


# ---------------------------------------------------------------- metrics
def daily_returns(equity: pd.Series) -> pd.Series:
    """UTC-day returns from an equity curve (last equity of each day)."""
    eod = equity.resample("1D").last().dropna()
    first = equity.iloc[0]
    prev = pd.concat([pd.Series([first]), eod.iloc[:-1]]).values
    return pd.Series(eod.values / prev - 1.0, index=eod.index)


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak)) if len(equity) else 0.0


def ratios(equity: pd.Series, mar=0.0):
    """Sortino, Sharpe, Calmar (annualised on daily returns, 365d) and the composite 0.4/0.3/0.3."""
    r = daily_returns(equity).values
    if len(r) < 2:
        return dict(sortino=np.nan, sharpe=np.nan, calmar=np.nan, comp=np.nan, ret=np.nan, mdd=np.nan)
    mu = r.mean()
    sd = r.std(ddof=1)
    dd = np.sqrt(np.mean(np.minimum(r - mar, 0.0) ** 2))
    sharpe = mu / sd * np.sqrt(ANN) if sd > 0 else (np.inf if mu > 0 else 0.0)
    sortino = (mu - mar) / dd * np.sqrt(ANN) if dd > 0 else (np.inf if mu > 0 else 0.0)
    e = equity.values
    mdd = max_drawdown(e)
    tot = e[-1] / e[0] - 1.0
    days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86400.0, 1.0)
    ann = (1 + tot) ** (ANN / days) - 1 if tot > -1 else -1.0
    calmar = ann / mdd if mdd > 1e-9 else (np.inf if ann > 0 else 0.0)
    # cap infs/huge values so a single window can't dominate averages
    def cap(x, c):
        return float(np.clip(x, -c, c))
    sortino, sharpe, calmar = cap(sortino, 60), cap(sharpe, 40), cap(calmar, 200)
    comp = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar
    return dict(sortino=sortino, sharpe=sharpe, calmar=calmar, comp=comp, ret=tot, mdd=mdd)


def rolling_windows(equity: pd.Series, days=14, step_days=1):
    """Score every rolling window (starting each `step_days`); returns DataFrame."""
    rows = []
    t0 = equity.index[0].normalize() + pd.Timedelta(days=1)
    end = equity.index[-1]
    t = t0
    while t + pd.Timedelta(days=days) <= end:
        seg = equity.loc[t: t + pd.Timedelta(days=days)]
        seg = seg / seg.iloc[0]
        m = ratios(seg)
        m["start"] = t
        rows.append(m)
        t += pd.Timedelta(days=step_days)
    return pd.DataFrame(rows).set_index("start")


def summarize_windows(w: pd.DataFrame):
    return dict(
        n=len(w),
        med_ret=w.ret.median(), mean_ret=w.ret.mean(), p10=w.ret.quantile(.1), p90=w.ret.quantile(.9),
        pos=(w.ret > 0).mean(), gt3=(w.ret > .03).mean(), gt5=(w.ret > .05).mean(),
        med_mdd=w.mdd.median(), p90_mdd=w.mdd.quantile(.9),
        med_comp=w.comp.median(), mean_comp=w.comp.mean(),
    )


# ---------------------------------------------------------------- simulator
def simulate(close: pd.DataFrame, weights: pd.DataFrame, cost=0.0012, lag=1):
    """Weight-based simulator. weights[t] are decided at close of bar t, executed at close of bar t
    (lag handled by shift: return of bar t+1 uses weights[t]). Cost = |dw| * cost (per side).
    Returns equity Series starting at 1.0 and turnover info."""
    w = weights.reindex(close.index).fillna(0.0)
    ret = close.pct_change().fillna(0.0)
    wl = w.shift(lag).fillna(0.0)
    gross = (wl * ret).sum(axis=1)
    dw = wl.diff().abs().fillna(wl.abs())
    tc = dw.sum(axis=1) * cost
    net = gross - tc
    eq = (1 + net).cumprod()
    return eq, dict(turnover_per_day=float(dw.sum(axis=1).sum() / (len(close) / 24)),
                    avg_gross=float(wl.abs().sum(axis=1).mean()))
