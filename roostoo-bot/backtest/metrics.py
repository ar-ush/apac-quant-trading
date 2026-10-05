"""Competition metrics: Sortino / Sharpe / Calmar on UTC-daily returns (365-day annualisation, 0 risk-free) and the
composite 0.4*Sortino + 0.3*Sharpe + 0.3*Calmar. The organisers have not published their exact conventions, so the
conventions are explicit here and ratios are capped to stop a single degenerate window dominating averages."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 365.0


def daily_returns(equity: pd.Series) -> pd.Series:
    eod = equity.resample("1D").last().dropna()
    prev = np.concatenate([[equity.iloc[0]], eod.values[:-1]])
    return pd.Series(eod.values / prev - 1.0, index=eod.index)


def max_drawdown(values: np.ndarray) -> float:
    peak = np.maximum.accumulate(values)
    return float(np.max(1.0 - values / peak)) if len(values) else 0.0


def ratios(equity: pd.Series) -> dict:
    r = daily_returns(equity).values
    e = equity.values
    tot = e[-1] / e[0] - 1.0
    mdd = max_drawdown(e)
    if len(r) < 2:
        return dict(sortino=np.nan, sharpe=np.nan, calmar=np.nan, comp=np.nan, ret=tot, mdd=mdd)
    mu, sd = r.mean(), r.std(ddof=1)
    dd = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    sharpe = mu / sd * np.sqrt(ANN) if sd > 0 else (np.inf if mu > 0 else 0.0)
    sortino = mu / dd * np.sqrt(ANN) if dd > 0 else (np.inf if mu > 0 else 0.0)
    days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86400.0, 1.0)
    ann = (1 + tot) ** (ANN / days) - 1 if tot > -1 else -1.0
    calmar = ann / mdd if mdd > 1e-9 else (np.inf if ann > 0 else 0.0)
    cap = lambda x, c: float(np.clip(x, -c, c))
    sortino, sharpe, calmar = cap(sortino, 60), cap(sharpe, 40), cap(calmar, 200)
    return dict(sortino=sortino, sharpe=sharpe, calmar=calmar, comp=0.4 * sortino + 0.3 * sharpe + 0.3 * calmar,
                ret=tot, mdd=mdd)


def rolling_windows(equity: pd.Series, days: int = 14, step_days: int = 1) -> pd.DataFrame:
    rows = []
    t = equity.index[0].normalize() + pd.Timedelta(days=1)
    while t + pd.Timedelta(days=days) <= equity.index[-1]:
        seg = equity.loc[t: t + pd.Timedelta(days=days)]
        m = ratios(seg / seg.iloc[0])
        m["start"] = t
        rows.append(m)
        t += pd.Timedelta(days=step_days)
    return pd.DataFrame(rows).set_index("start")


def summarize(w: pd.DataFrame) -> dict:
    return dict(windows=len(w), mean_ret=w.ret.mean(), median_ret=w.ret.median(), p10=w.ret.quantile(.1),
                p90=w.ret.quantile(.9), pos=(w.ret > 0).mean(), gt5=(w.ret > .05).mean(),
                median_mdd=w.mdd.median(), p90_mdd=w.mdd.quantile(.9), mean_comp=w.comp.mean(), median_comp=w.comp.median())
