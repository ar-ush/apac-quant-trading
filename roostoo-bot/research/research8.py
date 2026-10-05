"""Research batch 8: Grossman-Zhou / CPPI drawdown-controlled exposure on top of the momentum rotation, evaluated per 14d window
(peak resets at each window start, as in the live competition)."""
import itertools

import numpy as np
import pandas as pd

import research1 as r1
import research4 as r4
from lib import *

C = r1.C


def base_series(k=3, lb=336, reg="ema168_672", rank="raw", cost=0.0012):
    w = r4.build(lb, k, 24, reg, rank)
    eq, info = simulate(C, w, cost=cost)
    r = eq.pct_change().fillna(0.0)
    g = w.shift(1).fillna(0.0).abs().sum(axis=1)
    return r, g


def cppi_window(r, g, alpha, m, cost=0.0012, floor_mode="gz", lock=None, lock_scale=0.25):
    """Simulate exposure scale on a window of strategy returns r (net of base costs) & gross exposure g."""
    eq = 1.0
    peak = 1.0
    s_prev = 1.0
    eqs = np.empty(len(r))
    for i in range(len(r)):
        cushion = 1.0 - (1.0 - alpha) * peak / eq  # fraction of equity above floor
        s = min(1.0, max(0.0, m * cushion))
        if lock is not None and eq - 1.0 >= lock:
            s = min(s, lock_scale)
        ret = s * r[i] - cost * abs(s - s_prev) * g[i]
        eq *= 1 + ret
        peak = max(peak, eq)
        s_prev = s
        eqs[i] = eq
    return eqs


def evaluate(r, g, params, days=14, step_h=24):
    n = days * 24
    comps, rets, mdds = [], [], []
    rv, gv = r.values, g.values
    idx = r.index
    for st in range(24 * 60, len(r) - n, step_h):
        seg = slice(st, st + n)
        if params is None:
            eq = np.cumprod(1 + rv[seg])
        else:
            eq = cppi_window(rv[seg], gv[seg], **params)
        e = pd.Series(np.concatenate([[1.0], eq]), index=[idx[st - 1]] + list(idx[seg]))
        m = ratios(e)
        comps.append(m["comp"]); rets.append(m["ret"]); mdds.append(m["mdd"])
    rets, comps, mdds = np.array(rets), np.array(comps), np.array(mdds)
    return dict(mean_ret=rets.mean(), med_ret=np.median(rets), p10=np.percentile(rets, 10), p90=np.percentile(rets, 90),
                pos=(rets > 0).mean(), gt5=(rets > .05).mean(), mdd_med=np.median(mdds), mdd_p90=np.percentile(mdds, 90),
                mcomp=comps.mean(), medcomp=np.median(comps), ret_over_mdd=rets.mean() / mdds.mean())


if __name__ == "__main__":
    rows = []
    for k in [2, 3, 4]:
        r, g = base_series(k=k)
        rows.append(dict(name=f"k{k} base", **evaluate(r, g, None)))
        for alpha, m in itertools.product([0.06, 0.10, 0.15], [3, 5, 8]):
            rows.append(dict(name=f"k{k} gz a{alpha} m{m}", **evaluate(r, g, dict(alpha=alpha, m=m))))
        for lock in [0.08, 0.15]:
            rows.append(dict(name=f"k{k} gz a0.10 m5 lock{lock}", **evaluate(r, g, dict(alpha=0.10, m=5, lock=lock))))
        print("done k", k, flush=True)
    pd.set_option("display.width", 250)
    df = pd.DataFrame(rows).set_index("name")
    df.to_csv("out_research8.csv")
    print(df.round(3).to_string())
