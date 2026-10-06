"""research11_wf: walk-forward / out-of-sample validation of v1's parameter selection + cost sensitivity.
Grid is evaluated once on the full history; selection uses ONLY windows that end before the test period starts."""
import itertools
from multiprocessing import Pool
import numpy as np, pandas as pd
from research9_alt_lib import *
from lib import rolling_windows, summarize_windows

GRID = list(itertools.product([168, 240, 336, 504, 672], [2, 3, 5], [(48, 336), (120, 480), (168, 672)], [0, 2]))

def run_cfg(a, cost=0.0012):
    lb, k, (gf, gs), hy = a
    gate = gate_series(gf, gs)
    ML, WL, _ = v1_inputs(lb, k, hy, gate)
    eq, _ = run(ML=ML, WL=WL, regime=np.where(gate, 1, 0), cost_l=cost)
    return eq

def job(a):
    return a, run_cfg(a)

def win_stats(eq, t0, t1):
    w = rolling_windows(eq)
    w = w[(w.index >= t0) & (w.index < t1)]
    return dict(n=len(w), mean=w.ret.mean(), med=w.ret.median(), p10=w.ret.quantile(.1), p90=w.ret.quantile(.9),
                pos=(w.ret > 0).mean(), comp=w.comp.mean() if "comp" in w else np.nan)

if __name__ == "__main__":
    with Pool(8) as p:
        res = dict(p.map(job, GRID))
    eqs = res
    v1 = (336, 3, (168, 672), 2)
    # BTC hold + equal-weight-pool benchmarks (same windows)
    btc = (BTC / BTC[BTC.index >= START].iloc[0]); btc = btc[btc.index >= START]
    w0 = rolling_windows(eqs[v1]); print("window index sample", w0.index[:2].tolist(), w0.columns.tolist())
    TS = lambda s: pd.Timestamp(s, tz="UTC")
    splits = [("2024-01-01", "2025-01-01"), ("2025-01-01", "2026-01-01"), ("2026-01-01", "2027-01-01")]
    rows = []
    for t0, t1 in splits:
        t0, t1 = TS(t0), TS(t1)
        # in-sample = windows starting before t0 - 14d (no overlap with the test period)
        best = None
        for a, e in eqs.items():
            w = rolling_windows(e); w = w[w.index < t0 - pd.Timedelta(days=14)]
            sc = w.ret.mean()
            if best is None or sc > best[0]: best = (sc, a)
        a = best[1]
        o = win_stats(eqs[a], t0, t1); f = win_stats(eqs[v1], t0, t1); b = win_stats(btc, t0, t1)
        rows.append(dict(test=f"{t0.date()}..", picked=str(a), IS_mean=best[0], **{f"WF_{k}": v for k, v in o.items()},
                         **{f"v1_{k}": v for k, v in f.items()}, **{f"btc_{k}": v for k, v in b.items()}))
    df = pd.DataFrame(rows); df.to_csv("out_research11_wf.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(df.round(3).T.to_string())
    # single split: pick on 2023-03..2024-06, test 2024-07..2026-10
    t0 = TS("2024-07-01")
    best = max(((rolling_windows(e).pipe(lambda w: w[w.index < t0 - pd.Timedelta(days=14)]).ret.mean(), a) for a, e in eqs.items()))
    print("single split picked", best[1], "IS mean", round(best[0], 4))
    for nm, e in [("picked", eqs[best[1]]), ("v1", eqs[v1]), ("btc_hold", btc)]:
        print(nm, {k: round(v, 3) for k, v in win_stats(e, t0, TS("2027-01-01")).items()})
    # rank of v1 within the grid, out-of-sample (2024-07+) vs in-sample
    oos = {a: win_stats(e, t0, TS("2027-01-01"))["mean"] for a, e in eqs.items()}
    ins = {a: win_stats(e, TS("2023-03-01"), t0 - pd.Timedelta(days=14))["mean"] for a, e in eqs.items()}
    s = pd.Series(oos); print("v1 OOS-mean rank in grid:", int(s.rank(ascending=False)[[v1]].iloc[0]), "of", len(s), "| grid OOS mean median", round(s.median(), 4), "max", round(s.max(), 4), "| fraction of grid with OOS mean>0:", round((s > 0).mean(), 2))
    ci = pd.Series(ins); print("corr(IS mean, OOS mean) across grid:", round(ci.corr(s), 3))
    # cost sensitivity for v1
    out = []
    for c in (0.0012, 0.0020, 0.0030, 0.0040):
        e = run_cfg(v1, c); w = rolling_windows(e)
        out.append(dict(cost_per_side=c, total=e.iloc[-1] - 1, mdd=max_drawdown(e.values), **{k: round(v, 3) for k, v in win_stats(e, TS("2023-01-01"), TS("2027-01-01")).items()}))
    print(pd.DataFrame(out).round(3).to_string())
