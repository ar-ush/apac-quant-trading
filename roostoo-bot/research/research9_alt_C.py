"""research9_alt_C: SECOND SLEEVE for the gate-off regime (v1 long sleeve unchanged in gate-on).
Candidates for the idle-cash regime: long BTC, long BTC/ETH, long low-vol alts, long short-term losers (reversal),
ungated momentum, and the A-style shorts (reference). Compared with v1 alone (cash when gate off)."""
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

from research9_alt_lib import *
import research9_alt_B as B

_G = {}
NEG = -1e9


def prep():
    if not _G:
        g = B.S()
        ML, WL = B.build(("v1",))
        _G.update(ML=ML, WL=WL, g=g)
        _G["ref"] = B.REF()
    return _G


def second(spec):
    g = prep()["g"]
    kind = spec[0]
    mom = g["ret"][336]
    if kind == "cash":
        return np.zeros((T, N), bool), np.zeros((T, N)), +1.0, 0.0
    if kind in ("btc", "btceth", "eth"):
        _, gross = spec
        cols = {"btc": ["BTC"], "btceth": ["BTC", "ETH"], "eth": ["ETH"]}[kind]
        M = np.zeros((T, N), bool); W = np.zeros((T, N))
        for c in cols:
            j = COLS.index(c); M[:, j] = True; W[:, j] = gross / len(cols)
        return M, W, +1.0, 0.0
    if kind == "lowvol":  # lowest 168h-vol names (k) in the pool
        _, k, gross = spec
        M = membership(-g["vol168"], k, 0, NEG, +1, reset=GATE)
        return M, np.where(M, gross / k, 0.0), +1.0, 0.0
    if kind == "reversal":  # buy worst h-hour performers
        _, h, k, gross = spec
        M = membership(-g["ret"][h], k, 0, NEG, +1, reset=GATE)
        return M, np.where(M, gross / k, 0.0), +1.0, 0.0
    if kind == "mom_ungated":
        _, k, gross = spec
        M = membership(mom, k, 2, 0.0, +1, reset=GATE)
        return M, np.where(M, gross / k, 0.0), +1.0, 0.0
    if kind == "short_sel":  # A-style bottom-k short with 15% stop
        _, lb, k, gross = spec
        ret = g["ret"][lb] if lb in g["ret"] else C / C.shift(lb) - 1
        M = membership(ret, k, 0, 0.0, -1, reset=GATE)
        return M, np.where(M, gross / k, 0.0), -1.0, 0.15
    if kind == "short_btc":
        _, gross = spec
        M = np.zeros((T, N), bool); W = np.zeros((T, N)); j = COLS.index("BTC"); M[:, j] = True; W[:, j] = gross
        return M, W, -1.0, 0.0
    raise ValueError(spec)


def job(spec):
    G = prep()
    M, W, sign, stop = second(spec)
    reg = np.where(GATE, 1, -1)
    eq, info = run(ML=G["ML"], WL=G["WL"], MS=M, WS=W, regime=reg, sign_alt=sign, stop_s=stop)
    r = evaluate(eq, " ".join(map(str, spec)), dict(turn=info["turn"], gross=info["gross"]))
    r["tstat_vs_v1"] = B.paired_t(eq, G["ref"]) if spec[0] != "cash" else np.nan
    # sleeve-only (gate-off) stats
    only, _ = run(MS=M, WS=W, regime=np.where(GATE, 0, -1), sign_alt=sign, stop_s=stop)
    r["off_tot"] = only.iloc[-1] - 1
    for y in [2023, 2024, 2025, 2026]:
        e = only[only.index.year == y]
        r[f"off{y}"] = e.iloc[-1] / e.iloc[0] - 1
    return r


def grid():
    G = [("cash",)]
    for gr in [0.5, 1.0]:
        G += [("btc", gr), ("btceth", gr), ("eth", gr), ("short_btc", gr)]
        G += [("lowvol", k, gr) for k in [3, 5]]
        G += [("reversal", h, 3, gr) for h in [24, 72]]
        G += [("mom_ungated", 3, gr)]
        G += [("short_sel", lb, 3, gr) for lb in [168, 336]]
    return G


if __name__ == "__main__":
    G = grid()
    print(len(G), "configs", flush=True)
    with Pool(6) as p:
        res = p.map(job, G, chunksize=1)
    df = pd.DataFrame(res)
    df.to_csv("out_research9_alt_C.csv", index=False)
    print(fmt(df, ["label", "tot", "mdd", "mean14", "p10", "p90", "pos1", "mcomp", "y2023", "y2024", "y2025", "y2026", "off_tot",
                   "off2023", "off2024", "off2025", "off2026", "tstat_vs_v1"]))
