"""research10_alt_3b: tradeable version of the market-wide liquidation-cascade bounce (suggested by the TEST 3 event study).
When the BTC gate is OFF (v1 is in cash) and >=X of the liquid alt pool dropped more than Y within `win` hours, buy the equal-weight liquid-pool
basket at the close of the trigger bar, hold H hours (0.12%/side), 72h refractory. v1 itself is untouched; while an overlay position is
open its return replaces v1's (v1 is flat when the gate is off; if the gate flips on inside the window v1 simply waits until the overlay exits).
This family was selected AFTER looking at the event study, so it carries the whole multiple-testing penalty (see report)."""
import numpy as np
import pandas as pd
from multiprocessing import Pool

from research10_alt_lib import *

ALTS = [c for c in COLS if c != "BTC"]
_G = {}


def S():
    if _G:
        return _G
    _G["Cv"] = C.ffill().values
    _G["pool"] = POOL.values
    _G["ref"] = v1_ref()
    return _G


def triggers(win, Y, X, gate_off_only=True):
    rw = (C[ALTS] / C[ALTS].shift(win) - 1).values
    ap = POOL[ALTS].values
    share = np.where(ap, rw < -Y, False).sum(axis=1) / np.maximum(ap.sum(axis=1), 1)
    ok = (share >= X) & (ap.sum(axis=1) >= 10) & (IDX >= START)
    if gate_off_only:
        ok &= ~GATE
    return np.where(ok)[0]


def overlay_equity(trig, H, tgt="basket", cost=0.0012):
    g = S()
    Cv, pool = g["Cv"], g["pool"]
    eq0 = g["ref"]
    r = eq0.pct_change().fillna(0.0).reindex(IDX).fillna(0.0).values.copy()
    pos = {t: i for i, t in enumerate(IDX)}
    last_end = -1
    n_ev = 0
    ev_ret = []
    for t in trig:
        if t <= last_end or t + H >= T:
            continue
        mem = pool[t].copy()
        if tgt == "BTC":
            mem[:] = False; mem[COLS.index("BTC")] = True
        ok = mem & (Cv[t] > 0)
        if ok.sum() == 0:
            continue
        path = (Cv[t + 1: t + H + 1][:, ok] / Cv[t, ok]).mean(axis=1)  # basket value (starts 1.0 at t)
        prev = np.concatenate([[1.0], path[:-1]])
        ov = path / prev - 1.0
        ov[0] -= cost
        ov[-1] -= cost
        r[t + 1: t + H + 1] = ov
        last_end = t + H
        n_ev += 1
        ev_ret.append(path[-1] - 1 - 2 * cost)
    s = pd.Series(np.cumprod(1 + r), index=IDX)
    s = s[s.index >= START]
    return s / s.iloc[0], n_ev, ev_ret


def job(spec):
    win, Y, X, H, tgt = spec
    g = S()
    trig = triggers(win, Y, X)
    eq, n_ev, ev = overlay_equity(trig, H, tgt)
    label = f"cascade win{win} Y{Y} X{X} H{H} {tgt}"
    r = evaluate2(eq, label, dict(n_events=n_ev, ev_mean=float(np.mean(ev)) if ev else np.nan, ev_hit=float(np.mean(np.array(ev) > 0)) if ev else np.nan))
    r["t_vs_v1"] = paired_t(eq, g["ref"])
    return r


if __name__ == "__main__":
    G = []
    for win, Ys in [(24, [0.05, 0.08, 0.10]), (4, [0.03, 0.05, 0.07])]:
        for Y in Ys:
            for X in [0.5, 0.7, 0.85]:
                for H in [24, 72]:
                    G.append((win, Y, X, H, "basket"))
    G += [(24, 0.08, 0.7, 72, "BTC"), (24, 0.10, 0.5, 72, "BTC"), (4, 0.05, 0.7, 72, "BTC")]
    print(len(G), "configs")
    with Pool(14) as p:
        res = p.map(job, G, chunksize=1)
    df = pd.DataFrame(res)
    ref = evaluate2(v1_ref(), "v1")
    df = pd.concat([pd.DataFrame([ref]), df], ignore_index=True)
    df.to_csv("out_research10_alt_3b.csv", index=False)
    cols = ["label", "n_events", "ev_mean", "ev_hit", "mean14", "med14", "p10", "p90", "pos", "gt20", "mcomp", "tot", "mdd", "y2023", "y2024", "y2025", "y2026", "t_vs_v1"]
    print(fmt(df, cols))
