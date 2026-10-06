"""Tests 1-4: concentration/weighting, pool construction, rebalance phase noise floor + tranches, lookback robustness."""
import sys
import numpy as np
import pandas as pd
import research10_tour_lib as T
import research9_alt_lib as L

C = T.C
PD = {}
def pool(n):
    if n not in PD:
        PD[n] = T.pool_mask(n)
    return PD[n]

rows = []
def add(label, eq, **extra):
    r = T.evaluate(eq, label, extra)
    rows.append(r)
    return r

COLS = ["label", "P10", "P20", "P30", "P50", "mean", "med", "p10", "p90", "comp", "tot", "mdd", "tail", "tail2023", "tail2024", "tail2025", "tail2026", "gP10", "gP20", "gP30", "gmean"]
def show(df, title):
    print("\n==", title)
    print(T.fmt(df, [c for c in COLS if c in df.columns]))

ret336 = C / C.shift(336) - 1
base_eq, _ = T.run_v1()
add("v1 k3", base_eq)

# ------------------------------------------------ TEST 1: concentration + weighting
t1 = []
for k in [1, 2, 3, 4, 5]:
    for hyst in ([0, 1, 2] if k == 1 else [2]):
        eq, _ = T.run_v1(k=k, hyst=hyst)
        t1.append(add(f"T1 k{k} h{hyst} eq", eq))
def rank_weights(ML, k, kind):
    W = np.zeros(ML.shape)
    sc = ret336.values
    for t in np.where(L.REBAL)[0]:
        idx = np.where(ML[t])[0]
        if len(idx) == 0:
            continue
        o = idx[np.argsort(-sc[t, idx])]
        if kind == "inv_rank":
            w = 1.0 / np.arange(1, len(o) + 1)
        elif kind == "inv_rank2":
            w = 1.0 / np.arange(1, len(o) + 1) ** 2
        elif kind == "mom":
            w = np.maximum(sc[t, o], 1e-3)
        W[t, o] = w / w.sum()
    return W
for k in [2, 3, 5]:
    ML = L.membership(ret336, k, 2, 0.0, +1, reset=~T.GATE)
    for kind in ["inv_rank", "inv_rank2", "mom"]:
        W = rank_weights(ML, k, kind)
        eq, _ = T.run_v1(k=k, ML=ML, W=W, cap_mult=1.6 * k * 0 + 1e9)  # no trim cap for non-equal weights
        t1.append(add(f"T1 k{k} {kind}", eq))
show(pd.DataFrame(t1), "TEST1 concentration & weighting")

# ------------------------------------------------ TEST 2: pool
t2 = []
for n in [10, 15, 25, 40, 100]:
    for k in [1, 2, 3, 5]:
        eq, _ = T.run_v1(k=k, pool=pool(n))
        t2.append(add(f"T2 pool{n} k{k}", eq))
# momentum-first: liquid top-40 universe, momentum top-M, then rank those by 30d median volume, hold top-k by volume
dq = r1_qv = None
import research1 as r1
dqv = r1.QV.resample("1D").sum().rolling(30, min_periods=20).median().shift(1).reindex(C.index, method="ffill")
for M in [5, 8, 12]:
    for k in [3]:
        p40 = pool(40)
        s = ret336.where(p40 & (ret336 > 0))
        mrank = s.rank(axis=1, ascending=False)
        score = dqv.where(mrank <= M)
        # score is $vol (positive); membership picks highest volume among momentum-top-M
        eq, _ = T.run_v1(k=k, pool=p40, score=score, thr=0.0)
        t2.append(add(f"T2 momfirst M{M} then-vol k{k}", eq))
# vol-adjusted ranking
R1 = C.pct_change()
hv = R1.rolling(168, min_periods=84).std() * np.sqrt(24)
zs = ret336 / (hv * np.sqrt(14))
for k in [2, 3, 5]:
    ML = L.membership(zs.where(ret336 > 0), k, 2, 0.0, +1, reset=~T.GATE)
    eq, _ = T.run_v1(k=k, ML=ML)
    t2.append(add(f"T2 volAdj-z rank k{k}", eq))
show(pd.DataFrame(t2), "TEST2 pool")

# ------------------------------------------------ TEST 3: phase noise floor
t3 = []
for k in [3, 1]:
    for freq in [12, 24, 48]:
        for off in range(24):
            if freq == 12 and off >= 12:
                continue
            rm = T.rebal_mask(off, freq)
            eq, _ = T.run_v1(k=k, rebal=rm)
            r = add(f"T3 k{k} f{freq} off{off}", eq, k=k, freq=freq, off=off)
            t3.append(r)
d3 = pd.DataFrame(t3)
d3.to_csv("out_research10_tour_A_phase.csv", index=False)
print("\n== TEST3 phase spread (min/median/max/std) over offsets")
for k in [3, 1]:
    for freq in [12, 24, 48]:
        g = d3[(d3.k == k) & (d3.freq == freq)]
        print(f"k{k} f{freq} n={len(g)}")
        print(g[["P10", "P20", "P30", "P50", "mean", "p10", "p90", "comp", "tot", "tail", "gP20", "gmean"]].describe().loc[["min", "50%", "max", "std"]].round(3).to_string())
# tranches: average of equity curves of sleeves at offsets
def tranche(offs, k=3, freq=24):
    es = [T.run_v1(k=k, rebal=T.rebal_mask(o, freq))[0] for o in offs]
    return np.mean(es, axis=0)
for nt in [2, 4, 8, 24]:
    offs = list(range(0, 24, 24 // nt))
    eq = tranche(offs)
    t3.append(add(f"T3 tranche{nt} k3 (avg of {nt} phases, 24h)", eq))
for nt in [2, 4]:
    offs = list(range(0, 24, 24 // nt))
    eq = tranche(offs, k=1)
    t3.append(add(f"T3 tranche{nt} k1", eq))
show(pd.DataFrame(t3[-6:]), "TEST3 tranches")
pd.DataFrame(t3).to_csv("out_research10_tour_A_phase.csv", index=False)

# ------------------------------------------------ TEST 4: lookback robustness
t4 = []
for lb in [168, 240, 288, 336, 384, 432, 504]:
    for k in [3, 1]:
        eq, _ = T.run_v1(lb=lb, k=k)
        t4.append(add(f"T4 lb{lb} k{k}", eq))
def blend(lbs):
    return sum(C / C.shift(l) - 1 for l in lbs) / len(lbs)
for name, lbs in [("240-336-432", [240, 336, 432]), ("288-336-384", [288, 336, 384]), ("168-336-672", [168, 336, 672]), ("240-432", [240, 432])]:
    for k in [3, 1]:
        eq, _ = T.run_v1(k=k, score=blend(lbs))
        t4.append(add(f"T4 blend{name} k{k}", eq))
# phases for lb neighbourhood (lb x 4 phases) to separate lb effect from phase noise
for lb in [240, 288, 336, 384, 432]:
    ps = []
    for off in [0, 6, 12, 18]:
        eq, _ = T.run_v1(lb=lb, rebal=T.rebal_mask(off, 24))
        ps.append(T.evaluate(eq, ""))
    d = pd.DataFrame(ps)
    t4.append(dict(label=f"T4 lb{lb} k3 mean-over-4-phases", **{c: d[c].mean() for c in d.columns if c != "label"}))
show(pd.DataFrame(t4), "TEST4 lookback")

allr = pd.DataFrame(rows + t4[-5:])
allr.to_csv("out_research10_tour_A.csv", index=False)
print("configs evaluated:", len(rows))
