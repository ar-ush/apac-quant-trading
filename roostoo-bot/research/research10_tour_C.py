"""Pool-size deep dive (the one big difference found in test 2): neighbourhood, phase-averaging, cost stress, drop-one-coin, liquidity of picks, lock overlay on top."""
import numpy as np
import pandas as pd
import research10_tour_lib as T
import research9_alt_lib as L
import research1 as r1

C = T.C
PH = [0, 6, 12, 18]
COLS = ["label", "P10", "P20", "P30", "P50", "mean", "p10", "p90", "comp", "tot", "mdd", "tail", "tail2023", "tail2024", "tail2025", "tail2026", "gP10", "gP20", "gP30", "gmean"]
rows = []
def multi(label, mk):
    """average metrics over 4 rebalance phases; mk(rebal_mask)->(eq)"""
    ds = [T.evaluate(mk(T.rebal_mask(o, 24)), "") for o in PH]
    d = pd.DataFrame(ds)
    r = dict(label=label, **{c: d[c].mean() for c in d.columns if c != "label"})
    r["P20_min_ph"] = d.P20.min(); r["P20_max_ph"] = d.P20.max(); r["tot_min_ph"] = d.tot.min()
    rows.append(r); return r

print("coins per day in pool(n) and avg liquidity rank info")
qv = r1.QV.resample("1D").sum().rolling(30, min_periods=20).median().shift(1)
med = qv.median()
print("median 30d $vol (millions/day) by coin, sorted:")
print((med.sort_values(ascending=False) / 1e6).round(1).to_string())
for n in [20, 25, 30, 35, 41]:
    m = T.pool_mask(n)
    print(n, "avg pool size", m.sum(axis=1)[T._START_IDX:].mean().round(1))

# ---- pool neighbourhood (4-phase mean) k=3 and k=2,5
for n in [20, 25, 30, 35, 41]:
    for k in [3]:
        pm = T.pool_mask(n)
        multi(f"C pool{n} k{k}", lambda rm, pm=pm, k=k: T.run_v1(k=k, pool=pm, rebal=rm)[0])
for k in [1, 2, 5]:
    pm = T.pool_mask(41)
    multi(f"C pool41 k{k}", lambda rm, pm=pm, k=k: T.run_v1(k=k, pool=pm, rebal=rm)[0])

# ---- cost stress for pool41 vs pool25 (cost per side)
def run_cost(pm, cost, rm, k=3):
    ML = L.membership(C / C.shift(336) - 1, k, 2, 0.0, +1, pool=pm, reset=~T.GATE, rebal=rm)
    eq, _ = L.run(ML=ML, WL=L.eqw(ML, k), regime=np.where(T.GATE, 1, 0), rebal=rm, cost_l=cost, cost_s=cost)
    return eq.values
for n in [25, 41]:
    for cost in [0.0012, 0.0025, 0.005]:
        pm = T.pool_mask(n)
        multi(f"C cost{cost} pool{n}", lambda rm, pm=pm, cost=cost: run_cost(pm, cost, rm))
# extra-cost only for coins outside top-25 (illiquid): approx by using pool41 at 0.5% per side
d_cost = pd.DataFrame(rows)

# ---- drop-one-coin for pool41 (phase 0 only) to check dependence on one coin
base41 = T.evaluate(T.run_v1(k=3, pool=T.pool_mask(41))[0], "base41")
pm41 = T.pool_mask(41)
drops = []
for j, cn in enumerate(C.columns):
    pm = pm41.copy(); pm[:, j] = False
    r = T.evaluate(T.run_v1(k=3, pool=pm)[0], f"drop {cn}")
    drops.append(dict(coin=cn, P10=r["P10"], P20=r["P20"], P30=r["P30"], mean=r["mean"], tot=r["tot"], tail=r["tail"]))
dd = pd.DataFrame(drops).sort_values("tail")
print("\nbase pool41 phase0:", {k: round(base41[k], 3) for k in ["P10", "P20", "P30", "mean", "tot", "tail"]})
print("drop-one-coin (worst 8 by tail, best 3):")
print(pd.concat([dd.head(8), dd.tail(3)]).round(3).to_string())
print("tail range over drop-one: ", round(dd["tail"].min(),3), round(dd["tail"].max(),3), " P20 range", round(dd.P20.min(),3), round(dd.P20.max(),3))
# drop top-3 contributors jointly (worst-3 by tail): are pool25-equivalent results still better than pool25?
worst3 = list(dd.head(3).coin)
pm = pm41.copy()
for cn in worst3:
    pm[:, list(C.columns).index(cn)] = False
r = T.evaluate(T.run_v1(k=3, pool=pm)[0], "drop worst3")
print("drop 3 most-valuable coins", worst3, {k: round(r[k], 3) for k in ["P10", "P20", "P30", "mean", "tot", "tail"]})
# pool25 minus same coins for like-for-like
pm25 = T.pool_mask(25).copy()
for cn in worst3:
    pm25[:, list(C.columns).index(cn)] = False
r = T.evaluate(T.run_v1(k=3, pool=pm25)[0], "pool25 w/o worst3")
print("pool25 minus same", {k: round(r[k], 3) for k in ["P10", "P20", "P30", "mean", "tot", "tail"]})

# ---- which coins get picked (share of gate-on rebal days held), by liquidity rank
ML = L.membership(C / C.shift(336) - 1, 3, 2, 0.0, +1, pool=pm41, reset=~T.GATE)
held = pd.DataFrame(ML, index=C.index, columns=C.columns)
held = held[L.REBAL & (held.index >= L.START)]
share = held.sum() / max(1, (held.sum(axis=1) > 0).sum())
rank25 = T.pool_mask(25)[L.REBAL & (C.index >= L.START)]
in25 = pd.Series((ML[L.REBAL & (C.index >= L.START)] & rank25).sum(axis=0), index=C.columns)
tot_h = pd.Series(ML[L.REBAL & (C.index >= L.START)].sum(axis=0), index=C.columns)
print("\nshare of pool41 picks that were inside the 25-pool at the time:", round(float(in25.sum() / tot_h.sum()), 3))
print("top pick frequency (pool41):")
print((tot_h.sort_values(ascending=False).head(12) / tot_h.sum()).round(3).to_string())

# ---- lock overlay on pool41 base
eq41 = T.run_v1(k=3, pool=pm41)[0]
c = T.ctx(len(eq41)); s = c.s; nw = len(s)
r = np.empty_like(eq41); r[0] = 0; r[1:] = eq41[1:] / eq41[:-1] - 1
Rw = r[s[:, None] + 1 + np.arange(T.WL_)[None, :]]
def lock(Rw, X, Y):
    E = np.ones(nw); P = np.empty((nw, T.WL_ + 1)); P[:, 0] = 1; lk = np.zeros(nw, bool); fp = np.ones(nw)
    for t in range(T.WL_):
        lk |= E >= 1 + X
        f = np.where(lk, Y, 1.0)
        inv = Rw[:, t] != 0
        E = E * (1 + f * Rw[:, t]) - E * 0.0012 * np.abs(f - fp) * inv * 0.98
        fp = f; P[:, t + 1] = E
    return P
out = [base41]
for X in [0.20, 0.30, 0.40]:
    for Y in [0.0, 0.5]:
        m = T.metrics_from_E(lock(Rw, X, Y))
        out.append(T.summarize(m, s, c.gate_on, f"pool41 lock X{int(X*100)} Y{Y}"))
print("\n== pool41 + lock overlay (phase 0)")
print(T.fmt(pd.DataFrame(out), [x for x in COLS if x in out[0] and x not in ("tot", "mdd")]))

d = pd.DataFrame(rows)
print("\n== pool neighbourhood / cost (4-phase mean)")
print(T.fmt(d, [x for x in COLS + ["P20_min_ph", "P20_max_ph", "tot_min_ph"] if x in d.columns]))
d.to_csv("out_research10_tour_C.csv", index=False)

# ---- paired block bootstrap (14-day blocks of window starts) of tail differences, pool41 vs pool25 (4-phase pooled), and lock X30 vs none
def win_rets(eq):
    E = T.window_matrix(eq, s); return T.metrics_from_E(E)["ret"]
rng = np.random.default_rng(0)
def paired(a, b, thr_list=(0.10, 0.20, 0.30), nb=2000, blk=14):
    n = len(a); nbk = int(np.ceil(n / blk)); out = {}
    for th in thr_list:
        d = (a > th).astype(float) - (b > th).astype(float)
        bs = []
        for _ in range(nb):
            st = rng.integers(0, n - blk, nbk)
            idx = (st[:, None] + np.arange(blk)[None, :]).ravel()[:n]
            bs.append(d[idx].mean())
        out[th] = (round(d.mean(), 3), round(np.quantile(bs, .05), 3), round(np.quantile(bs, .95), 3))
    return out
ra = np.mean([ (win_rets(T.run_v1(k=3, pool=T.pool_mask(41), rebal=T.rebal_mask(o, 24))[0]) > 0.2) for o in PH], axis=0)
for o in PH:
    a = win_rets(T.run_v1(k=3, pool=T.pool_mask(41), rebal=T.rebal_mask(o, 24))[0])
    b = win_rets(T.run_v1(k=3, pool=T.pool_mask(25), rebal=T.rebal_mask(o, 24))[0])
    print("bootstrap pool41-pool25 phase", o, "diff P(>x) [mean, 5%, 95%]:", paired(a, b))
a = win_rets(eq41)
Em = lock(Rw, 0.30, 0.0); al = Em[:, -1] - 1
print("bootstrap lockX30Y0 - nolock (pool41):", paired(al, a))
al2 = None
