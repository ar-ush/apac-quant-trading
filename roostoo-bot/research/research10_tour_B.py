"""Test 5: tournament-aware overlays on v1 (k=3): (a) ratchet/lock + catch-up, (b) gate-off partial exposure, (c) exposure scaling by alt-season strength."""
import numpy as np
import pandas as pd
import research10_tour_lib as T
import research9_alt_lib as L

C = T.C
S = T.window_starts  # noqa
base_eq, _ = T.run_v1(k=3)
c = T.ctx(len(base_eq))
s = c.s
nw = len(s)
rows = []
COLS = ["label", "P10", "P15", "P20", "P30", "P50", "mean", "med", "p10", "p90", "p99", "comp", "tail", "tail2023", "tail2024", "tail2025", "tail2026", "gP10", "gP20", "gP30", "gmean", "gp10"]
def rec(label, E=None, eq=None, extra=None):
    if eq is not None:
        r = T.evaluate(eq, label, extra)
    else:
        r = T.eval_from_E(E, label, extra)
    rows.append(r)
    return r
def show(rs, title):
    print("\n==", title)
    print(T.fmt(pd.DataFrame(rs), [x for x in COLS if x in rs[0]]))

def hourly_win(eq):
    r = np.empty_like(eq); r[0] = 0; r[1:] = eq[1:] / eq[:-1] - 1
    return r[s[:, None] + 1 + np.arange(T.WL_)[None, :]]

R3 = hourly_win(base_eq)
R1w = hourly_win(T.run_v1(k=1)[0])
R5w = hourly_win(T.run_v1(k=5)[0])
COST = 0.0012

def path(Rm, f_fn):
    """generic sequential overlay. f_fn(t, E, state) -> (f vector, returns-source Rm index vector) handled via closures."""
    raise NotImplementedError

def ratchet(Rw, X, Y, sticky=True, d0=0, Z=None, Rcatch=None, Zday=7, Rlock=None):
    """Lock: once E>=1+X (sticky or current), exposure -> Y (on Rw, or on Rlock if given, e.g. diversified k5 book).
    Catch-up: if Z is not None, E<=1-Z at/after day Zday -> use Rcatch (e.g. k1 concentrated) returns."""
    E = np.ones(nw)
    paths = np.empty((nw, T.WL_ + 1)); paths[:, 0] = 1
    locked = np.zeros(nw, bool)
    fprev = np.ones(nw)
    prev_catch = np.zeros(nw, bool)
    for t in range(T.WL_):
        if sticky:
            locked |= (E >= 1 + X) & (t >= d0 * 24)
            lk = locked
        else:
            lk = (E >= 1 + X) & (t >= d0 * 24)
        f = np.where(lk, Y, 1.0)
        src = Rlock[:, t] if Rlock is not None else Rw[:, t]
        r = np.where(lk, f * (src if Rlock is not None else Rw[:, t]), Rw[:, t])
        if Z is not None:
            catch = (E <= 1 - Z) & (t >= Zday * 24) & ~lk
            r = np.where(catch, Rcatch[:, t], r)
            sw = catch != prev_catch
            prev_catch = catch
        else:
            sw = np.zeros(nw, bool)
        inv = (Rw[:, t] != 0)
        cost = COST * np.abs(f - fprev) * inv * 0.98 + COST * 2 * 0.98 * sw * inv
        fprev = f
        E = E * (1 + r) - E * cost
        paths[:, t + 1] = E
    return paths

# ------------------------------------------------------------ 5a
rec("v1 baseline (k3)", eq=base_eq)
rec("k1 baseline", eq=T.run_v1(k=1)[0])
rec("k5 baseline", eq=T.run_v1(k=5)[0])
ra = [rows[0]]
for X in [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
    for Y in [0.0, 0.5]:
        for sticky in [True, False]:
            if not sticky and Y == 0.0:
                pass
            ra.append(rec(f"5a lock X{int(X*100)} Y{Y} {'sticky' if sticky else 'live'}", E=ratchet(R3, X, Y, sticky)))
# only in second half of window
for X in [0.15, 0.30]:
    for Y in [0.0, 0.5]:
        ra.append(rec(f"5a lock X{int(X*100)} Y{Y} sticky d>=7", E=ratchet(R3, X, Y, True, d0=7)))
# lock into diversified k5 instead of cash
for X in [0.15, 0.30]:
    ra.append(rec(f"5a lock X{int(X*100)} -> k5 book", E=ratchet(R3, X, 1.0, True, Rlock=R5w)))
# catch-up: when behind, go concentrated k1 (anti-Browne control: stop-out to cash at -Z)
for Z in [0.05, 0.10]:
    ra.append(rec(f"5a catchup Z{int(Z*100)} -> k1 (day>=7)", E=ratchet(R3, 9, 1.0, True, Z=Z, Rcatch=R1w, Zday=7)))
for Z in [0.05, 0.10]:
    ra.append(rec(f"5a CONTROL stop Z{int(Z*100)} -> cash (day>=0)", E=ratchet(R3, 9, 1.0, True, Z=Z, Rcatch=np.zeros_like(R3), Zday=0)))
show(ra, "5a ratchet / lock / catch-up")

# ------------------------------------------------------------ 5b gate-off partial exposure
ret336 = C / C.shift(336) - 1
ret72 = C / C.shift(72) - 1
ML_on = L.membership(ret336, 3, 2, 0.0, +1, reset=~T.GATE)
W_on = L.eqw(ML_on, 3)
rb = [rows[0]]
cfgs = [(k, g, th) for k in (1, 2) for g in (0.3, 0.5) for th in (0.25,)] + [(1, 0.5, 0.15), (1, 0.5, 0.40), (2, 0.5, 0.15), (2, 0.5, 0.40)]
goff = ~c.gate_on
for (k, g, th) in cfgs:
    sc = ret336.where(ret72 > 0)
    MS = L.membership(sc, k, 2, th, +1, reset=T.GATE)
    WS = np.where(MS, g / k, 0.0)
    regime = np.where(T.GATE, 1, -1)
    eq, info = L.run(ML=ML_on, WL=W_on, MS=MS, WS=WS, regime=regime, sign_alt=+1.0)
    r = rec(f"5b gateoff k{k} g{g} thr{th}", eq=eq.values, extra=dict(gate_off_frac=goff.mean()))
    # stats on gate-off windows only
    E = T.window_matrix(eq.values, s)
    m = T.metrics_from_E(E)
    r["off_mean"] = m["ret"][goff].mean(); r["off_P10"] = (m["ret"][goff] > .10).mean(); r["off_P20"] = (m["ret"][goff] > .20).mean()
    r["off_p10"] = np.quantile(m["ret"][goff], .1); r["off_n_active"] = float((np.abs(m["ret"][goff]) > 1e-9).mean())
    rb.append(r)
show(rb, "5b gate-off partial exposure")
print(pd.DataFrame(rb)[["label", "off_mean", "off_P10", "off_P20", "off_p10", "off_n_active", "gate_off_frac"]].round(3).to_string())

# ------------------------------------------------------------ 5c exposure scaling by alt-season strength
pool = L.POOL
sc = ret336.where(pool & (ret336 > 0))
top3 = np.sort(np.nan_to_num(sc.values, nan=0.0), axis=1)[:, -3:]   # positive-momentum top-3 (0 if none)
top3m = pd.Series(top3.mean(axis=1), index=C.index)
disp = ret336.where(pool).std(axis=1)
breadth = (ret336.where(pool) > 0).sum(axis=1) / pool.sum(axis=1).replace(0, np.nan)
def f_from(sig, thr, flow):
    f = np.where(sig.values >= thr, 1.0, flow)
    f = pd.Series(f, index=C.index).where(L.REBAL).ffill().fillna(1.0).values  # decided at 00:00 close, held 24h
    return f
def overlay_f(f_full, base):
    """E from base return windows with exposure f[t] known at bar close t, applied to bar t+1 return."""
    f_win = f_full[T._START_IDX + s[:, None] + np.arange(T.WL_)[None, :]]
    E = np.ones(nw); paths = np.empty((nw, T.WL_ + 1)); paths[:, 0] = 1
    fprev = np.ones(nw)
    for t in range(T.WL_):
        f = f_win[:, t]
        inv = base[:, t] != 0
        E = E * (1 + f * base[:, t]) - E * COST * np.abs(f - fprev) * inv * 0.98
        fprev = f
        paths[:, t + 1] = E
    return paths
rc = [rows[0]]
print("\ntop3-mean ret336 quantiles (gate-on bars):", np.quantile(top3m.values[T.GATE], [.1, .25, .5, .75, .9]).round(3), " today:", round(float(top3m.iloc[-1]), 3), " disp today", round(float(disp.iloc[-1]), 3), "breadth", round(float(breadth.iloc[-1]), 3))
print("disp quantiles gate-on:", np.nanquantile(disp.values[T.GATE], [.1, .25, .5, .75, .9]).round(3), " breadth q:", np.nanquantile(breadth.values[T.GATE], [.1, .25, .5, .75, .9]).round(3))
for nm, sig, thrs in [("top3mom", top3m, [0.20, 0.30, 0.45, 0.60]), ("disp", disp, [0.10, 0.15, 0.25]), ("breadth", breadth.fillna(0), [0.4, 0.6])]:
    for thr in thrs:
        for flow in [0.0, 0.5]:
            f = f_from(sig, thr, flow)
            rc.append(rec(f"5c {nm}>={thr} else {flow}", E=overlay_f(f, R3), extra=dict(frac_full=float((f[T.GATE] == 1).mean()))))
# upside-tilt controls (opposite sign): scale DOWN when strength is HIGH
for thr in (0.40, 0.60):
    f = np.where(top3m.values >= thr, 0.5, 1.0)
    f = pd.Series(f, index=C.index).where(L.REBAL).ffill().fillna(1.0).values
    rc.append(rec(f"5c CONTROL top3mom>={thr} -> 0.5 (derisk when hot)", E=overlay_f(f, R3)))
show(rc, "5c exposure scaling")

pd.DataFrame(rows).to_csv("out_research10_tour_B.csv", index=False)
print("configs:", len(rows))
