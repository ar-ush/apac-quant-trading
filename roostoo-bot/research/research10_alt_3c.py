"""research10_alt_3c: robustness of the cascade-bounce overlay (family stats from 3b + entry delay, higher cost, half-sample)."""
import numpy as np, pandas as pd
import research10_alt_3b as M
from research10_alt_lib import *

df = pd.read_csv("out_research10_alt_3b.csv")
v1 = df.iloc[0]; fam = df.iloc[1:]
yrs = ["y2023", "y2024", "y2025", "y2026"]
print("family n=%d" % len(fam))
print(" share mean14>v1: %.2f  mcomp>v1: %.2f  tot>v1: %.2f  pos>v1: %.2f  p10>=v1: %.2f" % tuple(
    (fam[c] > v1[c]).mean() if c != "p10" else (fam[c] >= v1[c]).mean() for c in ["mean14", "mcomp", "tot", "pos", "p10"]))
print(" years better (>=3 of 4):", ((fam[yrs].values > v1[yrs].values.astype(float)).sum(axis=1) >= 3).mean(), " all4:", ((fam[yrs].values > v1[yrs].values.astype(float)).sum(axis=1) == 4).mean())
print(" median mean14 %.4f median mcomp %.2f median tot %.1f median t %.2f" % (fam.mean14.median(), fam.mcomp.median(), fam.tot.median(), fam.t_vs_v1.median()))

def run(win, Y, X, H, delay=0, cost=0.0012, half=None):
    trig = M.triggers(win, Y, X)
    trig = trig + delay
    trig = trig[trig < T - H - 1]
    if half:
        trig = trig[(IDX[trig] < pd.Timestamp("2025-01-01", tz="UTC"))] if half == 1 else trig[(IDX[trig] >= pd.Timestamp("2025-01-01", tz="UTC"))]
    eq, n, ev = M.overlay_equity(trig, H, "basket", cost=cost)
    r = evaluate2(eq, f"{win},{Y},{X},{H} d{delay} c{cost} h{half}", dict(n=n))
    r["t"] = paired_t(eq, v1_ref())
    return r

rows = []
for spec in [(24, .08, .5, 72), (24, .08, .7, 72), (24, .10, .5, 72), (24, .10, .7, 72), (4, .05, .7, 72), (4, .07, .5, 72)]:
    for kw in [dict(), dict(delay=1), dict(delay=4), dict(cost=0.0020), dict(half=1), dict(half=2)]:
        rows.append(run(*spec, **kw))
d = pd.DataFrame(rows)
d.to_csv("out_research10_alt_3c.csv", index=False)
print(fmt(d, ["label", "n", "mean14", "pos", "mcomp", "tot", "y2023", "y2024", "y2025", "y2026", "t"]))
