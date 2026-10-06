"""research10_ls: long-short book while BTC gate is ON (long top-kL by 336h + short bottom-kS), gate off = cash.
Roostoo shorts use 1x collateral from USD, so gross_long + gross_short <= 1.0 is enforced in the grid."""
import itertools
from multiprocessing import Pool
import numpy as np, pandas as pd
from research9_alt_lib import *

VOLD = (R1.rolling(168, min_periods=84).std() * np.sqrt(24))
_G = {}
def base():
    if not _G:
        ML, WL, ret = v1_inputs()
        _G.update(ML=ML, WL=WL, REG=np.where(GATE, 2, 0))
    return _G

def job(a):
    lbS, kS, gL, gS, stop, bor = a
    G = base()
    ret = C / C.shift(lbS) - 1
    MS = membership(ret, kS, 0, 0.0, -1, pool=POOL, reset=~GATE)   # bottom-kS, any sign (thr on side*score>0 => negative ret only)
    WS = np.where(MS, gS / kS, 0.0)
    WL = G["WL"] / 0.98 * gL
    eq, inf = run(ML=G["ML"], WL=WL, MS=MS, WS=WS, regime=G["REG"], stop_s=stop, borrow_day=bor)
    return evaluate(eq, f"LS lb{lbS} kS{kS} gL{gL} gS{gS} stop{stop}", dict(nstop=inf["nstop"], gross=inf["gross"]))

if __name__ == "__main__":
    grid = [("none",)]
    jobs = []
    for lbS, kS, (gL, gS), stop in itertools.product([168, 336], [3, 5], [(0.98, 0.0), (0.65, 0.3), (0.5, 0.5), (0.75, 0.25)], [0, 0.15, 0.30]):
        if gS == 0 and (lbS != 168 or kS != 3 or stop != 0): continue
        jobs.append((lbS, kS, gL, gS, stop, 0.0002))
    print(len(jobs), flush=True)
    with Pool(8) as p:
        rows = p.map(job, jobs)
    df = pd.DataFrame(rows); df.to_csv("out_research10_ls.csv", index=False)
    cols = ["label", "tot", "mdd", "mean14", "med14", "p10", "p90", "pos", "mcomp", "y2023", "y2024", "y2025", "y2026", "nstop"]
    print(fmt(df, cols))
