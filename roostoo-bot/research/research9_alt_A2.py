"""research9_alt_A2: diagnostics for the short sleeve: gross (cost-free) edge, deeper bear sub-regimes, sensitivity to borrow."""
import numpy as np, pandas as pd
from research9_alt_A import *

rows = []
G = base()
ret = C / C.shift(336) - 1
MS, WS = short_inputs(336, 3, 0.5, "eq", "none")
btc_below = (BTC < BTC.ewm(span=168, adjust=False).mean()).values
btc_ret168 = (BTC / BTC.shift(168) - 1).values
regimes = {
    "gateoff": ~GATE,
    "gateoff&BTC<ema168": (~GATE) & btc_below,
    "gateoff&BTCret168<-5%": (~GATE) & (btc_ret168 < -0.05),
}
for rn, offmask in regimes.items():
    reg_only = np.where(offmask, -1, 0)
    for cs, bd in [(0.0012, 0.0002), (0.0, 0.0), (0.001, 0.0), (0.0012, 0.0005)]:
        for nm, (m, w) in {"sel336k3": (MS, WS)}.items():
            eq, info = run(MS=m, WS=w, regime=reg_only, cost_s=cs, borrow_day=bd)
            r = evaluate(eq, f"ONLY {nm} {rn} cost{cs} borrow{bd}")
            rows.append(r)
    # BTC only
    mb = np.zeros((T, N), bool); wb = np.zeros((T, N)); j = COLS.index("BTC"); mb[:, j] = True; wb[:, j] = 0.5
    for cs, bd in [(0.0012, 0.0002), (0.0, 0.0)]:
        eq, info = run(MS=mb, WS=wb, regime=reg_only, cost_s=cs, borrow_day=bd)
        rows.append(evaluate(eq, f"ONLY BTC {rn} cost{cs} borrow{bd}"))
df = pd.DataFrame(rows)
df.to_csv("out_research9_alt_A2.csv", index=False)
print(fmt(df, ["label", "tot", "mdd", "mean14", "y2023", "y2024", "y2025", "y2026"]))
print("share of time: gateoff", (~GATE[IDX >= START]).mean())
for rn, m in regimes.items(): print(rn, m[IDX >= START].mean())
