import numpy as np, pandas as pd
import research10_tour_lib as T
import research9_alt_lib as L
C, IDX, GATE = T.C, T.IDX, T.GATE
rows = []
def add(label, eq, **extra):
    r = T.evaluate(eq, label, extra); rows.append(r); return r
COLS = ["label","P10","P20","P30","mean","med","p10","p90","comp","tot","mdd","mean2023","mean2024","mean2025","mean2026"]
def table(rs=None):
    df = pd.DataFrame(rs if rs is not None else rows)
    base = df.iloc[0]
    df["yrwin"] = sum((df[f"mean{y}"] > base[f"mean{y}"]).astype(int) for y in (2023,2024,2025,2026))
    pd.set_option("display.width",300); pd.set_option("display.max_columns",60)
    return df[COLS+["yrwin"]].round(3).to_string()
