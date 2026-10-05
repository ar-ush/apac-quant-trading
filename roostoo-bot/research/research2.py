"""Research batch 2: blends + portfolio-level overlays (vol target, DD brake, profit lock) on 1h panel."""
import sys
import numpy as np
import pandas as pd
from lib import *
import research1 as r1

C = r1.C


def overlay(close, w, cost=0.0012, vol_target=None, vol_win=72, dd_brake=None, dd_floor=0.3, lock_gain=None, lock_scale=0.3,
            window_h=24 * 14, max_scale=1.5):
    """Sequential sim. w decided at close t applied to bar t+1 (scaled by overlay state known at t).
    vol_target: target daily vol of strategy (e.g. .015). dd_brake: drawdown (from peak within 14d window) at which exposure
    is cut to dd_floor. lock_gain: once rolling-window gain >= lock_gain scale exposure to lock_scale."""
    ret = close.pct_change().fillna(0.0).values
    wv = w.reindex(close.index).fillna(0.0).values
    n = len(close)
    eq = np.ones(n)
    pos = np.zeros(wv.shape[1])
    strat_ret = np.zeros(n)
    scale = 1.0
    peak = 1.0
    for t in range(1, n):
        # P&L for bar t from positions set at t-1
        g = float(np.dot(pos, ret[t]))
        # target at t-1 closing -> decide positions for bar t+1 now; first realise pnl
        tgt = wv[t] * scale_at(t, strat_ret, eq, vol_target, vol_win, dd_brake, dd_floor, lock_gain, lock_scale, window_h, max_scale)
        dw = np.abs(tgt - pos).sum()
        net = g - dw * cost
        strat_ret[t] = net
        eq[t] = eq[t - 1] * (1 + net)
        pos = tgt * (1 + ret[t] * 0)  # drift ignored (small); weights refreshed each bar by overlay
    return pd.Series(eq, index=close.index)


def scale_at(t, sr, eq, vol_target, vol_win, dd_brake, dd_floor, lock_gain, lock_scale, window_h, max_scale):
    s = 1.0
    if vol_target is not None and t > vol_win:
        d = sr[t - vol_win:t].std() * np.sqrt(24)
        if d > 1e-6:
            s = min(max_scale, vol_target / d)
    lo = max(0, t - window_h)
    if dd_brake is not None:
        pk = eq[lo:t].max()
        dd = 1 - eq[t - 1] / pk
        if dd >= dd_brake:
            s *= dd_floor
    if lock_gain is not None:
        gain = eq[t - 1] / eq[lo] - 1
        if gain >= lock_gain:
            s *= lock_scale
    return s


def report(name, eq):
    win = rolling_windows(eq)
    s = summarize_windows(win)
    return dict(name=name, tot=eq.iloc[-1] - 1, mdd=max_drawdown(eq.values), med14=s["med_ret"], mean14=s["mean_ret"],
                pos=s["pos"], p10=s["p10"], p90=s["p90"], gt5=s["gt5"], mdd14=s["med_mdd"], comp=s["med_comp"], mcomp=s["mean_comp"])


if __name__ == "__main__":
    cost = 0.0012
    base = {
        "xs14": r1.xsmom_topk(24 * 14, 3),
        "xs7": r1.xsmom_topk(24 * 7, 3),
        "btc_tr": r1.btc_trend(),
        "don20": r1.donchian(),
    }
    base["blend_xs14_btc"] = 0.5 * base["xs14"] + 0.5 * base["btc_tr"]
    base["blend_all"] = (base["xs14"] + base["btc_tr"] + base["don20"] + base["xs7"]) / 4
    rows = []
    for k, w in base.items():
        eq, _ = simulate(C, w, cost=cost)
        rows.append(report(k, eq))
    for k in ["xs14", "blend_xs14_btc", "blend_all"]:
        w = base[k]
        for tag, kw in [("vt1.5", dict(vol_target=0.015)), ("vt2.5", dict(vol_target=0.025)),
                        ("dd5", dict(dd_brake=0.05)), ("lock6", dict(lock_gain=0.06)),
                        ("vt2.5+dd5+lock8", dict(vol_target=0.025, dd_brake=0.05, lock_gain=0.08))]:
            eq = overlay(C, w, cost=cost, **kw)
            rows.append(report(f"{k}|{tag}", eq))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).set_index("name").round(3).to_string())
