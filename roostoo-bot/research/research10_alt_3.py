"""research10_alt_3: TEST 3 - crash-reversal event study (1h/4h/24h z-score drops in liquid coins; market-wide liquidation cascades).
Entry = close of the event bar (same convention as the bot), exit at close +H, round-trip cost 0.24%.
Events: coin in the liquid POOL at t, z = h-bar return / (trailing-168h 1h-vol measured BEFORE the window * sqrt(h)) < -thr; per-coin 24h refractory
(first bar of a cluster only). t-stats are cluster-robust (clusters = 3-day blocks, because crash events across coins are simultaneous and
forward windows overlap). Excess = event fwd minus unconditional mean fwd of all pool-coin hours in the same gate state (and 'mkt-adj' = minus equal-weight pool basket)."""
import numpy as np
import pandas as pd

from research10_alt_lib import *

COST = 0.0024
HS = [4, 24, 72]
ALTS = [c for c in COLS if c != "BTC"]


def cluster_t(x, cl):
    """t-stat of mean(x) = 0 with cluster-robust SE (clusters = integer labels)."""
    x = np.asarray(x, float); cl = np.asarray(cl)
    ok = ~np.isnan(x); x, cl = x[ok], cl[ok]
    n = len(x)
    if n < 10:
        return np.nan, 0
    m = x.mean()
    d = pd.Series(x - m).groupby(cl).sum()
    se = np.sqrt((d ** 2).sum()) / n
    return m / se, len(d)


def main():
    fwd = {H: C.shift(-H) / C - 1 for H in HS}
    basket = {H: fwd[H].where(POOL).mean(axis=1) for H in HS}
    sig = R1.rolling(168, min_periods=100).std()
    blk = (np.arange(T) // 72)  # 3-day blocks
    blk_df = np.repeat(blk[:, None], N, axis=1)
    rows = []
    valid_T = np.arange(T) < T - max(HS)
    for gate_name, gm in [("gate_on", GATE), ("any", np.ones(T, bool))]:
        base_mask = POOL.values & gm[:, None] & valid_T[:, None] & (IDX >= START)[:, None]
        unc = {H: np.nanmean(fwd[H].values[base_mask]) for H in HS}
        unc_alt = {H: np.nanmean((fwd[H] - basket[H].values[:, None]).values[base_mask]) for H in HS}
        for h in [1, 4, 24]:
            ret_h = C / C.shift(h) - 1
            z = ret_h / (sig.shift(h) * np.sqrt(h))
            for thr in [3, 4, 5]:
                ev = (z < -thr).values & base_mask
                # per-coin refractory 24h: keep first of cluster
                keep = np.zeros_like(ev)
                for j in range(N):
                    idx = np.where(ev[:, j])[0]
                    last = -10 ** 9
                    for t in idx:
                        if t - last >= 24:
                            keep[t, j] = True
                        last = t
                ev = keep
                for H in HS:
                    f = fwd[H].values[ev]
                    m_adj = (fwd[H].values - basket[H].values[:, None])[ev]
                    cl = blk_df[ev]
                    n_ev = int(np.isfinite(f).sum())
                    if n_ev < 15:
                        continue
                    tt, ncl = cluster_t(f - COST - (unc[H] - COST), cl)  # excess vs unconditional (cost cancels)
                    t0, _ = cluster_t(f - COST, cl)
                    ta, _ = cluster_t(m_adj, cl)
                    yr = pd.Series(f - COST).groupby(IDX.year.values[np.where(ev)[0]][np.isfinite(f)] if False else IDX.year.values[np.where(ev)[0]]).mean()
                    rows.append(dict(kind="coin", gate=gate_name, h=h, thr=thr, H=H, n=n_ev, clusters=ncl,
                                     mean_net=np.nanmean(f) - COST, med_net=np.nanmedian(f) - COST, hit=np.nanmean(f - COST > 0),
                                     uncond_net=unc[H] - COST, excess=np.nanmean(f) - unc[H], t_excess=tt, t_net=t0,
                                     mktadj=np.nanmean(m_adj), mktadj_excess=np.nanmean(m_adj) - unc_alt[H], t_mktadj=ta,
                                     **{f"y{y}": yr.get(y, np.nan) for y in [2023, 2024, 2025, 2026]}))
    df = pd.DataFrame(rows)
    df.to_csv("out_research10_alt_3_coin.csv", index=False)
    print("COIN EVENTS"); print(fmt(df))

    # ---------------------------------------------------- market-wide cascades
    rows = []
    ALT_POOL = POOL[ALTS].values
    for win in [4, 24]:
        rw = (C[ALTS] / C[ALTS].shift(win) - 1).values
        for Y in ([0.03, 0.05, 0.07] if win == 4 else [0.05, 0.08, 0.10]):
            share = np.where(ALT_POOL, rw < -Y, False).sum(axis=1) / np.maximum(ALT_POOL.sum(axis=1), 1)
            for X in [0.5, 0.7, 0.85]:
                for gate_name, gm in [("gate_on", GATE), ("any", np.ones(T, bool))]:
                    trig = (share >= X) & gm & valid_T & (IDX >= START) & (ALT_POOL.sum(axis=1) >= 10)
                    idx = np.where(trig)[0]
                    keep = []; last = -10 ** 9
                    for t in idx:
                        if t - last >= 72:
                            keep.append(t)
                        last = t
                    keep = np.array(keep, int)
                    if len(keep) < 3:
                        continue
                    for tgt in ["basket", "BTC"]:
                        for H in HS:
                            if tgt == "basket":
                                x = basket[H].values; u = np.nanmean(x[(IDX >= START) & valid_T & gm])
                            else:
                                x = fwd[H]["BTC"].values; u = np.nanmean(x[(IDX >= START) & valid_T & gm])
                            f = x[keep]
                            t_n, _ = cluster_t(f - COST, np.arange(len(f)))
                            t_x, _ = cluster_t(f - u, np.arange(len(f)))
                            rows.append(dict(kind="mkt", win=win, Y=Y, X=X, gate=gate_name, tgt=tgt, H=H, n=len(keep),
                                             mean_net=np.nanmean(f) - COST, med_net=np.nanmedian(f) - COST, hit=np.mean(f - COST > 0),
                                             uncond_net=u - COST, excess=np.nanmean(f) - u, t_net=t_n, t_excess=t_x,
                                             first=str(IDX[keep[0]].date()), last=str(IDX[keep[-1]].date())))
    dm = pd.DataFrame(rows)
    dm.to_csv("out_research10_alt_3_mkt.csv", index=False)
    print("MARKET EVENTS"); print(fmt(dm))


if __name__ == "__main__":
    main()
