"""Extra jump-model variant (return-only trend features, closer to a trend regime than the crash-cluster default)."""
import numpy as np, pandas as pd
import research9_regime_models as m


def feats(px):
    r = np.log(px).diff()
    return pd.DataFrame({"ew5": r.ewm(halflife=5).mean(), "ew21": r.ewm(halflife=21).mean(),
                         "ew63": r.ewm(halflife=63).mean(), "lv": np.log(r.rolling(14).std())}).dropna()


m.jm_features = feats  # swap feature set; column 1 (ew21) still used to label the bull state
if __name__ == "__main__":
    px = m.btc_daily()
    refit = list(pd.date_range(pd.Timestamp("2023-04-01", tz="UTC"), px.index[-1], freq="MS"))
    P = pd.read_csv(m.HERE / "out_research9_regime_post.csv", index_col=0, parse_dates=True)
    for lam in (5, 15, 40):
        h, s = m.jm_walk(px, lam, refit, T_soft=1.0)
        P[f"jmr{lam}h"] = h; P[f"jmr{lam}s"] = s
        print(lam, h.loc["2023-04-01":].groupby(h.loc["2023-04-01":].index.year).mean().round(2).to_dict(), flush=True)
    P.to_csv(m.HERE / "out_research9_regime_post.csv")
