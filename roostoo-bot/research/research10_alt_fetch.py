"""research10_alt_fetch: download + cache derivatives/sentiment data to work/data/deriv/ (public no-key endpoints)."""
import json, sys, time
from pathlib import Path
import pandas as pd, requests

D = Path(__file__).parent / "data/deriv"
D.mkdir(parents=True, exist_ok=True)
FAPI = "https://fapi.binance.com"
S = requests.Session()
T0 = int(pd.Timestamp("2022-11-15", tz="UTC").timestamp() * 1000)
NOW = int(time.time() * 1000)


def get(url, params, tries=5):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 418):
                time.sleep(5 * (i + 1)); continue
            if r.status_code in (400, 404):
                return None
            if r.status_code in (403, 451):
                print("GEOBLOCK", url, r.status_code); return None
        except Exception as e:
            time.sleep(2)
    return None


def sym_for(c):
    return {"FLOKI": "1000FLOKIUSDT", "PEPE": "1000PEPEUSDT", "BONK": "1000BONKUSDT", "SHIB": "1000SHIBUSDT"}.get(c, c + "USDT")


def fetch_funding(sym):
    out, st = [], T0
    while st < NOW:
        j = get(FAPI + "/fapi/v1/fundingRate", dict(symbol=sym, startTime=st, limit=1000))
        if not j: break
        out += j
        st = j[-1]["fundingTime"] + 1
        if len(j) < 1000: break
        time.sleep(0.15)
    if not out: return None
    df = pd.DataFrame(out)
    df["t"] = pd.to_datetime(df.fundingTime, unit="ms", utc=True).dt.round("1min")
    return df.set_index("t")["fundingRate"].astype(float)


def fetch_prem(sym, kind="premiumIndexKlines", interval="1h"):
    out, st = [], T0
    while st < NOW:
        j = get(FAPI + f"/fapi/v1/{kind}", dict(symbol=sym, interval=interval, startTime=st, limit=1500))
        if not j: break
        out += j
        st = j[-1][0] + 1
        if len(j) < 1500: break
        time.sleep(0.15)
    if not out: return None
    df = pd.DataFrame(out).iloc[:, :5]
    df.columns = ["t", "o", "h", "l", "c"]
    df["t"] = pd.to_datetime(df.t, unit="ms", utc=True)
    return df.set_index("t").astype(float)


def main():
    from lib import load_panel
    import research1 as r1
    coins = list(r1.C.columns)
    fund, prem = {}, {}
    for c in coins:
        s = sym_for(c)
        f = D / f"fund_{c}.csv"; p = D / f"prem_{c}.csv"
        if f.exists() and p.exists():
            fund[c] = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]; prem[c] = pd.read_csv(p, index_col=0, parse_dates=True)
            continue
        fs = fetch_funding(s); pm = fetch_prem(s)
        print(c, s, None if fs is None else (len(fs), fs.index[0]), None if pm is None else (len(pm), pm.index[0]), flush=True)
        if fs is not None and pm is not None:
            fs.to_csv(f); pm.to_csv(p); fund[c] = fs; prem[c] = pm
    # sentiment / macro
    j = get("https://api.alternative.me/fng/", dict(limit=0))
    if j:
        d = pd.DataFrame(j["data"]); d["t"] = pd.to_datetime(d.timestamp.astype(int), unit="s", utc=True)
        d.set_index("t")["value"].astype(float).sort_index().to_csv(D / "fng.csv")
    j = get("https://stablecoins.llama.fi/stablecoincharts/all", {})
    if j:
        d = pd.DataFrame({"t": pd.to_datetime([int(x["date"]) for x in j], unit="s", utc=True),
                          "usd": [x["totalCirculatingUSD"]["peggedUSD"] for x in j]}).set_index("t")
        d.to_csv(D / "stable.csv")
    rows = []
    st = int(pd.Timestamp("2022-11-01", tz="UTC").timestamp() * 1000)
    while st < NOW:
        en = st + 30 * 24 * 3600 * 1000 * 2
        r = get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                dict(currency="BTC", start_timestamp=st, end_timestamp=en, resolution=3600))
        if r and "result" in r: rows += r["result"]["data"]
        st = en; time.sleep(0.2)
    if rows:
        d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c"]); d["t"] = pd.to_datetime(d.t, unit="ms", utc=True)
        d.drop_duplicates("t").set_index("t").sort_index().to_csv(D / "dvol.csv")
    print("done")


if __name__ == "__main__":
    main()
