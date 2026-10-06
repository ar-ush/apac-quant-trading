"""Fetch free auxiliary daily data (read-only public APIs): Yahoo cross-asset, DefiLlama stablecoin supply, Fear&Greed."""
import json, urllib.request, time
import pandas as pd
from pathlib import Path
D = Path(__file__).parent / "data/aux"
def get(u):
    r = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=60)
    return json.loads(r.read())
syms = {"SPX": "^GSPC", "NDX": "^IXIC", "VIX": "^VIX", "DXY": "DX-Y.NYB", "GOLD": "GC=F", "TNX": "^TNX", "QQQ": "QQQ", "HYG": "HYG", "IBIT": "IBIT"}
for k, s in syms.items():
    try:
        j = get(f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(s)}?period1=1640995200&period2=1791300000&interval=1d")
        r = j["chart"]["result"][0]
        idx = pd.to_datetime(r["timestamp"], unit="s").normalize()
        df = pd.DataFrame({"close": r["indicators"]["quote"][0]["close"]}, index=idx)
        df = df[~df.index.duplicated(keep="last")].dropna()
        df.to_csv(D / f"{k}.csv"); print(k, len(df), df.index[0].date(), df.index[-1].date())
    except Exception as e:
        print(k, "ERR", e)
    time.sleep(0.5)
j = get("https://stablecoins.llama.fi/stablecoincharts/all")
df = pd.DataFrame([{"date": pd.to_datetime(int(x["date"]), unit="s"), "usd": x["totalCirculatingUSD"].get("peggedUSD")} for x in j]).set_index("date")
df.to_csv(D / "stablecoin_total.csv"); print("stable", len(df), df.index[-1].date())
j = get("https://api.alternative.me/fng/?limit=0&format=json")["data"]
df = pd.DataFrame([{"date": pd.to_datetime(int(x["timestamp"]), unit="s"), "fng": float(x["value"])} for x in j]).set_index("date").sort_index()
df.to_csv(D / "fng.csv"); print("fng", len(df), df.index[0].date(), df.index[-1].date())
