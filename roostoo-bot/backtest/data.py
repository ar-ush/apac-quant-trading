"""Historical data: Binance spot klines (Roostoo prices are streamed from Binance, so Binance history is the proxy).

    python -m backtest.data --since 2023-01-01          # downloads hourly klines for every Roostoo crypto pair
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).resolve().parent.parent / "data_cache"
HOSTS = ["https://data-api.binance.vision", "https://api.binance.com"]
H_MS = 3_600_000
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_vol", "trades", "tb_base", "tb_quote"]


def fetch_klines(symbol: str, start_ms: int, end_ms: int, interval: str = "1h") -> pd.DataFrame:
    out, cur, host = [], start_ms, 0
    while cur < end_ms:
        for attempt in range(6):
            try:
                r = requests.get(f"{HOSTS[host]}/api/v3/klines", timeout=20,
                                 params={"symbol": symbol, "interval": interval, "startTime": cur, "limit": 1000})
                if r.status_code == 400:
                    return pd.DataFrame(columns=COLS)
                if r.status_code in (403, 451):
                    host = (host + 1) % len(HOSTS)
                    continue
                r.raise_for_status()
                rows = r.json()
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"giving up on {symbol}")
        if not rows:
            break
        out.extend(rows)
        cur = rows[-1][0] + H_MS
        if len(rows) < 1000:
            break
    df = pd.DataFrame(out, columns=COLS + ["ignore"]).drop(columns="ignore") if out else pd.DataFrame(columns=COLS)
    for c in COLS:
        df[c] = pd.to_numeric(df[c])
    return df


def download(coins, since: str, outdir: Path = CACHE) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    start = int(datetime.fromisoformat(since).replace(tzinfo=timezone.utc).timestamp() * 1000)
    end = int(time.time() * 1000)

    def one(c):
        df = fetch_klines(f"{c}USDT", start, end)
        if len(df):
            df.to_csv(outdir / f"{c}.csv.gz", index=False)
        return c, len(df)

    with ThreadPoolExecutor(6) as ex:
        for c, n in ex.map(one, coins):
            print(f"{c:12s}{n}")


def load_panel(coins=None, cache: Path = CACHE):
    close, qv = {}, {}
    for f in sorted(cache.glob("*.csv.gz")):
        c = f.name.split(".")[0]
        if coins and c not in coins:
            continue
        df = pd.read_csv(f)
        idx = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        close[c] = pd.Series(df["close"].values, index=idx)
        qv[c] = pd.Series(df["quote_vol"].values, index=idx)
    return pd.DataFrame(close).sort_index(), pd.DataFrame(qv).sort_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2023-01-01")
    ap.add_argument("--exchange-info", default="", help="path to a saved /v3/exchangeInfo json; default: query Roostoo")
    a = ap.parse_args()
    if a.exchange_info:
        info = json.load(open(a.exchange_info))
    else:
        info = requests.get("https://mock-api.roostoo.com/v3/exchangeInfo", timeout=20).json()
    coins = [v["Coin"] for v in info["TradePairs"].values() if v.get("AssetType", "crypto") == "crypto"]
    download(coins, a.since)
