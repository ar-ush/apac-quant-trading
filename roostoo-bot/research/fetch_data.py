"""Download Binance spot klines for every Roostoo crypto pair (research data).

Roostoo prices are streamed from Binance, so Binance history is the backtest proxy.
Usage: python fetch_data.py --interval 1h --since 2023-01-01
Writes work/data/klines_<interval>/<COIN>.csv.gz  (open_time_ms,open,high,low,close,volume,quote_vol,trades,taker_buy_quote)
"""
import argparse
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).parent
HOSTS = ["https://data-api.binance.vision", "https://api.binance.com"]
STEP_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def klines(symbol, interval, start_ms, end_ms):
    out = []
    cur = start_ms
    host_i = 0
    while cur < end_ms:
        for attempt in range(6):
            try:
                r = requests.get(
                    f"{HOSTS[host_i]}/api/v3/klines",
                    params={"symbol": symbol, "interval": interval, "startTime": cur, "limit": 1000},
                    timeout=20,
                )
                if r.status_code == 400:
                    return out  # symbol does not exist / not yet listed
                if r.status_code in (403, 451):
                    host_i = (host_i + 1) % len(HOSTS)
                    continue
                r.raise_for_status()
                rows = r.json()
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"failed {symbol} {interval} {cur}")
        if not rows:
            break
        out.extend(rows)
        cur = rows[-1][0] + STEP_MS[interval]
        if len(rows) < 1000:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--since", default="2023-01-01")
    ap.add_argument("--coins", default="")
    args = ap.parse_args()

    ei_path = HERE / "data/raw/exchangeInfo.json"
    if not ei_path.exists():  # first run: pull the current Roostoo universe
        ei_path.parent.mkdir(parents=True, exist_ok=True)
        ei_path.write_text(requests.get("https://mock-api.roostoo.com/v3/exchangeInfo", timeout=20).text)
    ei = json.load(open(ei_path))["TradePairs"]
    coins = [v["Coin"] for v in ei.values() if v["AssetType"] == "crypto"]
    if args.coins:
        coins = args.coins.split(",")
    start_ms = int(datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(time.time() * 1000)
    outdir = HERE / f"data/klines_{args.interval}"
    outdir.mkdir(parents=True, exist_ok=True)

    def work(coin):
        sym = f"{coin}USDT"
        rows = klines(sym, args.interval, start_ms, end_ms)
        if not rows:
            return coin, 0
        with gzip.open(outdir / f"{coin}.csv.gz", "wt") as f:
            f.write("open_time,open,high,low,close,volume,close_time,quote_vol,trades,taker_buy_base,taker_buy_quote\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]},{r[7]},{r[8]},{r[9]},{r[10]}\n")
        return coin, len(rows)

    with ThreadPoolExecutor(6) as ex:
        for coin, n in ex.map(work, coins):
            print(f"{coin:12s} {n}", flush=True)


if __name__ == "__main__":
    main()
