"""Live market data from Binance public endpoints (Roostoo's prices are streamed from Binance).

Only this module talks to Binance. Hosts fail over (data-api.binance.vision mirror) because the main host answers 451
from some regions. Only CLOSED bars are ever returned."""
from __future__ import annotations

import logging
import time
from typing import List, Optional

import pandas as pd
import requests

log = logging.getLogger("binance")
HOSTS = ["https://data-api.binance.vision", "https://api.binance.com", "https://api1.binance.com"]
H_MS = 3_600_000


class BinanceUnavailable(Exception):
    pass


class BinanceData:
    def __init__(self, session: Optional[requests.Session] = None, timeout: float = 15.0):
        self.s = session or requests.Session()
        self.timeout = timeout
        self._host = 0

    def _get(self, path: str, params: dict):
        last = None
        for attempt in range(len(HOSTS) * 2):
            host = HOSTS[(self._host + attempt) % len(HOSTS)]
            try:
                r = self.s.get(host + path, params=params, timeout=self.timeout)
                if r.status_code == 400:
                    raise BinanceUnavailable(f"bad request {params.get('symbol')}: {r.text[:100]}")
                if r.status_code in (403, 418, 429, 451) or r.status_code >= 500:
                    last = f"HTTP {r.status_code} from {host}"
                    time.sleep(0.5)
                    continue
                r.raise_for_status()
                self._host = (self._host + attempt) % len(HOSTS)
                return r.json()
            except BinanceUnavailable:
                raise
            except Exception as exc:  # network error: try the next host
                last = repr(exc)
                time.sleep(0.5)
        raise BinanceUnavailable(last or "unknown error")

    def klines(self, coin: str, interval: str, n: int, now_ms: Optional[int] = None) -> pd.DataFrame:
        """Last n CLOSED klines for COINUSDT (chunked backwards for n > 1000)."""
        now_ms = now_ms or int(time.time() * 1000)
        step = {"1h": H_MS, "1d": 24 * H_MS}[interval]
        rows: List[list] = []
        end = now_ms
        while len(rows) < n:
            take = min(1000, n - len(rows) + 2)
            chunk = self._get("/api/v3/klines", {"symbol": f"{coin}USDT", "interval": interval, "limit": take, "endTime": end})
            if not chunk:
                break
            rows = chunk + rows
            end = chunk[0][0] - 1
            if len(chunk) < take:
                break
        df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_vol",
                                         "trades", "tb_base", "tb_quote", "ignore"])
        if df.empty:
            return df
        for c in ["open", "high", "low", "close", "volume", "quote_vol"]:
            df[c] = pd.to_numeric(df[c])
        df = df[df["open_time"] + step <= now_ms].drop_duplicates("open_time").tail(n)
        df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df

    def hourly_closes(self, coins: List[str], n_by_coin: dict, now_ms: Optional[int] = None) -> pd.DataFrame:
        series = {}
        for c in coins:
            try:
                df = self.klines(c, "1h", n_by_coin.get(c, 400), now_ms)
                if len(df):
                    series[c] = df["close"]
            except BinanceUnavailable as exc:
                log.warning("no hourly data for %s: %s", c, exc)
        return pd.DataFrame(series).sort_index()

    def daily_stats(self, coin: str, days: int = 60):
        """(median daily quote volume over the last 30 closed days, number of closed days available)."""
        df = self.klines(coin, "1d", days)
        if df.empty:
            return float("nan"), 0
        return float(df["quote_vol"].tail(30).median()), int(len(df))
