"""Roostoo REST client: HMAC signing, server-clock offset, rolling call budget, retries, request log.

The only module that talks to Roostoo. Key and signature travel in headers and are never logged.
Order-creating calls are never retried (a timed-out request may still have been executed).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import requests

log = logging.getLogger("roostoo.client")
api_log = logging.getLogger("roostoo.api")

EMPTY_RESULT_HINTS = ("no pending order", "no order matched", "no order")


class RoostooError(Exception):
    def __init__(self, message: str, status: Optional[int] = None, payload: Any = None, ambiguous: bool = False):
        super().__init__(message)
        self.status = status
        self.payload = payload
        self.ambiguous = ambiguous  # True: the request may have been executed although we saw an error


class _Retryable(Exception):
    pass


def build_query(params: Dict[str, Any]) -> str:
    """Key-sorted `k=v&k=v`: exactly the string that is signed and sent."""
    return "&".join(f"{k}={params[k]}" for k in sorted(params))


def sign(query: str, secret: str) -> str:
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


class CallBudget:
    """At most `max_calls` in any rolling `period` seconds (blocks when exhausted)."""

    def __init__(self, max_calls: int, period: float = 60.0, clock=time.monotonic, sleep=time.sleep):
        self.max_calls, self.period, self.clock, self.sleep = max_calls, period, clock, sleep
        self.calls: Deque[float] = deque()

    def acquire(self) -> None:
        while True:
            now = self.clock()
            while self.calls and now - self.calls[0] >= self.period:
                self.calls.popleft()
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return
            self.sleep(max(self.period - (now - self.calls[0]), 0.05))

    def used(self) -> int:
        now = self.clock()
        return sum(1 for c in self.calls if now - c < self.period)


class RoostooClient:
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://mock-api.roostoo.com",
                 max_calls_per_minute: int = 20, timeout: float = 10.0, retries: int = 3, backoff: float = 1.0,
                 session: Optional[requests.Session] = None):
        self.api_key, self._secret = api_key, secret_key
        self.base_url = base_url.rstrip("/")
        self.timeout, self.retries, self.backoff = timeout, retries, backoff
        self.session = session or requests.Session()
        self.budget = CallBudget(max_calls_per_minute)
        self.clock_offset_ms = 0
        self.recent: Deque[bool] = deque(maxlen=30)  # True = call succeeded; feeds the API-error guard

    def __repr__(self) -> str:
        return f"RoostooClient({self.base_url})"

    # ---------------------------------------------------------------- clock
    def now_ms(self) -> int:
        return int(time.time() * 1000) + self.clock_offset_ms

    def sync_clock(self) -> int:
        t0 = time.time()
        server = self.server_time()
        t1 = time.time()
        self.clock_offset_ms = server - int((t0 + t1) / 2 * 1000)
        return self.clock_offset_ms

    # ---------------------------------------------------------------- public
    def server_time(self) -> int:
        return int(self._request("GET", "/v3/serverTime")["ServerTime"])

    def exchange_info(self) -> Dict[str, Any]:
        return self._request("GET", "/v3/exchangeInfo")

    def ticker(self, pair: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        params = {"pair": pair} if pair else {}
        return self._request("GET", "/v3/ticker", params, timestamp=True)["Data"]

    # ---------------------------------------------------------------- signed
    def balance(self) -> Dict[str, Dict[str, float]]:
        body = self._request("GET", "/v3/balance", signed=True)
        wallet = body.get("SpotWallet", body.get("Wallet"))
        if wallet is None:
            raise RoostooError("balance response has no wallet", payload=body)
        return wallet

    def place_order(self, pair: str, side: str, quantity: str, order_type: str = "MARKET",
                    price: Optional[str] = None) -> Dict[str, Any]:
        params = {"pair": pair, "side": side, "type": order_type, "quantity": quantity}
        if order_type == "LIMIT":
            if price is None:
                raise ValueError("LIMIT order needs a price")
            params["price"] = price
        body = self._request("POST", "/v3/place_order", params, signed=True, retry=False)
        return body.get("OrderDetail", body)

    def query_order(self, order_id: Optional[int] = None, pair: Optional[str] = None,
                    pending_only: Optional[bool] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if order_id is not None:
            params["order_id"] = order_id  # the API accepts no other filter together with order_id
        else:
            if pair:
                params["pair"] = pair
            if pending_only is not None:
                params["pending_only"] = "TRUE" if pending_only else "FALSE"
            if limit is not None:
                params["limit"] = limit
        body = self._request("POST", "/v3/query_order", params, signed=True)
        return body.get("OrderMatched") or []

    def cancel_order(self, order_id: Optional[int] = None, pair: Optional[str] = None) -> List[int]:
        params: Dict[str, Any] = {}
        if order_id is not None:
            params["order_id"] = order_id
        elif pair:
            params["pair"] = pair
        body = self._request("POST", "/v3/cancel_order", params, signed=True)
        return body.get("CanceledList") or []

    # ---------------------------------------------------------------- transport
    def error_rate(self) -> float:
        return 0.0 if not self.recent else 1.0 - sum(self.recent) / len(self.recent)

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, signed: bool = False,
                 timestamp: bool = False, retry: bool = True) -> Dict[str, Any]:
        attempts = 1 + (self.retries if retry else 0)
        for attempt in range(1, attempts + 1):
            try:
                out = self._send(method, path, dict(params or {}), signed, timestamp or signed)
                self.recent.append(True)
                return out
            except _Retryable as exc:
                self.recent.append(False)
                if attempt == attempts:
                    raise RoostooError(f"{method} {path} failed after {attempts} attempt(s): {exc}",
                                       ambiguous=not retry)
                time.sleep(self.backoff * 2 ** (attempt - 1))
            except RoostooError:
                self.recent.append(False)
                raise
        raise AssertionError("unreachable")

    def _send(self, method: str, path: str, params: Dict[str, Any], signed: bool, timestamp: bool) -> Dict[str, Any]:
        if timestamp:
            params["timestamp"] = str(self.now_ms())
        query = build_query(params)
        headers: Dict[str, str] = {}
        if signed:
            headers["RST-API-KEY"] = self.api_key
            headers["MSG-SIGNATURE"] = sign(query, self._secret)
        url = self.base_url + path
        self.budget.acquire()
        t0 = time.monotonic()
        try:
            if method == "GET":
                resp = self.session.get(url + ("?" + query if query else ""), headers=headers, timeout=self.timeout)
            else:
                headers["Content-Type"] = "application/x-www-form-urlencoded"
                resp = self.session.post(url, data=query, headers=headers, timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            self._log(method, path, params, None, t0, repr(exc))
            raise _Retryable(repr(exc))
        self._log(method, path, params, resp.status_code, t0, resp.text)
        if resp.status_code == 429 or resp.status_code >= 500:
            raise _Retryable(f"HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise RoostooError(f"HTTP {resp.status_code} from {path}: {resp.text[:200]}", status=resp.status_code)
        try:
            body = resp.json()
        except ValueError:
            raise RoostooError(f"non-JSON response from {path}: {resp.text[:200]}", status=200)
        if not isinstance(body, dict):
            raise RoostooError(f"unexpected response from {path}: {resp.text[:200]}", status=200)
        if body.get("Success") is False:  # HTTP 200 can still carry a failure
            msg = str(body.get("ErrMsg") or "")
            if any(h in msg.lower() for h in EMPTY_RESULT_HINTS):
                return body
            raise RoostooError(f"{method} {path}: {msg}", status=200, payload=body)
        return body

    @staticmethod
    def _log(method, path, params, status, t0, text) -> None:
        api_log.info(json.dumps({"method": method, "path": path, "params": params, "status": status,
                                 "ms": int((time.monotonic() - t0) * 1000), "response": text[:2000]},
                                separators=(",", ":")))
