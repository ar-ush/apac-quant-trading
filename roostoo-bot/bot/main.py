"""Entry point.

    python -m bot.main --check            # read-only connectivity / account check, places nothing
    python -m bot.main                    # DRY-RUN (default): full loop, orders are only logged
    python -m bot.main --live             # real orders on the account selected by ROOSTOO_ACCOUNT
"""
from __future__ import annotations

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

from bot.config import ROOT, load_config
from bot.data.binance import BinanceData
from bot.engine import Bot
from bot.execution.client import RoostooClient
from bot.journal import Journal
from bot.state import StateStore


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = RotatingFileHandler(log_dir / "bot.log", maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    api = logging.getLogger("roostoo.api")
    ah = RotatingFileHandler(log_dir / "api.log", maxBytes=10_000_000, backupCount=20)
    ah.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    api.addHandler(ah)
    api.propagate = False  # request/response bodies go to api.log only (never to the console)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--account", choices=["test", "competition"], default=None)
    ap.add_argument("--live", action="store_true", help="send real orders (default is dry-run)")
    ap.add_argument("--check", action="store_true", help="read-only account/connectivity check")
    ap.add_argument("--once", action="store_true", help="run a single tick and exit")
    ap.add_argument("--flatten", action="store_true", help="TEST account only: sell every position (never allowed on the competition account)")
    a = ap.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = load_config(a.config, a.account)
    log_dir = ROOT / cfg.runtime.log_dir
    setup_logging(log_dir)
    if not cfg.api_key or not cfg.secret_key:
        print("API keys missing: fill .env (see .env.example)")
        return 2
    client = RoostooClient(cfg.api_key, cfg.secret_key, cfg.base_url, cfg.execution.max_calls_per_minute)
    journal = Journal(log_dir)
    store = StateStore(ROOT / cfg.runtime.state_dir, f"{cfg.account}-{cfg.strategy.name}")
    bot = Bot(cfg, client, BinanceData(), journal, store, dry_run=not a.live)

    if a.check:
        bot.startup()
        wallet, quotes, equity, held = bot.snapshot_account()
        print(f"account={cfg.account} equity={equity:,.2f} USD_free={wallet.get('USD', {}).get('Free')} held={held}")
        print(f"clock offset ms={client.clock_offset_ms}  pairs={len(bot.rules)}  tickers={len(quotes)}")
        return 0
    if a.flatten:
        if cfg.account != "test":
            print("--flatten is refused on the competition account (no manual intervention allowed)")
            return 2
        bot.startup()
        wallet, quotes, equity, held = bot.snapshot_account()
        orders = bot.executor.rebalance({}, {c: "test_flatten" for c in wallet if c != "USD"}, equity, wallet, quotes)
        print(f"flatten: {len(orders)} orders; dry_run={bot.dry_run}")
        return 0
    if a.once:
        bot.startup()
        bot.tick()
        return 0
    bot.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
