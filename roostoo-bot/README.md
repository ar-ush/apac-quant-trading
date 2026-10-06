# Roostoo Quant Bot — BTC-gated momentum rotation

An autonomous, rule-based crypto trading bot for the **HK vs AU vs IN Quant Trading Hackathon** (Roostoo mock exchange).
Every order is produced by code in this repository; there is no manual trading, override or discretionary API call.

## 1. Project overview

* **Strategy family:** cross-sectional momentum with a market-regime filter (long-only, spot, no leverage).
* **Idea in one paragraph:** every day at 00:00 UTC the bot ranks the 25 most liquid Roostoo coins by their 14-day return and
  holds the top three (equal weight) — but only while Bitcoin's slow trend is up (EMA 168h > EMA 672h, checked hourly). When the
  gate closes it goes to cash. Research showed this is the only structure with a robust edge across parameters and years;
  brakes, stops, ML rankers and short-horizon tricks all lowered return or the competition score (see `docs/RESEARCH.md`).
* **Key features:** one strategy class shared by the live bot and the backtester (no research/production drift) · all versions
  kept in `strategies/` · maker-first execution with market fallback and a chase cap · stateless about positions (the wallet is the
  truth, so restarts are safe) · hard client-side API budget (≤ 20 calls/min; the limit is 30) · full audit journal with the strategy
  reason on every order · catastrophe kill-switch · 24 tests.

## 2. Architecture

```
Binance klines (public) ──► bot/data/binance.py ─┐
Roostoo ticker/balance ──► bot/execution/client.py ─┤
                                                    ▼
                     bot/engine.py  (hourly bar → Snapshot)
                                                    │
                     strategies/<version>.decide(Snapshot) → target weights   (pure, no I/O)
                                                    │
                     bot/risk/guards.py  (kill-switch, stale data, price sanity)
                                                    │
                     bot/execution/executor.py  (sells → buys, limit → cancel → market)
                                                    │
                     bot/journal.py  → logs/{decisions,orders,equity,events}-YYYYMMDD.jsonl
```

| component | path |
|---|---|
| data | `bot/data/binance.py` (closed bars only, host failover), `bot/data/universe.py` (liquid pool) |
| strategy | `strategies/` — `base.py`, `v1_momentum_rotation.py`, registry, `VERSIONS.md` |
| execution | `bot/execution/client.py` (signing, rate budget), `rules.py` (precision), `executor.py` |
| risk | `bot/risk/guards.py`, activity rule in `bot/engine.py` |
| logging | `bot/journal.py`, `tools/report.py` |
| backtest / research | `backtest/` (replays the live strategy class), `research/`, `docs/RESEARCH.md` |
| config | `config/default.yaml` (every change is a commit) |
| tests | `tests/` (`python -m pytest tests -q`) |

Tech stack: Python 3.11 (the version installed by `deploy/setup_ec2.sh`), pandas, numpy, requests, PyYAML. The Dockerfile is provided but not used for the live deployment (the bot runs under tmux on the organisers' EC2).

## 3. Strategy explanation (v1)

Full write-up with maths and evidence: [`strategies/V1_MOMENTUM_ROTATION.md`](strategies/V1_MOMENTUM_ROTATION.md).

* **Universe:** top-25 Roostoo crypto pairs by 30-day median Binance dollar volume, spread ≤ 10 bps, ≥ 60 days of history, no stablecoins/PAXG/tokenised stocks. Refreshed daily.
* **Entry:** at 00:00 UTC buy the top-3 coins by 336-hour return that are also positive, equal weight 1/3 each (gross ≤ 98%) — only if the BTC gate is on.
* **Exit:** a holding is sold when its rank drops below 5, its return turns ≤ 0, or the gate turns off (any hour). Winners are not topped up and are trimmed above 1.6× target.
* **Risk management:** no leverage; BTC regime gate; catastrophe kill-switch at −22% from peak (cash for 24 h); refuses stale data; drops entries whose Roostoo price disagrees with Binance by > 3%; client-side API budget; never retries an order POST after a timeout.
* **Position sizing:** 1/3 of equity per slot; fewer slots ⇒ cash.
* **Activity rule:** the contest needs trading on ≥ 8 days. If nothing filled by 20:00 UTC, the bot makes one small maintenance trade (0.3% of equity in BTC, reversed the next time it is needed). It is logged as `activity_probe_*` and excluded from the strategy's own holdings.
* **Assumptions:** Roostoo prices mirror Binance (confirmed by the organisers); fees 0.1% taker / 0.05% maker; we measured round-trip cost ≈ 0.2–0.3% on small caps including slippage.

### Backtest summary (0.12% per side, 2023-04 → 2026-10)
Rolling 14-day windows: mean +5.4%, p10 −11%, p90 +32%, positive in 39% (cash ~40% of the time). Total return +3,109%, max drawdown 59%. These are regime-dependent, survivorship-biased numbers — see the caveats in the strategy doc. Reproduce: `python -m backtest.run_backtest`.

## 4. Setup and how to run

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # fill in the keys; NEVER commit .env
python -m pytest tests -q                              # 24 tests, no network
python -m bot.main --check                             # read-only: connectivity + account equity
python -m bot.main --once                              # dry-run decision cycle (default is dry-run)
python -m bot.main --live                              # real orders on the account in ROOSTOO_ACCOUNT
python -m tools.report                                 # summary of equity / fills / reasons from logs/
```

Backtest: `python -m backtest.data --since 2023-01-01` then `python -m backtest.run_backtest --start 2023-04-01 --cost 0.0012`.

### Deploy on the organisers' EC2 (Sydney, Session Manager)
```bash
git clone <this-repo> && cd <repo>/roostoo-bot && bash deploy/setup_ec2.sh   # the bot lives in the roostoo-bot/ subfolder
cp .env.example .env && nano .env        # ROOSTOO_ACCOUNT=competition + competition keys
python -m bot.main --check               # must print account=competition and equity ~ 100,000
curl -s -o /dev/null -w "%{http_code}\n" "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=2"   # expect 200
tmux new -s bot
bash deploy/run_bot.sh                   # auto-restarts the bot; detach: Ctrl+B then D
```
Updating the strategy = edit code/config → commit → `git pull` on the box → bot restarts (stateless positions).

## 5. Compliance notes
* Trades only through `bot/execution/executor.py`, driven by `Strategy.decide()`; no ad-hoc order scripts exist (the only manual tool, `--flatten`, is hard-refused on the competition account).
* Every order row in `logs/orders-*.jsonl` carries `reason`, limit price, reference price, fill, fee and role.
* No HFT, market making or arbitrage: ≤ ~1 decision per hour, positions held for days.
* Strategy versions live in `strategies/` with `VERSIONS.md`; parameter changes live in `config/default.yaml` (one commit each).

## 6. Live results
_Updated during the competition (Oct 4–17)._
