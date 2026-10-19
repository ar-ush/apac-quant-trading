# Roostoo Quant Bot — BTC-gated momentum rotation

An autonomous, rule-based crypto trading bot for the **HK vs AU vs IN Quant Trading Hackathon** (Roostoo mock exchange).
Every order is produced by code in this repository; there is no manual trading, override or discretionary API call.

**Contents:** 1 Overview · 2 How the portfolio is curated · 3 Strategy and the thinking behind it · 4 Implementation (code map) ·
5 Backtesting · 6 How the code reflects the idea · 7 Maker and taker fees · 8 Risk management · 9 Setup and running ·
10 Compliance · 11 Live results · 12 Limitations and honesty notes

---

## 1. Overview

* **Strategy family:** cross-sectional momentum with a market-regime filter. Long-only, spot, no leverage, no shorting.
* **Idea in one paragraph:** every day at 00:00 UTC the bot ranks the 25 most liquid Roostoo coins by their 14-day return and
  holds the top three (equal weight), but only while Bitcoin's slow trend is up (EMA 168h > EMA 672h, checked hourly). When the
  gate closes, the bot goes to cash. Our research (17 rounds, `docs/RESEARCH.md`) found this was the only structure with a
  robust edge across parameters and years. Brakes, stops, ML rankers, shorts and short-horizon tricks all lowered return or the
  competition score.
* **Key features:** one strategy class shared by the live bot and the backtester (no research/production drift) · every strategy
  version kept in `strategies/` · maker-first execution with market fallback and a chase cap · positions are never stored by the
  bot (the wallet is the source of truth, so restarts are safe) · hard client-side API budget (≤ 20 calls/min, the limit is 30) ·
  full audit journal with the strategy reason on every order · catastrophe kill-switch · unit tests.

## 2. How the portfolio is curated

The portfolio is rebuilt by rules, never by hand. Each step below maps to a function in the code.

1. **Pick the investable pool (daily).** From all Roostoo crypto pairs keep the **top 25 by 30-day median Binance dollar volume**,
   with bid/ask spread ≤ 10 bps and ≥ 60 days of history. Stablecoins, PAXG, WBTC and tokenised stocks are excluded.
   *Why:* a multi-day edge is wiped out by wide spreads on thin coins, so liquidity filters come before any signal
   (`bot/data/universe.py`).
2. **Check the regime (every hour).** `risk_on = EMA(BTC, 168h) > EMA(BTC, 672h)`. If off, every position is sold and the
   portfolio is 100% cash. If on, continue.
3. **Rank the pool (daily at 00:00 UTC).** For each coin compute the 336-hour (14-day) return from closed hourly bars.
4. **Select.** Hold the **top 3 coins whose 14-day return is also positive**. A coin already held is kept while its rank ≤ 5
   and its return is > 0 (hysteresis, so we do not churn on rank 3 ↔ 4 noise).
5. **Size.** Equal weight, 1/3 of equity per slot, total invested ≤ 98%. If fewer than three coins qualify, the rest stays
   in cash. Winners are not topped up, and a winner that grows above 1.6× its target weight is trimmed.
6. **Execute.** Sells first, then buys, using passive limit orders first (maker fee) and a market fallback (taker fee).

Example of a day's curation: gate on, the pool is ranked, say, `NEAR +31%, FET +27%, AAVE +19%, ...`; the three are bought at
~33% each. Next day `FET` falls to rank 6 → sold; the next qualifying coin is bought; the other two are held untouched.

## 3. Strategy and the thinking behind it

Full write-up with the maths and evidence: [`strategies/V1_MOMENTUM_ROTATION.md`](strategies/V1_MOMENTUM_ROTATION.md).

**Thesis**
1. **Cross-sectional momentum exists in crypto at 1–4 week horizons** (Liu, Tsyvinski & Wu, *J. Finance* 2022; Fieberg et al.,
   *JFQA* 2024). The strongest liquid coins over two weeks tend to keep outperforming for days, because retail attention and
   liquidity flow into winners and information spreads slowly across a fragmented market.
2. **It only pays in up-trending regimes.** In down-trends the "winners" are bounces that fail. A slow BTC trend filter separates
   the regimes and flips only a few times a month.
3. **Liquidity matters more than the signal.** Restricting to the 25 most liquid coins keeps spreads and slippage small relative
   to the edge.
4. **The competition scoring favours return first.** Per the FAQ the top 20 per region are chosen by return, then ranked on a
   composite of Sortino, Sharpe and Calmar. Overlays that cut variance by cutting exposure lowered the return and did not
   help the composite, so the bot keeps risk controls wide and relies on the regime gate for protection.

**How we got here.** We tested time-series momentum, Donchian breakouts, low-vol trend, BTC trend, vol-targeting, drawdown
brakes, profit locks, 15-minute burst continuation, ML rankers, regime models (HMM), funding-rate and sentiment signals,
stat-arb pairs, and long-short books. Details, numbers and the failures are in [`docs/RESEARCH.md`](docs/RESEARCH.md) and
the scripts in `research/`. We also studied other public bots for ideas and failure modes only (no code was copied), and
re-tested their main ideas (multi-horizon ranking, inverse-vol / efficiency-ratio weights, per-coin trend filters, ATR stops).
None beat v1 beyond noise.

**Version control of strategies.** Strategies are separate modules in `strategies/`, listed with dates and reasons in
[`strategies/VERSIONS.md`](strategies/VERSIONS.md). Parameter changes happen only in `config/default.yaml`, one commit each.

| Rule | Value |
|---|---|
| Universe | top-25 liquid Roostoo coins, refreshed daily |
| Regime gate | BTC EMA168h > EMA672h (hourly) |
| Signal | 336h return, must be > 0 |
| Positions | top 3, equal weight, hysteresis 2, trim at 1.6× |
| Rebalance | daily, 00:00 UTC (the gate can force an exit at any hour) |
| Exit | rank > 5, return ≤ 0, or gate off |
| Leverage / shorts | none |

## 4. Implementation (code map)

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

| Component | Path | What it does |
|---|---|---|
| Entry point | `bot/main.py` | `--check`, `--once` (dry-run), `--live` |
| Engine loop | `bot/engine.py` | waits for each closed hourly bar, builds a `Snapshot`, calls the strategy, applies guards, executes, journals; also the daily activity rule |
| Data | `bot/data/binance.py`, `bot/data/universe.py` | closed bars only (no look-ahead), host failover; liquid pool |
| Strategy | `strategies/base.py`, `strategies/v1_momentum_rotation.py` | pure function `decide(snapshot) → target weights + reasons` |
| Execution | `bot/execution/client.py`, `rules.py`, `executor.py` | HMAC signing, rate budget, price/quantity precision, order lifecycle |
| Risk | `bot/risk/guards.py` | kill-switch, price deviation and spread checks |
| Logging | `bot/journal.py`, `tools/report.py` | audit trail and summary report |
| Backtest / research | `backtest/`, `research/`, `docs/RESEARCH.md` | replays the live strategy class on history |
| Config | `config/default.yaml` | every tunable parameter |
| Tests | `tests/` | client signing, executor, strategy (including a no-look-ahead test), universe and risk |

**Tech stack:** Python 3.11, pandas, numpy, requests, PyYAML. Deployed under tmux on the organisers' EC2 (Sydney). A Dockerfile
is provided but was not used for the live run.

## 5. Backtesting

* **Data:** Binance hourly klines for every Roostoo pair, 2023-01 → 2026-10. The pool is **point-in-time** (rebuilt each day
  from trailing data) so the backtest never picks a coin using future volume.
* **Engine:** `backtest/engine.py` feeds the **same `MomentumRotation` class** as the live bot with the same `Snapshot`
  object; trades happen at the bar close with a per-side cost (see §7). Minimum-trade size and exit rules mirror the live ones.
* **Scoring:** the 14-day rolling window (the competition length), with the organisers' metrics on UTC-daily returns:
  `0.4·Sortino + 0.3·Sharpe + 0.3·Calmar`, 365-day annualisation.
* **Results (0.12% per side, 2023-04 → 2026-10-05):**

| | |
|---|---|
| Total return / max drawdown | +3,109% / 59% (regime-biased, see §12) |
| By year | 2023 +102% · 2024 +401% · 2025 +18% · 2026 YTD +163% |
| Rolling 14-day windows (1,269) | mean +5.4% · median 0.0% · p10 −11.3% · p90 +32% · positive 39% |
| Median / p90 max drawdown inside a 14-day window | 10.4% / 21.8% |
| Trades | 880 in 3.5 years (≈ 0.7 per day) |

* **What the backtests taught us:** the median 14-day return is 0 because the strategy sits in cash ~40% of the time; the mean is
  carried by a right tail. We therefore expect a live outcome to be one draw from this wide distribution, not the mean.
* **Robustness checks (round 17 and the reference-bot review):** lookback ensembles, gate variants (EMA vs SMA, 240/960),
  weighting schemes (inverse-vol, efficiency-ratio), rebalance frequency and hour, and k = 1–5. Several variants were
  marginally better in some windows but within noise (paired-difference confidence intervals included zero) and weaker in
  2026. Rebalance-hour luck alone moves v1's total return by more than any variant gained. We therefore kept the simple rule.
* **Reproduce:**
  ```bash
  python -m backtest.data --since 2023-01-01
  python -m backtest.run_backtest --start 2023-04-01 --cost 0.0012
  ```

## 6. How the code reflects the idea

| Idea | Where it lives |
|---|---|
| "Only trade in up-trending markets" | `MomentumRotation.gate_on()` → `ema_last(btc, 168) > ema_last(btc, 672)`; when false, `decide()` returns `{}` with reason `gate_off_exit` for every holding |
| "Rank by 14-day return" | `MomentumRotation.momentum()` → `close[-1] / close[-337] - 1` per pool coin |
| "Top 3 with positive momentum" | `decide()` → `rank <= k and ret > 0` |
| "Do not churn" | `decide()` → keep a holding while `rank <= k + hysteresis and ret > 0` |
| "Let winners run, but do not over-concentrate" | `decide()` → `cap = max_weight_multiple / k`; a holding above cap is trimmed (`trim_winner`) |
| "Same logic in research and live" | `backtest/engine.py` and `bot/engine.py` both call `Strategy.decide(Snapshot)` |
| "No look-ahead" | the data layer returns closed bars only; `tests/test_strategy_v1.py` has a no-look-ahead test |
| "Every trade is explainable" | `Decision.reasons` is passed through to `Order.reason` and written to `logs/orders-*.jsonl` |

The strategy is a pure function (no network, no clock, no state), which is why it can be unit-tested and backtested
without changes.

## 7. How maker and taker fees are handled

Roostoo charges **0.05% maker** and **0.10% taker**.

* **Execution policy** (`bot/execution/executor.py`): place a passive **limit order at the touch** (best ask for sells, best bid
  for buys), which earns the maker fee if it fills. If it is not filled after `limit_timeout_sec` (150 s for entries, 60 s for
  exits) it is cancelled and the remainder is sent as a **market order** (taker fee). Entries are skipped if the price has
  moved more than 1.5% from the decision price, so we never pay up to chase.
* **Backtest cost:** 0.12% per side = 0.10% taker fee + 2 bps slippage. This is deliberately the **worst case** (every trade
  taker), so live results with maker fills should be slightly better. A round trip therefore costs ~0.24% in research. We
  measured roughly 0.2–0.3% round-trip on small caps live, including slippage.
* **Cost-control rules in the strategy:** hysteresis cut trades from 489 to 320 at the same return in the event simulator; the
  daily rebalance (not hourly); `min_trade_usd = 150` skips tiny rebalances; winners are not topped up; a 0.4% cash buffer
  covers fees and rounding so buys are never rejected for insufficient funds.
* **Fee audit:** every order row in `logs/orders-*.jsonl` records the `role` (MAKER/TAKER), `fee`, `fee_coin`, fill price and the
  reference price, so realised fees and slippage can be reconstructed exactly (`python -m tools.report`).
* **Exchange quirk handled:** a resting limit order can report `FilledQuantity == Quantity` while still `PENDING`. `Order.apply()`
  therefore trusts the fill size only when the status is `FILLED`, otherwise it uses `CoinChange`, so fills and fees are not
  over-reported.

## 8. Risk management

The risk philosophy: **protect against catastrophe and bad data, but do not throttle the return**, because the competition ranks
return first and every tested tight brake lowered it.

| Layer | Rule |
|---|---|
| Market regime | BTC EMA168h/EMA672h gate: go fully to cash when the slow trend turns down (evaluated every hour) |
| Diversification | 3 positions, equal weight, from 25 liquid coins; a winner above 1.6× target is trimmed |
| Leverage | none; gross exposure ≤ 98% of equity |
| Liquidity | pool filter (30-day volume, spread ≤ 10 bps); `min_trade_usd`; exchange precision and minimum-order rules in `rules.py` |
| Catastrophe kill-switch | equity down 22% from peak → sell everything, pause 24 h, restart from a fresh peak |
| Data safety | refuse to decide on bars older than 3 h; drop entries whose Roostoo price differs from Binance by > 3% |
| Execution safety | chase cap 1.5%; stale pending orders are swept before each rebalance; a failed order POST is never blindly retried |
| API safety | client-side budget of 20 calls/min (limit 30); one batched poll per interval; pause 5 min after repeated API errors |
| Crash safety | positions are read from the wallet, not from a file, so a restart cannot double-buy |
| Compliance | activity rule: if nothing filled by 20:00 UTC, one 0.3%-of-equity BTC maintenance trade is made so the day counts as active |

**Deliberately not used:** per-position stop-losses, take-profits, vol-targeting and drawdown brakes. In backtests every
variant reduced return without improving the composite score (tight ATR stops 3–5 were far worse; the best drawdown brake
improved Sortino but cost about 20% of mean return).

## 9. Setup and how to run

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # fill in the keys; NEVER commit .env
python -m pytest tests -q                              # no network needed
python -m bot.main --check                             # read-only: connectivity + account equity
python -m bot.main --once                              # dry-run decision cycle (default is dry-run)
python -m bot.main --live                              # real orders on the account in ROOSTOO_ACCOUNT
python -m tools.report                                 # summary of equity / fills / reasons from logs/
```

### Deploy on the organisers' EC2 (Sydney, Session Manager)
```bash
git clone <this-repo> && cd <repo>/roostoo-bot && bash deploy/setup_ec2.sh   # the bot lives in the roostoo-bot/ subfolder
cp .env.example .env && nano .env        # ROOSTOO_ACCOUNT=competition + competition keys
python -m bot.main --check               # must print account=competition and equity ~ 100,000
tmux new -s bot
bash deploy/run_bot.sh                   # auto-restarts the bot; detach: Ctrl+B then D
```
Updating the strategy = edit code/config → commit → `git pull` on the box → bot restarts (positions are read from the wallet).

## 10. Compliance notes
* All trades go through `bot/execution/executor.py`, driven by `Strategy.decide()`. No ad-hoc order scripts exist, and the only
  manual tool (`--flatten`) is hard-refused on the competition account.
* Every order row carries `reason`, limit price, reference price, fill, fee and role.
* No HFT, market making or arbitrage: about one decision per hour, positions held for days.
* Strategy versions live in `strategies/` with `VERSIONS.md`; parameters live in `config/default.yaml`.
* API keys are read from `.env` and are never committed.

## 11. Live results

> **TODO before submitting: fill this in from the EC2 logs. Do not estimate.**
> On the box, from `roostoo-bot/`: `python -m tools.report` (equity curve, fills, fees, reasons) and the Roostoo app
> leaderboard for the final return and rank.

| | |
|---|---|
| Live period | 2026-10-06 ~14:10 UTC (bot start) → 2026-10-17 |
| Starting / final equity | _fill in_ |
| Return, max drawdown | _fill in_ |
| Number of fills, maker vs taker share, total fees | _fill in_ |
| Distinct active UTC days | _fill in_ |
| Regime gate: how many times it was on or off | _fill in_ |

Commentary to add: what the market did (a broad alt sell-off in the second week, where most pool coins fell > 8%), how the
strategy responded (the gate stayed on, so the long-only rotation carried market beta), and what we would change next.

## 12. Limitations and honesty notes

* The backtest window holds two large alt-seasons (2024, 2026). Results are **regime-dependent and survivorship-biased**
  (only coins Roostoo lists today). A 14-day outcome is a draw from a wide distribution (p10 −11%, p90 +32%), not the mean.
* Binance prices proxy Roostoo prices (the organisers confirmed they mirror Binance). Real fills can differ.
* The strategy is long-only, so a broad market fall produces a loss unless the slow BTC gate has already turned off. The gate is
  slow by design and does not react to a fast sell-off.
* Research variants were evaluated on one 3.6-year price path, with simplified costs and overlapping windows. Small
  improvements are not statistically distinguishable.
* Strategy changes were allowed during the contest only as committed code or config; no manual stops, overrides or
  discretionary trades were made.
