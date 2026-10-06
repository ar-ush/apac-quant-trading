# Strategy versions

Every version stays in this folder, forever. A new version is a new module + a new row here; the previous one is never edited
except for bug fixes (noted below). The live version is whatever `strategy.name` says in `config/default.yaml`; every switch is a commit.

| version | module | status | idea | headline backtest (0.12%/side) |
|---|---|---|---|---|
| v1 | `v1_momentum_rotation.py` | **live candidate** (2026-10-05) | BTC-gated 14d momentum rotation, top-3, hysteresis 2 | total +3,109%, rolling-14d mean +5.4%, p10 −11%, p90 +32% |

## Change log
* **2026-10-05** v1 created after 8 research rounds (`docs/RESEARCH.md`); live-API canary on the test account found and fixed a
  fill-parsing quirk (resting limit orders report FilledQuantity = Quantity while PENDING).

* **2026-10-06** pre-deploy audit: v1 strategy logic and parameters UNCHANGED. Engine/execution fixes only: activity probe
  could never trade again after a rebalance sold its BTC; a missing ticker quote could trip the kill-switch; a held coin with no
  Binance data was sold as a ranking exit; a missed 00:00 rebalance was never caught up; activity guard is now a rolling 18h window.
  Added raw API response to order rows, hourly clock resync, single-instance lock. 30 tests.

## Ideas queue
* Shadow signal (log only, review after Oct 17): "BTC **or** ETH EMA168>EMA672" gate. Looked slightly better in 2026 only; not traded.
* ~~short sleeve~~, ~~probabilistic regime layer~~, ~~Monte Carlo/conformal ranking and sizing~~: tested 2026-10-05, none beat v1 (`docs/RESEARCH.md` rounds 9–11). Not to be rebuilt unless new data changes the picture.
* v2c: rank-weighted sizing inside the top-3 (untested).
* Execution: measure live maker-fill rate and slippage once v1 trades; tune `entry_timeout_sec`.
* Possible: tick data (LSE) for a real slippage model — only if live fills disagree with the 0.12% assumption.
