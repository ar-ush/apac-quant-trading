# Strategy versions

Every version stays in this folder, forever. A new version is a new module + a new row here; the previous one is never edited
except for bug fixes (noted below). The live version is whatever `strategy.name` says in `config/default.yaml`; every switch is a commit.

| version | module | status | idea | headline backtest (0.12%/side) |
|---|---|---|---|---|
| v1 | `v1_momentum_rotation.py` | **live candidate** (2026-10-05) | BTC-gated 14d momentum rotation, top-3, hysteresis 2 | total +3,109%, rolling-14d mean +5.4%, p10 −11%, p90 +32% |

## Change log
* **2026-10-05** v1 created after 8 research rounds (`docs/RESEARCH.md`); live-API canary on the test account found and fixed a
  fill-parsing quirk (resting limit orders report FilledQuantity = Quantity while PENDING).

## Ideas queue (not yet built)
* v2: a short sleeve when the gate is off (Roostoo `/v6` shorts, 0.1% open + 0.1% close) — competitors found shorts mostly lose; needs an explicit squeeze filter.
* v2b: probabilistic regime layer (Bayesian / HMM posterior of "trend regime") replacing the hard EMA gate.
* v2c: rank-weighted sizing (Kelly-style on momentum z-score) inside the top-3.
