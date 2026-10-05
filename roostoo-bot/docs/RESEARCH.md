# Research log

Method: Binance hourly (and 15-minute) klines for every Roostoo crypto pair, 2023-01 → 2026-10, point-in-time liquid pool
(top-25 by trailing 30-day median dollar volume), costs 0.12% per side, scored on **rolling 14-day windows** (the length of the
live competition) with the competition metrics (0.4·Sortino + 0.3·Sharpe + 0.3·Calmar on UTC-daily returns, 365-day annualisation).
Scripts: `research/`. Reference implementations from other teams were read for ideas and failure modes only.

A caution that shaped everything: **bars on Binance are labelled by open time**. Our first Donchian test showed an absurd
8,000,000× return — a look-ahead bug from using a resampled 4h bar before it had closed. Fixed, and the production strategy has a
dedicated no-look-ahead unit test.

| # | Question | Result | Decision |
|---|---|---|---|
| 1 | Do classic families (time-series momentum, cross-sectional momentum, Donchian, low-vol trend, BTC trend) beat holding? | Median 14-day return ≈ 0 for all. Cross-sectional 14d momentum with a BTC gate had by far the best mean/tail (mean14 +4–6%, p90 +27–36%) but deep drawdowns. | carry momentum forward |
| 2 | Do portfolio overlays (vol-target, drawdown brake, profit lock) help? | Vol-targeting trades return for drawdown ~linearly; brakes and locks reduce return without lifting the composite. | no overlays |
| 3 | Is there a 15-minute burst-continuation edge (z-score of 15–60 min returns vs 24h vol, vol-scaled TP/SL)? | Average net trade ≈ 0.0%, win rate 42–52%, −70…−100% total over 2023–26 and catastrophic in 2026. | rejected |
| 4 | How robust is the momentum rotation to its parameters (896 configs: lookback × k × rebalance × regime filter × rank type)? | 336h lookback best on every marginal; BTC EMA168/672 best gate; raw return rank beats vol-adjusted. A plateau, not a spike. | lb=336, gate 168/672 |
| 5 | Do position-level stops or vol-scaled sizing improve it? | Trailing stops 10–30% reduce return (total 20.8× → 4.8–9.3×) with no better p10; vol-scaled sizing cuts max drawdown 58% → 35% but cuts mean return 1.9% → 1.1%; rank hysteresis 2 cuts trades 489 → 320 at equal return. | hysteresis only |
| 6 | Does a walk-forward ML model (ridge / logistic / GBM on ~25 features) rank coins better? | Statistically real OOS rank-IC (0.08, t≈4.8) but its top-3/5/8 long-only portfolios *lose* money (−27…−61%) while plain momentum made +605% in the same period: average-rank skill ≠ top-of-book edge. | not used alone |
| 7 | Does trend *quality* (regression t-stat, Clenow slope×R²) or market breadth improve selection/gating? | No better than raw momentum; breadth gates reduce return and the share of positive windows. | not used |
| 8 | Grossman–Zhou / CPPI drawdown control (floor = (1−α)·peak, exposure ∝ cushion)? | Lowers mean 14-day return 4.6% → 0.5–2.8% and mean composite 16 → 7–13 at every (α, m). | not used |

**Conclusions**
* At a 14-day horizon the composite is governed by the sign and size of the window return; protection that cuts return does not pay.
* The only structure with a consistent, parameter-robust edge in our data is *momentum among liquid coins, active only in a BTC up-trend*.
* The honest expected outcome of v1 is a draw from a wide distribution (14-day p10 −11%, p90 +32%), not a steady line.

**Next (v2 candidates):** a probabilistic regime layer (Bayesian/HMM posterior instead of a hard EMA cross), rank-weighted sizing,
and a squeeze-aware short sleeve for the gate-off regime.
