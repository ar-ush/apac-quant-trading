# v1 — BTC-gated 14-day momentum rotation

Code: `strategies/v1_momentum_rotation.py` · Config: `config/default.yaml` · Backtest: `python -m backtest.run_backtest`

## Thesis

1. **Cross-sectional momentum exists in crypto at 1–4 week horizons** (Liu, Tsyvinski & Wu, *J. Finance* 2022; Fieberg et al., *JFQA* 2024).
   The strongest liquid coins over the last two weeks tend to keep outperforming for days, because retail attention and
   liquidity flow into winners and information diffuses slowly across a fragmented market.
2. **It only pays in up-trending regimes.** In down-trends the "winners" are bounces that fail. A slow BTC trend filter
   (EMA168h > EMA672h) separates the regimes; it flips only a few times a month (3 flips in the last 60 days).
3. **Liquidity matters more than the signal**: restricting to the 25 most liquid Roostoo coins (30-day median dollar volume,
   spread ≤ 10 bps) avoids names whose spreads and slippage would eat a multi-day edge.

## Rules

| | |
|---|---|
| Universe | top-25 Roostoo crypto pairs by 30-day median Binance dollar volume; spread ≤ 10 bps; ≥ 60 days of history; no stablecoins / PAXG / tokenised stocks. Refreshed daily. |
| Regime gate | `risk_on = EMA(BTC, 168h) > EMA(BTC, 672h)` on hourly closes, evaluated every hour. Off ⇒ sell everything, stay in cash. |
| Signal | `r_i = P_i(t) / P_i(t-336h) − 1` for each pool coin at 00:00 UTC (close of the 23:00 bar). |
| Entry | top-3 by `r_i`, only if `r_i > 0` (absolute momentum filter), equal weight 1/3 each (98% gross cap). |
| Hold | a holding stays while its rank ≤ 3 + 2 (hysteresis) and `r_i > 0`. Winners are never topped up; one that exceeds 1.6× its target weight is trimmed. |
| Exit | rank falls below 5, return turns ≤ 0, or the gate turns off (immediately, any hour). |
| Sizing | 1/3 of equity per slot, fully invested when three candidates qualify, cash otherwise. No leverage. |
| Execution | passive limit at the touch (maker 0.05%); unfilled after 150 s (exits 60 s) ⇒ cancel ⇒ market (taker 0.1%); entries are skipped if price ran > 1.5% from the decision price. |
| Risk | catastrophe kill-switch at −22% from peak (cash for 24 h); stale-data and API-error guards; Roostoo-vs-Binance price sanity check on new entries. |

There are deliberately **no per-position stops** and **no drawdown brake**: every variant tested reduced return without
improving the competition score (see `docs/RESEARCH.md`, rounds 5 and 8).

## Maths behind the choices

* **Why equal weight, concentrated (k=3)?** Momentum payoff is convex in dispersion; at 14-day horizons the 14-day-return
  distribution of crypto winners is heavy-tailed to the right. Rank-weighted portfolios beyond k=5 dilute the tail
  (mean 14-day return: k=2 6.2%, k=3 4.6%, k=4 3.5%) and a single-name book (k=1) was not adopted because of its tail risk (not tested here; competitors report a negative holdout median).
* **Why a regime gate?** The strategy's drawdown in 2022-style bears is ~−45%; the gate cuts the whole-period max drawdown
  from ~76% (no gate, same rotation) to ~56-59%, at the cost of sitting in cash ~40% of the time.
* **Why hysteresis?** A rank buffer of 2 reduced trades 489 → 320 at the same return in the event simulator (research5):
  turnover costs ≈ 0.24% per round trip.
* **Composite score (0.4·Sortino + 0.3·Sharpe + 0.3·Calmar).** At a 14-day horizon the composite is dominated by the sign
  and size of the window return. The research shows brakes that reduce return lower the composite (CPPI: mean composite
  16 → 7–13), so v1 maximises return conditional on the regime filter.

## Backtest (Binance hourly bars, 0.12% per side, 2023-04-01 → 2026-10-05, point-in-time pool)

| | |
|---|---|
| Total return / max drawdown | +3,109% / 59% (regime-biased, see caveats) |
| By year | 2023 +102% · 2024 +401% · 2025 +18% · 2026 YTD +163% |
| Rolling 14-day windows (1,269) | mean +5.4% · median 0.0% (in cash ~40% of the time) · p10 −11.3% · p90 +32% · positive 39% · >+5% in 29% |
| Median / p90 max drawdown inside a 14-day window | 10.4% / 21.8% |
| Trades | 880 in 3.5 years (≈ 0.7 per day) |

## Caveats (stated plainly)

* The backtest period contains two big alt-seasons (2024, 2026); results are regime dependent and survivorship-biased
  (only coins Roostoo lists today). Expect the live 14-day outcome to be a draw from the window distribution above, not the mean.
* Binance bars proxy Roostoo prices (Roostoo streams Binance); live fills can differ (we measured round-trip cost ≈ 0.2–0.3%
  including slippage on small caps).
* Median 14-day return is 0 because the gate keeps the book in cash when BTC trends down; the strategy is long-only by design.
