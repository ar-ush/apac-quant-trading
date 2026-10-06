# research10 literature scan - notes (2026-10-05)
Scripts: research10_lit_lib.py (engine wrapper), research10_lit_fetch.py (aux data -> data/aux), research10_lit1_momentum_variants.py,
research10_lit2_gates.py, research10_lit3_construction.py, research10_lit3b_pool_diag.py, research10_lit4_max_neighbourhood.py.
Outputs: out_research10_lit1.csv, lit2, lit3, lit3_lock, lit4. Baseline v1 reproduced: tot +28.2x, mdd 62.5%, mean14 +5.2%, med 0, p10 -11.8%, p90 +31.1%, pos 39.4%, P(>20%) 15%.
Configs: lit1 37, lit2 43, lit3 41 + 9 lock, lit4 15 = ~145 (+ diagnostics). Engine: research9_alt_lib (numba), cost 0.12%/side, trade at signal-bar close, point-in-time pool.
Sources/priors are in the final report; ranking of candidates is below.

## Candidate list (source | claimed effect | free data? | prior survives costs @14d)
1. Nearness-to-52w-high (Jia-Simkins-Yan-Zhang, JBF 2026, ssrn 5386180) | ~130bp/wk L-S VW | yes | 15% -> TESTED, worse than raw momentum (W 30d..365d).
2. MAX / lottery (Li-Urquhart-Wang-Zhang IRFA 2021; Intraday lottery SEF 2024) | >1.5%/wk quintile spread, sign disputed (momentum vs reversal) | yes | 15% -> TESTED: MAX-momentum filter +0.2..0.6pp mean14, unstable years.
3. Volume-weighted momentum (Huang-Sangiorgi-Urquhart ssrn 4825389) | 0.94%/day, SR 2.17 (TS) | yes | 15% -> TESTED: worse (mean14 3.1-3.6%).
4. Cross-sectional dispersion state (Makgolo-Zhang ssrn 6648082; roostoo-hackathon DECISIONS#competition-oos) | dispersion-scaled SR .63->.80 | yes | 15% -> TESTED both directions: all lose.
5. Size / illiquidity (Liu-Tsyvinski-Wu; arXiv 2510.14435; Fieberg et al.) | small coins -0.9..-2.3%/wk | yes | 20% -> pool breadth gradient monotone (mean14 5.2% -> 7.9%), but FLOKI/mid-cap driven, survivorship, 2/4 years.
6. Risk-managed momentum (Yang FRL 2025 vol-scaling; Grobys et al FMPM 2025) | SR 1.12 -> 1.42 | yes | 10% -> TESTED capped at 1x: reduces return.
7. UP-UP market-state momentum (FRL 2025 'State transitions and momentum') | momentum only in UP-UP | yes | 10% -> TESTED: reduces return (BTC EMA gate already captures it).
8. Cross-asset (NDX/SPX/VIX/DXY/gold/HYG lead-lag) | unproven | Yahoo free | 8% -> TESTED: nothing > v1.
9. Stablecoin supply / Fear&Greed | mixed | DefiLlama/alternative.me free | 5% -> TESTED: no gain (competitor repo also null).
10. Token unlock / listing-delisting events (Keyrock, ssrn 6632838, Binance listing study github Asalio123) | -5..-17% pre/post unlock; new listings -22% at 3m | calendar data NOT free/point-in-time (tokenomist) -> SKIPPED. Pool already needs >=60d history.
11. Seasonality: Sunday 22-23 UTC, weekend momentum (acr-journal 1514; quantpedia) | ~1bp/h | yes | 3% -> rebalance-hour/weekday TESTED: v1 not hour-fragile, no hour wins.
12. Rank-weight / HRP / vol-managed sizing | n/a | yes | 5% -> TESTED: worse than equal weight.
13. Pairs / cointegration | gross small, net negative at hourly (Tadi 2025; roostoo-hackathon S2) | yes | 2% -> not run (competitor repo showed net -8..-34bp/trade).
14. Tournament objective: M6 rank-optimal policy (arXiv 2412.04490), target lock (roostoo-hackathon) | P(14d>2%) .53->.68 | yes | objective-shape, not alpha. TESTED lock: pos 39%->50-52%, mean14 5.2%->0.8-3.6%, P(>20%)->0-12%.
15. Variance risk premium (DVOL) - Deribit history free via API but not tested (no cross-section; BTC only) | prior 3%.
