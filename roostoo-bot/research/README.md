# research/

Exploratory research scripts behind the strategy choice (kept for transparency; the production code lives in `bot/`, `strategies/`
and `backtest/`). Data is downloaded locally and is not committed.

```bash
cd research
python fetch_data.py --interval 1h --since 2023-01-01      # hourly Binance klines for every Roostoo crypto pair
python fetch_data.py --interval 15m --since 2023-06-01 --coins BTC,ETH,SOL    # 15m bars (research3)
python research1.py        # classic families on a 1h panel
```

| script | question | headline result (details: `../docs/RESEARCH.md`) |
|---|---|---|
| research1 | trend / momentum / Donchian / low-vol families | nothing beats beta by a wide margin; median 14-day return ~0 for all |
| research2 | portfolio overlays (vol target, DD brake, profit lock) | brakes cut return without lifting the composite |
| research3 | 15m burst-continuation (event simulator) | no edge after costs over 2023-26 |
| research4 | momentum-rotation parameter scan (896 configs) | 14-day lookback + BTC EMA168/672 gate is a plateau, not a spike |
| research5 | rotation + trailing stops / vol-scaled sizing | stops reduce return; sizing trades return for DD linearly |
| research6 | walk-forward ML ranking (ridge / logit / GBM) | real rank-IC (0.08) but top-k portfolios lose |
| research7 | trend-quality rankings (t-stat, Clenow), breadth gates | no better than raw momentum |
| research8 | Grossman-Zhou / CPPI drawdown control | lowers return and composite at every setting |
