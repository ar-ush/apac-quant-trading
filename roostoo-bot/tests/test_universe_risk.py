import pandas as pd

from bot.config import RiskConfig, load_config
from bot.data.universe import select_pool
from bot.risk.guards import KillSwitch, price_deviation_bps, spread_bps


def test_pool_filters_spread_history_and_exclusions():
    med = pd.Series({"A": 100.0, "B": 90.0, "C": 80.0, "D": 70.0, "STB": 1000.0, "NEW": 60.0})
    hist = pd.Series({"A": 200, "B": 200, "C": 200, "D": 200, "STB": 200, "NEW": 5})
    sp = pd.Series({"A": 1.0, "B": 50.0, "C": 3.0})
    pool = select_pool(med, hist, sp, size=3, min_history_days=60, max_spread_bps=10, exclude=["STB"])
    assert pool == ["A", "C", "D"]    # B too wide, STB excluded, NEW too young; unknown spread (D) not penalised


def test_kill_switch_triggers_pauses_and_resumes_from_new_peak():
    ks = KillSwitch(RiskConfig(kill_switch_drawdown=0.2, kill_switch_pause_hours=24))
    peak, halted, trig = ks.update(100_000, 100_000, 0, 0)
    assert not trig
    peak, halted, trig = ks.update(79_000, peak, 1000, halted)
    assert trig and halted == 1000 + 24 * 3_600_000
    _, h2, t2 = ks.update(79_000, peak, 2000, halted)
    assert not t2 and h2 == halted                              # stays paused
    peak3, h3, t3 = ks.update(80_000, peak, halted + 1, halted)
    assert h3 == 0 and peak3 == 80_000 and not t3               # resumes with a fresh peak


def test_helpers():
    assert abs(price_deviation_bps(100.0, 100.5) - 49.75) < 0.5
    assert abs(spread_bps({"MaxBid": 99.99, "MinAsk": 100.01}) - 2.0) < 0.01
    assert spread_bps({"MaxBid": 0, "MinAsk": 1}) is None


def test_default_config_loads_and_rejects_typos(tmp_path, monkeypatch):
    monkeypatch.setenv("ROOSTOO_ACCOUNT", "test")
    cfg = load_config()
    assert cfg.strategy.name == "v1_momentum_rotation" and cfg.universe.size == 25
    bad = tmp_path / "bad.yaml"
    bad.write_text("execution:\n  limit_timeoutt_sec: 5\n")
    try:
        load_config(bad)
        assert False, "typo should have been rejected"
    except ValueError:
        pass
