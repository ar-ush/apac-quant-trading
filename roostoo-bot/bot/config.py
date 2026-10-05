"""Configuration: YAML file (strategy/risk parameters, committed) + environment (secrets, never committed)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


@dataclass
class UniverseConfig:
    size: int = 25
    history_days: int = 60
    max_spread_bps: float = 10.0
    exclude: List[str] = field(default_factory=list)


@dataclass
class ExecutionConfig:
    use_limit_orders: bool = True
    limit_timeout_sec: int = 150
    exit_timeout_sec: int = 60
    poll_interval_sec: int = 15
    min_trade_usd: float = 150.0
    cash_buffer: float = 0.004
    entry_chase_cap: float = 0.015
    max_calls_per_minute: int = 20


@dataclass
class RiskConfig:
    kill_switch_drawdown: float = 0.22
    kill_switch_pause_hours: int = 24
    max_price_deviation_bps: float = 150.0
    max_data_age_hours: int = 3
    api_error_pause_sec: int = 300


@dataclass
class ActivityConfig:
    enabled: bool = True
    deadline_hour_utc: int = 20
    probe_usd_fraction: float = 0.003


@dataclass
class RuntimeConfig:
    loop_seconds: int = 60
    bar_close_delay_sec: int = 75
    state_dir: str = "state"
    log_dir: str = "logs"


@dataclass
class StrategyConfig:
    name: str = "v1_momentum_rotation"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    strategy: StrategyConfig
    universe: UniverseConfig
    execution: ExecutionConfig
    risk: RiskConfig
    activity: ActivityConfig
    runtime: RuntimeConfig
    base_url: str = "https://mock-api.roostoo.com"
    account: str = "test"
    api_key: str = ""
    secret_key: str = ""


def _section(cls, raw: Dict[str, Any] | None):
    raw = raw or {}
    unknown = set(raw) - set(cls.__dataclass_fields__)
    if unknown:  # a typo must never silently fall back to a default
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path | None = None, account: str | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = Config(
        strategy=_section(StrategyConfig, raw.get("strategy")),
        universe=_section(UniverseConfig, raw.get("universe")),
        execution=_section(ExecutionConfig, raw.get("execution")),
        risk=_section(RiskConfig, raw.get("risk")),
        activity=_section(ActivityConfig, raw.get("activity")),
        runtime=_section(RuntimeConfig, raw.get("runtime")),
    )
    cfg.base_url = os.environ.get("ROOSTOO_BASE_URL", cfg.base_url)
    cfg.account = (account or os.environ.get("ROOSTOO_ACCOUNT", "test")).lower()
    if cfg.account not in ("test", "competition"):
        raise ValueError("ROOSTOO_ACCOUNT must be 'test' or 'competition'")
    prefix = "ROOSTOO_TEST" if cfg.account == "test" else "ROOSTOO_COMPETITION"
    cfg.api_key = os.environ.get(f"{prefix}_API_KEY", "").strip()
    cfg.secret_key = os.environ.get(f"{prefix}_SECRET_KEY", "").strip()
    return cfg
