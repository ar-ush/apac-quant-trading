"""Portfolio-level safety rules. These are catastrophe guards, not alpha: research showed that tighter brakes cut return
without improving the competition score, so thresholds are deliberately wide."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from bot.config import RiskConfig


@dataclass
class KillSwitch:
    cfg: RiskConfig

    def update(self, equity: float, peak: float, now_ms: int, halted_until_ms: int):
        """Returns (new_peak, halted_until_ms, triggered_now)."""
        if now_ms < halted_until_ms:
            return max(peak, equity), halted_until_ms, False
        if halted_until_ms and now_ms >= halted_until_ms:
            peak = equity  # resume from a fresh peak after the pause
            halted_until_ms = 0
        peak = max(peak, equity)
        dd = 1.0 - equity / peak if peak > 0 else 0.0
        if dd >= self.cfg.kill_switch_drawdown:
            return peak, now_ms + self.cfg.kill_switch_pause_hours * 3_600_000, True
        return peak, halted_until_ms, False


def price_deviation_bps(roostoo_mid: float, binance_close: float) -> float:
    return abs(roostoo_mid / binance_close - 1.0) * 1e4 if binance_close > 0 else float("inf")


def spread_bps(quote: Dict[str, float]) -> Optional[float]:
    bid, ask = float(quote.get("MaxBid") or 0), float(quote.get("MinAsk") or 0)
    if bid <= 0 or ask <= 0:
        return None
    return (ask - bid) / ((ask + bid) / 2) * 1e4
