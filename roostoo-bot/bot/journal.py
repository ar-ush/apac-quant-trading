"""Append-only audit trail: decisions, orders, equity snapshots, risk events (JSON lines, one file per UTC day).

These files are the 'trade log integrity' evidence: every order row carries the strategy reason that produced it."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class Journal:
    def __init__(self, log_dir: str | Path):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, kind: str) -> Path:
        return self.dir / f"{kind}-{datetime.now(timezone.utc):%Y%m%d}.jsonl"

    def write(self, kind: str, row: Dict[str, Any]) -> None:
        row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row}
        line = json.dumps(row, default=str, separators=(",", ":"))
        with self._lock, open(self._path(kind), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def decision(self, row): self.write("decisions", row)
    def order(self, row): self.write("orders", row)
    def equity(self, row): self.write("equity", row)
    def event(self, row): self.write("events", row)
