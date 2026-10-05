"""Small persisted state so a restart resumes cleanly. Positions are NEVER read from here: the wallet is the truth."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class State:
    peak_equity: float = 0.0
    last_bar_id: Optional[int] = None          # last hourly bar we made a decision on
    halted_until_ms: int = 0                   # kill-switch pause
    last_fill_day: str = ""                    # UTC date (YYYY-MM-DD) of the last filled order
    probe_qty: float = 0.0                     # BTC held only as the daily-activity maintenance position
    pool: List[str] = field(default_factory=list)
    pool_day: str = ""
    started: bool = False                      # False until the first decision of this deployment


class StateStore:
    def __init__(self, directory: str | Path, name: str):
        self.path = Path(directory) / f"state-{name}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> State:
        if not self.path.exists():
            return State()
        try:
            raw: Dict = json.loads(self.path.read_text())
            return State(**{k: v for k, v in raw.items() if k in State.__dataclass_fields__})
        except Exception:
            return State()

    def save(self, st: State) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(st)))
        os.replace(tmp, self.path)
