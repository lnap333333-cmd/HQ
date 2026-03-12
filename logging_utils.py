"""Logging helpers (JSONL, metrics)."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict


def _default(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=_default)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

