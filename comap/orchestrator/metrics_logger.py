from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import write_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MetricsLogger:
    path: Path
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_round(cls, path: str | Path, round_id: int) -> "MetricsLogger":
        logger = cls(Path(path).expanduser().resolve())
        logger.payload = {
            "round_id": round_id,
            "created_at": _utc_now(),
            "stages": [],
        }
        return logger

    def log_stage(self, name: str, **details: Any) -> None:
        self.payload.setdefault("stages", []).append(
            {
                "name": name,
                "timestamp": _utc_now(),
                **details,
            }
        )

    def finalize(self, **extra: Any) -> None:
        self.payload.update(extra)
        self.payload["updated_at"] = _utc_now()
        write_json(self.payload, self.path)
