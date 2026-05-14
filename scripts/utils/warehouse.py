#!/usr/bin/env python3
"""Small helper for stage-level warehouse logs.

Each stage writes/updates a `warehouse.csv` file with lightweight
provenance records.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class WarehouseRecord:
    input_file: str
    output_file: str
    script: str
    date_utc: str
    params_hash: str
    condition: str = ""
    stage: str = ""


def params_hash(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_warehouse(stage_dir: Path, records: list[WarehouseRecord]) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    out_file = stage_dir / "warehouse.csv"

    new_df = pd.DataFrame([r.__dict__ for r in records])
    if out_file.exists():
        old_df = pd.read_csv(out_file)
        df = pd.concat([old_df, new_df], axis=0, ignore_index=True)
    else:
        df = new_df

    df.to_csv(out_file, index=False)
    return out_file
