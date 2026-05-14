#!/usr/bin/env python3
"""Compatibility helpers for reading legacy h5ad files."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Any

import h5py
import scanpy as sc


def read_h5ad_compat(path: Path):
    try:
        return sc.read_h5ad(path)
    except Exception as exc:
        msg = str(exc)
        is_known = (
            "encoding_type='null'" in msg
            or "encoding-type='null'" in msg
            or 'encoding_type="null"' in msg
        )
        if not is_known:
            raise

        print(
            "WARNING: detected legacy null log1p base metadata; "
            f"repairing temporary copy of {path.name}"
        )

        with tempfile.TemporaryDirectory(prefix="cobra_h5ad_fix_") as tmpd:
            tmp_file = Path(tmpd) / path.name
            shutil.copy2(path, tmp_file)

            def _enc_name(value: Any) -> str:
                if isinstance(value, bytes):
                    return value.decode("utf-8", errors="ignore")
                if value is None:
                    return ""
                return str(value)

            def _is_null_encoded(node: Any) -> bool:
                if not hasattr(node, "attrs"):
                    return False
                attrs = node.attrs
                enc = attrs.get("encoding-type", attrs.get("encoding_type"))
                return _enc_name(enc) == "null"

            def _strip_null_nodes(group: h5py.Group) -> int:
                removed = 0
                for key in list(group.keys()):
                    node = group.get(key)
                    if node is None:
                        continue
                    if _is_null_encoded(node):
                        del group[key]
                        removed += 1
                        continue
                    if isinstance(node, h5py.Group):
                        removed += _strip_null_nodes(node)
                return removed

            with h5py.File(tmp_file, "r+") as handle:
                removed_count = _strip_null_nodes(handle)

            print(f"WARNING: removed {removed_count} null-encoded entries")
            return sc.read_h5ad(tmp_file)
