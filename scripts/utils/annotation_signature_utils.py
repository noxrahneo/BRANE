#!/usr/bin/env python3
"""Signature loading and normalization helpers for annotation."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SIGNATURES: dict[str, list[str]] = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "MSLN"],
    "Basal_Epi": ["KRT5", "KRT14", "TP63", "ITGA6", "KRT17"],
    "Luminal_Epi": ["KRT8", "KRT18", "ESR1", "PGR", "GATA3"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA"],
    "Endothelial": ["PECAM1", "VWF", "KDR", "EMCN", "CLDN5"],
    "TCell": ["CD3D", "CD3E", "TRAC", "IL7R", "LTB"],
    "NK": ["NKG7", "GNLY", "KLRD1", "TRBC1", "TRBC2"],
    "BCell": ["MS4A1", "CD79A", "CD74", "CD19", "HLA-DRA"],
    "Plasma": ["MZB1", "JCHAIN", "SDC1", "XBP1", "DERL3"],
    "Myeloid": ["LYZ", "LST1", "FCER1G", "CTSS", "TYROBP"],
    "Mast": ["TPSAB1", "TPSB2", "KIT", "MS4A2", "HDC"],
    "Cycling": ["MKI67", "TOP2A", "PCNA", "CENPF", "TYMS"],
}


SIGNATURE_ALIAS_MAP: dict[str, str] = {
    "Endo": "Endothelial",
    "Fibro": "Fibroblast",
    "Fibro2": "Fibroblast",
    "Macro": "Myeloid",
    "Basal": "Basal_Epi",
    "LP": "Luminal_Epi",
    "ML": "Luminal_Epi",
    "Str": "Stroma",
}


def load_signature_table(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    need = {"CellType", "Signatures"}
    if not need.issubset(set(df.columns)):
        return {}

    out: dict[str, list[str]] = {}
    for celltype, sub in df.groupby("CellType"):
        genes = (
            sub["Signatures"]
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .tolist()
        )
        if genes:
            out[str(celltype)] = list(dict.fromkeys(genes))
    return out


def normalize_gene_vector(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, pd.DataFrame):
        if values.empty:
            return []
        series = values.iloc[:, 0]
    elif isinstance(values, pd.Series):
        series = values
    elif isinstance(values, np.ndarray):
        series = pd.Series(values.reshape(-1))
    elif isinstance(values, (list, tuple, set)):
        series = pd.Series(list(values))
    else:
        series = pd.Series([values])

    genes = (
        series.astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .tolist()
    )
    return list(dict.fromkeys(genes))


def load_lineage_rdata(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}

    try:
        pyreadr = importlib.import_module("pyreadr")
    except ImportError:
        print(
            "WARNING: pyreadr not installed; skipping lineage RData. "
            "Install with: pip install pyreadr"
        )
        return {}

    try:
        loaded = pyreadr.read_r(str(path))
    except Exception as exc:
        print(f"WARNING: failed to read lineage RData ({path}): {exc}")
        return {}

    key_map = {
        "Basal": "Basal",
        "LP": "LP",
        "ML": "ML",
        "Str": "Stroma",
        "Stroma": "Stroma",
    }
    out: dict[str, list[str]] = {}
    for src_key, out_key in key_map.items():
        if src_key not in loaded:
            continue
        genes = normalize_gene_vector(loaded[src_key])
        if genes:
            out[out_key] = genes
    return out


def load_pam50(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    need = {"Gene", "Subtype"}
    if not need.issubset(df.columns):
        return {}

    out: dict[str, list[str]] = {}
    for subtype, sub in df.groupby("Subtype"):
        genes = (
            sub["Gene"]
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .tolist()
        )
        if genes:
            out[f"PAM50_{subtype}"] = list(dict.fromkeys(genes))
    return out


def collapse_signature_aliases(
    signatures: dict[str, list[str]],
) -> tuple[dict[str, list[str]], int]:
    collapsed: dict[str, list[str]] = {}
    merged = 0
    for label, genes in signatures.items():
        target = SIGNATURE_ALIAS_MAP.get(label, label)
        if target in collapsed and target != label:
            merged += 1
        if target not in collapsed:
            collapsed[target] = []
        collapsed[target].extend(genes)

    for key, genes in collapsed.items():
        collapsed[key] = list(dict.fromkeys(genes))
    return collapsed, merged


def prune_redundant_signatures(
    signatures: dict[str, list[str]],
) -> tuple[dict[str, list[str]], list[str]]:
    out = dict(signatures)
    removed: list[str] = []

    has_lineage = all(k in out for k in ("Basal", "LP", "ML", "Stroma"))
    if has_lineage:
        for key in ("Epithelial", "Basal_Epi", "Luminal_Epi"):
            if key in out:
                removed.append(key)
                del out[key]

    for key in ("Endo", "Fibro", "Fibro2", "Macro"):
        if key in out:
            removed.append(key)
            del out[key]

    for key in ("DC", "Mega", "TCell2"):
        if key in out:
            removed.append(key)
            del out[key]

    return out, removed


def build_signatures(
    sig_path: Path,
    lineage_rdata_path: Path,
    use_lineage_rdata: bool,
    pam50_path: Path,
    include_pam50: bool,
    collapse_aliases: bool,
    prune_redundant: bool,
) -> dict[str, list[str]]:
    external = load_signature_table(sig_path)
    lineage = load_lineage_rdata(lineage_rdata_path) if use_lineage_rdata else {}
    pam50 = load_pam50(pam50_path) if include_pam50 else {}

    signatures = {k: list(v) for k, v in DEFAULT_SIGNATURES.items()}
    for source in (external, lineage, pam50):
        for key, genes in source.items():
            if key in signatures:
                signatures[key] = list(dict.fromkeys([*signatures[key], *genes]))
            else:
                signatures[key] = list(genes)

    merged_aliases = 0
    if collapse_aliases:
        signatures, merged_aliases = collapse_signature_aliases(signatures)

    removed_labels: list[str] = []
    if prune_redundant:
        signatures, removed_labels = prune_redundant_signatures(signatures)

    print(
        "Signatures loaded: "
        f"default={len(DEFAULT_SIGNATURES)}, "
        f"immune_external={len(external)}, "
        f"lineage_rdata={len(lineage)}, "
        f"pam50={len(pam50)}, "
        f"merged_aliases={merged_aliases}, "
        f"pruned_labels={len(removed_labels)}, "
        f"total={len(signatures)}"
    )
    if removed_labels:
        print("Pruned redundant labels: " + ", ".join(sorted(removed_labels)))

    return signatures
