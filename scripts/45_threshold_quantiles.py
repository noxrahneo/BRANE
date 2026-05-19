"""95th-percentile CSD/correlation thresholds and per-edge empirical FWER p-values from permutation maxima."""


from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils.network_utils import resolve_base


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "results/25_threshold_quantiles"

SINGLE_THRESHOLD_ROOT = REPO_ROOT / "results/12_single_condition_thresholds"
SINGLE_NETWORK_ROOT = REPO_ROOT / "results/13_single_condition_networks"

DIFF_THRESHOLD_ROOT = REPO_ROOT / "results/13_csd_thresholds"
DIFF_NETWORK_ROOT = REPO_ROOT / "results/14_csd_networks"


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute permutation quantile thresholds and edge-level FWER p-values")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--single-threshold-dir", default=str(SINGLE_THRESHOLD_ROOT))
    parser.add_argument("--single-network-dir", default=str(SINGLE_NETWORK_ROOT))
    parser.add_argument("--differential-threshold-dir", default=str(DIFF_THRESHOLD_ROOT))
    parser.add_argument("--differential-network-dir", default=str(DIFF_NETWORK_ROOT))
    parser.add_argument("--condition", action="append", default=[], help="Limit to specific single-condition network(s)")
    parser.add_argument("--pair", action="append", default=[], help="Limit to specific differential pair(s)")
    return parser.parse_args()


def _load_threshold_json(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in payload.items() if isinstance(v, (int, float))}


def _upper_tail_pvalues(values: np.ndarray, maxima: np.ndarray) -> np.ndarray:
    maxima = np.asarray(maxima, dtype=float)
    maxima = maxima[np.isfinite(maxima)]
    out = np.full(np.asarray(values).shape, np.nan, dtype=float)
    if maxima.size == 0:
        return out

    sorted_max = np.sort(maxima)
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return out

    idx = np.searchsorted(sorted_max, vals[finite], side="left")
    hits = sorted_max.size - idx
    out[finite] = (hits + 1.0) / (sorted_max.size + 1.0)
    return out


def _single_rows(threshold_root: Path, network_root: Path, quantile: float, keep_conditions: set[str]) -> tuple[list[dict], list[tuple[str, pd.DataFrame]]]:
    rows: list[dict] = []
    edge_tables: list[tuple[str, pd.DataFrame]] = []

    for maxima_file in sorted(threshold_root.glob("*/*_permutation_maxima.csv")):
        condition = maxima_file.parent.name
        if keep_conditions and condition not in keep_conditions:
            continue

        threshold_file = maxima_file.parent / f"{condition}_permutation_threshold.json"
        edge_file = network_root / condition / f"{condition}_edges.tsv"
        if not threshold_file.exists() or not edge_file.exists():
            log.warning("%s: missing threshold or edge file", condition)
            continue

        maxima_df = pd.read_csv(maxima_file)
        thresholds = _load_threshold_json(threshold_file)
        q95 = float(np.quantile(maxima_df["max_abs_r"].to_numpy(dtype=float), quantile))
        current = float(thresholds.get("threshold_abs_r", np.nan))

        edges_df = pd.read_csv(edge_file, sep="\t")
        obs = edges_df["abs_r"].abs().to_numpy(dtype=float) if "abs_r" in edges_df.columns else edges_df["r"].abs().to_numpy(dtype=float)
        edge_out = edges_df.copy()
        edge_out["p_fwer_abs_r"] = _upper_tail_pvalues(obs, maxima_df["max_abs_r"].to_numpy(dtype=float))
        edge_out["threshold_abs_r_q95"] = q95
        edge_out["passes_abs_r_current"] = obs >= current
        edge_out["passes_abs_r_q95"] = obs >= q95

        edge_tables.append((condition, edge_out))

        rows.append(
            {
                "family": "single",
                "name": condition,
                "metric": "abs_r",
                "quantile": quantile,
                "n_permutations": int(maxima_df.shape[0]),
                "current_threshold": current,
                "q95_threshold": q95,
                "n_edges_total": int(obs.size),
                "n_edges_surviving_current": int(np.sum(obs >= current)),
                "n_edges_surviving_q95": int(np.sum(obs >= q95)),
                "maxima_file": str(maxima_file),
                "threshold_file": str(threshold_file),
                "edge_file": str(edge_file),
            }
        )

    return rows, edge_tables


def _differential_rows(threshold_root: Path, network_root: Path, quantile: float, keep_pairs: set[str]) -> tuple[list[dict], list[tuple[str, pd.DataFrame]]]:
    rows: list[dict] = []
    edge_tables: list[tuple[str, pd.DataFrame]] = []

    for maxima_file in sorted(threshold_root.glob("*/*_permutation_maxima.csv")):
        pair = maxima_file.parent.name
        if keep_pairs and pair not in keep_pairs:
            continue

        threshold_file = maxima_file.parent / f"{pair}_permutation_thresholds.json"
        edge_file = network_root / pair / f"{pair}_differential_edges_permutation.csv"
        if not threshold_file.exists() or not edge_file.exists():
            log.warning("%s: missing threshold or edge file", pair)
            continue

        maxima_df = pd.read_csv(maxima_file)
        thresholds = _load_threshold_json(threshold_file)
        edges_df = pd.read_csv(edge_file)
        edge_out = edges_df.copy()

        for metric in ("C", "S", "D"):
            max_col = f"max_{metric}"
            q95 = float(np.quantile(maxima_df[max_col].to_numpy(dtype=float), quantile))
            current = float(thresholds.get(f"threshold_{metric}", np.nan))
            obs = pd.to_numeric(edges_df[metric], errors="coerce").to_numpy(dtype=float)
            edge_out[f"p_fwer_{metric}"] = _upper_tail_pvalues(obs, maxima_df[max_col].to_numpy(dtype=float))
            edge_out[f"threshold_{metric}_q95"] = q95
            edge_out[f"passes_{metric}_current"] = obs >= current
            edge_out[f"passes_{metric}_q95"] = obs >= q95

            rows.append(
                {
                    "family": "differential",
                    "name": pair,
                    "metric": metric,
                    "quantile": quantile,
                    "n_permutations": int(maxima_df.shape[0]),
                    "current_threshold": current,
                    "q95_threshold": q95,
                    "n_edges_total": int(obs.size),
                    "n_edges_surviving_current": int(np.sum(obs >= current)),
                    "n_edges_surviving_q95": int(np.sum(obs >= q95)),
                    "maxima_file": str(maxima_file),
                    "threshold_file": str(threshold_file),
                    "edge_file": str(edge_file),
                }
            )

        edge_tables.append((pair, edge_out))

    return rows, edge_tables


def main() -> int:
    args = parse_args()
    output_dir = resolve_base(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    single_rows, single_edge_tables = _single_rows(
        resolve_base(args.single_threshold_dir),
        resolve_base(args.single_network_dir),
        float(args.quantile),
        set(args.condition),
    )
    diff_rows, diff_edge_tables = _differential_rows(
        resolve_base(args.differential_threshold_dir),
        resolve_base(args.differential_network_dir),
        float(args.quantile),
        set(args.pair),
    )

    summary_rows = single_rows + diff_rows
    if not summary_rows:
        log.warning("No maxima files were processed")
        return 0

    summary_df = pd.DataFrame(summary_rows).sort_values(["family", "name", "metric"]).reset_index(drop=True)
    summary_df.to_csv(output_dir / "threshold_quantiles_summary.csv", index=False)

    edge_dir = output_dir / "edge_pvalues"
    edge_dir.mkdir(parents=True, exist_ok=True)
    for name, edge_df in single_edge_tables:
        edge_df.to_csv(edge_dir / f"{name}_edge_fwer_pvalues.csv", index=False)
    for name, edge_df in diff_edge_tables:
        edge_df.to_csv(edge_dir / f"{name}_edge_fwer_pvalues.csv", index=False)

    log.info("Wrote %d summary rows to %s", len(summary_df), output_dir)
    log.info("Edge-level p-values written to %s", edge_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())