#!/usr/bin/env python3
"""WGCNA-style soft-threshold scan and adjacency/TOM preparation.

This script consumes per-condition correlation outputs from:
`results/09_correlation/pearson/<Condition>/*_pearson_corr.npz`

For each condition it:
1) scans candidate soft-threshold powers,
2) computes scale-free fit diagnostics,
3) selects a power (lowest meeting target signed R^2 when possible),
4) builds adjacency and TOM matrices for the selected power.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.warehouse import (
    WarehouseRecord,
    append_warehouse,
    params_hash,
    utc_now_iso,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan soft-threshold powers and build adjacency/TOM matrices"
    )
    parser.add_argument(
        "--input-dir",
        default="results/09_correlation/pearson",
        help="Root with per-condition correlation outputs",
    )
    parser.add_argument(
        "--output-dir",
        default="results/11_coexpression_prep/single",
        help="Root output directory for power scan and TOM artifacts",
    )
    parser.add_argument(
        "--condition",
        default="all",
        help="Condition name or 'all'",
    )
    parser.add_argument(
        "--network-type",
        choices=["signed", "unsigned", "both"],
        default="signed",
        help="Adjacency transform type",
    )
    parser.add_argument(
        "--powers",
        default="1,2,3,4,5,6,7,8,9,10,12,14,16,18,20",
        help="Comma-separated soft-threshold powers",
    )
    parser.add_argument(
        "--target-signed-r2",
        type=float,
        default=0.70,
        help="Target signed R^2 for power selection",
    )
    parser.add_argument(
        "--min-mean-connectivity",
        type=float,
        default=5.0,
        help="Minimum mean connectivity for acceptable power",
    )
    parser.add_argument(
        "--r2-plateau-delta",
        type=float,
        default=0.03,
        help=(
            "Plateau tolerance: pick the first eligible power whose "
            "future max signed R2 improves by at most this delta"
        ),
    )
    parser.add_argument(
        "--r2-near-best-delta",
        type=float,
        default=0.02,
        help=(
            "Fallback tolerance: among powers near best signed R2, "
            "pick smallest power"
        ),
    )
    parser.add_argument(
        "--degree-bins",
        type=int,
        default=20,
        help="Histogram bins for scale-free fit estimation",
    )
    parser.add_argument(
        "--force-power",
        type=int,
        default=None,
        help="If set, skip selection and force this power for adjacency/TOM",
    )
    parser.add_argument(
        "--skip-tom",
        action="store_true",
        help="Skip TOM computation/export (faster, lower memory)",
    )
    return parser.parse_args()


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (REPO_ROOT / path).resolve()


def parse_powers(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        val = int(p)
        if val > 0:
            values.append(val)
    if not values:
        raise ValueError("No valid powers parsed from --powers")
    return sorted(set(values))


def list_condition_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def resolve_conditions(root: Path, requested: str) -> list[Path]:
    dirs = list_condition_dirs(root)
    if not dirs:
        raise FileNotFoundError(f"No condition directories in {root}")
    if requested.strip().lower() == "all":
        return dirs
    match = [d for d in dirs if d.name == requested]
    if not match:
        raise ValueError(f"Condition '{requested}' not found; available={[d.name for d in dirs]}")
    return match


def find_corr_npz(cond_dir: Path) -> Path:
    matches = sorted(cond_dir.glob("*_pearson_corr.npz"))
    if not matches:
        raise FileNotFoundError(f"No *_pearson_corr.npz in {cond_dir}")
    return matches[0]


def adjacency_from_corr(corr: np.ndarray, power: int, network_type: str) -> np.ndarray:
    c = np.asarray(corr, dtype=np.float64)
    c = np.clip(c, -1.0, 1.0)
    if network_type == "signed":
        sim = (1.0 + c) / 2.0
    else:
        sim = np.abs(c)
    adj = np.power(sim, int(power), dtype=np.float64)
    np.fill_diagonal(adj, 0.0)
    return adj


def scale_free_fit_from_connectivity(k: np.ndarray, bins: int) -> dict[str, float]:
    k = np.asarray(k, dtype=np.float64)
    k = k[np.isfinite(k)]
    k = k[k > 0]
    if k.size < 5:
        return {
            "slope": float("nan"),
            "r2": float("nan"),
            "signed_r2": float("nan"),
            "n_bins_used": 0,
        }

    counts, edges = np.histogram(k, bins=int(max(5, bins)))
    centers = 0.5 * (edges[:-1] + edges[1:])
    probs = counts / max(1, counts.sum())

    mask = (counts > 0) & (centers > 0) & (probs > 0)
    if int(mask.sum()) < 3:
        return {
            "slope": float("nan"),
            "r2": float("nan"),
            "signed_r2": float("nan"),
            "n_bins_used": int(mask.sum()),
        }

    x = np.log10(centers[mask])
    y = np.log10(probs[mask])
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    signed_r2 = float(np.sign(-slope) * r2) if np.isfinite(r2) else float("nan")

    return {
        "slope": float(slope),
        "r2": float(r2),
        "signed_r2": signed_r2,
        "n_bins_used": int(mask.sum()),
    }


def choose_power(
    scan_df: pd.DataFrame,
    target_signed_r2: float,
    min_mean_k: float,
    r2_plateau_delta: float,
    r2_near_best_delta: float,
) -> tuple[int, str]:
    ordered = scan_df.sort_values("power").reset_index(drop=True)
    eligible = ordered[
        (ordered["signed_r2"] >= float(target_signed_r2))
        & (ordered["mean_connectivity"] >= float(min_mean_k))
    ].copy()
    if not eligible.empty:
        plateau_rows: list[pd.Series] = []
        for row in eligible.itertuples(index=False):
            cur_p = int(getattr(row, "power"))
            cur_r2 = float(getattr(row, "signed_r2"))
            future = ordered[ordered["power"] >= cur_p]
            future_r2 = pd.to_numeric(
                future["signed_r2"],
                errors="coerce",
            )
            future_r2 = future_r2[np.isfinite(future_r2)]
            if future_r2.empty:
                continue
            max_future = float(future_r2.max())
            gain = max_future - cur_r2
            if gain <= float(r2_plateau_delta):
                plateau_rows.append(ordered[ordered["power"] == cur_p].iloc[0])

        if plateau_rows:
            chosen = pd.DataFrame(plateau_rows).sort_values("power").iloc[0]
            return (
                int(chosen["power"]),
                "lowest eligible power at scale-free fit plateau",
            )

        p = int(eligible.sort_values("power").iloc[0]["power"])
        return (
            p,
            "lowest power meeting signed R2 and mean connectivity thresholds",
        )

    fallback_pool = ordered[
        ordered["mean_connectivity"] >= float(min_mean_k)
    ].copy()
    if fallback_pool.empty:
        fallback_pool = ordered.copy()

    r2_vals = pd.to_numeric(fallback_pool["signed_r2"], errors="coerce")
    r2_vals = r2_vals[np.isfinite(r2_vals)]
    if r2_vals.empty:
        p = int(fallback_pool.sort_values("power").iloc[0]["power"])
        return p, "fallback to smallest power (signed R2 unavailable)"

    best_r2 = float(r2_vals.max())
    near_best = fallback_pool[
        fallback_pool["signed_r2"] >= best_r2 - float(r2_near_best_delta)
    ].sort_values("power")
    p = int(near_best.iloc[0]["power"])
    return (
        p,
        "fallback to lowest power near best signed R2 under connectivity floor",
    )


def plot_soft_threshold(scan_df: pd.DataFrame, target_r2: float, out_file: Path, title_prefix: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)

    x = scan_df["power"].to_numpy()
    y1 = scan_df["signed_r2"].to_numpy()
    y2 = scan_df["mean_connectivity"].to_numpy()

    axes[0].plot(x, y1, marker="o")
    axes[0].axhline(float(target_r2), color="red", linestyle="--", linewidth=1.3)
    for xi, yi in zip(x, y1):
        axes[0].text(xi, yi + 0.01, str(int(xi)), ha="center", va="bottom", fontsize=8)
    axes[0].set_xlabel("Power")
    axes[0].set_ylabel("Scale-free topology fit (signed R^2)")
    axes[0].set_title(f"{title_prefix}: Scale-free fit vs power")

    axes[1].plot(x, y2, marker="o")
    for xi, yi in zip(x, y2):
        axes[1].text(xi, yi + (0.01 * max(1.0, np.nanmax(y2))), str(int(xi)), ha="center", va="bottom", fontsize=8)
    axes[1].set_xlabel("Power")
    axes[1].set_ylabel("Mean connectivity")
    axes[1].set_title(f"{title_prefix}: Mean connectivity vs power")

    fig.savefig(out_file, dpi=190, bbox_inches="tight")
    plt.close(fig)


def compute_tom(adjacency: np.ndarray) -> np.ndarray:
    a = np.asarray(adjacency, dtype=np.float64)
    np.fill_diagonal(a, 0.0)
    k = a.sum(axis=1)
    l = a @ a
    denom = np.minimum.outer(k, k) + 1.0 - a
    tom = (l + a) / np.maximum(denom, 1e-12)
    np.fill_diagonal(tom, 1.0)
    tom = np.clip(tom, 0.0, 1.0)
    return tom


def matrix_sparsity_stats(arr: np.ndarray, eps: float = 1e-12) -> dict[str, float]:
    x = np.asarray(arr)
    if x.size == 0:
        return {
            "zero_fraction": float("nan"),
            "lt_0p01_fraction": float("nan"),
            "lt_0p1_fraction": float("nan"),
            "nonfinite_fraction": float("nan"),
        }

    abs_x = np.abs(x)
    return {
        "zero_fraction": float(np.mean(abs_x <= float(eps))),
        "lt_0p01_fraction": float(np.mean(abs_x < 0.01)),
        "lt_0p1_fraction": float(np.mean(abs_x < 0.1)),
        "nonfinite_fraction": float(np.mean(~np.isfinite(x))),
    }


def main() -> None:
    args = parse_args()
    in_root = resolve_base(args.input_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    powers = parse_powers(args.powers)
    cond_dirs = resolve_conditions(in_root, args.condition)
    records: list[WarehouseRecord] = []

    network_types = ["signed", "unsigned"] if args.network_type == "both" else [args.network_type]
    print(f"Network prep | types={network_types} | powers={powers}")

    for network_type in network_types:
        type_out_root = out_root / network_type if args.network_type == "both" else out_root
        type_out_root.mkdir(parents=True, exist_ok=True)

        for cdir in cond_dirs:
            condition = cdir.name
            npz_file = find_corr_npz(cdir)
            payload = np.load(npz_file, allow_pickle=True)

            corr = np.asarray(payload["corr"], dtype=np.float64)
            genes = payload["genes"].astype(str)
            n_genes = int(corr.shape[0])

            cond_out = type_out_root / condition
            cond_out.mkdir(parents=True, exist_ok=True)

            rows: list[dict[str, float]] = []
            for power in powers:
                adj = adjacency_from_corr(corr, power=power, network_type=network_type)
                k = adj.sum(axis=1)
                sf = scale_free_fit_from_connectivity(k, bins=args.degree_bins)
                rows.append(
                    {
                        "condition": condition,
                        "network_type": network_type,
                        "power": int(power),
                        "n_genes": n_genes,
                        "mean_connectivity": float(np.mean(k)),
                        "median_connectivity": float(np.median(k)),
                        "max_connectivity": float(np.max(k)),
                        **sf,
                    }
                )

            scan_df = pd.DataFrame(rows).sort_values("power").reset_index(drop=True)
            scan_csv = cond_out / f"{condition}_soft_threshold_scan.csv"
            scan_df.to_csv(scan_csv, index=False)

            scan_png = cond_out / f"{condition}_soft_threshold_scan.png"
            plot_soft_threshold(
                scan_df,
                target_r2=args.target_signed_r2,
                out_file=scan_png,
                title_prefix=f"{condition} [{network_type}]",
            )

            if args.force_power is not None:
                selected_power = int(args.force_power)
                select_reason = "forced by --force-power"
            else:
                selected_power, select_reason = choose_power(
                    scan_df,
                    target_signed_r2=args.target_signed_r2,
                    min_mean_k=args.min_mean_connectivity,
                    r2_plateau_delta=args.r2_plateau_delta,
                    r2_near_best_delta=args.r2_near_best_delta,
                )

            adj = adjacency_from_corr(corr, power=selected_power, network_type=network_type)
            adj_diag = matrix_sparsity_stats(adj)
            adj_file = cond_out / f"{condition}_adjacency_beta{selected_power}_{network_type}.npz"
            np.savez_compressed(
                adj_file,
                adjacency=np.asarray(adj, dtype=np.float32),
                genes=genes,
            )

            tom_file = cond_out / f"{condition}_tom_beta{selected_power}_{network_type}.npz"
            tom_diag = {
                "zero_fraction": float("nan"),
                "lt_0p01_fraction": float("nan"),
                "lt_0p1_fraction": float("nan"),
                "nonfinite_fraction": float("nan"),
            }
            if args.skip_tom:
                tom_path_out = ""
            else:
                tom = compute_tom(adj)
                tom_diag = matrix_sparsity_stats(tom)
                np.savez_compressed(
                    tom_file,
                    tom=np.asarray(tom, dtype=np.float32),
                    genes=genes,
                )
                tom_path_out = str(tom_file)

            sparsity_file = cond_out / f"{condition}_network_matrix_sparsity.csv"
            pd.DataFrame(
                [
                    {
                        "condition": condition,
                        "network_type": network_type,
                        "selected_power": int(selected_power),
                        "n_genes": int(n_genes),
                        "adj_zero_fraction": float(adj_diag["zero_fraction"]),
                        "adj_lt_0p01_fraction": float(adj_diag["lt_0p01_fraction"]),
                        "adj_lt_0p1_fraction": float(adj_diag["lt_0p1_fraction"]),
                        "adj_nonfinite_fraction": float(adj_diag["nonfinite_fraction"]),
                        "tom_zero_fraction": float(tom_diag["zero_fraction"]),
                        "tom_lt_0p01_fraction": float(tom_diag["lt_0p01_fraction"]),
                        "tom_lt_0p1_fraction": float(tom_diag["lt_0p1_fraction"]),
                        "tom_nonfinite_fraction": float(tom_diag["nonfinite_fraction"]),
                    }
                ]
            ).to_csv(sparsity_file, index=False)

            selected_meta = {
                "condition": condition,
                "network_type": network_type,
                "selected_power": int(selected_power),
                "selection_reason": select_reason,
                "target_signed_r2": float(args.target_signed_r2),
                "min_mean_connectivity": float(args.min_mean_connectivity),
                "source_corr_file": str(npz_file),
                "adjacency_file": str(adj_file),
                "tom_file": tom_path_out,
                "sparsity_file": str(sparsity_file),
            }
            selected_json = cond_out / f"{condition}_selected_power.json"
            selected_json.write_text(json.dumps(selected_meta, indent=2), encoding="utf-8")

            records.append(
                WarehouseRecord(
                    input_file=str(npz_file),
                    output_file=str(selected_json),
                    script=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                    date_utc=utc_now_iso(),
                    params_hash=params_hash(vars(args)),
                    condition=f"{condition}::{network_type}",
                    stage="08c_network_power_tom_prep",
                )
            )

            print(
                f"[{condition}::{network_type}] selected_beta={selected_power} "
                f"mean_k={scan_df.loc[scan_df.power == selected_power, 'mean_connectivity'].iloc[0]:.3f} "
                f"signed_R2={scan_df.loc[scan_df.power == selected_power, 'signed_r2'].iloc[0]:.3f}"
            )

    append_warehouse(out_root, records)
    print(f"Done. Network prep outputs: {out_root}")


if __name__ == "__main__":
    main()
