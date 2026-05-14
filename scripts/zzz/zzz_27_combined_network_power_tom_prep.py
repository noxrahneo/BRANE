#!/usr/bin/env python3
"""Build combined-condition network prep artifacts from per-condition correlations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.warehouse import WarehouseRecord, append_warehouse, params_hash, utc_now_iso

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_COMBOS: dict[str, list[str]] = {
    "normal_plus_precancerous": ["Normal", "Normal_BRCA1_-_pre-neoplastic"],
    "normal_plus_er_tumor": ["Normal", "ER_tumor"],
    "normal_plus_her2_tumor": ["Normal", "HER2_tumor"],
    "normal_plus_tn_brca1_tumor": ["Normal", "Triple_negative_BRCA1_tumor"],
    "normal_plus_tn_tumor": ["Normal", "Triple_negative_tumor"],
    "precancerous_plus_tn_brca1_tumor": ["Normal_BRCA1_-_pre-neoplastic", "Triple_negative_BRCA1_tumor"],
    "tn_tumor_plus_tn_brca1_tumor": ["Triple_negative_tumor", "Triple_negative_BRCA1_tumor"],
    "normal_plus_precancerous_plus_tn_brca1_tumor": [
        "Normal",
        "Normal_BRCA1_-_pre-neoplastic",
        "Triple_negative_BRCA1_tumor",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan powers and build adjacency/TOM for combined condition networks"
    )
    parser.add_argument(
        "--input-dir",
        default="results/09_correlation/pearson",
        help="Root with per-condition correlation outputs",
    )
    parser.add_argument(
        "--output-dir",
        default="results/11_coexpression_prep/zzz_combined",
        help="Output root directory for combined network prep outputs",
    )
    parser.add_argument(
        "--combo",
        default="all",
        help="Named combo key (or 'all'). Choices are listed by --list-combos",
    )
    parser.add_argument(
        "--list-combos",
        action="store_true",
        help="Print available default combos and exit",
    )
    parser.add_argument(
        "--network-type",
        choices=["signed", "unsigned"],
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
        default=0.80,
        help="Target signed R^2 for power selection",
    )
    parser.add_argument(
        "--min-mean-connectivity",
        type=float,
        default=1.0,
        help="Minimum mean connectivity for acceptable power",
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
        help="Skip TOM computation/export",
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


def sanitize_name(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return out or "combined"


def build_combo_output_name(source_conditions: list[str]) -> str:
    return "__plus__".join(sanitize_name(c) for c in source_conditions)


def find_corr_npz(in_root: Path, condition: str) -> Path:
    cond_dir = in_root / condition
    if not cond_dir.exists():
        raise FileNotFoundError(f"Condition directory not found: {cond_dir}")
    matches = sorted(cond_dir.glob("*_pearson_corr.npz"))
    if not matches:
        raise FileNotFoundError(f"No *_pearson_corr.npz in {cond_dir}")
    return matches[0]


def load_corr_payload(npz_file: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(npz_file, allow_pickle=True)
    corr = np.asarray(payload["corr"], dtype=np.float64)
    genes = payload["genes"].astype(str)
    norm_genes = np.array([g.strip().upper() for g in genes], dtype=str)
    return corr, norm_genes


def combine_corr_from_conditions(in_root: Path, source_conditions: list[str]) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    corr_by_condition: dict[str, np.ndarray] = {}
    gene_idx_by_condition: dict[str, dict[str, int]] = {}
    source_files: list[Path] = []

    for cond in source_conditions:
        npz_file = find_corr_npz(in_root, cond)
        source_files.append(npz_file)
        corr, norm_genes = load_corr_payload(npz_file)

        idx_map: dict[str, int] = {}
        for i, g in enumerate(norm_genes):
            if g and g not in idx_map:
                idx_map[g] = i

        corr_by_condition[cond] = corr
        gene_idx_by_condition[cond] = idx_map

    common_genes: set[str] | None = None
    for cond in source_conditions:
        genes = set(gene_idx_by_condition[cond].keys())
        common_genes = genes if common_genes is None else common_genes.intersection(genes)

    if not common_genes:
        raise ValueError(f"No shared genes across conditions: {source_conditions}")

    ordered_genes = np.array(sorted(common_genes), dtype=str)
    stacks: list[np.ndarray] = []

    for cond in source_conditions:
        idx_map = gene_idx_by_condition[cond]
        sel = np.array([idx_map[g] for g in ordered_genes], dtype=int)
        sub = corr_by_condition[cond][np.ix_(sel, sel)]
        stacks.append(sub)

    combined = np.mean(np.stack(stacks, axis=0), axis=0)
    combined = np.clip((combined + combined.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(combined, 1.0)

    return combined, ordered_genes, source_files


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
        return {"slope": float("nan"), "r2": float("nan"), "signed_r2": float("nan"), "n_bins_used": 0}

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


def choose_power(scan_df: pd.DataFrame, target_signed_r2: float, min_mean_k: float) -> tuple[int, str]:
    eligible = scan_df[
        (scan_df["signed_r2"] >= float(target_signed_r2))
        & (scan_df["mean_connectivity"] >= float(min_mean_k))
    ].sort_values("power")
    if not eligible.empty:
        return int(eligible.iloc[0]["power"]), "lowest power meeting signed R2 and mean connectivity thresholds"

    fallback = scan_df.copy().sort_values(["signed_r2", "mean_connectivity", "power"], ascending=[False, False, True])
    return int(fallback.iloc[0]["power"]), "fallback to best signed R2 / mean connectivity trade-off"


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
    return np.clip(tom, 0.0, 1.0)


def resolve_combos(requested: str) -> dict[str, list[str]]:
    if requested.strip().lower() == "all":
        return dict(DEFAULT_COMBOS)
    if requested not in DEFAULT_COMBOS:
        raise ValueError(
            f"Unknown combo '{requested}'. Use --list-combos for options."
        )
    return {requested: DEFAULT_COMBOS[requested]}


def main() -> None:
    args = parse_args()
    if args.list_combos:
        print("Available combos:")
        for key, conds in DEFAULT_COMBOS.items():
            print(f"  - {key}: {' + '.join(conds)}")
        return

    in_root = resolve_base(args.input_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    powers = parse_powers(args.powers)
    combos = resolve_combos(args.combo)
    records: list[WarehouseRecord] = []

    print(f"Combined network prep | type={args.network_type} | powers={powers}")

    for combo_key, source_conditions in combos.items():
        combo_name = build_combo_output_name(source_conditions)
        cond_out = out_root / combo_name
        cond_out.mkdir(parents=True, exist_ok=True)

        corr, genes, source_files = combine_corr_from_conditions(in_root, source_conditions)
        n_genes = int(corr.shape[0])

        rows: list[dict[str, float | int | str]] = []
        for power in powers:
            adj = adjacency_from_corr(corr, power=power, network_type=args.network_type)
            k = adj.sum(axis=1)
            sf = scale_free_fit_from_connectivity(k, bins=args.degree_bins)
            rows.append(
                {
                    "combo_key": combo_key,
                    "combined_condition": combo_name,
                    "network_type": args.network_type,
                    "power": int(power),
                    "n_genes": n_genes,
                    "mean_connectivity": float(np.mean(k)),
                    "median_connectivity": float(np.median(k)),
                    "max_connectivity": float(np.max(k)),
                    **sf,
                }
            )

        scan_df = pd.DataFrame(rows).sort_values("power").reset_index(drop=True)
        scan_csv = cond_out / f"{combo_name}_soft_threshold_scan.csv"
        scan_df.to_csv(scan_csv, index=False)

        scan_png = cond_out / f"{combo_name}_soft_threshold_scan.png"
        plot_soft_threshold(scan_df, target_r2=args.target_signed_r2, out_file=scan_png, title_prefix=combo_name)

        if args.force_power is not None:
            selected_power = int(args.force_power)
            select_reason = "forced by --force-power"
        else:
            selected_power, select_reason = choose_power(
                scan_df,
                target_signed_r2=args.target_signed_r2,
                min_mean_k=args.min_mean_connectivity,
            )

        adj = adjacency_from_corr(corr, power=selected_power, network_type=args.network_type)
        adj_file = cond_out / f"{combo_name}_adjacency_beta{selected_power}_{args.network_type}.npz"
        np.savez_compressed(adj_file, adjacency=np.asarray(adj, dtype=np.float32), genes=genes)

        tom_file = cond_out / f"{combo_name}_tom_beta{selected_power}_{args.network_type}.npz"
        if args.skip_tom:
            tom_path_out = ""
        else:
            tom = compute_tom(adj)
            np.savez_compressed(tom_file, tom=np.asarray(tom, dtype=np.float32), genes=genes)
            tom_path_out = str(tom_file)

        selected_meta = {
            "combo_key": combo_key,
            "combined_condition": combo_name,
            "source_conditions": source_conditions,
            "network_type": args.network_type,
            "selected_power": int(selected_power),
            "selection_reason": select_reason,
            "target_signed_r2": float(args.target_signed_r2),
            "min_mean_connectivity": float(args.min_mean_connectivity),
            "n_source_conditions": int(len(source_conditions)),
            "n_genes_intersection": int(n_genes),
            "source_corr_files": [str(p) for p in source_files],
            "adjacency_file": str(adj_file),
            "tom_file": tom_path_out,
        }
        selected_json = cond_out / f"{combo_name}_selected_power.json"
        selected_json.write_text(json.dumps(selected_meta, indent=2), encoding="utf-8")

        records.append(
            WarehouseRecord(
                input_file=";".join(str(p) for p in source_files),
                output_file=str(selected_json),
                script=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                date_utc=utc_now_iso(),
                params_hash=params_hash(vars(args)),
                condition=combo_name,
                stage="08c_combined_network_power_tom_prep",
            )
        )

        selected_row = scan_df.loc[scan_df.power == selected_power].iloc[0]
        print(
            f"[{combo_name}] selected_beta={selected_power} "
            f"mean_k={selected_row['mean_connectivity']:.3f} "
            f"signed_R2={selected_row['signed_r2']:.3f} "
            f"genes={n_genes}"
        )

    append_warehouse(out_root, records)
    print(f"Done. Combined network prep outputs: {out_root}")


if __name__ == "__main__":
    main()
