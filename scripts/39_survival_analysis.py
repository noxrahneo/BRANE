#!/usr/bin/env python3
"""
Stage 15 — Survival Analysis (Kaplan-Meier).

For each comparison pair:
  1. Compute top 50 hub genes by weighted degree from persistent overlap network.
  2. Pull TCGA-BRCA expression + clinical data via UCSC Xena (cached locally).
  3. Filter to subtype-matched TCGA cohort.
  4. Per-gene: KM curves, log-rank test, Cox HR, BH-FDR correction.
  5. Multi-gene composite risk score: same 50 hubs, z-score normalised mean.
  6. UP-hub and DOWN-hub sub-scores.
  7. Forest plot + per-gene KM plots for significant genes + summary tables.

Design decisions (confirmed):
  - Hub gene pool : top 50 by weighted degree from persistent overlap edges.
  - Risk score    : same 50 hubs (internally consistent).
  - TCGA cohort   : subtype-matched primary; all-BRCA fallback for BRCA1 pairs.
  - Endpoints     : DSS (disease-specific survival) only.
  - Median split  : known limitation, documented in output metadata.
  - Min events    : skip gene if either arm has <3 events.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_ANNOT_DIR = REPO_ROOT / "results/20_node_annotation"
OUTPUT_ROOT = REPO_ROOT / "results/23_survival"
CACHE_DIR = OUTPUT_ROOT / "tcga_cache"

TIERS = ["D", "S_case", "S_ctrl"]

PAIR_SHORT = {
    "ER_tumor__vs__Normal": "ER",
    "HER2_tumor__vs__Normal": "HER2",
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal": "NormalBRCA1",
    "Triple_negative_BRCA1_tumor__vs__Normal": "TNBC_BRCA1",
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic": "TNBC_BRCA1_vs_NormalBRCA1",
    "Triple_negative_tumor__vs__Normal": "TNBC",
}

N_HUBS = 50
MIN_EVENTS = 3
FDR_THRESH = 0.1

# UCSC Xena hubs — confirmed working endpoints
XENA_TCGA_HUB = "https://tcga.xenahubs.net"
XENA_PANCAN_HUB = "https://pancanatlas.xenahubs.net"

# Expression: TCGA BRCA hub (log2 normalised counts)
EXPR_DATASET = "TCGA.BRCA.sampleMap/HiSeqV2"

# Survival: Liu et al. 2018 pan-cancer curated endpoints (OS, DSS, DFI)
SURV_DATASET = "Survival_SupplementalTable_S1_20171025_xena_sp"

# Phenotype: ER/PR/HER2 status for subtype matching
PHENO_DATASET = "TCGA.BRCA.sampleMap/BRCA_clinicalMatrix"

# Phenotype field names (decoded values: Positive/Negative/Indeterminate)
PHENO_ER_FIELD = "breast_carcinoma_estrogen_receptor_status"
PHENO_PR_FIELD = "breast_carcinoma_progesterone_receptor_status"
PHENO_HER2_FIELD = "HER2_Final_Status_nature2012"

PAIR_NAMES = [
    "ER_tumor__vs__Normal",
    "HER2_tumor__vs__Normal",
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic",
    "Triple_negative_tumor__vs__Normal",
]

# Subtype filters applied to TCGA phenotype columns.
# None → use all BRCA patients (fallback for BRCA1 pairs without matched cohort).
PAIR_SUBTYPE: dict[str, Optional[dict[str, list[str]]]] = {
    "ER_tumor__vs__Normal": {
        PHENO_ER_FIELD: ["Positive"],
    },
    "HER2_tumor__vs__Normal": {
        PHENO_HER2_FIELD: ["Positive"],
    },
    "Triple_negative_tumor__vs__Normal": {
        PHENO_ER_FIELD: ["Negative"],
        PHENO_PR_FIELD: ["Negative"],
        PHENO_HER2_FIELD: ["Negative"],
    },
    "Triple_negative_BRCA1_tumor__vs__Normal": {
        PHENO_ER_FIELD: ["Negative"],
        PHENO_PR_FIELD: ["Negative"],
        PHENO_HER2_FIELD: ["Negative"],
    },
    # No matched TCGA cohort for BRCA1 germline carriers → all BRCA fallback
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic": None,
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal": None,
}

DIRECTION_COLOURS = {
    "up": "#d62728",
    "down": "#1f77b4",
    "stable": "#7f7f7f",
    "unchanged": "#7f7f7f",
}

_ENDPOINT_LABEL = {
    "DSS": "Disease-Specific Survival",
}

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---- Hub gene loading -------------------------------------------------------

def load_tier_hubs(pair_name: str, tier: str) -> pd.DataFrame:
    """Load pre-computed hub gene list for a given tier from node_annotation outputs."""
    prefix = PAIR_SHORT.get(pair_name, pair_name)
    hub_file = NODE_ANNOT_DIR / pair_name / f"{prefix}_hubs_{tier}.csv"
    if not hub_file.exists():
        raise FileNotFoundError(f"Hub file not found: {hub_file}")
    df = pd.read_csv(hub_file, low_memory=False)
    # Normalise column names: tier_degree → weighted_degree, lnFC → lfc
    if "tier_degree" in df.columns and "weighted_degree" not in df.columns:
        df = df.rename(columns={"tier_degree": "weighted_degree"})
    if "lnFC" in df.columns and "lfc" not in df.columns:
        df = df.rename(columns={"lnFC": "lfc"})
    if "deg_direction" not in df.columns and "direction" in df.columns:
        df = df.rename(columns={"direction": "deg_direction"})
    df["deg_direction"] = df.get("deg_direction", pd.Series("unknown", index=df.index)).fillna("unknown").str.lower()
    return df


# ---- TCGA data via UCSC Xena (cached) --------------------------------------

def _xena_field_values(
    hub: str, dataset: str, fields: list[str], samples: list[str]
) -> pd.DataFrame:
    """Fetch fields from a Xena clinical/phenotype dataset, decoding categorical codes."""
    import xenaPython as xena
    rows: dict[str, list] = {"sample": samples}
    for field in fields:
        try:
            _, [vals] = xena.dataset_probe_values(hub, dataset, samples, [field])
            codes = xena.field_codes(hub, dataset, [field])
            if codes and codes[0].get("code"):
                # Categorical field: decode integer index → string label
                code_list = codes[0]["code"].split("\t")
                decoded = []
                for v in vals:
                    try:
                        if str(v) == "NaN" or np.isnan(float(v)):
                            decoded.append(None)
                        else:
                            decoded.append(code_list[int(float(v))])
                    except (ValueError, TypeError):
                        decoded.append(None)
                rows[field] = decoded
            else:
                # Numeric field: coerce to float (replaces 'NaN' strings)
                rows[field] = pd.to_numeric(vals, errors="coerce").tolist()
        except Exception as exc:
            log.warning("Could not fetch field %s: %s", field, exc)
            rows[field] = [None] * len(samples)
    return pd.DataFrame(rows)


def _xena_gene_expression(
    hub: str, dataset: str, samples: list[str], genes: list[str],
    batch_size: int = 20,
) -> pd.DataFrame:
    """
    Return (samples × genes) DataFrame from a Xena gene expression dataset.

    Fetches in batches of batch_size to avoid API size limits.
    dataset_gene_probe_avg returns a list of dicts:
      {'gene': str, 'position': [...], 'scores': [[probe1_vals], [probe2_vals], ...]}
    Multiple probes per gene are averaged.
    """
    import xenaPython as xena

    all_expr: dict[str, list] = {}
    n_found = 0

    for i in range(0, len(genes), batch_size):
        batch = genes[i : i + batch_size]
        try:
            result = xena.dataset_gene_probe_avg(hub, dataset, samples, batch)
            for entry in result:
                gene_name = entry["gene"]
                scores = entry["scores"]
                if not scores or not scores[0]:
                    continue
                if len(scores) == 1:
                    vals = [float(v) if str(v) != "NaN" else float("nan") for v in scores[0]]
                else:
                    arr = np.array(
                        [[float(v) if str(v) != "NaN" else float("nan") for v in row] for row in scores]
                    )
                    vals = np.nanmean(arr, axis=0).tolist()
                # Only keep if length matches sample count
                if len(vals) != len(samples):
                    continue
                all_expr[gene_name] = vals
                n_found += 1
        except Exception as exc:
            log.warning("Expression batch %d–%d failed: %s", i, i + batch_size, exc)

    if not all_expr:
        log.error("No gene expression data retrieved for any gene in the batch.")
        return pd.DataFrame(index=samples)

    log.info("Expression: %d/%d genes retrieved from Xena", n_found, len(genes))
    return pd.DataFrame(all_expr, index=samples)


def load_tcga(genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load TCGA-BRCA expression + survival + phenotype (Xena, cached).

    Data sources:
      Expression  : tcga.xenahubs.net  — TCGA.BRCA.sampleMap/HiSeqV2
      Survival    : pancanatlas.xenahubs.net — Liu et al. 2018 (DSS)
      Phenotype   : tcga.xenahubs.net  — TCGA.BRCA.sampleMap/BRCA_clinicalMatrix

    Returns:
        expr_df — DataFrame(index=sample_id, columns=gene_symbols)
        clin_df — DataFrame with DSS, DSS.time, ER/PR/HER2 status
    """
    import xenaPython as xena

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    expr_cache = CACHE_DIR / "tcga_brca_expression.parquet"
    clin_cache = CACHE_DIR / "tcga_brca_clinical.parquet"

    if clin_cache.exists():
        log.info("Loading clinical data from cache")
        clin_df = pd.read_parquet(clin_cache)
    else:
        log.info("Downloading TCGA-BRCA clinical data from Xena ...")

        # BRCA sample IDs come from the expression dataset (BRCA-specific hub)
        brca_samples = xena.dataset_samples(XENA_TCGA_HUB, EXPR_DATASET, None)
        log.info("  %d BRCA samples in expression dataset", len(brca_samples))

        # Survival: Liu et al. pan-cancer curated (OS, DSS, DFI) from pancanAtlas
        surv_df = _xena_field_values(
            XENA_PANCAN_HUB, SURV_DATASET,
            ["DSS", "DSS.time"],
            brca_samples,
        )

        # Phenotype: ER/PR/HER2 status from BRCA clinical matrix
        pheno_df = _xena_field_values(
            XENA_TCGA_HUB, PHENO_DATASET,
            [PHENO_ER_FIELD, PHENO_PR_FIELD, PHENO_HER2_FIELD],
            brca_samples,
        )

        clin_df = surv_df.merge(
            pheno_df.drop(columns=["sample"], errors="ignore"),
            left_index=True, right_index=True, how="left",
        )
        clin_df.to_parquet(clin_cache, index=False)
        log.info("Clinical cache saved (%d samples)", len(clin_df))

    cached_genes: set[str] = set()
    if expr_cache.exists():
        log.info("Loading expression data from cache")
        expr_df = pd.read_parquet(expr_cache)
        cached_genes = set(expr_df.columns)
    else:
        expr_df = pd.DataFrame()

    missing = [g for g in genes if g not in cached_genes]
    if missing:
        log.info("Fetching %d new genes from Xena expression dataset ...", len(missing))
        brca_samples = (
            list(clin_df["sample"])
            if "sample" in clin_df.columns
            else xena.dataset_samples(XENA_TCGA_HUB, EXPR_DATASET, None)
        )
        new_expr = _xena_gene_expression(XENA_TCGA_HUB, EXPR_DATASET, brca_samples, missing)
        if not new_expr.empty:
            expr_df = pd.concat([expr_df, new_expr], axis=1)
            expr_df.to_parquet(expr_cache)
            log.info("Expression cache updated (%d genes total)", expr_df.shape[1])

    return expr_df, clin_df


# ---- Subtype filtering ------------------------------------------------------

def filter_subtype(
    clin_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    pair_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (filtered_clin, filtered_expr, cohort_note)."""
    subtype_filter = PAIR_SUBTYPE.get(pair_name)
    if subtype_filter is None:
        cohort_note = "all_BRCA_fallback"
        mask = pd.Series(True, index=clin_df.index)
    else:
        cohort_note = "subtype_matched"
        mask = pd.Series(True, index=clin_df.index)
        for col, allowed in subtype_filter.items():
            if col in clin_df.columns:
                mask &= clin_df[col].isin(allowed)
            else:
                log.warning("Phenotype column %s not found in clinical data", col)

    clin_sub = clin_df[mask].copy()
    sample_col = "sample" if "sample" in clin_sub.columns else None
    if sample_col:
        expr_sub = expr_df.loc[expr_df.index.isin(clin_sub[sample_col])]
    else:
        expr_sub = expr_df

    log.info("%s: %d patients (%s)", pair_name, len(clin_sub), cohort_note)
    return clin_sub, expr_sub, cohort_note


# ---- KM / Cox helpers -------------------------------------------------------

def _prepare_km_data(
    gene: str,
    expr_df: pd.DataFrame,
    clin_df: pd.DataFrame,
    endpoint: str,
) -> Optional[tuple]:
    """Merge expression + survival, median split. Returns None if skipped."""
    time_col = f"{endpoint}.time"
    event_col = endpoint

    if gene not in expr_df.columns:
        return None
    if time_col not in clin_df.columns or event_col not in clin_df.columns:
        return None

    sample_col = "sample" if "sample" in clin_df.columns else None
    clin_idx = clin_df.set_index(sample_col) if sample_col else clin_df.copy()

    merged = pd.DataFrame({
        "expr": expr_df[gene].reindex(clin_idx.index),
        "T": pd.to_numeric(clin_idx[time_col], errors="coerce"),
        "E": pd.to_numeric(clin_idx[event_col], errors="coerce"),
    }).dropna()
    merged = merged[merged["T"] > 0]

    if len(merged) < 20:
        return None

    median_expr = merged["expr"].median()
    high = merged[merged["expr"] >= median_expr]
    low = merged[merged["expr"] < median_expr]

    if min(len(high), len(low)) < 5:
        return None
    if high["E"].sum() < MIN_EVENTS or low["E"].sum() < MIN_EVENTS:
        return None

    return (
        high["T"].values, high["E"].values,
        low["T"].values, low["E"].values,
        len(high), len(low),
    )


def _run_logrank_cox(T_high, E_high, T_low, E_low) -> dict:
    from lifelines.statistics import logrank_test
    from lifelines import CoxPHFitter

    lr = logrank_test(T_high, T_low, event_observed_A=E_high, event_observed_B=E_low)

    cox_data = pd.DataFrame({
        "T": np.concatenate([T_high, T_low]),
        "E": np.concatenate([E_high, E_low]),
        "group": np.concatenate([np.ones(len(T_high)), np.zeros(len(T_low))]),
    })
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col="T", event_col="E")
    s = cph.summary

    return {
        "p_logrank": lr.p_value,
        "hr": float(np.exp(s.loc["group", "coef"])),
        "hr_lower": float(np.exp(s.loc["group", "coef lower 95%"])),
        "hr_upper": float(np.exp(s.loc["group", "coef upper 95%"])),
    }


# ---- Plots ------------------------------------------------------------------

def plot_km_curve(
    T_high, E_high, T_low, E_low,
    gene: str, pair_name: str, endpoint: str,
    p_logrank: float, hr: float, n_high: int, n_low: int,
    out_path: Path,
) -> None:
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts

    fig, ax = plt.subplots(figsize=(8, 6))
    kmf_high = KaplanMeierFitter().fit(T_high, E_high, label=f"High expression (n={n_high})")
    kmf_low  = KaplanMeierFitter().fit(T_low,  E_low,  label=f"Low expression (n={n_low})")
    kmf_high.plot_survival_function(ax=ax, ci_show=True, color="#d62728")
    kmf_low.plot_survival_function(ax=ax, ci_show=True, color="#1f77b4")

    add_at_risk_counts(kmf_high, kmf_low, ax=ax, fontsize=7)

    ax.set_title(f"{gene}  |  {pair_name.replace('__vs__', ' vs ')}", fontsize=9)
    ax.set_xlabel("Time (days)", fontsize=9)
    ax.set_ylabel(f"{_ENDPOINT_LABEL.get(endpoint, endpoint)} probability", fontsize=9)
    ax.text(
        0.97, 0.97, f"Log-rank p={p_logrank:.3g}\nHR={hr:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_top5_km(
    results: list[dict],
    hubs_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    clin_df: pd.DataFrame,
    pair_name: str,
    tier: str,
    endpoint: str,
    out_path: Path,
    n_top: int = 6,
    panel_kind: str = "top_hubs",  # "top_hubs" or "sig_genes"
) -> None:
    """Multi-panel KM figure.

    panel_kind="top_hubs"  — top N hubs by degree, always generated.
    panel_kind="sig_genes" — FDR-significant genes sorted by p_adj.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts

    top_genes = (
        hubs_df.nlargest(n_top, "weighted_degree")["gene"].tolist()
        if "weighted_degree" in hubs_df.columns
        else hubs_df.head(n_top)["gene"].tolist()
    )
    n_plots = len(top_genes)
    if n_plots == 0:
        return

    stats_by_gene = {r["gene"]: r for r in results}
    ep_label = _ENDPOINT_LABEL.get(endpoint, endpoint)
    pair_short = pair_name.replace("__vs__", " vs ")

    ncols = 2
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 5.0))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, gene in zip(axes_flat, top_genes):
        km_data = _prepare_km_data(gene, expr_df, clin_df, endpoint)
        if km_data is None:
            ax.set_visible(False)
            continue
        T_h, E_h, T_l, E_l, n_h, n_l = km_data

        kmf_h = KaplanMeierFitter().fit(T_h, E_h, label=f"High (n={n_h})")
        kmf_l = KaplanMeierFitter().fit(T_l, E_l, label=f"Low (n={n_l})")
        kmf_h.plot_survival_function(ax=ax, ci_show=True, color="#d62728")
        kmf_l.plot_survival_function(ax=ax, ci_show=True, color="#1f77b4")
        add_at_risk_counts(kmf_h, kmf_l, ax=ax, fontsize=6)

        stats = stats_by_gene.get(gene, {})
        p_lr  = stats.get("p_logrank", float("nan"))
        hr    = stats.get("hr", float("nan"))
        p_adj = stats.get("p_adj", float("nan"))
        deg_d = str(stats.get("deg_direction", "")).lower()
        sig_mark = "*" if p_adj < 0.1 else ""
        deg_sym  = {"up": "↑", "down": "↓"}.get(deg_d, "")

        ax.set_title(
            f"{gene}{deg_sym}  [{tier}]  {sig_mark}",
            fontsize=9,
            fontweight="bold" if p_adj < 0.1 else "normal",
        )
        ax.set_xlabel("Time (days)", fontsize=8)
        ax.set_ylabel(f"{ep_label} prob.", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.text(
            0.97, 0.97,
            f"p={p_lr:.3g}  adj={p_adj:.3g}\nHR={hr:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75),
        )

    # Hide unused axes; center the last plot when it's alone in its row
    n_unused = len(axes_flat) - n_plots
    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    if n_plots % ncols == 1 and n_plots > 1:
        # One plot alone in the last row — center it by shifting its position
        last_ax = axes_flat[n_plots - 1]
        pos = last_ax.get_position()
        last_ax.set_position([
            pos.x0 + pos.width / 2,  # shift right by half a column
            pos.y0,
            pos.width,
            pos.height,
        ])

    if panel_kind == "sig_genes":
        n_total_sig = len(results)
        shown_note = f"top {n_plots} of {n_total_sig}" if n_total_sig > n_plots else f"n={n_plots}"
        suptitle = (
            f"{pair_short}  |  {tier}  |  {endpoint}"
            f"  — FDR<0.1 significant hub genes  ({shown_note}, ranked by p-adj)"
        )
    else:
        suptitle = (
            f"{pair_short}  |  {tier}  |  {endpoint}"
            f"  — top-{n_plots} hubs by degree"
        )

    fig.suptitle(suptitle, fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved top-%d KM panel (2×%d): %s", n_top, nrows, out_path.name)


FOREST_XLIM = (0.3, 3.0)  # cap x-axis; genes beyond this get an arrow annotation

def plot_forest(results_df: pd.DataFrame, pair_name: str, endpoint: str, out_path: Path) -> None:
    """
    HR forest plot. Sorted by HR.
    Colour = deg_direction. CI line width encodes confidence
    (wide CI → thin line, common in small cohorts like HER2+ / TNBC).
    X-axis capped at FOREST_XLIM; genes exceeding the cap get an arrow + exact value.
    """
    df = results_df.dropna(subset=["hr", "hr_lower", "hr_upper"]).sort_values("hr")
    if df.empty:
        return

    n = len(df)
    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.28)))
    x_min, x_max = FOREST_XLIM

    for i, (_, row) in enumerate(df.iterrows()):
        ci_width = row["hr_upper"] - row["hr_lower"]
        lw = max(0.5, 3.5 - ci_width * 0.8)
        colour = DIRECTION_COLOURS.get(str(row.get("deg_direction", "stable")).lower(), "#7f7f7f")

        # Clip CI to xlim for drawing; mark clipped genes with an arrow
        ci_lo = max(row["hr_lower"], x_min)
        ci_hi = min(row["hr_upper"], x_max)
        hr_plot = min(row["hr"], x_max)
        ax.plot([ci_lo, ci_hi], [i, i], color=colour, lw=lw, alpha=0.8)
        ax.scatter(hr_plot, i, color=colour, zorder=5, s=30)

        sig = (
            "**" if row.get("p_adj", 1) < FDR_THRESH
            else ("*" if row.get("p_logrank", 1) < 0.05 else "")
        )

        if row["hr"] > x_max:
            # Arrow pointing right + exact HR value
            ax.annotate(
                f"→ HR={row['hr']:.2f}",
                xy=(x_max, i), xytext=(x_max - 0.05, i),
                fontsize=6.5, va="center", ha="right", color=colour,
            )
            ax.text(x_max + 0.02, i, f"{row['gene']}  {sig}", va="center", fontsize=7)
        else:
            ax.text(ci_hi + 0.05, i, f"{row['gene']}  {sig}", va="center", fontsize=7)

    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlim(x_min, x_max + 0.6)  # extra space for gene labels
    ax.set_yticks([])
    ax.set_xlabel("Hazard Ratio (high vs low expression)", fontsize=9)
    ax.set_title(
        f"{pair_name.replace('__vs__', ' vs ')} — Hub gene HR "
        f"({_ENDPOINT_LABEL.get(endpoint, endpoint)})",
        fontsize=9,
    )
    handles = [
        mpatches.Patch(color=DIRECTION_COLOURS["up"], label="UP in tumor"),
        mpatches.Patch(color=DIRECTION_COLOURS["down"], label="DOWN in tumor"),
        mpatches.Patch(color=DIRECTION_COLOURS["stable"], label="Stable/unknown"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.text(
        0.01, 0.02, "** FDR<0.1  * p<0.05 (unadjusted)  |  x-axis capped at 3.0",
        transform=ax.transAxes, fontsize=7, color="grey",
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_risk_score_km(
    expr_df: pd.DataFrame,
    clin_df: pd.DataFrame,
    hubs: pd.DataFrame,
    pair_name: str,
    endpoint: str,
    direction_filter: Optional[str],
    out_path: Path,
    cohort_note: str,
) -> Optional[dict]:
    """
    Composite risk score = mean z-score across hub genes, split at median.
    direction_filter: None (all hubs), 'up', or 'down'.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    from lifelines.plotting import add_at_risk_counts

    if direction_filter is not None:
        subset_genes = hubs.loc[
            hubs["deg_direction"].str.lower() == direction_filter, "gene"
        ].tolist()
    else:
        subset_genes = hubs["gene"].tolist()

    available = [g for g in subset_genes if g in expr_df.columns]
    if len(available) < 3:
        log.warning(
            "%s: %d genes available for %s risk score — skipping",
            pair_name, len(available), direction_filter or "all",
        )
        return None

    time_col = f"{endpoint}.time"
    event_col = endpoint
    sample_col = "sample" if "sample" in clin_df.columns else None
    clin_idx = clin_df.set_index(sample_col) if sample_col else clin_df.copy()

    expr_sub = expr_df[available].reindex(clin_idx.index)
    z = (expr_sub - expr_sub.mean()) / (expr_sub.std() + 1e-9)
    risk_score = z.mean(axis=1)

    merged = pd.DataFrame({
        "risk": risk_score,
        "T": pd.to_numeric(clin_idx[time_col], errors="coerce"),
        "E": pd.to_numeric(clin_idx[event_col], errors="coerce"),
    }).dropna()
    merged = merged[merged["T"] > 0]

    if len(merged) < 20:
        return None

    median_risk = merged["risk"].median()
    high = merged[merged["risk"] >= median_risk]
    low = merged[merged["risk"] < median_risk]

    if high["E"].sum() < MIN_EVENTS or low["E"].sum() < MIN_EVENTS:
        return None

    lr = logrank_test(high["T"], low["T"], event_observed_A=high["E"], event_observed_B=low["E"])
    p = lr.p_value

    # Median follow-up: median observed time across all patients (Kaplan-Meier estimator of follow-up)
    median_followup = int(merged["T"].median())

    fig, ax = plt.subplots(figsize=(8, 6))
    kmf_high = KaplanMeierFitter().fit(high["T"], high["E"], label=f"High risk (n={len(high)})")
    kmf_low  = KaplanMeierFitter().fit(low["T"],  low["E"],  label=f"Low risk (n={len(low)})")
    kmf_high.plot_survival_function(ax=ax, ci_show=True, color="#d62728")
    kmf_low.plot_survival_function(ax=ax, ci_show=True, color="#1f77b4")

    add_at_risk_counts(kmf_high, kmf_low, ax=ax, fontsize=7)

    dir_label = {None: "all hubs", "up": "UP hubs", "down": "DOWN hubs"}.get(direction_filter, direction_filter)
    ax.set_title(
        f"Hub risk score ({dir_label})  |  {pair_name.replace('__vs__', ' vs ')}\n"
        f"{cohort_note}  |  n_genes={len(available)}  |  median follow-up={median_followup} days",
        fontsize=9,
    )
    ax.set_xlabel("Time (days)", fontsize=9)
    ax.set_ylabel(f"{_ENDPOINT_LABEL.get(endpoint, endpoint)} probability", fontsize=9)
    ax.text(
        0.97, 0.97, f"Log-rank p={p:.3g}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {"n_genes": len(available), "p_logrank": p, "direction_filter": direction_filter or "all",
            "median_followup_days": median_followup}


# ---- Per-pair runner --------------------------------------------------------

def run_pair(
    pair_name: str,
    tier: str,
    expr_df: pd.DataFrame,
    clin_df: pd.DataFrame,
) -> list[dict]:
    out_dir = OUTPUT_ROOT / pair_name / tier
    out_dir.mkdir(parents=True, exist_ok=True)

    hubs = load_tier_hubs(pair_name, tier)
    log.info("%s [%s]: %d hub genes", pair_name, tier, len(hubs))

    clin_sub, expr_sub, cohort_note = filter_subtype(clin_df, expr_df, pair_name)
    n_patients = len(clin_sub)

    if n_patients < 200:
        log.warning(
            "%s: only %d patients — individual gene FDR results are exploratory "
            "(low power). Interpret absence of significance cautiously.",
            pair_name, n_patients,
        )

    all_results: list[dict] = []

    for endpoint in ["DSS"]:
        log.info("%s: running %s ...", pair_name, endpoint)
        per_gene_rows: list[dict] = []

        for _, hub_row in hubs.iterrows():
            gene = hub_row["gene"]
            base = {
                "pair_name": pair_name,
                "tier": tier,
                "gene": gene,
                "endpoint": endpoint,
                "deg_direction": hub_row.get("deg_direction", "unknown"),
                "lfc": hub_row.get("lfc", float("nan")),
                "weighted_degree": hub_row.get("weighted_degree", float("nan")),
                "cohort_note": cohort_note,
                "n_patients": n_patients,
            }

            km_data = _prepare_km_data(gene, expr_sub, clin_sub, endpoint)
            if km_data is None:
                per_gene_rows.append({**base, "skipped": True, "skip_reason": "insufficient data/events"})
                continue

            T_high, E_high, T_low, E_low, n_high, n_low = km_data
            try:
                stats = _run_logrank_cox(T_high, E_high, T_low, E_low)
            except Exception as exc:
                log.debug("%s %s Cox failed: %s", pair_name, gene, exc)
                per_gene_rows.append({**base, "skipped": True, "skip_reason": "Cox fit failed"})
                continue

            per_gene_rows.append({
                **base,
                "n_high": n_high, "n_low": n_low,
                "n_events_high": int(E_high.sum()), "n_events_low": int(E_low.sum()),
                "p_logrank": stats["p_logrank"],
                "hr": stats["hr"], "hr_lower": stats["hr_lower"], "hr_upper": stats["hr_upper"],
                "skipped": False,
            })

        # BH-FDR correction across all tested genes
        tested = [r for r in per_gene_rows if not r.get("skipped", True)]
        if tested:
            from statsmodels.stats.multitest import multipletests
            _, padj, _, _ = multipletests([r["p_logrank"] for r in tested], method="fdr_bh")
            for r, pa in zip(tested, padj):
                r["p_adj"] = pa

        results_df = pd.DataFrame(per_gene_rows)

        # Directionality consistency flag:
        # "consistent"   = high expression → worse survival AND gene is UP in tumor
        #                   OR high expression → better survival AND gene is DOWN in tumor
        # "paradoxical"  = the survival direction conflicts with tumor expression direction
        #                   e.g. gene is DOWN in tumor but high expression → worse survival (like EMP1)
        # "not_assessed" = gene was skipped or direction is unknown
        if "hr" in results_df.columns and "deg_direction" in results_df.columns:
            def _consistency(row):
                if row.get("skipped", True) or pd.isna(row.get("hr")):
                    return "not_assessed"
                d = str(row.get("deg_direction", "")).lower()
                if d not in ("up", "down"):
                    return "not_assessed"
                hr_high = row["hr"] > 1  # True = high expression → worse survival
                if (hr_high and d == "up") or (not hr_high and d == "down"):
                    return "consistent"
                return "paradoxical"
            results_df["direction_consistency"] = results_df.apply(_consistency, axis=1)

        prefix = PAIR_SHORT.get(pair_name, pair_name)
        results_df.to_csv(out_dir / f"{prefix}_{tier}_{endpoint}_km_results.csv", index=False)
        all_results.extend(results_df.to_dict(orient="records"))

        # KM plots for significant genes
        sig_genes = [r for r in tested if r.get("p_adj", 1) < FDR_THRESH]
        MAX_PANEL = 18  # beyond this a grid becomes unreadable

        if sig_genes:
            sig_dir = out_dir / f"{endpoint}_km_plots"
            for r in sig_genes:
                km_data = _prepare_km_data(r["gene"], expr_sub, clin_sub, endpoint)
                if km_data:
                    T_h, E_h, T_l, E_l, n_h, n_l = km_data
                    plot_km_curve(
                        T_h, E_h, T_l, E_l,
                        gene=r["gene"], pair_name=pair_name, endpoint=endpoint,
                        p_logrank=r["p_logrank"], hr=r["hr"],
                        n_high=n_h, n_low=n_l,
                        out_path=sig_dir / f"{r['gene']}_{endpoint}_km.png",
                    )

            # Combined panel when number of significant genes is manageable
            if len(sig_genes) <= MAX_PANEL:
                # Sort by p_adj ascending so most significant genes come first
                sig_sorted = sorted(sig_genes, key=lambda r: r.get("p_adj", 1))
                plot_top5_km(
                    results=sig_sorted,
                    hubs_df=pd.DataFrame(
                        [{"gene": r["gene"],
                          "weighted_degree": len(sig_sorted) - i}  # rank preserves p_adj order
                         for i, r in enumerate(sig_sorted)]
                    ),
                    expr_df=expr_sub,
                    clin_df=clin_sub,
                    pair_name=pair_name,
                    tier=tier,
                    endpoint=endpoint,
                    out_path=out_dir / f"{prefix}_{tier}_{endpoint}_sig_km_panel.png",
                    n_top=min(len(sig_sorted), 6),
                    panel_kind="sig_genes",
                )

        log.info("%s %s: %d FDR<%.1f genes, plots saved", pair_name, endpoint, len(sig_genes), FDR_THRESH)

        # Forest plot
        tested_df = results_df[~results_df["skipped"].fillna(True)]
        if not tested_df.empty:
            plot_forest(
                tested_df, pair_name=pair_name, endpoint=endpoint,
                out_path=out_dir / f"{prefix}_{tier}_{endpoint}_forest_plot.png",
            )

        # Top-5 hub KM panel — always generated regardless of FDR
        plot_top5_km(
            results=tested,
            hubs_df=hubs,
            expr_df=expr_sub,
            clin_df=clin_sub,
            pair_name=pair_name,
            tier=tier,
            endpoint=endpoint,
            out_path=out_dir / f"{prefix}_{tier}_{endpoint}_top6_km.png",
        )

    return all_results


# ---- Main -------------------------------------------------------------------

def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Collect all hub genes across all pairs × tiers for a single TCGA load
    log.info("Loading hub gene lists for all pairs × tiers ...")
    run_plan: list[tuple[str, str, pd.DataFrame]] = []  # (pair, tier, hubs)
    all_genes: set[str] = set()
    for pair in PAIR_NAMES:
        for tier in TIERS:
            try:
                hubs = load_tier_hubs(pair, tier)
                run_plan.append((pair, tier, hubs))
                all_genes.update(hubs["gene"].dropna().tolist())
            except FileNotFoundError as exc:
                log.warning("Skipping %s / %s: %s", pair, tier, exc)

    log.info("Total unique hub genes across all pairs × tiers: %d", len(all_genes))

    log.info("Loading TCGA-BRCA data ...")
    expr_df, clin_df = load_tcga(list(all_genes))

    if expr_df.empty or clin_df.empty:
        log.error(
            "TCGA data could not be loaded. Check network connectivity to Xena "
            "or inspect %s for cached data.", CACHE_DIR,
        )
        return 1

    all_results: list[dict] = []
    for pair, tier, _ in tqdm(run_plan, desc="Pair/tier", unit="run"):
        try:
            all_results.extend(run_pair(pair, tier, expr_df, clin_df))
        except Exception as exc:
            log.error("Pair %s / tier %s failed: %s", pair, tier, exc, exc_info=True)

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(OUTPUT_ROOT / "survival_summary.csv", index=False)
        log.info("Survival summary: %d rows", len(summary_df))

        top_df = (
            summary_df[
                (~summary_df["skipped"].fillna(True))
                & (summary_df["p_adj"].fillna(1) < FDR_THRESH)
            ]
            .sort_values("hr", ascending=False, key=abs)
            .drop_duplicates(subset=["gene", "pair_name", "tier"])
        )
        top_df.to_csv(OUTPUT_ROOT / "top_prognostic_genes.csv", index=False)
        log.info("Top prognostic genes (FDR<%.1f): %d", FDR_THRESH, len(top_df))

    log.info("Done. Results in %s", OUTPUT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
