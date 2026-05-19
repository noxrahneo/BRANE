"""KM curves for druggable hub genes with HR and FDR annotation."""


from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from utils.network_utils import resolve_base


_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SURVIVAL_DIR = str(_REPO_ROOT / "results/23_survival")
DEFAULT_DRUG_DIR = str(_REPO_ROOT / "results/24_drug_targets")
DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "results/24_drug_targets/drug_survival_km")

PAIR_SUBTYPE_FILTERS = {
    "ER_tumor__vs__Normal": {"er": "positive"},
    "HER2_tumor__vs__Normal": {"her2": "positive"},
    "Triple_negative_tumor__vs__Normal": {"er": "negative", "pr": "negative", "her2": "negative"},
    "Triple_negative_BRCA1_tumor__vs__Normal": {"er": "negative", "pr": "negative", "her2": "negative"},
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic": None,
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal": None,
}

FDR_THRESHOLD = 0.1
HIGHLIGHT_COLOUR = "#e74c3c"
COLOURS = {"high": "#e74c3c", "low": "#2980b9"}
APPROVED_BADGE_COLOUR = "#27ae60"
UNAPPROVED_BADGE_COLOUR = "#95a5a6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drug-annotated KM curves for druggable hub genes")
    parser.add_argument("--survival-dir", default=DEFAULT_SURVIVAL_DIR)
    parser.add_argument("--drug-dir", default=DEFAULT_DRUG_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--p-threshold", type=float, default=0.15,
                        help="Nominal p-value threshold for individual KM plots (default 0.15)")
    parser.add_argument("--endpoint", default="DSS", choices=["DSS"],
                        help="Survival endpoint (DSS only)")
    return parser.parse_args()


def _subtype_mask(clin: pd.DataFrame, filters: dict | None) -> pd.Series:
    #return boolean mask for subtype-matched patients
    if filters is None:
        return pd.Series(True, index=clin.index)
    mask = pd.Series(True, index=clin.index)
    er_col = "breast_carcinoma_estrogen_receptor_status"
    pr_col = "breast_carcinoma_progesterone_receptor_status"
    her2_col = "HER2_Final_Status_nature2012"
    if "er" in filters:
        mask &= clin[er_col].str.lower().str.strip() == filters["er"]
    if "pr" in filters:
        mask &= clin[pr_col].str.lower().str.strip() == filters["pr"]
    if "her2" in filters:
        mask &= clin[her2_col].str.lower().str.strip() == filters["her2"]
    return mask


def _build_gene_drug_table(drug_dir: Path) -> pd.DataFrame:
    #return table of gene × drug × approval_status from ranked candidates
    ranked = pd.read_csv(drug_dir / "05_drug_candidates_ranked.csv")
    rows = []
    for _, r in ranked.iterrows():
        for gene in str(r["targeted_hub_genes"]).split("|"):
            gene = gene.strip()
            if gene:
                rows.append({
                    "gene": gene.upper(),
                    "drug_name": r["canonical_drug_name"],
                    "approved": bool(r["approved_any"]),
                    "candidate_rank": int(r["candidate_rank"]),
                    "candidate_score": float(r["candidate_score"]),
                })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def plot_km_with_drugs(
    gene: str,
    pair_name: str,
    endpoint: str,
    drug_rows: pd.DataFrame,
    expr: pd.DataFrame,
    clin: pd.DataFrame,
    p_logrank: float,
    p_adj: float,
    hr: float,
    deg_direction: str,
    direction_consistency: str,
    output_path: Path,
) -> None:
    #draw a single KM plot for one gene annotated with its drug candidates
    subtype_filter = PAIR_SUBTYPE_FILTERS.get(pair_name)
    mask = _subtype_mask(clin, subtype_filter)
    merged = clin[mask].copy()

    if gene not in expr.columns:
        logging.warning("Gene %s not in expression cache — skipping", gene)
        return

    merged = merged.join(expr[[gene]], how="inner")
    merged = merged.dropna(subset=[gene, f"{endpoint}.time", endpoint])
    merged[endpoint] = pd.to_numeric(merged[endpoint], errors="coerce")
    merged[f"{endpoint}.time"] = pd.to_numeric(merged[f"{endpoint}.time"], errors="coerce")
    merged = merged.dropna(subset=[f"{endpoint}.time", endpoint])
    merged = merged[merged[f"{endpoint}.time"] > 0]

    if len(merged) < 20:
        logging.warning("Too few patients for %s/%s after subtype filter (%d) — skipping", gene, pair_name, len(merged))
        return

    median_expr = merged[gene].median()
    merged["group"] = merged[gene].apply(lambda x: "High" if x >= median_expr else "Low")

    high = merged[merged["group"] == "High"]
    low = merged[merged["group"] == "Low"]

    if len(high) < 5 or len(low) < 5:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    kmf_high.fit(high[f"{endpoint}.time"], event_observed=high[endpoint], label=f"High {gene} (n={len(high)})")
    kmf_low.fit(low[f"{endpoint}.time"], event_observed=low[endpoint], label=f"Low {gene} (n={len(low)})")

    kmf_high.plot_survival_function(ax=ax, color=COLOURS["high"], ci_show=True, ci_alpha=0.12)
    kmf_low.plot_survival_function(ax=ax, color=COLOURS["low"], ci_show=True, ci_alpha=0.12)

    #at-risk table
    try:
        from lifelines.plotting import add_at_risk_counts
        add_at_risk_counts(kmf_high, kmf_low, ax=ax, fontsize=9)
    except Exception:
        pass

    #build drug annotation text
    approved_drugs = drug_rows[drug_rows["approved"]]["drug_name"].tolist()
    other_drugs = drug_rows[~drug_rows["approved"]]["drug_name"].tolist()

    drug_lines = []
    if approved_drugs:
        drug_lines.append("Approved: " + ", ".join(approved_drugs[:4]))
        if len(approved_drugs) > 4:
            drug_lines.append(f"  (+{len(approved_drugs)-4} more approved)")
    if other_drugs:
        drug_lines.append("Other: " + ", ".join(other_drugs[:3]))
        if len(other_drugs) > 3:
            drug_lines.append(f"  (+{len(other_drugs)-3} more)")

    fdr_label = f"FDR = {p_adj:.3f}" + (" ✓ significant" if p_adj < FDR_THRESHOLD else "")
    stats_text = (
        f"HR = {hr:.2f}  |  p = {p_logrank:.4f}\n"
        f"{fdr_label}\n"
        f"Direction: {deg_direction}  ({direction_consistency})"
    )

    #stats box (top right)
    ax.text(0.98, 0.96, stats_text, transform=ax.transAxes,
            fontsize=9, va="top", ha="right",
            bbox={"facecolor": "#f8f9fa", "edgecolor": "#dee2e6", "alpha": 0.9, "pad": 6})

    #drug annotation box (bottom left)
    drug_box_text = "Targeting drugs:\n" + "\n".join(drug_lines)
    colour = APPROVED_BADGE_COLOUR if approved_drugs else UNAPPROVED_BADGE_COLOUR
    ax.text(0.02, 0.04, drug_box_text, transform=ax.transAxes,
            fontsize=8.5, va="bottom", ha="left", color="black",
            bbox={"facecolor": colour, "edgecolor": "none", "alpha": 0.15, "pad": 6})

    pair_short = pair_name.replace("_tumor__vs__Normal", "").replace("__vs__Normal", "").replace("_", " ")
    ax.set_title(
        f"{gene} — {endpoint} by Expression Level\n{pair_short}",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel(f"Time (days)", fontsize=11)
    ax.set_ylabel("Survival Probability", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2)

    #interpretation note at very bottom
    note = ("Note: curves show gene expression as a prognostic biomarker. "
            "Drug annotation indicates known targeting agents — not treatment outcome data.")
    fig.text(0.5, -0.03, note, ha="center", fontsize=7.5, style="italic", color="#6c757d",
             wrap=True)

    plt.tight_layout()
    safe_gene = gene.replace("/", "_")
    safe_pair = pair_name.replace(" ", "_").replace("/", "_")[:50]
    fname = output_path / f"{safe_gene}__{safe_pair}__{endpoint}_drug_km.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Saved %s", fname.name)


def plot_prioritisation_summary(summary_df: pd.DataFrame, output_path: Path, endpoint: str) -> None:
    #one-page summary: forest plot of HR for all druggable genes, coloured by significance
    if summary_df.empty:
        return

    df = summary_df.sort_values("p_logrank").reset_index(drop=True)
    n = len(df)
    fig, ax = plt.subplots(figsize=(13, max(5, n * 0.55 + 2)))

    for i, row in df.iterrows():
        colour = HIGHLIGHT_COLOUR if row["p_adj"] < FDR_THRESHOLD else ("#f39c12" if row["p_logrank"] < 0.1 else "#95a5a6")
        ax.plot([row["hr_lower"], row["hr_upper"]], [i, i], color=colour, lw=2.5, alpha=0.7)
        ax.plot(row["hr"], i, "o", color=colour, ms=9, zorder=3)

    ax.axvline(1.0, color="black", lw=1, ls="--", alpha=0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{r['gene']}  ({r['pair_short']})" for _, r in df.iterrows()],
        fontsize=9
    )
    ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=11)
    ax.set_title(f"Druggable Hub Genes — {endpoint} Hazard Ratios\n(all gene–drug pairs with survival data)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0.05, max(5.0, df["hr_upper"].max() * 1.1))
    ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.2)

    #legend
    legend_els = [
        mpatches.Patch(color=HIGHLIGHT_COLOUR, label=f"FDR < {FDR_THRESHOLD}"),
        mpatches.Patch(color="#f39c12", label="p < 0.10 (nominal)"),
        mpatches.Patch(color="#95a5a6", label="p ≥ 0.10"),
    ]
    ax.legend(handles=legend_els, loc="upper right", fontsize=9)

    #drug labels on right side
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(
        [r["top_drug"] for _, r in df.iterrows()],
        fontsize=7.5, color="#2c3e50"
    )
    ax2.set_ylabel("Top drug candidate", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path / f"00_druggable_genes_forest_{endpoint}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Saved forest summary plot")


def run(args: argparse.Namespace) -> None:
    survival_dir = Path(resolve_base(args.survival_dir))
    drug_dir = Path(resolve_base(args.drug_dir))
    output_dir = Path(resolve_base(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    #load data
    expr = pd.read_parquet(survival_dir / "tcga_cache" / "tcga_brca_expression.parquet")
    clin = pd.read_parquet(survival_dir / "tcga_cache" / "tcga_brca_clinical.parquet")
    clin = clin.set_index("sample") if "sample" in clin.columns else clin

    surv = pd.read_csv(survival_dir / "survival_summary.csv")
    gene_drug = _build_gene_drug_table(drug_dir)

    if gene_drug.empty:
        logging.error("No gene–drug table built — check drug candidates CSV")
        return

    targeted_genes = set(gene_drug["gene"].str.upper())

    endpoint = args.endpoint
    T_col = f"{endpoint}.time"
    E_col = endpoint

    #filter survival results to drug-targeted genes and chosen endpoint
    sub = surv[
        (surv["gene"].str.upper().isin(targeted_genes))
        & (surv["endpoint"] == endpoint)
        & (~surv["skipped"].astype(bool))
    ].copy()

    sub["gene_upper"] = sub["gene"].str.upper()

    summary_rows = []
    for _, row in sub.sort_values("p_logrank").iterrows():
        gene = row["gene_upper"]
        pair = row["pair_name"]
        drugs_for_gene = gene_drug[gene_drug["gene"] == gene].sort_values("candidate_rank")

        if drugs_for_gene.empty:
            continue

        pair_short = pair.replace("_tumor__vs__Normal", "").replace("__vs__Normal", "").replace("_", " ")
        top_drug = drugs_for_gene.iloc[0]["drug_name"]

        summary_rows.append({
            "gene": gene,
            "pair_name": pair,
            "pair_short": pair_short,
            "p_logrank": row["p_logrank"],
            "p_adj": row["p_adj"],
            "hr": row["hr"],
            "hr_lower": max(0.05, row["hr_lower"]),
            "hr_upper": min(20.0, row["hr_upper"]),
            "deg_direction": row["deg_direction"],
            "direction_consistency": row["direction_consistency"],
            "n_drugs": len(drugs_for_gene),
            "top_drug": top_drug,
            "all_drugs": "|".join(drugs_for_gene["drug_name"].tolist()),
        })

        #only generate individual KM plots for genes below the p threshold
        if row["p_logrank"] <= args.p_threshold:
            plot_km_with_drugs(
                gene=gene,
                pair_name=pair,
                endpoint=endpoint,
                drug_rows=drugs_for_gene,
                expr=expr,
                clin=clin,
                p_logrank=row["p_logrank"],
                p_adj=row["p_adj"],
                hr=row["hr"],
                deg_direction=row["deg_direction"],
                direction_consistency=row["direction_consistency"],
                output_path=output_dir,
            )

    if not summary_rows:
        logging.warning("No druggable genes with survival data found")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / f"druggable_genes_survival_{endpoint}.csv", index=False)
    logging.info("Wrote summary CSV with %d rows", len(summary_df))

    plot_prioritisation_summary(summary_df, output_dir, endpoint)

    #print prioritisation table
    print(f"\n=== Druggable Hub Genes — {endpoint} Survival (sorted by p-value) ===\n")
    print(f"{'Gene':<12} {'p_logrank':>10} {'p_adj':>8} {'HR':>6}  {'Top drug':<35} {'Direction':<12} {'FDR sig'}")
    print("-" * 110)
    for _, r in summary_df.sort_values("p_logrank").iterrows():
        sig = "<-- FDR significant" if r["p_adj"] < FDR_THRESHOLD else ""
        print(f"{r['gene']:<12} {r['p_logrank']:>10.4f} {r['p_adj']:>8.3f} {r['hr']:>6.2f}  {r['top_drug']:<35} {r['deg_direction']:<12} {sig}")
    print(f"\nIndividual KM plots (p < {args.p_threshold}) saved to: {output_dir}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    run(args)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
