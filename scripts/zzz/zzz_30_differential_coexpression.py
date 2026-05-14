#!/usr/bin/env python3
"""Build article-style differential co-expression outputs from condition correlations.

For each condition pair (A vs B), this script:
1) aligns shared genes,
2) compares Pearson correlations per gene pair,
3) labels edges into PO/NO/OP/ON categories,
4) exports condition-specific edge tables and hub metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from utils.warehouse import WarehouseRecord, append_warehouse, params_hash, utc_now_iso

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Differential co-expression using PO/NO/OP/ON categories"
    )
    parser.add_argument(
        "--input-dir",
        default="results/09_correlation/pearson",
        help="Root with per-condition correlation outputs",
    )
    parser.add_argument(
        "--output-dir",
        default="results/07_network/zzz_10_differential_coexpression",
        help="Output root for pairwise differential co-expression outputs",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Condition pair in format 'case:control'. Repeat for multiple pairs.",
    )
    parser.add_argument(
        "--control",
        default="Normal",
        help="Control condition for automatic pair generation if --pair is omitted",
    )
    parser.add_argument(
        "--strong-pcc",
        type=float,
        default=0.7,
        help="Strong co-expression threshold on absolute PCC",
    )
    parser.add_argument(
        "--weak-pcc",
        type=float,
        default=0.3,
        help="No-correlation threshold on absolute PCC",
    )
    parser.add_argument(
        "--min-delta-r",
        type=float,
        default=0.0,
        help="Minimum absolute delta correlation |r_case-r_control|",
    )
    parser.add_argument(
        "--max-edges-per-category",
        type=int,
        default=0,
        help="If >0, keep top-N by |delta_r| per category",
    )
    parser.add_argument(
        "--top-hubs",
        type=int,
        default=30,
        help="Top genes to export by weighted degree and betweenness",
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


def list_conditions(in_root: Path) -> list[str]:
    if not in_root.exists():
        return []
    return sorted([p.name for p in in_root.iterdir() if p.is_dir()])


def parse_pairs(raw_pairs: list[str], all_conditions: list[str], control: str) -> list[tuple[str, str]]:
    if raw_pairs:
        pairs: list[tuple[str, str]] = []
        for text in raw_pairs:
            if ":" not in text:
                raise ValueError(f"Invalid --pair '{text}', expected case:control")
            case, ctrl = [x.strip() for x in text.split(":", 1)]
            pairs.append((case, ctrl))
    else:
        pairs = [(cond, control) for cond in all_conditions if cond != control]

    valid: list[tuple[str, str]] = []
    for case, ctrl in pairs:
        if case not in all_conditions or ctrl not in all_conditions:
            print(f"[warn] skipping pair {case}:{ctrl} (missing condition)")
            continue
        valid.append((case, ctrl))

    dedup = list(dict.fromkeys(valid))
    if not dedup:
        raise ValueError("No valid condition pairs available")
    return dedup


def load_corr_payload(in_root: Path, condition: str) -> tuple[np.ndarray, np.ndarray, Path]:
    cond_dir = in_root / condition
    matches = sorted(cond_dir.glob("*_pearson_corr.npz"))
    if not matches:
        raise FileNotFoundError(f"No *_pearson_corr.npz in {cond_dir}")
    npz_file = matches[0]
    payload = np.load(npz_file, allow_pickle=True)
    corr = np.asarray(payload["corr"], dtype=np.float64)
    genes = payload["genes"].astype(str)
    return corr, genes, npz_file


def align_corrs(
    corr_a: np.ndarray,
    genes_a: np.ndarray,
    corr_b: np.ndarray,
    genes_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_a = {str(g).strip(): i for i, g in enumerate(genes_a) if str(g).strip()}
    idx_b = {str(g).strip(): i for i, g in enumerate(genes_b) if str(g).strip()}

    common = sorted(set(idx_a.keys()).intersection(set(idx_b.keys())))
    if not common:
        raise ValueError("No shared genes between pair conditions")

    ia = np.array([idx_a[g] for g in common], dtype=int)
    ib = np.array([idx_b[g] for g in common], dtype=int)
    sub_a = corr_a[np.ix_(ia, ia)]
    sub_b = corr_b[np.ix_(ib, ib)]
    genes = np.array(common, dtype=str)
    return sub_a, sub_b, genes


def classify_edges(
    corr_case: np.ndarray,
    corr_control: np.ndarray,
    genes: np.ndarray,
    strong_pcc: float,
    weak_pcc: float,
    min_delta_r: float,
    max_edges_per_category: int,
) -> pd.DataFrame:
    tri_i, tri_j = np.triu_indices(corr_case.shape[0], k=1)
    r_case = corr_case[tri_i, tri_j]
    r_ctrl = corr_control[tri_i, tri_j]
    delta = r_case - r_ctrl

    abs_case = np.abs(r_case)
    abs_ctrl = np.abs(r_ctrl)
    abs_delta = np.abs(delta)

    is_case_pos = r_case >= float(strong_pcc)
    is_case_neg = r_case <= -float(strong_pcc)
    is_ctrl_pos = r_ctrl >= float(strong_pcc)
    is_ctrl_neg = r_ctrl <= -float(strong_pcc)
    is_case_weak = abs_case <= float(weak_pcc)
    is_ctrl_weak = abs_ctrl <= float(weak_pcc)
    is_delta_ok = abs_delta >= float(min_delta_r)

    po = is_case_pos & is_ctrl_weak & is_delta_ok
    no = is_case_neg & is_ctrl_weak & is_delta_ok
    op = is_case_weak & is_ctrl_pos & is_delta_ok
    on = is_case_weak & is_ctrl_neg & is_delta_ok

    categories = np.full(r_case.shape[0], "", dtype=object)
    categories[po] = "PO"
    categories[no] = "NO"
    categories[op] = "OP"
    categories[on] = "ON"

    keep = categories != ""
    if not np.any(keep):
        return pd.DataFrame(
            columns=[
                "gene_a",
                "gene_b",
                "r_case",
                "r_control",
                "delta_r",
                "abs_delta_r",
                "category",
                "specific_to",
                "edge_weight_specific",
            ]
        )

    df = pd.DataFrame(
        {
            "gene_a": genes[tri_i[keep]],
            "gene_b": genes[tri_j[keep]],
            "r_case": r_case[keep],
            "r_control": r_ctrl[keep],
            "delta_r": delta[keep],
            "abs_delta_r": abs_delta[keep],
            "category": categories[keep],
        }
    )
    df["specific_to"] = np.where(
        df["category"].isin(["PO", "NO"]),
        "case",
        "control",
    )
    df["edge_weight_specific"] = np.where(
        df["specific_to"] == "case",
        np.abs(df["r_case"]),
        np.abs(df["r_control"]),
    )

    if int(max_edges_per_category) > 0 and not df.empty:
        max_n = int(max_edges_per_category)
        out_frames: list[pd.DataFrame] = []
        for cat in ["PO", "NO", "OP", "ON"]:
            sub = df[df["category"] == cat].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("abs_delta_r", ascending=False).head(max_n)
            out_frames.append(sub)
        if out_frames:
            df = pd.concat(out_frames, axis=0, ignore_index=True)

    return df.sort_values("abs_delta_r", ascending=False).reset_index(drop=True)


def build_specific_graph(edges_df: pd.DataFrame, specific_to: str) -> nx.Graph:
    sub = edges_df[edges_df["specific_to"] == specific_to].copy()
    g = nx.Graph()
    for row in sub.itertuples(index=False):
        g.add_edge(
            str(row.gene_a),
            str(row.gene_b),
            weight=float(row.edge_weight_specific),
            category=str(row.category),
            delta_r=float(row.delta_r),
            r_case=float(row.r_case),
            r_control=float(row.r_control),
        )
    return g


def compute_hub_metrics(g: nx.Graph, top_hubs: int) -> pd.DataFrame:
    if g.number_of_nodes() == 0:
        return pd.DataFrame(
            columns=[
                "gene",
                "degree",
                "weighted_degree",
                "betweenness",
                "component_id",
            ]
        )

    weighted_degree = dict(g.degree(weight="weight"))
    degree = dict(g.degree())

    dist_g = g.copy()
    for _, _, d in dist_g.edges(data=True):
        w = float(d.get("weight", 0.0))
        d["distance"] = 1.0 / max(w, 1e-12)

    betweenness = nx.betweenness_centrality(dist_g, weight="distance")

    component_id: dict[str, int] = {}
    for idx, nodes in enumerate(nx.connected_components(g), start=1):
        for n in nodes:
            component_id[str(n)] = int(idx)

    df = pd.DataFrame(
        {
            "gene": list(g.nodes()),
            "degree": [int(degree[n]) for n in g.nodes()],
            "weighted_degree": [float(weighted_degree[n]) for n in g.nodes()],
            "betweenness": [float(betweenness[n]) for n in g.nodes()],
            "component_id": [int(component_id[str(n)]) for n in g.nodes()],
        }
    ).sort_values(["weighted_degree", "betweenness"], ascending=[False, False])

    if int(top_hubs) > 0:
        return df.head(int(top_hubs)).reset_index(drop=True)
    return df.reset_index(drop=True)


def component_summary(g: nx.Graph, label: str) -> pd.DataFrame:
    if g.number_of_nodes() == 0:
        return pd.DataFrame(
            [{"network": label, "component_id": 0, "n_nodes": 0, "n_edges": 0, "density": 0.0}]
        )

    rows: list[dict[str, object]] = []
    for idx, nodes in enumerate(nx.connected_components(g), start=1):
        sub = g.subgraph(nodes)
        n_nodes = int(sub.number_of_nodes())
        n_edges = int(sub.number_of_edges())
        density = (2.0 * n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0
        rows.append(
            {
                "network": label,
                "component_id": int(idx),
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "density": float(density),
            }
        )
    return pd.DataFrame(rows)


def write_pair_outputs(
    out_dir: Path,
    case: str,
    control: str,
    edges_df: pd.DataFrame,
    top_hubs: int,
) -> dict[str, str]:
    pair_name = f"{case}__vs__{control}"
    pair_dir = out_dir / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    all_edges_file = pair_dir / f"{pair_name}_differential_edges_all.csv"
    edges_df.to_csv(all_edges_file, index=False)

    for cat in ["PO", "NO", "OP", "ON"]:
        cat_file = pair_dir / f"{pair_name}_{cat}.csv"
        edges_df[edges_df["category"] == cat].to_csv(cat_file, index=False)

    case_g = build_specific_graph(edges_df, specific_to="case")
    control_g = build_specific_graph(edges_df, specific_to="control")

    case_edges_file = pair_dir / f"{pair_name}_{case}_specific_edges.csv"
    control_edges_file = pair_dir / f"{pair_name}_{control}_specific_edges.csv"
    edges_df[edges_df["specific_to"] == "case"].to_csv(case_edges_file, index=False)
    edges_df[edges_df["specific_to"] == "control"].to_csv(control_edges_file, index=False)

    case_hubs = compute_hub_metrics(case_g, top_hubs=top_hubs)
    control_hubs = compute_hub_metrics(control_g, top_hubs=top_hubs)
    case_hubs_file = pair_dir / f"{pair_name}_{case}_specific_hubs.csv"
    control_hubs_file = pair_dir / f"{pair_name}_{control}_specific_hubs.csv"
    case_hubs.to_csv(case_hubs_file, index=False)
    control_hubs.to_csv(control_hubs_file, index=False)

    comp_df = pd.concat(
        [
            component_summary(case_g, label=f"{case}_specific"),
            component_summary(control_g, label=f"{control}_specific"),
        ],
        axis=0,
        ignore_index=True,
    )
    comp_file = pair_dir / f"{pair_name}_component_summary.csv"
    comp_df.to_csv(comp_file, index=False)

    case_gexf = pair_dir / f"{pair_name}_{case}_specific.gexf"
    control_gexf = pair_dir / f"{pair_name}_{control}_specific.gexf"
    nx.write_gexf(case_g, case_gexf)
    nx.write_gexf(control_g, control_gexf)

    summary = {
        "pair": pair_name,
        "case": case,
        "control": control,
        "n_differential_edges_total": int(edges_df.shape[0]),
        "n_PO": int((edges_df["category"] == "PO").sum()),
        "n_NO": int((edges_df["category"] == "NO").sum()),
        "n_OP": int((edges_df["category"] == "OP").sum()),
        "n_ON": int((edges_df["category"] == "ON").sum()),
        "n_edges_case_specific": int((edges_df["specific_to"] == "case").sum()),
        "n_edges_control_specific": int((edges_df["specific_to"] == "control").sum()),
        "n_nodes_case_specific": int(case_g.number_of_nodes()),
        "n_nodes_control_specific": int(control_g.number_of_nodes()),
        "all_edges_file": str(all_edges_file),
        "case_hubs_file": str(case_hubs_file),
        "control_hubs_file": str(control_hubs_file),
        "component_summary_file": str(comp_file),
    }
    summary_file = pair_dir / f"{pair_name}_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "pair_dir": str(pair_dir),
        "summary_file": str(summary_file),
        "all_edges_file": str(all_edges_file),
    }


def main() -> None:
    args = parse_args()
    in_root = resolve_base(args.input_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_conditions = list_conditions(in_root)
    pairs = parse_pairs(args.pair, all_conditions, control=args.control)

    records: list[WarehouseRecord] = []
    index_rows: list[dict[str, object]] = []

    for case, control in pairs:
        corr_case, genes_case, case_file = load_corr_payload(in_root, case)
        corr_ctrl, genes_ctrl, ctrl_file = load_corr_payload(in_root, control)

        sub_case, sub_ctrl, common_genes = align_corrs(
            corr_case,
            genes_case,
            corr_ctrl,
            genes_ctrl,
        )

        edges_df = classify_edges(
            corr_case=sub_case,
            corr_control=sub_ctrl,
            genes=common_genes,
            strong_pcc=float(args.strong_pcc),
            weak_pcc=float(args.weak_pcc),
            min_delta_r=float(args.min_delta_r),
            max_edges_per_category=int(args.max_edges_per_category),
        )

        outputs = write_pair_outputs(
            out_dir=out_root,
            case=case,
            control=control,
            edges_df=edges_df,
            top_hubs=int(args.top_hubs),
        )

        pair_name = f"{case}__vs__{control}"
        index_rows.append(
            {
                "pair": pair_name,
                "case": case,
                "control": control,
                "n_common_genes": int(common_genes.size),
                "n_differential_edges": int(edges_df.shape[0]),
                "n_case_specific_edges": int((edges_df["specific_to"] == "case").sum()),
                "n_control_specific_edges": int((edges_df["specific_to"] == "control").sum()),
                "summary_file": outputs["summary_file"],
            }
        )

        records.append(
            WarehouseRecord(
                input_file=f"{case_file};{ctrl_file}",
                output_file=outputs["summary_file"],
                script=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                date_utc=utc_now_iso(),
                params_hash=params_hash(vars(args)),
                condition=pair_name,
                stage="08h_differential_coexpression",
            )
        )

        print(
            f"[{pair_name}] common_genes={common_genes.size} "
            f"diff_edges={edges_df.shape[0]}"
        )

    index_df = pd.DataFrame(index_rows).sort_values("pair") if index_rows else pd.DataFrame()
    index_file = out_root / "differential_coexpression_index.csv"
    index_df.to_csv(index_file, index=False)

    append_warehouse(out_root, records)
    print(f"Done. Differential co-expression outputs: {out_root}")


if __name__ == "__main__":
    main()
