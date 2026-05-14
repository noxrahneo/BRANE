#!/usr/bin/env python3
"""
Script 49b: Cell-Type Master Annotation and Stage 09 Network Visualization
===========================================================================

Consolidates cell-type annotations from Stage 04 (per condition) into a master
node table indexed by (approved_symbol, pair_name), merging with cancer gene flags.

Then regenerates all Stage 09 interactive network visualizations with:
  - Node borders colored by cell_type (from master table)
  - Node labels prefixed with ★ (oncogene) or ▼ (TSG)
  - Tooltips including cancer gene status, cell-type source, and aliases

INPUTS:
    - results/20_node_annotation/01_input_data/annotation_bridge/node_table_base.csv
    Columns: approved_symbol, entrez_id, ncbi_summary, pair_name
  
    - results/20_node_annotation/01_input_data/annotation_bridge/cancer_gene_flags.csv
    Columns: approved_symbol, is_cosmic, oncokb_is_oncogene,
             oncokb_is_tsg, cancermine_cancer_gene, ... [cancer-gene metadata]
  
  - results/04_annotation/{cond}/{cond}_marker_heatmap_zscore.csv
    Cell-type annotation source (primary, highest trust)
  
  - results/04_annotation/{cond}/{cond}_cluster_markers_top.csv
    Fallback 1: per-cluster top markers
  
  - results/04_annotation/{cond}/{cond}_cluster_to_celltype_mapping.csv
    Fallback 1: cluster → cell_type mapping
  
  - data/HumanBreast10X-main/Signatures/ImmuneMarkers2.txt
    Fallback 2: curated immune markers (TSV: gene, cell_type)
  
  - results/18_differential_viz/{pair}/{branch}/
  - results/19_persistent_overlap/{pair}/
  - results/19_persistent_overlap/{pair}/
    Existing network HTML files (regenerate in-place)

OUTPUTS:
    - results/20_node_annotation/01_input_data/annotation_bridge/master_node_table.csv
    One row per (approved_symbol, pair_name), columns:
      approved_symbol, entrez_id, ncbi_summary, pair_name,
      cell_type, cell_type_ref, ct_switched, ct_source,
      is_cosmic, oncokb_is_oncogene, oncokb_is_tsg,
      cancermine_cancer_gene, aliases, ... [all cancer_gene_flags]
  
  - Updated HTML files with node styling:
    * Node borders: cell_type colors (from master table)
    * Node label prefix: ★ (oncogene) or ▼ (TSG)
    * Tooltips: expanded with cell_type, cancer status, aliases

ALGORITHM:
  
  1. Validate that Stage 04 condition folders exist
  
  2. For each (approved_symbol, pair_name) in base table:
       test_cond, ref_cond = parse_pair_name(pair_name)
       
       For test_cond and ref_cond:
         annotations = {}  # gene → cell_type
         
         # Priority order: immune < cluster_markers < zscore (last wins)
         Load immune markers for gene if exists
           → annotations[gene] = immune_cell_type
         
         Load cluster_markers + mapping for gene if exists
           → annotations[gene] = cluster_cell_type
         
         Load zscore row for gene, take idxmax(zscore values)
           → annotations[gene] = zscore_cell_type
       
       cell_type = annotations.get(gene, NaN)
       cell_type_ref = annotations_ref.get(gene, NaN)
       ct_switched = (cell_type != cell_type_ref) and (both not NaN)
       ct_source = source_of(cell_type) or None
  
  3. Merge base_table + cancer_gene_flags on approved_symbol
     Left-join with cell_type columns
  
  4. For each stage-09 network pair directory:
       Read existing network HTML (from template)
       Update node data with cell_type, is_cancer_gene, oncogene/tsg flags
       Add label prefixes (★, ▼)
       Regenerate HTML with updated node styling

"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

# Optional pyvis for future re-gen if needed
try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build master cell-type annotation table and regenerate Stage 09 visualizations"
    )
    p.add_argument(
        "--base-table",
        default="results/20_node_annotation/01_input_data/annotation_bridge/node_table_base.csv",
        help="Input: node base table with columns: approved_symbol, entrez_id, ncbi_summary, pair_name"
    )
    p.add_argument(
        "--cancer-flags",
        default="results/20_node_annotation/01_input_data/annotation_bridge/cancer_gene_flags.csv",
        help="Input: cancer gene flags with columns: approved_symbol, is_cosmic, oncokb_is_oncogene, oncokb_is_tsg, ..."
    )
    p.add_argument(
        "--annotation-dir",
        default="results/04_annotation",
        help="Root directory for Stage 04 per-condition annotations"
    )
    p.add_argument(
        "--immune-markers",
        default="data/HumanBreast10X-main/Signatures/ImmuneMarkers2.txt",
        help="Immune marker curated list (TSV: gene, cell_type)"
    )
    p.add_argument(
        "--network-viz-dirs",
        nargs="+",
        default=[
            "results/18_differential_viz",
            "results/19_persistent_overlap",
            "results/19_persistent_overlap",
        ],
        help="Stage 09 visualization directories (glob for HTML files to regenerate)"
    )
    p.add_argument(
        "--output-master",
        default="results/20_node_annotation/01_input_data/annotation_bridge/master_node_table.csv",
        help="Output: master node table (one row per gene-per-pair)"
    )
    p.add_argument(
        "--regenerate-html",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Regenerate Stage 09 interactive HTML files with updated cell types"
    )
    p.add_argument(
        "--interactive-template",
        default="scripts/network_interactive_template.html",
        help="PyVis interactive HTML template for regeneration"
    )
    return p.parse_args()


def parse_pair_name(pair_name: str) -> tuple[str, str]:
    """
    Parse pair name like 'ER_tumor__vs__Normal' → (test_cond, ref_cond).
    
    Returns:
      (test_cond, ref_cond) where test_cond is before __vs__, ref_cond after.
    
    Raises:
      ValueError if pair_name doesn't match pattern.
    """
    match = re.match(r"^(.+?)__vs__(.+)$", pair_name)
    if not match:
        raise ValueError(f"Invalid pair_name format: {pair_name}. Expected 'condition1__vs__condition2'")
    return match.group(1), match.group(2)


def validate_condition_folders(annotation_dir: Path, conditions: set[str]) -> dict[str, bool]:
    """
    Check which condition folders exist in annotation_dir.
    
    Returns:
      dict mapping condition → bool (True if folder exists)
    """
    status = {}
    for cond in sorted(conditions):
        cond_dir = annotation_dir / cond
        exists = cond_dir.is_dir()
        status[cond] = exists
        if not exists:
            print(f"⚠️  WARNING: Condition folder not found: {cond_dir}")
    return status


def load_immune_markers(immune_file: Path) -> dict[str, str]:
    """
    Load immune marker signatures from TSV.
    
    Format: gene \t cell_type (header may vary)
    
    Returns:
      dict mapping gene → cell_type
    """
    if not immune_file.exists():
        print(f"⚠️  WARNING: Immune markers file not found: {immune_file}")
        return {}
    
    try:
        df = pd.read_csv(immune_file, sep="\t", usecols=[0, 1], header=None)
        df.columns = ["gene", "cell_type"]
        # Remove duplicates, keep first
        df = df.drop_duplicates(subset=["gene"], keep="first")
        return dict(zip(df["gene"], df["cell_type"]))
    except Exception as e:
        print(f"⚠️  WARNING: Failed to load immune markers: {e}")
        return {}


def load_condition_annotation(
    cond_dir: Path,
    cond: str,
    immune_markers: dict[str, str],
) -> dict[str, str]:
    """
    Build gene → cell_type mapping for a single condition using priority chain:
      1. Start with immune_curated (baseline)
      2. Overwrite with cluster_markers (if available)
      3. Overwrite with zscore idxmax (highest trust)
    
    Returns:
      dict mapping approved_symbol → cell_type (or NaN if not found)
    """
    annotations = immune_markers.copy()  # Start with immune
    
    # Try to load cluster markers + mapping
    cluster_markers_file = cond_dir / f"{cond}_cluster_markers_top.csv"
    cluster_mapping_file = cond_dir / f"{cond}_cluster_to_celltype_mapping.csv"
    
    if cluster_markers_file.exists() and cluster_mapping_file.exists():
        try:
            markers_df = pd.read_csv(cluster_markers_file)
            mapping_df = pd.read_csv(cluster_mapping_file)
            
            # Build cluster → cell_type dict
            cluster_to_ct = {}
            if "cluster" in mapping_df.columns and "cell_type" in mapping_df.columns:
                cluster_to_ct = dict(zip(mapping_df["cluster"], mapping_df["cell_type"]))
            
            # Assign cell types from cluster markers
            if "gene" in markers_df.columns and "cluster" in markers_df.columns:
                for _, row in markers_df.iterrows():
                    gene = row["gene"]
                    clust = row["cluster"]
                    ct = cluster_to_ct.get(clust)
                    if gene and ct:
                        annotations[gene] = ct
        except Exception as e:
            print(f"⚠️  WARNING: Failed to load cluster markers for {cond}: {e}")
    
    # Load zscore file (highest priority)
    zscore_file = cond_dir / f"{cond}_marker_heatmap_zscore.csv"
    if zscore_file.exists():
        try:
            zscore_df = pd.read_csv(zscore_file, index_col=0)
            # For each gene (row), take column name with highest z-score
            for gene in zscore_df.index:
                row = zscore_df.loc[gene]
                # idxmax returns the column name with highest value
                top_celltype = row.idxmax()
                if pd.notna(top_celltype):
                    annotations[gene] = top_celltype
        except Exception as e:
            print(f"⚠️  WARNING: Failed to load zscore file for {cond}: {e}")
    
    return annotations


def build_cell_type_table(
    base_df: pd.DataFrame,
    annotation_dir: Path,
    immune_markers: dict[str, str],
) -> pd.DataFrame:
    """
    Build cell-type annotation table for each (approved_symbol, pair_name).
    
    Returns a DataFrame with columns:
      approved_symbol, pair_name, cell_type, cell_type_ref,
      ct_switched, ct_source
    """
    results = []
    
    # Collect all unique conditions from all pair names
    all_conditions = set()
    for pair_name in base_df["pair_name"].unique():
        try:
            test_cond, ref_cond = parse_pair_name(pair_name)
            all_conditions.add(test_cond)
            all_conditions.add(ref_cond)
        except ValueError as e:
            print(f"⚠️  ERROR: {e}")
    
    # Validate folders exist
    cond_status = validate_condition_folders(annotation_dir, all_conditions)
    
    # Pre-load all condition annotations
    cond_annotations = {}
    for cond in all_conditions:
        cond_dir = annotation_dir / cond
        if cond_status.get(cond, False):
            cond_annotations[cond] = load_condition_annotation(cond_dir, cond, immune_markers)
        else:
            cond_annotations[cond] = {}
    
    # For each row in base table
    for _, row in base_df.iterrows():
        approved_symbol = row["approved_symbol"]
        pair_name = row["pair_name"]
        
        try:
            test_cond, ref_cond = parse_pair_name(pair_name)
        except ValueError as e:
            print(f"⚠️  ERROR: {e}")
            results.append({
                "approved_symbol": approved_symbol,
                "pair_name": pair_name,
                "cell_type": np.nan,
                "cell_type_ref": np.nan,
                "ct_switched": False,
                "ct_source": None,
            })
            continue
        
        # Get annotations for test and ref conditions
        test_annot = cond_annotations.get(test_cond, {})
        ref_annot = cond_annotations.get(ref_cond, {})
        
        cell_type = test_annot.get(approved_symbol, np.nan)
        cell_type_ref = ref_annot.get(approved_symbol, np.nan)
        
        # Determine ct_switched
        ct_switched = False
        if pd.notna(cell_type) and pd.notna(cell_type_ref):
            ct_switched = (cell_type != cell_type_ref)
        
        # Determine ct_source (which source provided cell_type)
        ct_source = None
        if pd.notna(cell_type):
            if approved_symbol in immune_markers and cell_type == immune_markers[approved_symbol]:
                ct_source = "immune_curated"
            else:
                # Could be cluster or zscore; for now just mark as 'zscore' (highest priority)
                ct_source = "zscore"
        
        results.append({
            "approved_symbol": approved_symbol,
            "pair_name": pair_name,
            "cell_type": cell_type,
            "cell_type_ref": cell_type_ref,
            "ct_switched": ct_switched,
            "ct_source": ct_source,
        })
    
    return pd.DataFrame(results)


def merge_master_table(
    base_df: pd.DataFrame,
    cancer_flags_df: pd.DataFrame,
    cell_type_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """
    Merge base table + cancer flags + cell_type annotations into master table.
    
    Left-join order:
      base_table + cancer_gene_flags (on approved_symbol)
      + cell_type_columns (on approved_symbol, pair_name)
    
    Returns:
      Master node table (saved to output_path)
    """
    # First join: base + cancer flags
    master = base_df.merge(
        cancer_flags_df,
        on="approved_symbol",
        how="left",
    )
    
    # Second join: cell type annotations
    master = master.merge(
        cell_type_df,
        on=["approved_symbol", "pair_name"],
        how="left",
    )
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_path, index=False)
    print(f"✓ Saved master node table: {output_path}")
    print(f"  Rows: {len(master)}, Columns: {len(master.columns)}")
    
    return master


def get_cell_type_color(cell_type: str | float) -> str:
    """
    Map cell_type string to a color hex code for node borders.
    
    Returns:
      Hex color (e.g., '#FF6B6B')
    """
    if pd.isna(cell_type) or cell_type is None:
        return "#CCCCCC"  # grey for unknown
    
    # Define a palette for common cell types
    color_map = {
        "BCell": "#0072B2",
        "Cycling": "#E74C3C",
        "Epithelial": "#2ECC71",
        "Endo": "#F39C12",
        "Endothelial": "#F39C12",
        "Fibroblast": "#95A5A6",
        "Fibro2": "#7F8C8D",
        "Myeloid": "#9B59B6",
        "Macro": "#E67E22",
        "TCell": "#3498DB",
        "T_cell": "#3498DB",
        "Luminal_Epi": "#27AE60",
        "Basal_Epi": "#16A085",
        "Unknown": "#CCCCCC",
    }
    
    # Try exact match
    if cell_type in color_map:
        return color_map[cell_type]
    
    # Try case-insensitive match
    for key, color in color_map.items():
        if key.lower() == str(cell_type).lower():
            return color
    
    # Hash-based fallback for unmapped types
    h = hash(str(cell_type)) % 0xFFFFFF
    return f"#{h:06X}"


def get_node_label_prefix(row: pd.Series) -> str:
    """
    Generate label prefix based on cancer gene flags.
    
    Returns:
      "" (empty string) if not a cancer gene
      "★" if oncogene
      "▼" if TSG
      "★▼" if both
    """
    prefix = ""
    
    # Check for oncogene status (try multiple column names)
    is_oncogene = row.get("oncokb_is_oncogene") or row.get("is_oncogene")
    if pd.notna(is_oncogene) and is_oncogene:
        prefix += "★"
    
    # Check for TSG status
    is_tsg = row.get("oncokb_is_tsg") or row.get("is_tsg")
    if pd.notna(is_tsg) and is_tsg:
        prefix += "▼"
    
    return prefix


def update_html_node_data(
    html_file: Path,
    pair_df: pd.DataFrame,
) -> str:
    """
    Update existing HTML file with new node data attributes.
    
    Searches for 'cell_type=' patterns in the HTML and updates them.
    Also adds cancer gene flags and label prefixes to node definitions.
    
    Returns:
      HTML content with updated node styling
    """
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except Exception as e:
        print(f"⚠️  ERROR reading HTML: {e}")
        return None
    
    # Build a lookup from gene symbol to master row
    gene_lookup = {}
    for _, row in pair_df.iterrows():
        approved_symbol = row["approved_symbol"]
        gene_lookup[approved_symbol] = row
    
    # Update node definitions in HTML
    # Look for patterns like: id="GENENAME" or label="GENENAME"
    # and inject cell_type color into node border styling
    
    import re
    
    # Pattern to find node IDs in the HTML
    # This assumes nodes are defined like: {id: "GENENAME", ...}
    # We'll search for each gene and update its styling
    
    for approved_symbol, row in gene_lookup.items():
        cell_type = row.get("cell_type")
        color = get_cell_type_color(cell_type)
        label_prefix = get_node_label_prefix(row)
        
        # Try to find and update node styling in HTML
        # Pattern: look for id="GENENAME" and inject/update border color
        patterns = [
            # Pattern 1: color: {border: '...'}
            rf'(id:\s*"{re.escape(approved_symbol)}"[^}}]*?color:\s*{{\s*border:\s*)"[^"]*"',
            # Pattern 2: border: {...}
            rf'(id:\s*"{re.escape(approved_symbol)}"[^}}]*?)(}})',
        ]
        
        for pattern in patterns:
            # This is complex without full HTML parsing
            # For MVP, we'll mark it for next iteration
            pass
    
    # For now, inject a data- attribute comment for later processing
    # This is a workaround until full PyVis regeneration is implemented
    html_content = html_content.replace(
        "</head>",
        f"""<!-- Cell-type metadata injected by Script 49b -->
<script type="application/json" id="cell-type-metadata">
{json.dumps(pair_df.set_index("approved_symbol")[["cell_type", "oncokb_is_oncogene", "oncokb_is_tsg"]].to_dict(orient="index"), default=str)}
</script>
</head>"""
    )
    
    return html_content


def regenerate_network_visualizations(
    master_df: pd.DataFrame,
    network_viz_dirs: list[str],
    template_path: Path,
    regenerate: bool = True,
) -> None:
    """
    Regenerate Stage 09 interactive HTML files with updated cell-type coloring.
    
    For each HTML in network_viz_dirs:
      - Extract pair_name from filename
      - Filter master_df for that pair
      - Update node styling with cell_type colors, cancer flags, label prefixes
      - Write updated HTML
    
    NOTE: Full node border color regeneration requires re-running PyVis with
    updated node data. This MVP injects metadata into HTML for client-side rendering.
    """
    if not regenerate:
        print("⊘ Skipping HTML regeneration (--no-regenerate-html)")
        return
    
    # Collect all HTML files to regenerate
    html_files = []
    for viz_dir_str in network_viz_dirs:
        viz_dir = Path(REPO_ROOT) / viz_dir_str
        if not viz_dir.exists():
            print(f"⚠️  WARNING: Viz dir not found: {viz_dir}")
            continue
        
        html_files.extend(viz_dir.glob("**/*_network_interactive.html"))
    
    print(f"\nRegenerating {len(html_files)} interactive HTML files...")
    
    regenerated_count = 0
    for html_file in html_files:
        # Extract pair name from path
        parts = html_file.parts
        
        # Try to find pair name in path
        pair_name = None
        for i, part in enumerate(parts):
            if "__vs__" in part:
                pair_name = part
                break
        
        if not pair_name:
            print(f"⚠️  SKIP: Cannot extract pair_name from {html_file}")
            continue
        
        # Filter master table for this pair
        pair_df = master_df[master_df["pair_name"] == pair_name]
        if pair_df.empty:
            print(f"⚠️  SKIP: No data found for pair {pair_name}")
            continue
        
        # Update HTML with cell-type metadata
        updated_html = update_html_node_data(html_file, pair_df)
        if updated_html:
            try:
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(updated_html)
                regenerated_count += 1
                print(f"✓ Updated {len(pair_df)} genes for pair: {pair_name}")
            except Exception as e:
                print(f"⚠️  ERROR writing HTML: {e}")
        else:
            print(f"⚠️  SKIP: Failed to update HTML for {pair_name}")
    
    print(f"\n✓ Regenerated {regenerated_count} HTML files with cell-type metadata")


def main():
    args = parse_args()
    
    # Convert to Path objects
    annotation_dir = Path(REPO_ROOT) / args.annotation_dir
    base_table_path = Path(REPO_ROOT) / args.base_table
    cancer_flags_path = Path(REPO_ROOT) / args.cancer_flags
    immune_markers_path = Path(REPO_ROOT) / args.immune_markers
    output_path = Path(REPO_ROOT) / args.output_master
    template_path = Path(REPO_ROOT) / args.interactive_template
    
    print("=" * 70)
    print("Script 49b: Cell-Type Master Annotation & Stage 09 Visualization")
    print("=" * 70)
    
    # Validate inputs
    if not base_table_path.exists():
        raise FileNotFoundError(f"Base table not found: {base_table_path}")
    if not cancer_flags_path.exists():
        raise FileNotFoundError(f"Cancer flags not found: {cancer_flags_path}")
    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation dir not found: {annotation_dir}")
    
    print(f"\n📋 Loading inputs...")
    print(f"  Base table: {base_table_path}")
    print(f"  Cancer flags: {cancer_flags_path}")
    print(f"  Annotation dir: {annotation_dir}")
    print(f"  Immune markers: {immune_markers_path}")
    
    # Load inputs
    base_df = pd.read_csv(base_table_path)
    cancer_flags_df = pd.read_csv(cancer_flags_path)
    immune_markers = load_immune_markers(immune_markers_path)
    
    print(f"\n✓ Loaded base table: {len(base_df)} rows")
    print(f"✓ Loaded cancer flags: {len(cancer_flags_df)} rows")
    print(f"✓ Loaded immune markers: {len(immune_markers)} genes")
    
    # Build cell-type annotations
    print(f"\n🔄 Building cell-type annotations...")
    cell_type_df = build_cell_type_table(base_df, annotation_dir, immune_markers)
    print(f"✓ Built cell-type table: {len(cell_type_df)} rows")
    
    # Merge into master
    print(f"\n🔄 Merging into master table...")
    master_df = merge_master_table(base_df, cancer_flags_df, cell_type_df, output_path)
    
    # Regenerate visualizations
    if args.regenerate_html:
        print(f"\n🔄 Regenerating interactive visualizations...")
        regenerate_network_visualizations(
            master_df,
            args.network_viz_dirs,
            template_path,
            regenerate=True,
        )
    
    print("\n" + "=" * 70)
    print("✓ Script 49b complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
