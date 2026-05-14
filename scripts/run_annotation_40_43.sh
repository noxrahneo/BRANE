#!/usr/bin/env bash
# Runner for scripts 40–43 (annotation pipeline).
# Extracts full gene lists from 14_csd_networks, then runs:
#   40_hgnc_normalisation  → 17_hgnc/
#   41_ncbi_gene_fetch     → 18_ncbi/
#   43_cancer_gene_annotation → 19_cancer_gene_annotation/
#   42_annotation_bridge   → 20_node_annotation/01_input_data/annotation_bridge/
#
# Run from repo root: bash BRANE/scripts/run_annotation_40_43.sh

set -e

REPO=/triumvirate/home/alexarol/breast_cancer_analysis
PYTHON=/triumvirate/home/alexarol/.conda/envs/breast_cancer_scrnaseq/bin/python
SCRIPTS=$REPO/BRANE/scripts
RESULTS=$REPO/BRANE/results

PAIRS=(
    "ER_tumor__vs__Normal"
    "HER2_tumor__vs__Normal"
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal"
    "Triple_negative_BRCA1_tumor__vs__Normal"
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic"
    "Triple_negative_tumor__vs__Normal"
)

cd "$REPO"

for PAIR in "${PAIRS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  PAIR: $PAIR"
    echo "============================================================"

    NET_DIR="$RESULTS/14_csd_networks/$PAIR"
    EDGE_FILE="$NET_DIR/${PAIR}_differential_edges_permutation.csv"

    HGNC_DIR="$RESULTS/17_hgnc/$PAIR"
    NCBI_DIR="$RESULTS/18_ncbi/$PAIR"
    CANCER_DIR="$RESULTS/19_cancer_gene_annotation/$PAIR"
    mkdir -p "$HGNC_DIR" "$NCBI_DIR" "$CANCER_DIR"

    # Extract unique gene list from edge file
    GENE_LIST="$HGNC_DIR/${PAIR}_input_genes.csv"
    echo "[prep] Extracting gene list from $EDGE_FILE"
    $PYTHON - <<PYEOF
import pandas as pd
df = pd.read_csv("$EDGE_FILE")
genes = pd.concat([df["gene_a"], df["gene_b"]]).drop_duplicates().sort_values()
genes.to_frame(name="gene").to_csv("$GENE_LIST", index=False)
print(f"  {len(genes)} unique genes written to $GENE_LIST")
PYEOF

    # Script 40: HGNC normalisation
    echo "[40] HGNC normalisation..."
    $PYTHON "$SCRIPTS/40_hgnc_normalisation.py" \
        --input   "$GENE_LIST" \
        --output  "$HGNC_DIR/${PAIR}_pair_nodes_hgnc_normalised.csv" \
        --log     "$HGNC_DIR/${PAIR}_pair_hgnc_normalisation_log.txt" \
        --cache   "$RESULTS/17_hgnc/hgnc_cache.json"

    # Script 41: NCBI fetch
    echo "[41] NCBI gene fetch..."
    $PYTHON "$SCRIPTS/41_ncbi_gene_fetch.py" \
        --input   "$HGNC_DIR/${PAIR}_pair_nodes_hgnc_normalised.csv" \
        --output  "$NCBI_DIR/${PAIR}_pair_nodes_ncbi_annotated.csv" \
        --log     "$NCBI_DIR/${PAIR}_pair_ncbi_fetch_log.txt" \
        --api-key    "d50cefd909547b59eecb814466078c6c8b08" \
        --batch-size 100

    # Script 43: Cancer gene annotation
    echo "[43] Cancer gene annotation..."
    $PYTHON "$SCRIPTS/43_cancer_gene_annotation.py" \
        --input  "$HGNC_DIR/${PAIR}_pair_nodes_hgnc_normalised.csv" \
        --output "$CANCER_DIR/${PAIR}_cancer_gene_master_annotation.csv" \
        --log    "$CANCER_DIR/${PAIR}_pair_cancer_gene_annotation_log.txt"

    echo "  done: $PAIR"
done

echo ""
echo "============================================================"
echo "  Running 42_annotation_bridge (consolidation)..."
echo "============================================================"
$PYTHON "$SCRIPTS/42_annotation_bridge.py"

echo ""
echo "ALL DONE — annotation pipeline 40–43 complete."
