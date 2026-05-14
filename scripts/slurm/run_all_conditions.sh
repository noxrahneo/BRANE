#!/usr/bin/env bash
set -e
PYTHON=/triumvirate/home/alexarol/.conda/envs/breast_cancer_scrnaseq/bin/python
SCRIPT=BRANE/scripts/26_networkx_visualization.py
INPUT=BRANE/results/11_coexpression_prep/single/signed
OUTPUT=BRANE/results/13_coexpression_viz/single/signed

cd /triumvirate/home/alexarol/breast_cancer_analysis

for cond in ER_tumor HER2_tumor "Normal_BRCA1_-_pre-neoplastic" Triple_negative_tumor Triple_negative_BRCA1_tumor; do
    echo "=== $cond ==="
    $PYTHON $SCRIPT \
        --input-dir $INPUT \
        --condition "$cond" \
        --network-type signed \
        --min-weight 0.10 \
        --max-edges 0 \
        --global-max-edges 0 \
        --min-degree 1 \
        --output-dir $OUTPUT
    echo "--- done ---"
done

echo "ALL CONDITIONS COMPLETE"
