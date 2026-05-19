# Interactive Co-expression Networks

Six interactive differential co-expression networks from the BRANE (BReast cAncer NEtworks) pipeline, one per contrast. Each file is a self-contained HTML page generated from the annotated network at pipeline stage 20 (node annotation).

Download and open in any browser — no server required.

## Files

| File | Contrast | Nodes | Edges displayed |
|------|----------|-------|-----------------|
| `ER_network_annotated.html` | ER+ tumour vs Normal | 2,563 | 14,311 |
| `HER2_network_annotated.html` | HER2+ tumour vs Normal | 3,194 | 35,781 |
| `TNBC_network_annotated.html` | Triple negative tumour vs Normal | 2,656 | 24,817 |
| `TNBC_BRCA1_network_annotated.html` | TN-BRCA1 tumour vs Normal | 1,735 | 6,278 |
| `NormalBRCA1_network_annotated.html` | N-BRCA1 pre-neoplastic vs Normal | 2,201 | 27,647 |
| `TNBC_BRCA1_vs_NormalBRCA1_network_annotated.html` | TN-BRCA1 tumour vs N-BRCA1 pre-neoplastic | 1,461 | 1,443 |

Edges displayed use a stricter visual threshold (mean + 1 SD of permutation maxima per CSD tier) applied for display only. The full edge sets remain in the pipeline results.

## Node and edge annotations

**Edge colours** reflect CSD tier:
- Blue — Conserved (C): co-expression present in both conditions
- Green — Specific-case (S_case, S_ctrl): co-expression gained in the case condition, co-expression lost from the control condition
- Red — Differentiated (D): co-expression rewired between conditions

**Node border colours** indicate annotated cell type (Epithelial, Fibroblast, Myeloid, BCell, TCell, NK, Cycling, etc.).

**Node size** scales with tier-weighted degree within the displayed tier.

Clicking a node shows its gene symbol, cancer gene annotation (OncoKB / CancerMine / NCG), cell type label, DEG direction (UP / DOWN / STABLE) and CSD zone distribution.

## Dataset

Source data: GSE161529 (Pal et al. 2021, *Nature Cell Biology*), 49 samples across six breast tissue conditions. Pseudobulk profiles were aggregated per patient per cell type; permutation-derived CSD thresholds were applied to Pearson correlation matrices to classify each gene pair.
