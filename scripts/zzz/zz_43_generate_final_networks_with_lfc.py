#!/usr/bin/env python3
"""
Script 51: Generate final thesis-ready networks with LFC metadata injection.

Takes:
  - Original HTMLs in: 12_tagging/final_networks_with_lfc/{pair}/
  - LFC-augmented data from: 12_tagging/output_with_lfc/{pair}_tagged_with_lfc.csv

Produces:
  - Updated HTMLs with embedded LFC + direction in: 12_tagging/final_networks_with_lfc/{pair}/

This script injects LFC and direction metadata + runtime helpers directly into
the HTML files, creating thesis-ready interactive visualizations that show:
  - Network topology (CSD-based rewiring)
  - Cell-type annotation (from Stage 09)
  - Expression direction (UP/DOWN/STABLE)
  - Combined visual badges (↑↓★▼)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE09_DIR = REPO_ROOT / "results" / "stages" / "09_differential_restructured"
LFC_DATA_DIR = STAGE09_DIR / "12_tagging" / "output_with_lfc"
FINAL_NETWORKS_DIR = STAGE09_DIR / "12_tagging" / "final_networks_with_lfc"

METADATA_BLOCK_PATTERN = re.compile(
    r"<!-- Cell-type metadata injected by Script.*? -->\s*"
    r"<script type=\"application/json\" id=\"cell-type-metadata\">.*?</script>",
    flags=re.DOTALL,
)

TOOLTIP_HELPER_MARKER = "var edges;"
HELPER_BLOCK_MARKER = "function applyMetadataToNodeTitles(nodeDataset)"

# JavaScript helper functions for LFC + direction badges
TOOLTIP_HELPER_JS = """
              function _parseCellTypeMetadata() {
                  try {
                      var el = document.getElementById('cell-type-metadata');
                      if (!el) return {};
                      var raw = (el.textContent || el.innerText || '').trim();
                      if (!raw) return {};
                      return JSON.parse(raw);
                  } catch (err) {
                      console.warn('Could not parse cell-type metadata:', err);
                      return {};
                  }
              }

              function _formatMetaValue(value) {
                  if (value === null || value === undefined || value === '') return null;
                  if (typeof value === 'number') {
                      if (Number.isFinite(value) && !Number.isInteger(value)) {
                          return value.toFixed(3);
                      }
                      return String(value);
                  }
                  if (typeof value === 'boolean') {
                      return value ? 'true' : 'false';
                  }
                  return String(value);
              }

              function _appendMetaLine(lines, label, value) {
                  var rendered = _formatMetaValue(value);
                  if (rendered === null) return;
                  lines.push(label + '=' + rendered);
              }

              function _compactPipeList(value, maxItems) {
                  if (value === null || value === undefined) return null;
                  var raw = String(value).trim();
                  if (!raw || raw.toLowerCase() === 'nan' || raw.toLowerCase() === 'none') return null;
                  var parts = raw.split('|').map(function (x) { return x.trim(); }).filter(Boolean);
                  if (!parts.length) return null;
                  var lim = Math.max(1, maxItems || 4);
                  return parts.slice(0, lim).join(', ');
              }

              function _isTruthyFlag(value) {
                  if (value === true || value === 1) return true;
                  var s = String(value || '').trim().toLowerCase();
                  return s === 'true' || s === '1' || s === 'yes';
              }

              function _badgeForMeta(meta) {
                  if (!meta || typeof meta !== 'object') return '';
                  var role = String(meta.cancer_role || '').trim().toLowerCase();
                  var isOnc = _isTruthyFlag(meta.oncokb_is_oncogene) || role === 'oncogene';
                  var isTsg = _isTruthyFlag(meta.oncokb_is_tsg) || role === 'tsg' || role === 'tumor_suppressor' || role === 'tumor suppressor';
                  var direction = String(meta.direction || '').trim().toUpperCase();
                  var marks = '';
                  if (direction === 'UP') marks += '\\u2191';
                  if (direction === 'DOWN') marks += '\\u2193';
                  if (isOnc) marks += '\\u2605';
                  if (isTsg) marks += '\\u25bc';
                  return marks;
              }

              function _baseNodeLabel(node) {
                  if (node && node.original_label) return String(node.original_label).trim();
                  if (node && node.label) {
                      var lbl = String(node.label).trim();
                      if (lbl.endsWith(' \\u2605') || lbl.endsWith(' \\u25bc') || lbl.endsWith(' \\u2191') || lbl.endsWith(' \\u2193')) {
                          lbl = lbl.replace(/\\s+[\\u2191\\u2193\\u2605\\u25bc]+$/g, '').trim();
                      }
                      return lbl;
                  }
                  return '';
              }

              function _cleanBaseTitle(baseTitle) {
                  if (!baseTitle) return '';
                  var text = String(baseTitle);
                  var rawLines = text.split('\\\\n');
                  var kept = [];
                  rawLines.forEach(function (line) {
                      var compact = String(line || '').toLowerCase().replace(/\\s+/g, '');
                      if (compact.indexOf('cell_type=') === 0 || compact.indexOf('celltype=') === 0) return;
                      if (!line || String(line).trim() === '') {
                          if (kept.length && kept[kept.length - 1] !== '') kept.push('');
                          return;
                      }
                      kept.push(String(line));
                  });
                  while (kept.length && kept[kept.length - 1] === '') kept.pop();
                  return kept.join('\\\\n').trim();
              }

              function _normalizeCellType(value) {
                  var rendered = _formatMetaValue(value);
                  if (!rendered) return 'Unknown';
                  return rendered;
              }

              function _cellTypeColor(cellType) {
                  var key = String(cellType || 'Unknown').toLowerCase();
                  var colorMap = {
                      'bcell': '#0072B2',
                      'cycling': '#E74C3C',
                      'epithelial': '#2ECC71',
                      'endo': '#F39C12',
                      'endothelial': '#F39C12',
                      'fibroblast': '#6D4C41',
                      'fibro': '#00ACC1',
                      'fibro2': '#00897B',
                      'myeloid': '#9B59B6',
                      'macro': '#E67E22',
                      'tcell': '#3498DB',
                      'tcell2': '#3498DB',
                      't_cell': '#3498DB',
                      'luminal_epi': '#27AE60',
                      'luminal': '#27AE60',
                      'basal_epi': '#16A085',
                      'basal': '#16A085',
                      'plasma': '#E74C3C',
                      'nk': '#F1C40F',
                      'unknown': '#9CA3AF'
                  };
                  return Object.prototype.hasOwnProperty.call(colorMap, key) ? colorMap[key] : '#9CA3AF';
              }

              function _buildEnhancedNodeTitle(node, meta) {
                  var base = _cleanBaseTitle((node && node.title) ? String(node.title) : '');
                  if (!meta || typeof meta !== 'object') return base;
                  var lines = [];
                  _appendMetaLine(lines, 'cell_type', meta.cell_type);
                  _appendMetaLine(lines, 'cell_type_ref', meta.cell_type_ref);
                  _appendMetaLine(lines, 'ct_switched', meta.ct_switched);
                  _appendMetaLine(lines, 'top_zscore', meta.top_zscore);
                  _appendMetaLine(lines, 'margin', meta.margin);
                  _appendMetaLine(lines, 'ct_source', meta.ct_source);
                  _appendMetaLine(lines, 'major_compartment', meta.major_compartment);
                  _appendMetaLine(lines, 'major_compartment_ref', meta.major_compartment_ref);
                  _appendMetaLine(lines, 'synonyms', _compactPipeList(meta.alias_symbols, 5));
                  _appendMetaLine(lines, 'prev_symbols', _compactPipeList(meta.prev_symbols, 4));
                  _appendMetaLine(lines, 'known_cancer_gene', meta.known_cancer_gene);
                  _appendMetaLine(lines, 'cancer_role', meta.cancer_role);
                  _appendMetaLine(lines, 'evidence_tier', meta.evidence_tier);
                  _appendMetaLine(lines, 'oncokb_is_oncogene', meta.oncokb_is_oncogene);
                  _appendMetaLine(lines, 'oncokb_is_tsg', meta.oncokb_is_tsg);
                  _appendMetaLine(lines, 'lfc', meta.lfc);
                  _appendMetaLine(lines, 'direction', meta.direction);
                  if (!lines.length) return base;
                  if (!base) return lines.join('\\n');
                  return base + '\\n' + lines.join('\\n');
              }

              function applyMetadataToNodeTitles(nodeDataset) {
                  if (!nodeDataset || typeof nodeDataset.get !== 'function' || typeof nodeDataset.update !== 'function') return;
                  var metadataMap = _parseCellTypeMetadata();
                  if (!metadataMap || !Object.keys(metadataMap).length) return;
                  var allNodes = nodeDataset.get();
                  if (!Array.isArray(allNodes) || !allNodes.length) return;
                  var updates = [];
                  allNodes.forEach(function (node) {
                      var key = null;
                      if (node && node.id !== undefined && node.id !== null) {
                          key = String(node.id);
                      } else if (node && node.label) {
                          key = String(node.label);
                      }
                      if (!key) return;
                      var meta = metadataMap[key];
                      if (!meta) return;
                      var enhanced = _buildEnhancedNodeTitle(node, meta);
                      var resolvedCellType = _normalizeCellType(meta.cell_type || node.cell_type);
                      var resolvedCellColor = _cellTypeColor(resolvedCellType);
                      var hasTitleChange = !!enhanced && enhanced !== node.title;
                      var hasCellTypeChange = node.cell_type !== resolvedCellType;
                      var hasCellColorChange = node.cell_color !== resolvedCellColor;
                      if (!hasTitleChange && !hasCellTypeChange && !hasCellColorChange) return;
                      var update = {
                          id: node.id,
                          cell_type: resolvedCellType,
                          cell_color: resolvedCellColor
                      };
                      var baseLabel = _baseNodeLabel(node);
                      var badge = _badgeForMeta(meta);
                      var labelWithBadge = badge ? (baseLabel + ' ' + badge) : baseLabel;
                      if (labelWithBadge && labelWithBadge !== node.label) {
                          update.label = labelWithBadge;
                      }
                      if (baseLabel && baseLabel !== node.original_label) {
                          update.original_label = baseLabel;
                      }
                      if (hasTitleChange) {
                          update.title = enhanced;
                      }
                      updates.push(update);
                  });
                  if (updates.length) {
                      nodeDataset.update(updates);
                  }
              }
""".rstrip("\n")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _sanitize_value(v: Any) -> Any:
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        return v
    return str(v)


def build_gene_metadata_map(lfc_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build metadata map from LFC-augmented tagged table."""
    keep_columns = [
        "approved_symbol",
        "alias_symbols",
        "prev_symbols",
        "hgnc_id",
        "entrez_id",
        "ensembl_gene_id",
        "ncbi_symbol",
        "full_name",
        "summary",
        "go_ids",
        "go_term_names",
        "omim_ids",
        "known_cancer_gene",
        "cancer_role",
        "evidence_tier",
        "oncokb_is_oncogene",
        "oncokb_is_tsg",
        "tier1_cosmic",
        "tier2_oncokb",
        "cancermine_cancer_gene",
        "ncg_gene",
        "cell_type",
        "cell_type_ref",
        "ct_switched",
        "top_zscore",
        "margin",
        "ct_source",
        "major_compartment",
        "major_compartment_ref",
        "lfc",
        "direction",
    ]

    present = [c for c in keep_columns if c in lfc_df.columns]
    if "approved_symbol" not in present:
        raise ValueError("approved_symbol column missing in LFC table")

    de = lfc_df[present].drop_duplicates(subset=["approved_symbol"], keep="first")
    out: dict[str, dict[str, Any]] = {}
    for _, row in de.iterrows():
        gene = str(row["approved_symbol"]).strip()
        if not gene:
            continue
        payload = {col: _sanitize_value(row[col]) for col in present if col != "approved_symbol"}
        out[gene] = payload

    return out


def inject_metadata_block(html_text: str, metadata_map: dict[str, dict[str, Any]]) -> str:
    """Inject or replace metadata JSON block in HTML."""
    json_payload = json.dumps(metadata_map, ensure_ascii=False)
    block = (
        "<!-- Cell-type metadata injected by Script 51 -->\n"
        '<script type="application/json" id="cell-type-metadata">\n'
        f"{json_payload}\n"
        "</script>"
    )

    if METADATA_BLOCK_PATTERN.search(html_text):
        return METADATA_BLOCK_PATTERN.sub(block, html_text)

    if "</head>" in html_text:
        return html_text.replace("</head>", block + "\n</head>")

    return html_text + "\n" + block + "\n"


def inject_helper_functions(html_text: str) -> str:
    """Inject or update helper JavaScript functions."""
    updated = html_text

    # Check if old helpers exist and remove them to force fresh injection
    if f"function _parseCellTypeMetadata()" in updated and f"function applyMetadataToNodeTitles(nodeDataset)" in updated:
        start = updated.find("function _parseCellTypeMetadata()")
        end = updated.find("var filter = {")
        if start != -1 and end != -1 and end > start:
            updated = updated[:start] + TOOLTIP_HELPER_JS + "\n\n              " + updated[end:]
        else:
            logging.debug("Could not locate exact helper block boundaries; appending new helpers")
            if TOOLTIP_HELPER_MARKER in updated:
                updated = updated.replace(
                    TOOLTIP_HELPER_MARKER,
                    TOOLTIP_HELPER_MARKER + "\n\n" + TOOLTIP_HELPER_JS,
                    1,
                )
    elif TOOLTIP_HELPER_MARKER in updated:
        updated = updated.replace(
            TOOLTIP_HELPER_MARKER,
            TOOLTIP_HELPER_MARKER + "\n\n" + TOOLTIP_HELPER_JS,
            1,
        )
    else:
        logging.warning("Could not find tooltip helper anchor")

    # Ensure metadata application hook is in place
    if "applyMetadataToNodeTitles(nodes);" not in updated:
        # Try to inject after edges dataset creation
        hook_pattern = re.compile(
            r"(\]\);\s*)(edges\s*=\s*new\s+vis\.DataSet\(\[)",
            re.DOTALL,
        )
        updated, n = hook_pattern.subn(
            r"\1                  applyMetadataToNodeTitles(nodes);\n\n                  \2",
            updated,
            count=1,
        )
        if n == 0:
            logging.warning("Could not inject node dataset metadata hook")

    return updated


def process_pair(pair_name: str, lfc_csv_path: Path, html_dir: Path) -> bool:
    """Process one condition pair: read LFC data, update HTML."""
    logging.info(f"\n--- Processing: {pair_name} ---")

    # Find HTML file
    html_files = list(html_dir.glob("*_network_interactive.html"))
    if not html_files:
        logging.warning(f"No HTML found in {html_dir}")
        return False

    html_file = html_files[0]
    logging.info(f"Found HTML: {html_file.name}")

    # Read LFC data
    try:
        lfc_df = pd.read_csv(lfc_csv_path)
        logging.info(f"Loaded {len(lfc_df)} genes from LFC CSV")
    except Exception as e:
        logging.error(f"Failed to read LFC CSV: {e}")
        return False

    # Build metadata map
    try:
        metadata_map = build_gene_metadata_map(lfc_df)
        logging.info(f"Built metadata for {len(metadata_map)} genes")
    except Exception as e:
        logging.error(f"Failed to build metadata map: {e}")
        return False

    # Read HTML
    try:
        html_text = html_file.read_text(encoding="utf-8")
    except Exception as e:
        logging.error(f"Failed to read HTML: {e}")
        return False

    # Inject metadata and helpers
    try:
        new_text = inject_metadata_block(html_text, metadata_map)
        new_text = inject_helper_functions(new_text)
    except Exception as e:
        logging.error(f"Failed to inject metadata: {e}")
        return False

    # Write updated HTML
    try:
        html_file.write_text(new_text, encoding="utf-8")
        logging.info(f"✓ Updated HTML with LFC + direction metadata")
        return True
    except Exception as e:
        logging.error(f"Failed to write HTML: {e}")
        return False


def main() -> None:
    configure_logging()

    logging.info("Script 51: Generate final networks with LFC metadata")
    logging.info(f"Input LFC data: {LFC_DATA_DIR}")
    logging.info(f"Output HTML files: {FINAL_NETWORKS_DIR}")

    # Find all LFC CSV files
    lfc_files = sorted(LFC_DATA_DIR.glob("*_tagged_with_lfc.csv"))
    if not lfc_files:
        logging.error(f"No LFC CSV files found in {LFC_DATA_DIR}")
        return

    logging.info("=" * 80)
    logging.info(f"Found {len(lfc_files)} LFC files to process")
    logging.info("=" * 80)

    success_count = 0
    for lfc_file in lfc_files:
        # Extract pair name from filename
        pair_name = lfc_file.name.replace("_tagged_with_lfc.csv", "")

        # Find corresponding HTML directory
        html_dir = FINAL_NETWORKS_DIR / pair_name
        if not html_dir.exists():
            logging.warning(f"HTML directory not found: {html_dir}")
            continue

        # Process pair
        if process_pair(pair_name, lfc_file, html_dir):
            success_count += 1

    logging.info("\n" + "=" * 80)
    logging.info(f"COMPLETE: {success_count}/{len(lfc_files)} pairs processed successfully")
    logging.info(f"Final networks ready in: {FINAL_NETWORKS_DIR}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
