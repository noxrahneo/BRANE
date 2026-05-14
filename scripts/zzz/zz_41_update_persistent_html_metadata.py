#!/usr/bin/env python3
"""
Script 49d: Update persistent overlap interactive HTMLs with Script 49c tagged metadata.

Reads per-pair tagged tables from:
  results/23_node_annotation/output/{pair}_tagged.csv

Injects/replaces metadata JSON block in existing interactive files:
  results/19_persistent_overlap/{pair}/*_network_interactive.html

Stage 04 remains read-only; this script only updates Stage 09 HTML outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TAGGED_DIR = (
    REPO_ROOT
    / "results"
    / "stages"
    / "09_differential_restructured"
    / "12_tagging"
    / "output"
)
TAGGED_WITH_LFC_DIR = (
    REPO_ROOT
    / "results"
    / "stages"
    / "09_differential_restructured"
    / "12_tagging"
    / "03_output_with_lfc"
)
PERSISTENT_VIZ_DIR = (
    REPO_ROOT
    / "results"
    / "stages"
    / "09_differential_restructured"
    / "08_persistent_overlap_viz"
)
FINAL_NETWORKS_DIR = (
    REPO_ROOT
    / "results"
    / "stages"
    / "09_differential_restructured"
    / "12_tagging"
    / "final_networks"
)

METADATA_BLOCK_PATTERN = re.compile(
    r"<!-- Cell-type metadata injected by Script 49b -->\s*"
    r"<script type=\"application/json\" id=\"cell-type-metadata\">.*?</script>",
    flags=re.DOTALL,
)

TOOLTIP_HELPER_ANCHOR = "var edges;"
TOOLTIP_HELPER_MARKER = "function applyMetadataToNodeTitles(nodeDataset)"

EXISTING_HELPER_BLOCK_PATTERN = re.compile(
    r"function _parseCellTypeMetadata\(\)[\s\S]*?"
    r"function applyMetadataToNodeTitles\(nodeDataset\)[\s\S]*?"
    r"\n\s*}\s*(?=\n\s*var filter\s*=\s*\{)",
    flags=re.DOTALL,
)

NODE_DATASET_HOOK_PATTERN = re.compile(
    r"(\]\);\s*)(edges\s*=\s*new\s+vis\.DataSet\(\[)",
    flags=re.DOTALL,
)

TOOLTIP_HELPER_JS = """
              function _parseCellTypeMetadata() {
                  try {
                      var el = document.getElementById('cell-type-metadata');
                      if (!el) {
                          return {};
                      }
                      var raw = (el.textContent || el.innerText || '').trim();
                      if (!raw) {
                          return {};
                      }
                      return JSON.parse(raw);
                  } catch (err) {
                      console.warn('Could not parse cell-type metadata block:', err);
                      return {};
                  }
              }

              function _formatMetaValue(value) {
                  if (value === null || value === undefined || value === '') {
                      return null;
                  }
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
                  if (rendered === null) {
                      return;
                  }
                  lines.push(label + '=' + rendered);
              }

              function _compactPipeList(value, maxItems) {
                  if (value === null || value === undefined) {
                      return null;
                  }
                  var raw = String(value).trim();
                  if (!raw || raw.toLowerCase() === 'nan' || raw.toLowerCase() === 'none') {
                      return null;
                  }
                  var parts = raw.split('|').map(function (x) { return x.trim(); }).filter(Boolean);
                  if (!parts.length) {
                      return null;
                  }
                  var lim = Math.max(1, maxItems || 4);
                  return parts.slice(0, lim).join(', ');
              }

              function _isTruthyFlag(value) {
                  if (value === true || value === 1) {
                      return true;
                  }
                  var s = String(value || '').trim().toLowerCase();
                  return s === 'true' || s === '1' || s === 'yes';
              }

              function _directionBadge(direction) {
                  var dir = String(direction || '').trim().toUpperCase();
                  if (dir === 'UP') {
                      return '↑';
                  }
                  if (dir === 'DOWN') {
                      return '↓';
                  }
                  return '';
              }

              function _badgeForMeta(meta) {
                  if (!meta || typeof meta !== 'object') {
                      return '';
                  }
                  var role = String(meta.cancer_role || '').trim().toLowerCase();
                  var isOnc = _isTruthyFlag(meta.oncokb_is_oncogene) || role === 'oncogene';
                  var isTsg = _isTruthyFlag(meta.oncokb_is_tsg) || role === 'tsg' || role === 'tumor_suppressor' || role === 'tumor suppressor';
                  var direction = String(meta.direction || '').trim().toUpperCase();
                  var marks = '';
                  if (direction === 'UP') {
                      marks += '↑';
                  }
                  if (direction === 'DOWN') {
                      marks += '↓';
                  }
                  if (isOnc) {
                      marks += '★';
                  }
                  if (isTsg) {
                      marks += '▼';
                  }
                  return marks;
              }

              function _baseNodeLabel(node) {
                  if (node && node.original_label) {
                      return String(node.original_label).trim();
                  }
                  if (node && node.label) {
                      var lbl = String(node.label).trim();
                      if (lbl.endsWith(' ★') || lbl.endsWith(' ▼') || lbl.endsWith(' ★▼')) {
                          lbl = lbl.replace(/\s+[★▼]+$/g, '').trim();
                      }
                      return lbl;
                  }
                  return '';
              }

              function _cleanBaseTitle(baseTitle) {
                  if (!baseTitle) {
                      return '';
                  }
                  var text = String(baseTitle);
                  var rawLines = text.split('\\n');
                  var kept = [];

                  rawLines.forEach(function (line) {
                      var compact = String(line || '').toLowerCase().replace(/\s+/g, '');
                      if (compact.indexOf('cell_type=') === 0 || compact.indexOf('celltype=') === 0) {
                          return;
                      }
                      if (!line || String(line).trim() === '') {
                          if (kept.length && kept[kept.length - 1] !== '') {
                              kept.push('');
                          }
                          return;
                      }
                      kept.push(String(line));
                  });

                  while (kept.length && kept[kept.length - 1] === '') {
                      kept.pop();
                  }
                  return kept.join('\\n').trim();
              }

              function _normalizeCellType(value) {
                  var rendered = _formatMetaValue(value);
                  if (!rendered) {
                      return 'Unknown';
                  }
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
                      'fibroblast': '#95A5A6',
                      'fibro': '#7F8C8D',
                      'fibro2': '#7F8C8D',
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
                  if (Object.prototype.hasOwnProperty.call(colorMap, key)) {
                      return colorMap[key];
                  }
                  return '#9CA3AF';
              }

              function _buildEnhancedNodeTitle(node, meta) {
                  var base = _cleanBaseTitle((node && node.title) ? String(node.title) : '');
                  if (!meta || typeof meta !== 'object') {
                      return base;
                  }

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

                  if (!lines.length) {
                      return base;
                  }
                  if (!base) {
                      return lines.join('\\n');
                  }
                  return base + '\\n' + lines.join('\\n');
              }

              function applyMetadataToNodeTitles(nodeDataset) {
                  if (!nodeDataset || typeof nodeDataset.get !== 'function' || typeof nodeDataset.update !== 'function') {
                      return;
                  }

                  var metadataMap = _parseCellTypeMetadata();
                  if (!metadataMap || !Object.keys(metadataMap).length) {
                      return;
                  }

                  var allNodes = nodeDataset.get();
                  if (!Array.isArray(allNodes) || !allNodes.length) {
                      return;
                  }

                  var updates = [];
                  allNodes.forEach(function (node) {
                      var key = null;
                      if (node && node.id !== undefined && node.id !== null) {
                          key = String(node.id);
                      } else if (node && node.label) {
                          key = String(node.label);
                      }
                      if (!key) {
                          return;
                      }

                      var meta = metadataMap[key];
                      if (!meta) {
                          return;
                      }

                      var enhanced = _buildEnhancedNodeTitle(node, meta);
                      var resolvedCellType = _normalizeCellType(meta.cell_type || node.cell_type);
                      var resolvedCellColor = _cellTypeColor(resolvedCellType);

                      var hasTitleChange = !!enhanced && enhanced !== node.title;
                      var hasCellTypeChange = node.cell_type !== resolvedCellType;
                      var hasCellColorChange = node.cell_color !== resolvedCellColor;

                      if (!hasTitleChange && !hasCellTypeChange && !hasCellColorChange) {
                          return;
                      }

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inject Script 49c tagged metadata into persistent overlap interactive HTML files"
    )
    p.add_argument(
        "--tagged-dir",
        default=str(TAGGED_DIR),
        help="Directory containing *_tagged.csv files",
    )
    p.add_argument(
        "--viz-dir",
        default=str(PERSISTENT_VIZ_DIR),
        help="Persistent overlap visualization root",
    )
    p.add_argument(
        "--use-lfc",
        action="store_true",
        default=True,
        help="Use LFC-enhanced data from output_with_lfc if available (default: True)",
    )
    p.add_argument(
        "--no-lfc",
        dest="use_lfc",
        action="store_false",
        help="Force use of basic tagged data from output (skip LFC)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report actions without writing files",
    )
    return p.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _sanitize_value(v: Any) -> Any:
    if pd.isna(v):
        return None
    # keep ints as ints where possible
    if isinstance(v, (int, float)):
        return v
    return str(v)


def build_gene_metadata_map(tagged_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    # include rich annotation payload for tooltip/client rendering
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
        "is_cosmic",
        "is_oncokb",
        "is_disgenet",
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

    present = [c for c in keep_columns if c in tagged_df.columns]
    if "approved_symbol" not in present:
        raise ValueError("approved_symbol column missing in tagged table")

    # deduplicate by approved_symbol to single metadata row
    de = tagged_df[present].drop_duplicates(subset=["approved_symbol"], keep="first")

    out: dict[str, dict[str, Any]] = {}
    for _, row in de.iterrows():
        gene = str(row["approved_symbol"]).strip()
        if not gene:
            continue
        payload = {col: _sanitize_value(row[col]) for col in present if col != "approved_symbol"}
        out[gene] = payload

    return out


def inject_metadata_block(html_text: str, metadata_map: dict[str, dict[str, Any]]) -> str:
    json_payload = json.dumps(metadata_map, ensure_ascii=False)
    block = (
        "<!-- Cell-type metadata injected by Script 49b -->\n"
        '<script type="application/json" id="cell-type-metadata">\n'
        f"{json_payload}\n"
        "</script>"
    )

    if METADATA_BLOCK_PATTERN.search(html_text):
        return METADATA_BLOCK_PATTERN.sub(block, html_text)

    # fallback insertion before </head>
    if "</head>" in html_text:
        return html_text.replace("</head>", block + "\n</head>")

    # final fallback append
    return html_text + "\n" + block + "\n"


def inject_tooltip_runtime(html_text: str) -> str:
    updated = html_text

    # Always prefer the newest helper implementation by replacing old injected blocks.
    updated, replaced = EXISTING_HELPER_BLOCK_PATTERN.subn(
        lambda _m: TOOLTIP_HELPER_JS,
        updated,
        count=1,
    )

    if replaced == 0 and TOOLTIP_HELPER_MARKER not in updated:
        if TOOLTIP_HELPER_ANCHOR in updated:
            updated = updated.replace(
                TOOLTIP_HELPER_ANCHOR,
                TOOLTIP_HELPER_ANCHOR + "\n\n" + TOOLTIP_HELPER_JS,
                1,
            )
        else:
            logging.warning("Could not find tooltip helper anchor '%s'", TOOLTIP_HELPER_ANCHOR)

    # Fallback: force-upgrade older helper blocks that may evade regex replacement.
    if "function _cellTypeColor(cellType)" not in updated:
        start = updated.find("function _parseCellTypeMetadata()")
        end = updated.find("var filter = {")
        if start != -1 and end != -1 and end > start:
            updated = updated[:start] + TOOLTIP_HELPER_JS + "\n\n              " + updated[end:]
        elif TOOLTIP_HELPER_MARKER not in updated and TOOLTIP_HELPER_ANCHOR in updated:
            updated = updated.replace(
                TOOLTIP_HELPER_ANCHOR,
                TOOLTIP_HELPER_ANCHOR + "\n\n" + TOOLTIP_HELPER_JS,
                1,
            )

    if "applyMetadataToNodeTitles(nodes);" not in updated:
        updated, n = NODE_DATASET_HOOK_PATTERN.subn(
            r"\1                  applyMetadataToNodeTitles(nodes);\n\n                  \2",
            updated,
            count=1,
        )
        if n == 0:
            logging.warning("Could not inject node dataset metadata hook")

    return updated


def find_pair_html(viz_root: Path, pair_name: str) -> Path | None:
    pair_dir = viz_root / pair_name
    if not pair_dir.exists():
        return None

    hits = sorted(pair_dir.glob("*_network_interactive.html"))
    if not hits:
        return None
    return hits[0]


def run(tagged_root: Path, viz_root: Path, dry_run: bool = False, use_lfc: bool = True) -> int:
    # Prefer output_with_lfc if it exists and has LFC data, otherwise fall back to output
    if use_lfc and TAGGED_WITH_LFC_DIR.exists():
        search_root = TAGGED_WITH_LFC_DIR
        pattern = "*_tagged_with_lfc.csv"
        logging.info("Using LFC-enhanced data from %s", search_root)
    else:
        search_root = tagged_root
        pattern = "*_tagged.csv"
        logging.info("Using basic tagged data from %s", search_root)

    tagged_files = sorted(search_root.glob(pattern))
    if not tagged_files:
        logging.error("No tagged CSV files found in %s", search_root)
        return 1

    updated = 0
    skipped = 0

    for tagged_file in tagged_files:
        # Handle both _tagged.csv and _tagged_with_lfc.csv naming
        if "_tagged_with_lfc.csv" in tagged_file.name:
            pair_name = tagged_file.name.replace("_tagged_with_lfc.csv", "")
        else:
            pair_name = tagged_file.name.replace("_tagged.csv", "")
        html_file = find_pair_html(viz_root, pair_name)
        if html_file is None:
            logging.warning("[%s] no interactive HTML found under %s", pair_name, viz_root)
            skipped += 1
            continue

        tagged_df = pd.read_csv(tagged_file)
        metadata_map = build_gene_metadata_map(tagged_df)

        html_text = html_file.read_text(encoding="utf-8")
        new_text = inject_metadata_block(html_text, metadata_map)
        new_text = inject_tooltip_runtime(new_text)

        if dry_run:
            logging.info("[DRY RUN] [%s] would update %s with %d genes", pair_name, html_file, len(metadata_map))
            updated += 1
            continue

        html_file.write_text(new_text, encoding="utf-8")
        logging.info("[%s] updated %s with %d genes", pair_name, html_file, len(metadata_map))
        updated += 1

    logging.info("Done. updated=%d skipped=%d", updated, skipped)
    return 0


def main() -> int:
    args = parse_args()
    configure_logging()

    tagged_root = Path(args.tagged_dir)
    viz_root = Path(args.viz_dir)

    return run(tagged_root=tagged_root, viz_root=viz_root, dry_run=args.dry_run, use_lfc=args.use_lfc)


if __name__ == "__main__":
    raise SystemExit(main())
