#!/usr/bin/env python3
"""Script 57: Clean rebuild of final interactive networks + stable direction filter.

Steps:
1. Restore final HTMLs from clean source HTMLs.
2. Add robust direction filter panel at bottom-left.
3. Wire filter to real vis datasets by exposing network/nodes/edges on window.

This script only touches files under final_networks_with_lfc.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE09 = REPO_ROOT / "results" / "stages" / "09_differential_restructured"
SRC_DIR = STAGE09 / "12_tagging" / "04_updated_persistent_networks"
DST_DIR = STAGE09 / "12_tagging" / "zzz_final_networks_with_lfc"

CSS_MARKER_START = "<!-- DirectionFilter CSS START -->"
CSS_MARKER_END = "<!-- DirectionFilter CSS END -->"
PANEL_MARKER_START = "<!-- DirectionFilter PANEL START -->"
PANEL_MARKER_END = "<!-- DirectionFilter PANEL END -->"
JS_MARKER_START = "<!-- DirectionFilter JS START -->"
JS_MARKER_END = "<!-- DirectionFilter JS END -->"

FILTER_CSS = f"""
{CSS_MARKER_START}
<style>
#directionFilterPanel {{
  position: fixed;
  left: 16px;
  bottom: 16px;
  z-index: 10002;
  width: 240px;
  background: rgba(255,255,255,0.97);
  border: 1px solid #cfd5df;
  border-radius: 10px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.22);
  color: #1f2937;
  font-family: "Segoe UI", Arial, sans-serif;
  font-size: 13px;
  padding: 10px;
}}
#directionFilterPanel h4 {{
  margin: 0 0 8px 0;
  font-size: 14px;
  font-weight: 700;
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
#directionFilterPanel.minimized .df-body {{
  display: none;
}}
#directionFilterPanel .df-mini {{
  border: none;
  background: transparent;
  cursor: pointer;
  font-size: 15px;
  line-height: 1;
  color: #334155;
}}
.df-row {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 8px 0;
}}
.df-label {{
  width: 70px;
  font-weight: 600;
}}
.df-switch {{
  position: relative;
  display: inline-block;
  width: 40px;
  height: 22px;
}}
.df-switch input {{
  opacity: 0;
  width: 0;
  height: 0;
}}
.df-slider {{
  position: absolute;
  inset: 0;
  background: #cbd5e1;
  border-radius: 999px;
  transition: 0.2s;
}}
.df-slider:before {{
  content: "";
  position: absolute;
  width: 16px;
  height: 16px;
  left: 3px;
  top: 3px;
  background: #fff;
  border-radius: 50%;
  transition: 0.2s;
}}
.df-switch input:checked + .df-slider {{
  background: #22c55e;
}}
.df-switch input:checked + .df-slider:before {{
  transform: translateX(18px);
}}
.df-count {{
  margin-left: auto;
  min-width: 54px;
  text-align: right;
  color: #475569;
  font-variant-numeric: tabular-nums;
}}
.df-total {{
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid #e2e8f0;
  display: flex;
  justify-content: space-between;
  color: #334155;
  font-size: 12px;
}}
.df-reset {{
  width: 100%;
  margin-top: 8px;
  border: 1px solid #2563eb;
  background: #2563eb;
  color: #fff;
  padding: 6px 8px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
}}
.df-reset:hover {{
  background: #1d4ed8;
}}
</style>
{CSS_MARKER_END}
""".strip()

FILTER_PANEL = f"""
{PANEL_MARKER_START}
<div id="directionFilterPanel">
  <h4>
    <span>Direction Filter</span>
    <button class="df-mini" type="button" onclick="window.dfTogglePanel()">-</button>
  </h4>
  <div class="df-body">
    <div class="df-row">
      <span class="df-label">UP</span>
      <label class="df-switch">
        <input id="df-up" type="checkbox" checked onchange="window.dfToggle('UP')" />
        <span class="df-slider"></span>
      </label>
      <span class="df-count" id="df-count-up">0 / 0</span>
    </div>
    <div class="df-row">
      <span class="df-label">DOWN</span>
      <label class="df-switch">
        <input id="df-down" type="checkbox" checked onchange="window.dfToggle('DOWN')" />
        <span class="df-slider"></span>
      </label>
      <span class="df-count" id="df-count-down">0 / 0</span>
    </div>
    <div class="df-row">
      <span class="df-label">STABLE</span>
      <label class="df-switch">
        <input id="df-stable" type="checkbox" checked onchange="window.dfToggle('STABLE')" />
        <span class="df-slider"></span>
      </label>
      <span class="df-count" id="df-count-stable">0 / 0</span>
    </div>
    <div class="df-total">
      <span>Visible</span>
      <span id="df-count-total">0 / 0</span>
    </div>
    <button class="df-reset" type="button" onclick="window.dfReset()">Reset All</button>
  </div>
</div>
{PANEL_MARKER_END}
""".strip()

FILTER_JS = f"""
{JS_MARKER_START}
<script>
(function() {{
  var state = {{ UP: true, DOWN: true, STABLE: true }};

  function getRefs() {{
    return {{
      network: window.__dirFilterNetwork || null,
      nodes: window.__dirFilterNodes || null,
      edges: window.__dirFilterEdges || null
    }};
  }}

  function parseDirectionFromTitle(title) {{
    if (!title) return 'UNKNOWN';
    var m = String(title).match(/(?:^|\\n)direction=([A-Z_]+)/);
    return m ? m[1] : 'UNKNOWN';
  }}

  function nodeDirection(node) {{
    if (node && node.direction) return String(node.direction).toUpperCase();
    return parseDirectionFromTitle(node ? node.title : '');
  }}

  function listNodes() {{
    var refs = getRefs();
    if (!refs.nodes || typeof refs.nodes.get !== 'function') return [];
    try {{
      return refs.nodes.get();
    }} catch (e) {{
      return [];
    }}
  }}

  function listEdges() {{
    var refs = getRefs();
    if (!refs.edges || typeof refs.edges.get !== 'function') return [];
    try {{
      return refs.edges.get();
    }} catch (e) {{
      return [];
    }}
  }}

  function setText(id, text) {{
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }}

  function refreshCounts() {{
    var nodes = listNodes();
    var totals = {{ UP: 0, DOWN: 0, STABLE: 0 }};
    var visible = {{ UP: 0, DOWN: 0, STABLE: 0 }};

    nodes.forEach(function(n) {{
      var d = nodeDirection(n);
      if (d !== 'UP' && d !== 'DOWN' && d !== 'STABLE') return;
      totals[d] += 1;
      if (state[d]) visible[d] += 1;
    }});

    setText('df-count-up', visible.UP + ' / ' + totals.UP);
    setText('df-count-down', visible.DOWN + ' / ' + totals.DOWN);
    setText('df-count-stable', visible.STABLE + ' / ' + totals.STABLE);
    setText('df-count-total', (visible.UP + visible.DOWN + visible.STABLE) + ' / ' + (totals.UP + totals.DOWN + totals.STABLE));
  }}

  function applyFilter() {{
    var refs = getRefs();
    if (!refs.nodes || !refs.edges) return;

    var nodes = listNodes();
    var edges = listEdges();
    if (!nodes.length) return;

    var vis = {{}};
    var nodeUpdates = [];
    nodes.forEach(function(n) {{
      var d = nodeDirection(n);
      var show = (d === 'UP' || d === 'DOWN' || d === 'STABLE') ? !!state[d] : true;
      vis[n.id] = show;
      nodeUpdates.push({{ id: n.id, hidden: !show }});
    }});
    refs.nodes.update(nodeUpdates);

    var edgeUpdates = [];
    edges.forEach(function(e) {{
      var show = !!vis[e.from] && !!vis[e.to];
      edgeUpdates.push({{ id: e.id, hidden: !show }});
    }});
    refs.edges.update(edgeUpdates);

    if (refs.network && typeof refs.network.redraw === 'function') refs.network.redraw();
    refreshCounts();
  }}

  window.dfToggle = function(kind) {{
    state[kind] = !state[kind];
    applyFilter();
  }};

  window.dfReset = function() {{
    state = {{ UP: true, DOWN: true, STABLE: true }};
    var up = document.getElementById('df-up');
    var down = document.getElementById('df-down');
    var stable = document.getElementById('df-stable');
    if (up) up.checked = true;
    if (down) down.checked = true;
    if (stable) stable.checked = true;
    applyFilter();
  }};

  window.dfTogglePanel = function() {{
    var p = document.getElementById('directionFilterPanel');
    if (p) p.classList.toggle('minimized');
  }};

  function bootstrap(attempt) {{
    var refs = getRefs();
    if (refs.nodes && refs.edges && listNodes().length) {{
      refreshCounts();
      return;
    }}
    if (attempt < 60) setTimeout(function() {{ bootstrap(attempt + 1); }}, 200);
  }}

  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', function() {{ bootstrap(0); }});
  }} else {{
    bootstrap(0);
  }}
}})();
</script>
{JS_MARKER_END}
""".strip()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def clean_old_filter_artifacts(text: str) -> str:
    # Marker-based cleanups first
    for start, end in [
        (CSS_MARKER_START, CSS_MARKER_END),
        (PANEL_MARKER_START, PANEL_MARKER_END),
        (JS_MARKER_START, JS_MARKER_END),
    ]:
        if start in text and end in text:
            pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), flags=re.DOTALL)
            text = pattern.sub("", text)

    # Legacy cleanups from prior attempts
    text = re.sub(r"<style>\s*#directionFilterPanel.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<div id=\"directionFilterPanel\">.*?</div>", "", text, flags=re.DOTALL)
    text = re.sub(r"<script>\s*\(function\(\)\s*\{.*?window\.toggleDirection.*?</script>", "", text, flags=re.DOTALL)

    # Remove orphaned toggle fragments previously inserted near search bar
    text = re.sub(r"\s*<div class=\"toggle-switch-container\">\s*<label class=\"toggle-label\">(?:↑\s*)?UP.*?</div>", "", text, flags=re.DOTALL)
    text = re.sub(r"\s*<div class=\"toggle-switch-container\">\s*<label class=\"toggle-label\">(?:↓\s*)?DOWN.*?</div>", "", text, flags=re.DOTALL)
    text = re.sub(r"\s*<div class=\"toggle-switch-container\">\s*<label class=\"toggle-label\">(?:—\s*)?STABLE.*?</div>", "", text, flags=re.DOTALL)
    text = re.sub(r"\s*<div class=\"filter-stats\">.*?</div>", "", text, flags=re.DOTALL)
    text = re.sub(r"\s*<button class=\"reset-btn\"[^>]*>.*?</button>", "", text, flags=re.DOTALL)

    return text


def expose_network_refs(text: str) -> str:
    needle = "return network;"
    replacement = (
        "window.__dirFilterNetwork = network;\n"
        "                  window.__dirFilterNodes = nodes;\n"
        "                  window.__dirFilterEdges = edges;\n\n"
        "                  return network;"
    )
    if "window.__dirFilterNetwork = network;" in text:
        return text
    return text.replace(needle, replacement, 1)


def inject_blocks(text: str) -> str:
    if "</head>" in text and CSS_MARKER_START not in text:
        text = text.replace("</head>", FILTER_CSS + "\n\n</head>", 1)

    body_match = re.search(r"<body[^>]*>", text)
    if body_match and PANEL_MARKER_START not in text:
        pos = body_match.end()
        text = text[:pos] + "\n\n" + FILTER_PANEL + "\n\n" + text[pos:]

    if "</body>" in text and JS_MARKER_START not in text:
        text = text.replace("</body>", "\n\n" + FILTER_JS + "\n\n</body>", 1)
    return text


def restore_clean_htmls() -> int:
    count = 0
    for src_html in sorted(SRC_DIR.glob("*/*_network_interactive.html")):
        pair = src_html.parent.name
        dst_pair = DST_DIR / pair
        dst_pair.mkdir(parents=True, exist_ok=True)
        dst_html = dst_pair / src_html.name
        shutil.copy2(src_html, dst_html)
        count += 1
    return count


def patch_final_htmls() -> tuple[int, int]:
    total = 0
    ok = 0
    for html in sorted(DST_DIR.glob("*/*_network_interactive.html")):
        total += 1
        raw = html.read_text(encoding="utf-8")
        text = clean_old_filter_artifacts(raw)
        text = expose_network_refs(text)
        text = inject_blocks(text)
        html.write_text(text, encoding="utf-8")
        ok += 1
    return ok, total


def main() -> None:
    configure_logging()
    logging.info("Script 57: clean rebuild + stable bottom-left direction filter")

    restored = restore_clean_htmls()
    logging.info("Restored clean HTML files: %s", restored)

    ok, total = patch_final_htmls()
    logging.info("Patched filter UI + JS in final HTML files: %s/%s", ok, total)


if __name__ == "__main__":
    main()
