#!/usr/bin/env python3
"""
Script 56: Clean direction filter fix

The direction data IS in the nodes already. Just need to:
1. Move filter panel to bottomleft (correct CSS positioning)
2. Hook filter to the actual network object properly
3. Add direction as a real node property for filtering
"""

from __future__ import annotations

import logging
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_NETWORKS_DIR = (
    REPO_ROOT
    / "results"
    / "stages"
    / "09_differential_restructured"
    / "12_tagging"
    / "final_networks_with_lfc"
)


# Clean, simple CSS for bottom-left panel
FILTER_CSS = """
<style>
#directionFilterPanel {
    position: fixed;
    bottom: 20px;
    left: 20px;
    background: white;
    border: 2px solid #333;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    z-index: 9999;
    font-family: Arial, sans-serif;
    font-size: 14px;
    min-width: 220px;
    user-select: none;
}
#directionFilterPanel.minimized {
    padding: 8px 15px;
    max-height: 40px;
}
#directionFilterPanel.minimized .filter-content {
    display: none;
}
#directionFilterPanel h4 {
    margin: 0 0 12px 0;
    padding: 0 0 8px 0;
    border-bottom: 1px solid #ccc;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.minimize-btn {
    background: none;
    border: none;
    cursor: pointer;
    color: #666;
    padding: 0;
}
.toggle-switch-container {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    gap: 10px;
}
.toggle-switch {
    position: relative;
    width: 50px;
    height: 24px;
}
.toggle-switch input {
    opacity: 0;
    width: 0;
    height: 0;
}
.toggle-slider {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #ccc;
    border-radius: 24px;
    transition: 0.3s;
}
.toggle-slider:before {
    position: absolute;
    content: "";
    height: 18px;
    width: 18px;
    left: 3px;
    bottom: 3px;
    background-color: white;
    border-radius: 50%;
    transition: 0.3s;
}
input:checked + .toggle-slider {
    background-color: #4CAF50;
}
input:checked + .toggle-slider:before {
    transform: translateX(26px);
}
.toggle-label {
    font-weight: 500;
    min-width: 70px;
}
.toggle-count {
    font-size: 12px;
    color: #666;
    margin-left: auto;
    min-width: 50px;
    text-align: right;
}
.filter-stats {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #ccc;
    font-size: 12px;
    color: #666;
}
.reset-btn {
    width: 100%;
    margin-top: 10px;
    padding: 6px;
    background-color: #007BFF;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-weight: bold;
}
.reset-btn:hover {
    background-color: #0056b3;
}
</style>
"""

# Simple, working filter JavaScript
FILTER_JS = """
<script>
(function() {
    var filterState = {UP: true, DOWN: true, STABLE: true};
    var countState = {UP: 0, DOWN: 0, STABLE: 0, total: 0};
    
    function getNetwork() {
        // Look for the network object in global scope
        if (window.network !== undefined) return window.network;
        if (window.net !== undefined) return window.net;
        return null;
    }
    
    function getNodesDataset() {
        var net = getNetwork();
        if (!net) return null;
        try {
            return net.body.data.nodes;
        } catch (e) {}
        return null;
    }
    
    function getEdgesDataset() {
        var net = getNetwork();
        if (!net) return null;
        try {
            return net.body.data.edges;
        } catch (e) {}
        return null;
    }
    
    function getAllNodes() {
        var ds = getNodesDataset();
        if (!ds || !ds.get) return [];
        try {
            return ds.get();
        } catch (e) {
            return [];
        }
    }
    
    function getAllEdges() {
        var ds = getEdgesDataset();
        if (!ds || !ds.get) return [];
        try {
            return ds.get();
        } catch (e) {
            return [];
        }
    }
    
    function updateCounts() {
        var nodes = getAllNodes();
        countState = {UP: 0, DOWN: 0, STABLE: 0, total: nodes.length};
        nodes.forEach(function(node) {
            var dir = node.direction || 'UNKNOWN';
            if (dir === 'UP' || dir === 'DOWN' || dir === 'STABLE') {
                countState[dir]++;
            }
        });
    }
    
    function updateDisplay() {
        updateCounts();
        var up = document.getElementById('count-UP');
        var dn = document.getElementById('count-DOWN');
        var st = document.getElementById('count-STABLE');
        
        var visible = 0;
        var nodes = getAllNodes();
        nodes.forEach(function(n) {
            var d = n.direction || 'UNKNOWN';
            if (filterState[d]) visible++;
        });
        
        if (up) up.textContent = (filterState.UP ? countState.UP : 0) + ' / ' + countState.UP;
        if (dn) dn.textContent = (filterState.DOWN ? countState.DOWN : 0) + ' / ' + countState.DOWN;
        if (st) st.textContent = (filterState.STABLE ? countState.STABLE : 0) + ' / ' + countState.STABLE;
        
        var tot = document.getElementById('stat-total');
        if (tot) tot.textContent = visible + ' / ' + countState.total;
    }
    
    function applyFilter() {
        var nodes = getAllNodes();
        var edges = getAllEdges();
        var nodeDs = getNodesDataset();
        var edgeDs = getEdgesDataset();
        
        if (!nodeDs || !edgeDs) return;
        
        var visNodeIds = {};
        nodes.forEach(function(n) {
            var dir = n.direction || 'UNKNOWN';
            visNodeIds[n.id] = filterState[dir] ? true : false;
        });
        
        // Update nodes
        nodeDs.update(nodes.map(function(n) {
            return {id: n.id, hidden: !visNodeIds[n.id]};
        }));
        
        // Update edges
        edgeDs.update(edges.map(function(e) {
            var fromVis = visNodeIds[e.from];
            var toVis = visNodeIds[e.to];
            return {id: e.id, hidden: !(fromVis && toVis)};
        }));
        
        var net = getNetwork();
        if (net && net.redraw) net.redraw();
        
        updateDisplay();
    }
    
    window.toggleDirection = function(dir) {
        filterState[dir] = !filterState[dir];
        applyFilter();
    };
    
    window.resetFilters = function() {
        filterState = {UP: true, DOWN: true, STABLE: true};
        document.getElementById('toggle-UP').checked = true;
        document.getElementById('toggle-DOWN').checked = true;
        document.getElementById('toggle-STABLE').checked = true;
        applyFilter();
    };
    
    window.minimizeFilterPanel = function() {
        var p = document.getElementById('directionFilterPanel');
        if (p) p.classList.toggle('minimized');
    };
    
    // Initialize when DOM is ready and network is drawn
    function initialize() {
        if (getNetwork() && getAllNodes().length > 0) {
            updateDisplay();
            console.log('[Filter] Initialized with ' + countState.total + ' nodes');
        } else {
            setTimeout(initialize, 200);
        }
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize);
    } else {
        setTimeout(initialize, 500);
    }
})();
</script>
"""

PANEL_HTML = """
<div id="directionFilterPanel">
    <h4>🔬 Direction Filter <button class="minimize-btn" onclick="window.minimizeFilterPanel()">−</button></h4>
    <div class="filter-content">
        <div class="toggle-switch-container">
            <label class="toggle-label">↑ UP</label>
            <label class="toggle-switch">
                <input type="checkbox" id="toggle-UP" checked onchange="window.toggleDirection('UP')">
                <span class="toggle-slider"></span>
            </label>
            <span class="toggle-count" id="count-UP">0 / 0</span>
        </div>
        <div class="toggle-switch-container">
            <label class="toggle-label">↓ DOWN</label>
            <label class="toggle-switch">
                <input type="checkbox" id="toggle-DOWN" checked onchange="window.toggleDirection('DOWN')">
                <span class="toggle-slider"></span>
            </label>
            <span class="toggle-count" id="count-DOWN">0 / 0</span>
        </div>
        <div class="toggle-switch-container">
            <label class="toggle-label">— STABLE</label>
            <label class="toggle-switch">
                <input type="checkbox" id="toggle-STABLE" checked onchange="window.toggleDirection('STABLE')">
                <span class="toggle-slider"></span>
            </label>
            <span class="toggle-count" id="count-STABLE">0 / 0</span>
        </div>
        <div class="filter-stats">
            <div class="stat-line"><span>Visible:</span><span id="stat-total">0 / 0</span></div>
        </div>
        <button class="reset-btn" onclick="window.resetFilters()">Reset All</button>
    </div>
</div>
"""


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def fix_html(html_file: Path) -> bool:
    """Clean up and inject proper filter."""
    logging.info(f"Processing: {html_file.name}")
    
    try:
        html_text = html_file.read_text(encoding="utf-8")
    except Exception as e:
        logging.error(f"Read failed: {e}")
        return False
    
    # Remove ALL old filter attempts
    html_text = re.sub(r'<style>\s*#directionFilterPanel.*?</style>', '', html_text, flags=re.DOTALL)
    html_text = re.sub(r'<div id="directionFilterPanel">.*?</div>', '', html_text, flags=re.DOTALL)
    html_text = re.sub(r'<script>.*?window\.toggleDirection.*?</script>', '', html_text, flags=re.DOTALL)
    
    # Add CSS to head
    if "</head>" in html_text:
        html_text = html_text.replace("</head>", FILTER_CSS + "\n</head>", 1)
    
    # Add filter panel right after <body>
    body_match = re.search(r'<body[^>]*>', html_text)
    if body_match:
        insert_pos = body_match.end()
        html_text = html_text[:insert_pos] + "\n" + PANEL_HTML + "\n" + html_text[insert_pos:]
    
    # Add JS before </body>
    if "</body>" in html_text:
        html_text = html_text.replace("</body>", FILTER_JS + "\n</body>", 1)
    else:
        html_text += "\n" + FILTER_JS
    
    try:
        html_file.write_text(html_text, encoding="utf-8")
        logging.info(f"✓ {html_file.name}")
        return True
    except Exception as e:
        logging.error(f"Write failed: {e}")
        return False


def main() -> None:
    configure_logging()
    logging.info("Script 56: Clean direction filter implementation")
    logging.info("=" * 80)
    
    html_files = sorted(FINAL_NETWORKS_DIR.glob("*/*_network_interactive.html"))
    if not html_files:
        logging.error("No HTMLs found")
        return
    
    success = sum(1 for f in html_files if fix_html(f))
    
    logging.info("=" * 80)
    logging.info(f"COMPLETE: {success}/{len(html_files)} networks fixed")


if __name__ == "__main__":
    main()
