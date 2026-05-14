/* network_fix.js — Clean responsive layout
   Strategy: shrink .app-shell to centre area; build fixed sidebars around it.
   The network canvas is NEVER moved and NEVER covered. */
(function () {

    var LEFT_W  = 'clamp(200px, 28%, 440px)';
    var RIGHT_W = 'clamp(180px, 23%, 280px)';
    var HDR_H   = '50px';

    /* ── 1. Inject CSS immediately ─────────────────────────── */
    var style = document.createElement('style');
    style.textContent = [

        'html,body { background:#fff !important; margin:0; overflow:hidden; height:100%; }',

        /* Shrink network canvas to the centre strip */
        '.app-shell {',
        '  position:fixed !important;',
        '  top:' + HDR_H + ' !important;',
        '  left:' + LEFT_W + ' !important;',
        '  right:' + RIGHT_W + ' !important;',
        '  bottom:0 !important;',
        '  width:auto !important;',
        '  height:auto !important;',
        '}',
        '.graph-wrap, #mynetwork { width:100% !important; height:100% !important; }',

        /* Fixed header bar */
        '#pg-header {',
        '  position:fixed; top:0; left:0; right:0; height:' + HDR_H + ';',
        '  display:flex; align-items:center; gap:10px; padding:0 14px;',
        '  background:#111827; border-bottom:1px solid rgba(229,231,235,0.2);',
        '  z-index:10005; box-sizing:border-box; overflow:hidden;',
        '}',

        /* Fixed left sidebar — physics */
        '#pg-left {',
        '  position:fixed; top:' + HDR_H + '; left:0; bottom:0;',
        '  width:' + LEFT_W + ';',
        '  overflow-y:auto; background:#fff;',
        '  border-right:1px solid #e5e7eb; z-index:10003; box-sizing:border-box;',
        '}',

        /* Fixed right sidebar — guide + controls + legend */
        '#pg-right {',
        '  position:fixed; top:' + HDR_H + '; right:0; bottom:0;',
        '  width:' + RIGHT_W + ';',
        '  overflow-y:auto; background:#111827;',
        '  border-left:1px solid rgba(229,231,235,0.25); z-index:10003; box-sizing:border-box;',
        '}',

        /* De-fix panels — let them flow inside their sidebar */
        '#config {',
        '  position:static !important; width:100% !important;',
        '  min-width:0 !important; max-width:100% !important;',
        '  max-height:none !important; border:none !important;',
        '  border-radius:0 !important; box-shadow:none !important;',
        '}',
        '#config .vis-configuration-wrapper { min-width:0 !important; }',

        /* Compact the physics panel contents */
        '#config { font-size:11px !important; }',
        '#config .vis-configuration { padding:4px 6px !important; }',
        '#config .vis-configuration-wrapper div { font-size:11px !important; line-height:1.4 !important; }',
        '#config input[type=range] { width:120px !important; height:4px !important; }',
        '#config input[type=text], #config select { font-size:11px !important; padding:2px 4px !important; }',
        '#config .vis-configuration-popup { width:180px !important; }',
        '#config label { font-size:11px !important; }',

        '.network-guide, .network-controls, .network-legend {',
        '  position:static !important; width:100% !important;',
        '  max-width:100% !important; max-height:none !important;',
        '  border:none !important; border-radius:0 !important;',
        '  box-shadow:none !important; backdrop-filter:none !important;',
        '  border-bottom:1px solid rgba(229,231,235,0.15) !important;',
        '}',

        /* Search bar flows in header */
        '.gene-search-bar {',
        '  position:static !important; transform:none !important;',
        '  flex:1; min-width:0; width:100% !important; max-width:none !important;',
        '  padding:0 !important; border:none !important; background:transparent !important;',
        '}',
        '.gene-search-bar input { background:rgba(255,255,255,0.12); color:#f9fafb; border-color:rgba(229,231,235,0.3); }',
        '.gene-search-btn { color:#f9fafb; }',

        /* Title flows in header */
        '.graph-title { position:static !important; white-space:nowrap; flex-shrink:0; }',

        /* Edge type filter stays as fixed overlay in centre */
        '#edgeTypeFilterPanel { z-index:10002 !important; }',

        /* Hide unwanted */
        '#directionFilterPanel { display:none !important; }',

        /* Responsive: collapse sidebars below 860px */
        '@media (max-width:860px) {',
        '  .app-shell { left:0 !important; right:0 !important; top:' + HDR_H + ' !important; bottom:0 !important; }',
        '  #pg-left, #pg-right { position:static; width:100%; height:auto; border:none; border-bottom:1px solid #e5e7eb; }',
        '  #pg-left { order:2; } #pg-right { order:3; }',
        '  html,body { overflow:auto !important; }',
        '}',

    ].join('\n');
    document.head.appendChild(style);

    /* ── 2. Build sidebars + header after load ─────────────── */
    window.addEventListener('load', function () {
        setTimeout(function () {

            /* Cosmetic */
            var bn = document.getElementById('bestNetworkBtn');
            if (bn) bn.remove();
            var sv = document.getElementById('stableViewBtn');
            if (sv) sv.textContent = 'Stable module view';

            /* Grab elements */
            var title  = document.querySelector('.graph-title');
            var search = document.querySelector('.gene-search-bar');
            var config = document.getElementById('config');
            var guide  = document.querySelector('.network-guide');
            var ctrl   = document.querySelector('.network-controls');
            var legend = document.getElementById('communityLegend');

            /* Header */
            var hdr = document.createElement('div');
            hdr.id = 'pg-header';
            if (title)  hdr.appendChild(title);
            if (search) hdr.appendChild(search);
            document.body.appendChild(hdr);

            /* Left sidebar */
            var left = document.createElement('div');
            left.id = 'pg-left';
            if (config) left.appendChild(config);
            document.body.appendChild(left);

            /* Right sidebar */
            var right = document.createElement('div');
            right.id = 'pg-right';
            if (guide)  right.appendChild(guide);
            if (ctrl)   right.appendChild(ctrl);
            if (legend) right.appendChild(legend);
            document.body.appendChild(right);

            /* Nudge network to redraw in its new size */
            setTimeout(function () {
                window.dispatchEvent(new Event('resize'));
                try { if (typeof network !== 'undefined') network.fit(); } catch (e) {}
            }, 300);

        }, 700);
    });

})();
