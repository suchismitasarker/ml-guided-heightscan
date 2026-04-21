#!/usr/bin/env python3
"""
HeightScan Line Plot App  —  CHESS QM2 / ID4B
==============================================
Standalone Flask app for analysing CBF detector-image series.

Tabs:
  Tab 1 · ROI Line Plot      — integrate user-defined pixel ROI over z range
  Tab 2 · Pixel Inspector    — global max pixel per image, position vs z
  Tab 3 · Max-Pixel ROI Plot — auto-detect peak location; plot ROI around it

New features v2:
  • Interactive Plotly heatmap for detector preview (AJAX, on-demand)
  • Image navigator  — step through every z index with Prev/Next + slider
  • Max-Pixel ROI tab — auto-locate the hot-spot, integrate a box around it

Usage:
    pip install flask fabio numpy matplotlib pillow
    python heightscan_app.py          # http://localhost:5050
"""

from flask import Flask, render_template_string, request, jsonify, Response
from urllib.parse import quote as urlquote
import os, io, re, glob, json, base64, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import fabio
    FABIO_OK = True
except ImportError:
    FABIO_OK = False

try:
    from PIL import Image as PILImage
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as _CK
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

app = Flask(__name__)

PORT         = 5050
CBF_ROOT     = "/nfs/chess/id4b/2026-1"
DEFAULT_PATH = "/nfs/chess/id4b/2026-1/de-la-to-4792-a/tiffs/4Hab_TaS2_001"

# ── Pilatus 6M detector geometry ─────────────────────────────────────────────
# Sensor: 5 × 12 modules, each 487 × 195 px, separated by 7 px (col) / 17 px (row) gaps.
# Total active area: 2463 × 2527 px.
# We add a ±5 px buffer on every gap edge to avoid picking up diffraction
# that straddles a gap boundary.
_GAP_BUF = 5   # extra buffer pixels on each side of every gap

# Column gaps in x (4 gaps between 5 module columns)
PILATUS_6M_COL_GAPS = [
    (487  - _GAP_BUF,  493  + _GAP_BUF),
    (981  - _GAP_BUF,  987  + _GAP_BUF),
    (1475 - _GAP_BUF,  1481 + _GAP_BUF),
    (1969 - _GAP_BUF,  1975 + _GAP_BUF),
]

# Row gaps in y (11 gaps between 12 module rows)
PILATUS_6M_ROW_GAPS = [
    (195  - _GAP_BUF,  211  + _GAP_BUF),
    (407  - _GAP_BUF,  423  + _GAP_BUF),
    (619  - _GAP_BUF,  635  + _GAP_BUF),
    (831  - _GAP_BUF,  847  + _GAP_BUF),
    (1043 - _GAP_BUF,  1059 + _GAP_BUF),
    (1255 - _GAP_BUF,  1271 + _GAP_BUF),
    (1467 - _GAP_BUF,  1483 + _GAP_BUF),
    (1679 - _GAP_BUF,  1695 + _GAP_BUF),
    (1891 - _GAP_BUF,  1907 + _GAP_BUF),
    (2103 - _GAP_BUF,  2119 + _GAP_BUF),
    (2315 - _GAP_BUF,  2331 + _GAP_BUF),
]

# Combined dead columns: detector edges + all Pilatus 6M column gaps
# Format: list of (x_start, x_end) inclusive column ranges.
DETECTOR_DEAD_COLS = (
    [(0, 100)]                  # left edge artefact
    + PILATUS_6M_COL_GAPS       # internal module column gaps
    + [(2400, 2600)]            # right edge artefact
)

# Row dead zones (y bands) — Pilatus 6M horizontal module gaps
DETECTOR_DEAD_ROWS = PILATUS_6M_ROW_GAPS

CHESS_LOGO_B64 = None
try:
    with open("/nfs/chess/id4baux/chesslogo.png", "rb") as _f:
        CHESS_LOGO_B64 = base64.b64encode(_f.read()).decode()
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════════════════
#  Shared CSS / HTML shell
# ══════════════════════════════════════════════════════════════════════════════
SHELL = """<!DOCTYPE html><html><head>
<title>HeightScan — CBF Series · CHESS QM2</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root{--bg:#f5f7fb;--bg-card:#fff;--bg-header:#0d1520;--bg-sidebar:#111827;
        --text-main:#222;--text-muted:#555;--text-header:#e8f0f8;
        --border-subtle:#d0d4e0;--accent:#1b4f72;--accent-hover:#163f5a;--accent-glow:#00c8ff;}
  body.dark{--bg:#050816;--bg-card:#0d1520;--bg-header:#020617;--bg-sidebar:#020617;
            --text-main:#e5e7eb;--text-muted:#9ca3af;--text-header:#e5e7eb;
            --border-subtle:#1e3a5f;--accent:#00c8ff;--accent-hover:#0082b8;}
  *{box-sizing:border-box;}
  body{margin:0;font-family:'IBM Plex Sans',-apple-system,BlinkMacSystemFont,sans-serif;
       background:var(--bg);color:var(--text-main);}
  .app-shell{display:flex;min-height:100vh;}
  /* ── sidebar ── */
  .sidebar{width:230px;background:var(--bg-sidebar);color:var(--text-header);
           display:flex;flex-direction:column;flex-shrink:0;}
  .sb-brand{padding:16px;display:flex;align-items:center;gap:10px;
            border-bottom:1px solid rgba(255,255,255,.06);}
  .logo-circle{width:34px;height:34px;border-radius:50%;
    background:conic-gradient(from 0deg,#00c8ff 0%,#0047ab 45%,#00c8ff 100%);
    display:flex;align-items:center;justify-content:center;font-size:13px;
    font-weight:700;color:#fff;box-shadow:0 0 14px rgba(0,200,255,.3);flex-shrink:0;}
  .logo-text{font-size:13px;font-weight:600;line-height:1.3;}
  .logo-sub{font-size:10px;opacity:.55;letter-spacing:.04em;}
  .nav-sec{font-size:10px;text-transform:uppercase;letter-spacing:.1em;
           color:rgba(255,255,255,.32);padding:13px 16px 4px;}
  .nav-link{display:flex;align-items:center;gap:8px;padding:8px 16px;
            color:rgba(255,255,255,.62);text-decoration:none;font-size:13px;
            border-left:2px solid transparent;transition:background .15s,color .15s;}
  .nav-link:hover{background:rgba(255,255,255,.06);color:#fff;}
  .nav-link.active{background:rgba(0,200,255,.1);color:#00c8ff;border-left-color:#00c8ff;}
  .nav-icon{font-size:14px;width:18px;text-align:center;}
  .sb-foot{margin-top:auto;padding:12px 16px;font-size:11px;
           color:rgba(255,255,255,.28);border-top:1px solid rgba(255,255,255,.06);}
  /* ── main column ── */
  .main-col{flex:1;display:flex;flex-direction:column;min-width:0;}
  header{background:var(--bg-header);color:var(--text-header);
         padding:12px 26px;display:flex;align-items:center;
         justify-content:space-between;box-shadow:0 2px 12px rgba(0,0,0,.4);
         position:sticky;top:0;z-index:50;}
  header h1{margin:0;font-size:16px;font-weight:500;}
  header p{margin:2px 0 0;font-size:11px;opacity:.55;}
  .hbadge{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#00c8ff;
          background:rgba(0,200,255,.08);border:1px solid rgba(0,200,255,.2);
          border-radius:20px;padding:3px 10px;}
  .theme-btn{border-radius:999px;border:1px solid rgba(148,163,184,.4);
             padding:4px 12px;background:transparent;color:var(--text-header);
             font-size:11px;cursor:pointer;}
  main{padding:22px 26px 36px;flex:1;overflow-y:auto;}
  footer{font-size:11px;padding:10px 26px;color:var(--text-muted);
         border-top:1px solid var(--border-subtle);}
  /* ── cards ── */
  .card{background:var(--bg-card);border-radius:8px;padding:18px 22px;
        margin-bottom:20px;box-shadow:0 1px 4px rgba(15,23,42,.2);
        border:1px solid var(--border-subtle);}
  .card-hdr{display:flex;align-items:center;gap:8px;padding-bottom:11px;
            margin-bottom:14px;border-bottom:1px solid var(--border-subtle);}
  .card-dot{width:6px;height:6px;border-radius:50%;background:var(--accent-glow);
            box-shadow:0 0 8px var(--accent-glow);}
  .card-title{font-size:11px;font-weight:600;letter-spacing:.08em;
              text-transform:uppercase;color:var(--text-muted);}
  .sec-title{margin:0 0 6px;font-size:17px;font-weight:600;}
  .subhead{font-size:13px;color:var(--text-muted);margin:0 0 10px;}
  /* ── form elements ── */
  label.fl{font-size:12px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
           color:var(--text-muted);margin-bottom:4px;display:block;}
  .fg{margin-bottom:14px;}
  input[type=number],input[type=text],select{
    padding:6px 10px;border-radius:5px;border:1px solid var(--border-subtle);
    font-size:13px;font-family:'IBM Plex Sans',sans-serif;
    background:var(--bg-card);color:var(--text-main);outline:none;
    transition:border-color .15s,box-shadow .15s;}
  input[type=number]:focus,input[type=text]:focus,select:focus{
    border-color:var(--accent);box-shadow:0 0 0 2px rgba(0,200,255,.12);}
  input[type=number]{width:110px;}
  input[type=text]{width:100%;max-width:520px;}
  input[type=checkbox],input[type=radio]{accent-color:var(--accent);}
  .row{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-end;}
  .row .fg{margin-bottom:0;}
  button,.btn{background:var(--accent);color:#f9fafb;border:none;border-radius:5px;
              padding:7px 16px;cursor:pointer;font-size:13px;
              font-family:'IBM Plex Sans',sans-serif;font-weight:500;
              transition:background .15s;}
  button:hover{background:var(--accent-hover);}
  .btn-primary{background:linear-gradient(135deg,#0082b8 0%,#00c8ff 100%);
               font-weight:600;letter-spacing:.04em;
               box-shadow:0 2px 14px rgba(0,200,255,.2);
               text-transform:uppercase;font-size:12px;padding:9px 24px;}
  .btn-primary:hover{opacity:.88;}
  .btn-sm{padding:4px 12px;font-size:12px;}
  .btn-ghost{background:transparent;color:var(--text-muted);border:1px solid var(--border-subtle);}
  .btn-ghost:hover{background:rgba(0,200,255,.08);color:#00c8ff;border-color:rgba(0,200,255,.3);}
  /* ── misc ── */
  .path-label{font-family:'IBM Plex Mono',monospace;background:rgba(148,163,184,.1);
              padding:2px 6px;border-radius:4px;font-size:12px;}
  .badge{display:inline-block;font-size:11px;padding:2px 8px;border-radius:999px;font-weight:600;margin-left:5px;}
  .badge-blue{background:rgba(56,189,248,.15);color:#38bdf8;}
  .badge-green{background:rgba(74,222,128,.15);color:#4ade80;}
  .badge-red{background:rgba(249,115,115,.15);color:#f97373;}
  .alert-info{color:#1e40af;background:#eff6ff;border:1px solid #93c5fd;
              border-radius:5px;padding:9px 14px;font-size:12px;margin-bottom:14px;}
  .alert-warn{color:#92400e;background:rgba(240,180,41,.08);
              border:1px solid rgba(240,180,41,.25);border-radius:5px;
              padding:8px 12px;font-size:12px;margin-bottom:12px;}
  .alert-error{color:#dc2626;background:rgba(255,79,79,.06);
               border:1px solid rgba(255,79,79,.2);border-radius:5px;
               padding:10px 12px;font-size:13px;}
  .meta-chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px;}
  .meta-chip{font-family:'IBM Plex Mono',monospace;font-size:11px;padding:3px 10px;
             border-radius:20px;background:rgba(0,200,255,.06);
             border:1px solid rgba(0,200,255,.15);color:var(--text-muted);}
  .meta-chip b{color:var(--accent-glow);}
  .stat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));
             gap:8px;margin:14px 0 6px;}
  .stat-box{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);
            border-radius:8px;padding:8px 10px;text-align:center;}
  .stat-lbl{font-size:10px;color:var(--text-muted);text-transform:uppercase;
            letter-spacing:.05em;margin-bottom:4px;}
  .stat-val{font-family:'IBM Plex Mono',monospace;font-size:13px;font-weight:600;
            color:var(--text-main);}
  .stat-val.hi{color:#4ade80;}  /* green — good SNR / contrast */
  .stat-val.lo{color:#f87171;}  /* red   — low SNR / contrast  */
  .two-col{display:grid;grid-template-columns:1fr 1fr;gap:20px;}
  img.det-img{width:100%;max-width:600px;border-radius:5px;
              border:1px solid var(--border-subtle);display:block;}
  table{border-collapse:collapse;width:100%;font-size:13px;}
  th,td{border:1px solid var(--border-subtle);padding:6px 10px;}
  th{background:rgba(148,163,184,.15);font-weight:600;font-size:11px;
     text-transform:uppercase;letter-spacing:.05em;}
  pre{background:#040810;border:1px solid var(--border-subtle);border-radius:5px;
      padding:10px 14px;font-family:'IBM Plex Mono',monospace;font-size:11px;
      color:#9ca3af;max-height:200px;overflow-y:auto;line-height:1.6;}
  .dl-btn{display:inline-flex;align-items:center;gap:4px;margin-top:8px;
          font-size:11px;font-family:'IBM Plex Mono',monospace;color:var(--text-muted);
          text-decoration:none;padding:3px 9px;border:1px solid var(--border-subtle);
          border-radius:4px;background:rgba(0,0,0,.04);transition:background .15s;}
  .dl-btn:hover{background:rgba(0,200,255,.08);color:#00c8ff;
                border-color:rgba(0,200,255,.3);text-decoration:none;}
  ::-webkit-scrollbar{width:6px;height:6px;}
  ::-webkit-scrollbar-thumb{background:var(--border-subtle);border-radius:3px;}
  #spinner{display:none;font-size:13px;color:var(--accent);margin-top:10px;}
  .prog-bar-wrap{width:100%;background:var(--border-subtle);border-radius:3px;
                 overflow:hidden;margin-top:8px;display:none;height:3px;}
  .prog-bar{height:100%;background:linear-gradient(90deg,var(--accent-hover),var(--accent-glow));
            animation:prog 1.8s ease-in-out infinite;}
  @keyframes prog{0%{margin-left:-40%;width:30%}50%{margin-left:30%;width:50%}100%{margin-left:100%;width:30%}}
  input[type=range]{-webkit-appearance:none;height:4px;border-radius:2px;
                    background:var(--border-subtle);outline:none;width:100%;margin:6px 0;}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;
    border-radius:50%;background:var(--accent);cursor:pointer;}
  /* ── Tab bar ── */
  .tab-bar{display:flex;gap:4px;margin-bottom:16px;border-bottom:2px solid var(--border-subtle);padding-bottom:0;}
  .tab-btn{background:transparent;color:var(--text-muted);border:none;border-bottom:2px solid transparent;
           padding:9px 18px;font-size:13px;font-family:'IBM Plex Sans',sans-serif;font-weight:500;
           cursor:pointer;margin-bottom:-2px;border-radius:5px 5px 0 0;transition:background .15s,color .15s;}
  .tab-btn:hover{background:rgba(0,200,255,.07);color:var(--accent-glow);}
  .tab-btn.active{color:var(--accent-glow);border-bottom-color:var(--accent-glow);
                  background:rgba(0,200,255,.06);}
  .tab-panel{display:none;}
  .tab-panel.active{display:block;}
  /* ── Image navigator ── */
  .nav-bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;
           padding:10px 14px;background:rgba(0,200,255,.04);
           border:1px solid rgba(0,200,255,.12);border-radius:6px;margin-bottom:14px;}
  .nav-bar input[type=range]{flex:1;min-width:120px;}
  .nav-bar input[type=number]{width:80px;}
  .nav-stat{font-family:'IBM Plex Mono',monospace;font-size:11px;
            color:var(--text-muted);white-space:nowrap;}
  .nav-stat b{color:var(--accent-glow);}
</style>
<script>
  function applyTheme(){if(localStorage.getItem('qm2_theme')==='dark')document.body.classList.add('dark');}
  function toggleTheme(){document.body.classList.toggle('dark');localStorage.setItem('qm2_theme',document.body.classList.contains('dark')?'dark':'light');}
  document.addEventListener('DOMContentLoaded', applyTheme);
</script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
</head><body>
<div class="app-shell">
  <aside class="sidebar">
    <div class="sb-brand">
      {LOGO}
      <div><div class="logo-text">QM2 HeightScan</div><div class="logo-sub">CHESS · Cornell</div></div>
    </div>
    <div class="nav-sec">HeightScan</div>
    <a class="nav-link active" href="/"><span class="nav-icon">📈</span> CBF Line Plot</a>
    <div class="sb-foot">HeightScan v2.0 · QM2 · CHESS</div>
  </aside>
  <div class="main-col">
    <header>
      <div>
        <h1>AI-Guided Data Collection @Quantum Materials Beamline</h1>
        <p>ROI integration · pixel inspector · max-pixel ROI tracking</p>
      </div>
      <div style="display:flex;align-items:center;gap:10px;">
        <span class="hbadge">ID4B · Pilatus</span>
        <button class="theme-btn" onclick="toggleTheme()">☀ / ☾</button>
      </div>
    </header>
    <main>{CONTENT}</main>
    <footer>Quantum Materials Beamline (QM2) · Cornell High Energy Synchrotron Source (CHESS)</footer>
  </div>
</div>
</body></html>
"""

def shell(content):
    if CHESS_LOGO_B64:
        logo = f'<img src="data:image/png;base64,{CHESS_LOGO_B64}" style="width:34px;height:34px;border-radius:50%;object-fit:contain;background:#fff;padding:2px;">'
    else:
        logo = '<div class="logo-circle">QM</div>'
    return SHELL.replace("{LOGO}", logo).replace("{CONTENT}", content)


# ══════════════════════════════════════════════════════════════════════════════
#  Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def scan_folder(folder):
    """Return sorted (indices, prefix, file_map) for prefix_NNN.cbf pattern."""
    pat = re.compile(r'^(.+?)_(\d+)\.cbf$', re.IGNORECASE)
    entries = []
    for fp in sorted(glob.glob(os.path.join(folder, "*.cbf"))):
        m = pat.match(os.path.basename(fp))
        if m:
            entries.append((int(m.group(2)), fp, m.group(1)))
    entries.sort(key=lambda e: e[0])
    if not entries:
        return [], "", {}
    prefix   = entries[0][2]
    file_map = {e[0]: e[1] for e in entries}
    return [e[0] for e in entries], prefix, file_map


def load_cbf(fp):
    """Load CBF → float32 ndarray  shape=(rows/Y, cols/X)."""
    return fabio.open(fp).data.astype(np.float32)


def roi_value(img, x0, x1, y0, y1, mode):
    roi = img[y0:y1 + 1, x0:x1 + 1]
    if mode == "mean":   return float(roi.mean())
    if mode == "max":    return float(roi.max())
    if mode == "median": return float(np.median(roi))
    return float(roi.sum())


def img_to_b64(img_arr, roi=None, log_scale=True, cmap="inferno", dpi=120,
               vmin=None, vmax=None):
    """Render a 2-D detector image (with optional ROI box) → base64 PNG.
    vmin/vmax are in display-space (i.e. after log10 transform if log_scale=True)."""
    h, w = img_arr.shape
    fig_w = min(7, max(4, w / 300))
    fig_h = fig_w * h / w
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    disp = np.log10(np.maximum(img_arr, 1)) if log_scale else img_arr
    im = ax.imshow(disp, cmap=cmap, aspect="auto", origin="lower",
                   interpolation="nearest", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log₁₀(counts)" if log_scale else "counts", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    if roi:
        x0r, x1r, y0r, y1r = roi
        rect = mpatches.FancyBboxPatch(
            (x0r, y0r), x1r - x0r, y1r - y0r,
            boxstyle="square,pad=0",
            linewidth=2, edgecolor="#FF4444", facecolor="#FF444418",
            linestyle="--", label="ROI"
        )
        ax.add_patch(rect)
        ax.legend(loc="upper right", fontsize=8,
                  facecolor="#1a2035", edgecolor="#444", labelcolor="white")
    ax.set_xlabel("x  (col / pixel)", fontsize=8)
    ax.set_ylabel("y  (row / pixel)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def img_to_plotly_json(img_arr, roi=None, log_scale=True, max_dim=480,
                       vmin=None, vmax=None):
    """
    Downsample a detector image and return a dict suitable for Plotly heatmap.
    vmin/vmax are in display-space (after log10 if log_scale=True).

    Coordinate convention (matches matplotlib origin='lower'):
      • zdata[j][i]  →  original pixel row j*step, col i*step
      • JS builds yscale = [0, step, 2*step, …]  (y increases upward)
      • ROI shape stored in original pixel coordinates — no inversion needed.
    """
    h, w = img_arr.shape
    step = max(1, max(h, w) // max_dim)
    ds   = img_arr[::step, ::step]
    if log_scale:
        ds = np.log10(np.maximum(ds, 1))
    # Do NOT flip rows — JS builds an increasing yscale so Plotly puts row 0
    # at y=0 (bottom of chart), matching matplotlib origin='lower'.
    zdata = ds.tolist()
    roi_shape = None
    if roi:
        x0, x1, y0, y1 = roi
        # Store in original pixel space; JS uses them as-is against the yscale/xscale
        roi_shape = {"x0": int(x0), "x1": int(x1), "y0": int(y0), "y1": int(y1)}
    return {"zdata": zdata, "step": step, "h_orig": h, "w_orig": w,
            "log": log_scale, "roi_shape": roi_shape,
            "zmin": vmin, "zmax": vmax}


def find_top_n_peaks(img_arr, n=3, nms_pad=100, dead_cols=None, dead_rows=None):
    """
    Return the top-N spatially distinct peak pixel locations using
    non-maximum suppression (NMS).

    nms_pad   — half-side of the exclusion box suppressed after each peak is
                found.  100 → 200×200 px dead zone; peaks 2/3 are always
                outside this region of the previous peak.
    dead_cols — list of (x_start, x_end) inclusive column ranges that are
                permanently zeroed before any search begins (detector edges,
                module gaps, artefact columns, …).
    dead_rows — list of (y_start, y_end) inclusive row ranges (horizontal
                module gaps, e.g. Pilatus 6M row gaps).

    Returns list of dicts (always length n):
        [{'x': int, 'y': int, 'value': int}, ...]
    """
    work = img_arr.astype(np.float64).copy()
    h, w = work.shape

    # Zero out permanently excluded column bands (x)
    if dead_cols:
        for x0, x1 in dead_cols:
            c0 = max(0, x0);  c1 = min(w, x1 + 1)
            work[:, c0:c1] = 0
    # Zero out permanently excluded row bands (y) — e.g. Pilatus 6M horizontal gaps
    if dead_rows:
        for y0, y1 in dead_rows:
            r0 = max(0, y0);  r1 = min(h, y1 + 1)
            work[r0:r1, :] = 0

    peaks = []
    for _ in range(n):
        if work.max() <= 0:
            break
        row, col = np.unravel_index(work.argmax(), work.shape)
        peaks.append({'x': int(col), 'y': int(row), 'value': int(img_arr[row, col])})
        # Suppress (2*nms_pad+1)² box — next peak must be outside this zone
        r0 = max(0, row - nms_pad);  r1 = min(h, row + nms_pad + 1)
        c0 = max(0, col - nms_pad);  c1 = min(w, col + nms_pad + 1)
        work[r0:r1, c0:c1] = 0

    # Always return exactly n entries so callers can index [0], [1], [2]
    while len(peaks) < n:
        peaks.append({'x': 0, 'y': 0, 'value': 0})
    return peaks


def _compute_fwhm(z, I, half_max):
    """
    Numerically estimate FWHM by finding where I crosses half_max.
    Uses linear interpolation between adjacent samples.
    Returns float Δz, or None if crossings cannot be found.
    """
    left_z = right_z = None
    for i in range(len(I) - 1):
        if I[i] < half_max <= I[i + 1]:          # rising crossing
            frac = (half_max - I[i]) / (I[i + 1] - I[i])
            left_z = z[i] + frac * (z[i + 1] - z[i])
            break
    for i in range(len(I) - 2, -1, -1):
        if I[i] >= half_max > I[i + 1]:          # falling crossing
            frac = (I[i] - half_max) / (I[i] - I[i + 1])
            right_z = z[i] + frac * (z[i + 1] - z[i])
            break
    if left_z is not None and right_z is not None and right_z > left_z:
        return right_z - left_z
    return None


def compute_peak_stats(z_arr, i_arr):
    """
    Return a dict of key statistical measures for a z/intensity series.
    Keys: mean, std, cv, peak_z, peak_i, centroid_z, fwhm,
          snr, contrast, area  (all floats or 'N/A' strings).
    """
    NA = 'N/A'
    if not z_arr or not i_arr:
        return {k: NA for k in
                ['mean','std','cv','peak_z','peak_i','centroid_z',
                 'fwhm','snr','contrast','area']}

    z = np.array(z_arr, dtype=float)
    I = np.array(i_arr, dtype=float)
    n = len(I)

    mean_i  = float(np.mean(I))
    std_i   = float(np.std(I))
    max_i   = float(np.max(I))
    peak_idx = int(np.argmax(I))
    peak_z   = float(z[peak_idx])

    # CV  (%)
    cv = (std_i / mean_i * 100.0) if mean_i != 0 else 0.0

    # Centroid z  (intensity-weighted mean)
    total_i = float(np.sum(I))
    centroid_z = float(np.sum(z * I) / total_i) if total_i > 0 else peak_z

    # Background: mean + std of the bottom-25th-percentile samples
    thresh = float(np.percentile(I, 25))
    bg_vals = I[I <= thresh]
    bg_mean = float(np.mean(bg_vals)) if len(bg_vals) > 0 else float(np.min(I))
    bg_std  = float(np.std(bg_vals))  if len(bg_vals) > 1 else (std_i if std_i > 0 else 1.0)

    # SNR  = (peak − background) / background_std
    snr = (max_i - bg_mean) / bg_std if bg_std > 0 else float('inf')

    # Contrast ratio  = peak / background_mean
    contrast = max_i / bg_mean if bg_mean > 0 else float('inf')

    # FWHM  (numerical, above background)
    half_max = bg_mean + (max_i - bg_mean) / 2.0
    fwhm = _compute_fwhm(z, I, half_max)

    # Integrated area  (trapezoidal)
    area = float(np.trapezoid(I, z)) if n > 1 else float(I[0])

    def _fmt(v, decimals=2):
        if v is None:
            return NA
        if abs(v) >= 1e5 or (abs(v) < 1e-3 and v != 0):
            return f"{v:.3e}"
        return f"{v:.{decimals}f}"

    return {
        'mean':       _fmt(mean_i),
        'std':        _fmt(std_i),
        'cv':         _fmt(cv, 1),
        'peak_z':     _fmt(peak_z, 1),
        'peak_i':     _fmt(max_i),
        'centroid_z': _fmt(centroid_z, 2),
        'fwhm':       _fmt(fwhm, 2),
        'snr':        _fmt(snr, 1),
        'contrast':   _fmt(contrast, 1),
        'area':       _fmt(area),
    }


def fit_gpr_profile(z_arr, i_arr):
    """
    Fit a Gaussian Process to the I(z) series.

    Returns a dict with keys:
        ok, z_fit, i_fit, i_upper, i_lower,
        opt_z (float), opt_z_str, gpr_fwhm
    or None if sklearn is unavailable or too few points.
    """
    if not SKLEARN_OK or len(z_arr) < 6:
        return None
    try:
        z = np.array(z_arr, dtype=float)
        I = np.array(i_arr, dtype=float)
        z_range = float(z.max() - z.min())
        if z_range < 1:
            return None

        # Subsample to max 200 points to keep fitting fast
        if len(z) > 150:
            idx = np.round(np.linspace(0, len(z) - 1, 150)).astype(int)
            z_fit_in, I_fit_in = z[idx], I[idx]
        else:
            z_fit_in, I_fit_in = z, I

        ls0 = z_range / 5.0
        kernel = (_CK(1.0, (1e-3, 1e3))
                  * RBF(ls0, (ls0 / 20, ls0 * 20))
                  + WhiteKernel(0.1, (1e-10, 10.0)))
        gpr = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=2,
            normalize_y=True, alpha=1e-10)
        gpr.fit(z_fit_in.reshape(-1, 1), I_fit_in)

        z_dense = np.linspace(z.min(), z.max(), 150)
        I_pred, I_sigma = gpr.predict(z_dense.reshape(-1, 1), return_std=True)

        opt_idx = int(np.argmax(I_pred))
        opt_z   = float(z_dense[opt_idx])

        # FWHM from the smooth GP curve
        bg_gp    = float(np.percentile(I_pred, 25))
        half_gp  = bg_gp + (float(I_pred.max()) - bg_gp) / 2.0
        fwhm_gp  = _compute_fwhm(z_dense, I_pred, half_gp)

        def _r4(arr):
            return [round(float(v), 4) for v in arr]

        return {
            'ok':        True,
            'z_fit':     _r4(z_dense),
            'i_fit':     _r4(I_pred),
            'i_upper':   _r4(I_pred + I_sigma),
            'i_lower':   _r4(I_pred - I_sigma),
            'opt_z':     opt_z,
            'opt_z_str': f"{opt_z:.2f}",
            'gpr_fwhm':  f"{fwhm_gp:.2f}" if fwhm_gp is not None else "N/A",
        }
    except Exception:
        return None


def detect_outlier_frames(z_arr, i_arr, threshold=3.5):
    """
    Flag frames whose intensity is a statistical outlier using the
    MAD-based modified Z-score (robust to non-Gaussian tails).
    Returns list of z indices that are outliers.
    """
    if len(i_arr) < 5:
        return []
    I = np.array(i_arr, dtype=float)
    median = np.median(I)
    mad    = np.median(np.abs(I - median))
    if mad == 0:
        return []
    mod_z = 0.6745 * np.abs(I - median) / mad
    return [int(z_arr[k]) for k in range(len(I)) if mod_z[k] > threshold]


# ══════════════════════════════════════════════════════════════════════════════
#  Landing page  (GET /)
# ══════════════════════════════════════════════════════════════════════════════

FORM_HTML = """
<div class="card">
  <div class="card-hdr">
    <div class="card-dot"></div>
    <span class="card-title">HeightScan — CBF Image Series</span>
    <span class="badge badge-blue">NX Projection Panel concept</span>
  </div>
  <h2 class="sec-title">1D Line Plot  ·  X-Axis = z (image index), Y-Axis = integrated ROI intensity</h2>
  <p class="subhead">
    Load a series of <code>.cbf</code> detector images, define a pixel ROI on the detector, and
    track the integrated intensity across the image-index (z) range — equivalent to the
    <em>NX Projection Panel</em> with X-Axis=z, Y-Axis=None.
  </p>
</div>

{WARN}

<form action="/run" method="post" onsubmit="startSpin()">

  <!-- Step 1: Folder -->
  <div class="card">
    <div class="card-hdr">
      <div class="card-dot"></div>
      <span class="card-title">Step 1 · Data Folder</span>
    </div>
    <div class="fg">
      <label class="fl">Path to CBF folder</label>
      <div style="display:flex;gap:8px;align-items:center;">
        <input type="text" name="folder" id="folder-input" value="{FOLDER}"
               placeholder="/nfs/chess/id4b/2026-1/.../tiffs/4Hab_TaS2_001"
               style="flex:1;max-width:580px;">
        <a href="/browse?path={FOLDER_ENC}">
          <button type="button">📁 Browse</button>
        </a>
      </div>
      <div style="font-size:11px;color:var(--text-muted);margin-top:4px;">
        Root: <span class="path-label">/nfs/chess/id4b/2026-1/</span>
        &nbsp;·&nbsp; Navigate to your <code>tiffs/</code> subfolder and click <b>Select this folder</b>.
      </div>
    </div>
    {FILE_INFO}
  </div>

  <!-- Step 2: z range -->
  <div class="card">
    <div class="card-hdr">
      <div class="card-dot"></div>
      <span class="card-title">Step 2 · Image Index Range (z)</span>
    </div>
    <p class="subhead" style="margin-bottom:10px;">
      Corresponds to the <em>z</em> axis in the NX Projection Panel (e.g. z min = 1, z max = 30).
    </p>
    <div class="row">
      <div class="fg">
        <label class="fl">z min  <span style="font-weight:400;text-transform:none;">(first image index)</span></label>
        <input type="number" name="z_min" value="{Z_MIN}" min="0" step="1">
      </div>
      <div class="fg">
        <label class="fl">z max  <span style="font-weight:400;text-transform:none;">(last image index)</span></label>
        <input type="number" name="z_max" value="{Z_MAX}" min="0" step="1">
      </div>
      <div class="fg">
        <label class="fl">Preview image index</label>
        <input type="number" name="z_preview" value="{Z_PREV}" min="0" step="1"
               title="Which image to show as detector preview">
      </div>
    </div>
  </div>

  <!-- Step 3: ROI -->
  <div class="card">
    <div class="card-hdr">
      <div class="card-dot"></div>
      <span class="card-title">Step 3 · Detector ROI  (pixel coordinates)</span>
    </div>
    <p class="subhead" style="margin-bottom:10px;">
      x = detector column,  y = detector row.  Maps to NX Projection Panel x/y range fields.
    </p>
    <div class="row">
      <div class="fg"><label class="fl">x min  (col)</label>
        <input type="number" name="x_min" value="{X_MIN}" min="0" step="1"></div>
      <div class="fg"><label class="fl">x max  (col)</label>
        <input type="number" name="x_max" value="{X_MAX}" min="0" step="1"></div>
      <div class="fg"><label class="fl">y min  (row)</label>
        <input type="number" name="y_min" value="{Y_MIN}" min="0" step="1"></div>
      <div class="fg"><label class="fl">y max  (row)</label>
        <input type="number" name="y_max" value="{Y_MAX}" min="0" step="1"></div>
    </div>
  </div>

  <!-- Step 4: Options -->
  <div class="card">
    <div class="card-hdr">
      <div class="card-dot"></div>
      <span class="card-title">Step 4 · Integration &amp; Display Options</span>
    </div>
    <div class="row" style="margin-bottom:14px;">
      <div class="fg">
        <label class="fl">Integration mode</label>
        <select name="mode">
          <option value="sum"    {SEL_SUM}>Sum of ROI pixels</option>
          <option value="mean"   {SEL_MEAN}>Mean of ROI pixels</option>
          <option value="max"    {SEL_MAX}>Max of ROI pixels</option>
          <option value="median" {SEL_MED}>Median of ROI pixels</option>
        </select>
      </div>
      <div class="fg">
        <label class="fl">Image colormap</label>
        <select name="cmap">
          <option value="inferno" {SEL_INF}>Inferno</option>
          <option value="viridis" {SEL_VIR}>Viridis</option>
          <option value="hot"     {SEL_HOT}>Hot</option>
          <option value="gray"    {SEL_GRY}>Gray</option>
        </select>
      </div>
      <div class="fg">
        <label class="fl">Max-Pixel ROI pad <span style="font-weight:400;text-transform:none;">(px each side)</span></label>
        <div style="display:flex;gap:8px;align-items:center;">
          <span style="font-size:11px;color:#f97316;white-space:nowrap;">🥇 P1</span>
          <input type="number" name="maxroi_pad"  value="{MAXROI_PAD1}" min="1" max="500" step="1"
                 style="width:70px;" title="Half-width of ROI box for Peak 1 (brightest)">
          <span style="font-size:11px;color:#a855f7;white-space:nowrap;">🥈 P2</span>
          <input type="number" name="maxroi_pad2" value="{MAXROI_PAD2}" min="1" max="500" step="1"
                 style="width:70px;" title="Half-width of ROI box for Peak 2 (2nd brightest)">
          <span style="font-size:11px;color:#22d3ee;white-space:nowrap;">🥉 P3</span>
          <input type="number" name="maxroi_pad3" value="{MAXROI_PAD3}" min="1" max="500" step="1"
                 style="width:70px;" title="Half-width of ROI box for Peak 3 (3rd brightest)">
        </div>
      </div>
    </div>
    <div class="row">
      <div class="fg" style="align-self:flex-end;">
        <label style="font-size:13px;cursor:pointer;">
          <input type="checkbox" name="log_img" value="1" {LOG_CHK}> Log₁₀ scale on detector image
        </label>
      </div>
      <div class="fg" style="align-self:flex-end;">
        <label style="font-size:13px;cursor:pointer;">
          <input type="checkbox" name="show_table" value="1" {TBL_CHK}> Show data table
        </label>
      </div>
    </div>
  </div>

  <button type="submit" class="btn-primary" id="run-btn">▶  Run HeightScan</button>
  <div id="spinner">⟳ &nbsp;Processing images — please wait…</div>
  <div class="prog-bar-wrap" id="prog-wrap"><div class="prog-bar"></div></div>
</form>

<script>
function startSpin(){
  document.getElementById('spinner').style.display='block';
  document.getElementById('prog-wrap').style.display='block';
  document.getElementById('run-btn').disabled=true;
  document.getElementById('run-btn').textContent='⟳  Running…';
}
</script>
"""

# ══════════════════════════════════════════════════════════════════════════════
#  Results page template
#  Placeholder naming rules:
#    • All placeholders use {NAME} with braces
#    • Replace longer/more-specific names BEFORE shorter ones in the /run chain
#    • Pixel inspector: {PX_ZV_ESC} before {PX_ZV}, {PX_MV_ESC} before {PX_MV}
#    • Max-ROI:         {MR_ZV_ESC} before {MR_ZV}, {MR_IV_ESC} before {MR_IV}
#    • ROI plot:        {Z_JSON_ESC} before {Z_JSON}, {I_JSON_ESC} before {I_JSON}
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_HTML = """
<!-- meta chips -->
<div class="meta-chips">
  <div class="meta-chip">📁 <b>{PREFIX}</b></div>
  <div class="meta-chip">z &nbsp;<b>{Z_MIN} – {Z_MAX}</b></div>
  <div class="meta-chip">ROI x &nbsp;<b>{X_MIN}:{X_MAX}</b></div>
  <div class="meta-chip">ROI y &nbsp;<b>{Y_MIN}:{Y_MAX}</b></div>
  <div class="meta-chip">mode &nbsp;<b>{MODE}</b></div>
  <div class="meta-chip">images &nbsp;<b>{N_IMGS}</b></div>
  <div class="meta-chip">⏱ &nbsp;<b>{ELAPSED}s</b></div>
</div>

<!-- ══ Detector Preview + Navigator ═══════════════════════════════════════════ -->
<div class="card">
  <div class="card-hdr">
    <div class="card-dot"></div>
    <span class="card-title">Detector Image Preview  ·  z = {Z_PREV}</span>
    <span style="margin-left:auto;font-size:11px;color:var(--text-muted);">
      {PREV_FILE} &nbsp;|&nbsp; {IMG_SHAPE}
    </span>
    <!-- View toggle -->
    <div style="margin-left:16px;display:flex;gap:4px;">
      <button class="btn btn-sm btn-ghost" id="btn-png-view"
              onclick="showPngView()" title="Static PNG view">🖼 PNG</button>
      <button class="btn btn-sm btn-ghost" id="btn-plotly-view"
              onclick="showPlotlyView()" title="Interactive Plotly heatmap">🔲 Plotly</button>
    </div>
  </div>

  <!-- PNG view (default) -->
  <div id="det-png-view">
    <div style="display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start;">
      <img id="det-static-img" class="det-img" src="data:image/png;base64,{IMG_B64}" alt="detector"
           style="max-width:480px;">
      <div style="font-size:13px;line-height:2;">
        <div>ROI sum &nbsp; <b id="prev-roi-sum" style="color:var(--accent-glow);font-family:'IBM Plex Mono',monospace;">{ROI_SUM}</b></div>
        <div>ROI mean &nbsp;<b id="prev-roi-mean" style="color:var(--accent-glow);font-family:'IBM Plex Mono',monospace;">{ROI_MEAN}</b></div>
        <div>ROI max &nbsp; <b id="prev-roi-max" style="color:var(--accent-glow);font-family:'IBM Plex Mono',monospace;">{ROI_MAX_V}</b></div>
        <hr style="border:none;border-top:1px solid var(--border-subtle);margin:8px 0;">
        <div>Global max &nbsp;<b style="color:#f0b429;font-family:'IBM Plex Mono',monospace;">{PREV_GMAX}</b></div>
        <div style="font-size:11px;color:var(--text-muted);">
          at pixel &nbsp;<b style="font-family:'IBM Plex Mono',monospace;">x={PREV_GMAX_X}, y={PREV_GMAX_Y}</b>
        </div>
      </div>
    </div>
  </div>

  <!-- Plotly view (lazy loaded) -->
  <div id="det-plotly-view" style="display:none;">
    <div id="det-plotly" style="width:100%;height:520px;"></div>
    <div id="det-plotly-spinner" style="text-align:center;padding:40px;color:var(--text-muted);">
      ⟳ Loading interactive view…
    </div>
  </div>

  <!-- ── Image Navigator ── -->
  <div style="margin-top:16px;">
    <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;
                color:var(--text-muted);margin-bottom:8px;">🎞 Image Navigator</div>
    <div class="nav-bar">
      <button class="btn btn-sm btn-ghost" onclick="navStep(-1)" title="Previous image">◀ Prev</button>
      <input type="range" id="nav-slider" min="0" max="1" value="0"
             oninput="navGoIdx(parseInt(this.value))">
      <button class="btn btn-sm btn-ghost" onclick="navStep(1)" title="Next image">Next ▶</button>
      <span class="nav-stat">z = <b id="nav-z-display">{Z_PREV}</b></span>
      <input type="number" id="nav-z-input" value="{Z_PREV}" style="width:80px;"
             onchange="navGoZ(parseInt(this.value))" title="Jump to z index">
      <span class="nav-stat" id="nav-stats-block">
        sum=<b id="nav-roi-sum">{ROI_SUM}</b> &nbsp;
        mean=<b id="nav-roi-mean">{ROI_MEAN}</b> &nbsp;
        max=<b id="nav-roi-max">{ROI_MAX_V}</b>
      </span>
    </div>
  </div>
</div>

<!-- ══ Live ROI & Display Controls ══════════════════════════════════════════ -->
<div class="card" id="roi-ctrl-card">
  <div class="card-hdr" style="cursor:pointer;" onclick="toggleCtrl()">
    <div class="card-dot" style="background:#38bdf8;box-shadow:0 0 8px #38bdf8;"></div>
    <span class="card-title" style="color:#38bdf8;">Live ROI &amp; Display Controls</span>
    <span style="font-size:11px;color:var(--text-muted);margin-left:10px;">
      Adjust ROI coordinates &amp; contrast · changes apply to Navigator view in real time
    </span>
    <span id="ctrl-toggle-icon" style="margin-left:auto;font-size:14px;">▾</span>
  </div>
  <div id="roi-ctrl-body">
    <!-- ROI row -->
    <div class="row" style="gap:10px;margin-bottom:12px;">
      <div class="fg">
        <label class="fl">ROI x min</label>
        <input type="number" id="ctrl-xmin" value="{MR_X0}" min="0" step="1" style="width:90px;"
               oninput="ctrlLiveUpdate()">
      </div>
      <div class="fg">
        <label class="fl">ROI x max</label>
        <input type="number" id="ctrl-xmax" value="{MR_X1}" min="0" step="1" style="width:90px;"
               oninput="ctrlLiveUpdate()">
      </div>
      <div class="fg">
        <label class="fl">ROI y min</label>
        <input type="number" id="ctrl-ymin" value="{MR_Y0}" min="0" step="1" style="width:90px;"
               oninput="ctrlLiveUpdate()">
      </div>
      <div class="fg">
        <label class="fl">ROI y max</label>
        <input type="number" id="ctrl-ymax" value="{MR_Y1}" min="0" step="1" style="width:90px;"
               oninput="ctrlLiveUpdate()">
      </div>
      <div style="width:1px;background:var(--border-subtle);align-self:stretch;margin:0 4px;"></div>
      <!-- vmin/vmax -->
      <div class="fg">
        <label class="fl">vmin <span style="font-weight:400;text-transform:none;font-size:10px;">(display)</span></label>
        <input type="number" id="ctrl-vmin" value="{CTRL_VMIN_HTML}" placeholder="auto"
               step="any" style="width:100px;" oninput="ctrlLiveUpdate()">
      </div>
      <div class="fg">
        <label class="fl">vmax <span style="font-weight:400;text-transform:none;font-size:10px;">(display)</span></label>
        <input type="number" id="ctrl-vmax" value="{CTRL_VMAX_HTML}" placeholder="auto"
               step="any" style="width:100px;" oninput="ctrlLiveUpdate()">
      </div>
      <!-- Action buttons -->
      <div class="fg" style="display:flex;flex-direction:column;gap:6px;align-self:flex-end;">
        <button class="btn btn-sm btn-ghost" onclick="ctrlAutoROI(99,'com')"
                title="Auto-detect ROI from center of mass of top-1% pixels">🎯 Auto-ROI (COM)</button>
        <button class="btn btn-sm btn-ghost" onclick="ctrlAutoROI(99,'bbox')"
                title="Bounding box of top-1% pixels">📦 Auto-ROI (bbox)</button>
        <button class="btn btn-sm btn-ghost" onclick="ctrlAutoVminVmax()"
                title="Set vmin/vmax from 1st/99th percentile of current image">⚡ Auto vmin/vmax</button>
        <button class="btn btn-sm btn-ghost" onclick="ctrlReset()"
                title="Reset to original scan values">↩ Reset</button>
      </div>
    </div>
    <!-- ── Peak quick-navigate strip ── -->
    <div style="border-top:1px solid var(--border-subtle);margin-top:8px;padding-top:8px;
                display:flex;gap:10px;flex-wrap:wrap;align-items:center;">
      <span style="font-size:11px;font-weight:600;color:var(--text-muted);text-transform:uppercase;
                   letter-spacing:.06em;">Jump to peak:</span>
      <button class="btn btn-sm" onclick="ctrlSetPeak(0)"
              style="border-color:#f97316;color:#f97316;"
              title="Set navigator ROI to Peak 1 (brightest pixel)">
        🥇 Peak&nbsp;1 &nbsp;<span style="font-family:IBM Plex Mono,monospace;font-size:10px;">
        x={MR_X0}:{MR_X1} &nbsp;y={MR_Y0}:{MR_Y1}</span>
      </button>
      <button class="btn btn-sm" onclick="ctrlSetPeak(1)"
              style="border-color:#a855f7;color:#a855f7;"
              title="Set navigator ROI to Peak 2 (2nd brightest pixel)">
        🥈 Peak&nbsp;2 &nbsp;<span style="font-family:IBM Plex Mono,monospace;font-size:10px;">
        x={MR2_X0}:{MR2_X1} &nbsp;y={MR2_Y0}:{MR2_Y1}</span>
      </button>
      <button class="btn btn-sm" onclick="ctrlSetPeak(2)"
              style="border-color:#22d3ee;color:#22d3ee;"
              title="Set navigator ROI to Peak 3 (3rd brightest pixel)">
        🥉 Peak&nbsp;3 &nbsp;<span style="font-family:IBM Plex Mono,monospace;font-size:10px;">
        x={MR3_X0}:{MR3_X1} &nbsp;y={MR3_Y0}:{MR3_Y1}</span>
      </button>
    </div>
    <div id="ctrl-info" style="font-size:11px;color:var(--text-muted);min-height:18px;margin-top:6px;"></div>
  </div>
</div>

<!-- ── Tab bar ── -->
<div class="tab-bar">
  <button class="tab-btn active" id="tab-roi" onclick="switchTab('roi')">📈 ROI Line Plot</button>
  <button class="tab-btn" id="tab-px"  onclick="switchTab('px')">🔍 Pixel Inspector</button>
  <button class="tab-btn" id="tab-mr"  onclick="switchTab('mr')">🎯 Max-Pixel ROI</button>
</div>

<!-- ══ Tab 1: ROI line plot ══════════════════════════════════════════════════ -->
<div class="tab-panel active" id="panel-roi">
  <div class="card">
    <div class="card-hdr">
      <div class="card-dot" style="background:#f0b429;box-shadow:0 0 8px #f0b429;"></div>
      <span class="card-title" style="color:#f0b429;">
        HeightScan — {MODE_CAP} over ROI  ·  x={X_MIN}:{X_MAX}, y={Y_MIN}:{Y_MAX}
      </span>
      <span style="margin-left:auto;font-size:11px;color:var(--text-muted);">
        peak at z=<b>{PEAK_Z_VAL}</b> · I=<b>{PEAK_I}</b>
      </span>
    </div>
    <div id="hs-plot" style="width:100%;height:420px;"></div>
    <script>
    (function(){
      var zv  = {Z_JSON};
      var iv  = {I_JSON};
      var zpk = {PEAK_Z_VAL};
      Plotly.newPlot('hs-plot', [{
        x: zv, y: iv, type:'scatter', mode:'markers+lines',
        name:'ROI {MODE} intensity',
        line: {color:'#00c8ff', width:2},
        marker: {color:'#00c8ff', size:6, symbol:'circle',
                 line: {color:'#fff', width:1}},
        hovertemplate:'z=%{x}<br>I=%{y:.4e}<extra></extra>'
      }], {
        paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
        xaxis: {title:'z  (image index)', showgrid:true, gridcolor:'rgba(128,128,128,.2)',
                zeroline:false, autorange:true},
        yaxis: {title:'{MODE_CAP} Intensity  [ROI x={X_MIN}:{X_MAX}, y={Y_MIN}:{Y_MAX}]',
                showgrid:true, gridcolor:'rgba(128,128,128,.2)', zeroline:false},
        shapes: [{type:'line', x0:zpk, x1:zpk, y0:0, y1:1, yref:'paper',
                  line: {color:'#f0b429', width:1.5, dash:'dot'}}],
        annotations: [{x:zpk, y:1, yref:'paper', text:'peak z='+zpk,
                       showarrow:true, arrowhead:2, arrowcolor:'#f0b429',
                       font: {size:11, color:'#f0b429'}, ax:20, ay:-30}],
        margin: {t:20, r:20, b:65, l:90},
        font: {family:'IBM Plex Sans,Inter,sans-serif', size:12},
        hovermode:'x unified'
      }, {
        responsive:true, displaylogo:false, scrollZoom:true, displayModeBar:true,
        modeBarButtonsToAdd:['toggleSpikelines'],
        toImageButtonOptions: {format:'png', filename:'roi_heightscan_{PREFIX}', scale:2}
      });
    }())
    </script>
  </div>

  {TABLE_BLOCK}

  <div class="card">
    <div class="card-hdr">
      <div class="card-dot"></div><span class="card-title">Export — ROI scan</span>
    </div>
    <form action="/download_csv" method="post" style="display:inline;">
      <input type="hidden" name="z_json"  value="{Z_JSON_ESC}">
      <input type="hidden" name="i_json"  value="{I_JSON_ESC}">
      <input type="hidden" name="mode"    value="{MODE}">
      <input type="hidden" name="prefix"  value="{PREFIX}">
      <button type="submit">⬇ Download ROI CSV</button>
    </form>
    &nbsp;&nbsp;<a class="dl-btn" href="/">← New scan</a>
  </div>
</div>

<!-- ══ Tab 2: Pixel Inspector ════════════════════════════════════════════════ -->
<div class="tab-panel" id="panel-px">

  <div class="card">
    <div class="card-hdr">
      <div class="card-dot" style="background:#4ade80;box-shadow:0 0 8px #4ade80;"></div>
      <span class="card-title" style="color:#4ade80;">
        Max Pixel Value per Image  ·  full detector frame
      </span>
      <span style="margin-left:auto;font-size:11px;color:var(--text-muted);">
        peak at z=<b>{PX_PKZS}</b> · max=<b>{PX_PKVAL}</b>
      </span>
    </div>
    <div id="px-plot" style="width:100%;height:420px;"></div>
    <script>
    (function(){
      var zv  = {PX_ZV};
      var mv  = {PX_MV};
      var xv  = {PX_XV};
      var yv  = {PX_YV};
      var zpk = {PX_PKZN};
      Plotly.newPlot('px-plot',
        [
          { x:zv, y:mv, type:'scatter', mode:'markers+lines',
            name:'Global max (counts)',
            line: {color:'#4ade80', width:2},
            marker: {color:'#4ade80', size:7, symbol:'circle',
                     line: {color:'#fff', width:1.5}},
            hovertemplate:'z=%{x}<br>max=%{y} counts<extra></extra>',
            yaxis:'y' },
          { x:zv, y:xv, type:'scatter', mode:'lines',
            name:'x-col of max pixel',
            line: {color:'#f97316', width:1.5, dash:'dot'},
            hovertemplate:'z=%{x}<br>x-col=%{y}<extra></extra>',
            yaxis:'y2' },
          { x:zv, y:yv, type:'scatter', mode:'lines',
            name:'y-row of max pixel',
            line: {color:'#a78bfa', width:1.5, dash:'dot'},
            hovertemplate:'z=%{x}<br>y-row=%{y}<extra></extra>',
            yaxis:'y2' }
        ],
        { paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
          xaxis:  { title:'z  (image index)', showgrid:true,
                    gridcolor:'rgba(128,128,128,.2)', zeroline:false },
          yaxis:  { title:'Max pixel value (counts)', showgrid:true,
                    gridcolor:'rgba(128,128,128,.2)', zeroline:false, side:'left' },
          yaxis2: { title:'Pixel coordinate', overlaying:'y', side:'right',
                    showgrid:false, zeroline:false },
          shapes: [{ type:'line', x0:zpk, x1:zpk, y0:0, y1:1, yref:'paper',
                     line: {color:'#f0b429', width:1.5, dash:'dot'} }],
          annotations: [{ x:zpk, y:1, yref:'paper', text:'peak z='+zpk,
                          showarrow:true, arrowhead:2, arrowcolor:'#f0b429',
                          font: {size:11, color:'#f0b429'}, ax:20, ay:-30 }],
          legend: { orientation:'h', yanchor:'bottom', y:1.02, xanchor:'left', x:0 },
          margin: { t:40, r:90, b:65, l:90 },
          font: { family:'IBM Plex Sans,Inter,sans-serif', size:12 },
          hovermode:'x unified'
        },
        { responsive:true, displaylogo:false, scrollZoom:true, displayModeBar:true,
          modeBarButtonsToAdd:['toggleSpikelines'],
          toImageButtonOptions: { format:'png', filename:'maxpixel_{PX_PFX}', scale:2 }
        }
      );
    }())
    </script>
  </div>

  <!-- Per-image pixel stats table -->
  <div class="card">
    <div class="card-hdr">
      <div class="card-dot"></div>
      <span class="card-title">Per-Image Pixel Statistics</span>
    </div>
    <div style="max-height:360px;overflow-y:auto;">
    <table>
      <thead><tr>
        <th>z</th><th>Global Max</th><th>x col</th><th>y row</th>
        <th>Frame Mean</th><th>Frame Std</th><th>Filename</th>
      </tr></thead>
      <tbody>{PX_TROWS}</tbody>
    </table>
    </div>
  </div>

  <div class="card">
    <div class="card-hdr">
      <div class="card-dot"></div>
      <span class="card-title">Export — Pixel Inspector</span>
    </div>
    <form action="/download_csv" method="post" style="display:inline;">
      <input type="hidden" name="z_json"  value="{PX_ZV_ESC}">
      <input type="hidden" name="i_json"  value="{PX_MV_ESC}">
      <input type="hidden" name="mode"    value="global_max">
      <input type="hidden" name="prefix"  value="{PX_PFX}">
      <button type="submit">⬇ Download Pixel Inspector CSV</button>
    </form>
    &nbsp;&nbsp;<a class="dl-btn" href="/">← New scan</a>
  </div>
</div>

<!-- ══ Tab 3: Max-Pixel ROI Plots (Top-3 Peaks) ══════════════════════════════ -->
<div class="tab-panel" id="panel-mr">

  <!-- ── Peak 1: Brightest Pixel ─────────────────────────────────────────── -->
  <div class="card">
    <div class="card-hdr">
      <div class="card-dot" style="background:#f97316;box-shadow:0 0 8px #f97316;"></div>
      <span class="card-title" style="color:#f97316;">
        🥇 Brightest Pixel ROI · pad = {MR_PAD} px
      </span>
      <span style="margin-left:auto;font-size:11px;color:var(--text-muted);">
        peak at z=<b>{MR_PKZS}</b> · I=<b>{MR_PKIS}</b>
      </span>
      <button class="btn btn-sm" onclick="ctrlSetPeak(0)"
              style="margin-left:12px;border-color:#f97316;color:#f97316;"
              title="Set detector navigator ROI to this peak">📍 Navigate</button>
    </div>
    <div class="meta-chips" style="margin-bottom:14px;">
      <div class="meta-chip">Pixel &nbsp;<b>x={MR_GX}, y={MR_GY}</b></div>
      <div class="meta-chip">ROI x &nbsp;<b>{MR_X0} : {MR_X1}</b></div>
      <div class="meta-chip">ROI y &nbsp;<b>{MR_Y0} : {MR_Y1}</b></div>
      <div class="meta-chip">mode &nbsp;<b>{MR_MODE}</b></div>
    </div>
    <div id="mr-plot" style="width:100%;height:380px;"></div>
    <script>(function(){
      var zv={MR_ZV}, iv={MR_IV}, zpk={MR_PKZN};
      var gprOk={MR_GPR_OK}, gZ={MR_GPR_ZV}, gI={MR_GPR_IV}, gUp={MR_GPR_UP}, gLo={MR_GPR_LO}, gOpt={MR_GPR_OPTZ};
      var badZ={MR_BAD_ZV};
      var traces=[{x:zv,y:iv,type:'scatter',mode:'markers+lines',name:'Peak 1 {MR_MODE}',
        line:{color:'#f97316',width:2},
        marker:{color:'#f97316',size:7,symbol:'circle',line:{color:'#fff',width:1.5}},
        hovertemplate:'z=%{x}<br>I=%{y:.4e}<extra></extra>'}];
      if(gprOk){
        traces.push({x:gZ,y:gUp,type:'scatter',mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'});
        traces.push({x:gZ,y:gLo,type:'scatter',mode:'lines',fill:'tonexty',fillcolor:'rgba(249,115,22,0.13)',line:{width:0},name:'GP ±1σ',hoverinfo:'skip'});
        traces.push({x:gZ,y:gI,type:'scatter',mode:'lines',line:{color:'rgba(249,115,22,0.85)',width:2,dash:'dash'},name:'GP fit',hovertemplate:'z=%{x}<br>GP=%{y:.4e}<extra></extra>'});
      }
      if(badZ.length>0){
        var bI=badZ.map(function(bz){var k=zv.indexOf(bz);return k>=0?iv[k]:null;});
        traces.push({x:badZ,y:bI,type:'scatter',mode:'markers',marker:{color:'#f87171',size:10,symbol:'x-thin',line:{color:'#f87171',width:2}},name:'Outlier',hovertemplate:'z=%{x} ⚠ outlier<extra></extra>'});
      }
      var shapes=[{type:'line',x0:zpk,x1:zpk,y0:0,y1:1,yref:'paper',line:{color:'#f0b429',width:1.5,dash:'dot'}}];
      var annots=[{x:zpk,y:1,yref:'paper',text:'peak z='+zpk,showarrow:true,arrowhead:2,arrowcolor:'#f0b429',font:{size:11,color:'#f0b429'},ax:20,ay:-30}];
      if(gprOk){
        shapes.push({type:'line',x0:gOpt,x1:gOpt,y0:0,y1:1,yref:'paper',line:{color:'#86efac',width:1.5,dash:'dot'}});
        annots.push({x:gOpt,y:0.84,yref:'paper',text:'GP opt='+parseFloat(gOpt).toFixed(1),showarrow:true,arrowhead:2,arrowcolor:'#86efac',font:{size:11,color:'#86efac'},ax:-25,ay:-30});
      }
      Plotly.newPlot('mr-plot',traces,{
        paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
        xaxis:{title:'z (image index)',showgrid:true,gridcolor:'rgba(128,128,128,.2)',zeroline:false},
        yaxis:{title:'Peak 1 {MR_MODE} Intensity [pad={MR_PAD}px]',showgrid:true,gridcolor:'rgba(128,128,128,.2)',zeroline:false},
        shapes:shapes, annotations:annots,
        margin:{t:20,r:20,b:65,l:90}, font:{family:'IBM Plex Sans,Inter,sans-serif',size:12}, hovermode:'x unified'
      },{responsive:true,displaylogo:false,scrollZoom:true,displayModeBar:true,
         modeBarButtonsToAdd:['toggleSpikelines'],
         toImageButtonOptions:{format:'png',filename:'peak1_{MR_PFX}',scale:2}});
    }())</script>
    <!-- ── Peak 1 stats grid ── -->
    <div class="stat-grid">
      <div class="stat-box"><div class="stat-lbl">Peak z</div><div class="stat-val">{MR_S_PKZN}</div></div>
      <div class="stat-box"><div class="stat-lbl">Centroid z</div><div class="stat-val">{MR_S_CNTZ}</div></div>
      <div class="stat-box"><div class="stat-lbl">FWHM (Δz)</div><div class="stat-val">{MR_S_FWHM}</div></div>
      <div class="stat-box"><div class="stat-lbl">SNR</div><div class="stat-val {MR_S_SNR_CLS}">{MR_S_SNR}</div></div>
      <div class="stat-box"><div class="stat-lbl">Contrast</div><div class="stat-val {MR_S_CNTR_CLS}">{MR_S_CNTR}</div></div>
      <div class="stat-box"><div class="stat-lbl">Mean I</div><div class="stat-val">{MR_S_MEAN}</div></div>
      <div class="stat-box"><div class="stat-lbl">Std I</div><div class="stat-val">{MR_S_STD}</div></div>
      <div class="stat-box"><div class="stat-lbl">CV</div><div class="stat-val">{MR_S_CV}%</div></div>
      <div class="stat-box"><div class="stat-lbl">∫ I dz</div><div class="stat-val">{MR_S_AREA}</div></div>
    </div>
    <details style="margin-top:10px;">
      <summary style="cursor:pointer;font-size:12px;color:var(--text-muted);padding:4px 0;">
        📊 Peak 1 data table &nbsp;(x={MR_X0}:{MR_X1}, y={MR_Y0}:{MR_Y1})
      </summary>
      <div style="max-height:240px;overflow-y:auto;margin-top:6px;">
        <table><thead><tr>
          <th>z index</th><th>{MODE_CAP} Intensity</th><th>Filename</th>
        </tr></thead><tbody>{MR_ROWS}</tbody></table>
      </div>
    </details>
    <div style="margin-top:10px;">
      <form action="/download_csv" method="post" style="display:inline;">
        <input type="hidden" name="z_json"  value="{MR_ZV_ESC}">
        <input type="hidden" name="i_json"  value="{MR_IV_ESC}">
        <input type="hidden" name="mode"    value="peak1_{MR_MODE}">
        <input type="hidden" name="prefix"  value="{MR_PFX}">
        <button type="submit" class="btn btn-sm">⬇ CSV Peak 1</button>
      </form>
    </div>
  </div>

  <!-- ── Peak 2: 2nd Brightest Pixel ─────────────────────────────────────── -->
  <div class="card">
    <div class="card-hdr">
      <div class="card-dot" style="background:#a855f7;box-shadow:0 0 8px #a855f7;"></div>
      <span class="card-title" style="color:#a855f7;">
        🥈 2nd Brightest Pixel ROI · pad = {MR2_PAD} px
      </span>
      <span style="margin-left:auto;font-size:11px;color:var(--text-muted);">
        peak at z=<b>{MR2_PKZS}</b> · I=<b>{MR2_PKIS}</b>
      </span>
      <button class="btn btn-sm" onclick="ctrlSetPeak(1)"
              style="margin-left:12px;border-color:#a855f7;color:#a855f7;"
              title="Set detector navigator ROI to this peak">📍 Navigate</button>
    </div>
    <div class="meta-chips" style="margin-bottom:14px;">
      <div class="meta-chip">Pixel &nbsp;<b>x={MR2_GX}, y={MR2_GY}</b></div>
      <div class="meta-chip">ROI x &nbsp;<b>{MR2_X0} : {MR2_X1}</b></div>
      <div class="meta-chip">ROI y &nbsp;<b>{MR2_Y0} : {MR2_Y1}</b></div>
      <div class="meta-chip">mode &nbsp;<b>{MR_MODE}</b></div>
    </div>
    <div id="mr2-plot" style="width:100%;height:380px;"></div>
    <script>(function(){
      var zv={MR2_ZV}, iv={MR2_IV}, zpk={MR2_PKZN};
      var gprOk={MR2_GPR_OK}, gZ={MR2_GPR_ZV}, gI={MR2_GPR_IV}, gUp={MR2_GPR_UP}, gLo={MR2_GPR_LO}, gOpt={MR2_GPR_OPTZ};
      var badZ={MR2_BAD_ZV};
      var traces=[{x:zv,y:iv,type:'scatter',mode:'markers+lines',name:'Peak 2 {MR_MODE}',
        line:{color:'#a855f7',width:2},
        marker:{color:'#a855f7',size:7,symbol:'circle',line:{color:'#fff',width:1.5}},
        hovertemplate:'z=%{x}<br>I=%{y:.4e}<extra></extra>'}];
      if(gprOk){
        traces.push({x:gZ,y:gUp,type:'scatter',mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'});
        traces.push({x:gZ,y:gLo,type:'scatter',mode:'lines',fill:'tonexty',fillcolor:'rgba(168,85,247,0.13)',line:{width:0},name:'GP ±1σ',hoverinfo:'skip'});
        traces.push({x:gZ,y:gI,type:'scatter',mode:'lines',line:{color:'rgba(168,85,247,0.85)',width:2,dash:'dash'},name:'GP fit',hovertemplate:'z=%{x}<br>GP=%{y:.4e}<extra></extra>'});
      }
      if(badZ.length>0){
        var bI=badZ.map(function(bz){var k=zv.indexOf(bz);return k>=0?iv[k]:null;});
        traces.push({x:badZ,y:bI,type:'scatter',mode:'markers',marker:{color:'#f87171',size:10,symbol:'x-thin',line:{color:'#f87171',width:2}},name:'Outlier',hovertemplate:'z=%{x} ⚠ outlier<extra></extra>'});
      }
      var shapes=[{type:'line',x0:zpk,x1:zpk,y0:0,y1:1,yref:'paper',line:{color:'#c084fc',width:1.5,dash:'dot'}}];
      var annots=[{x:zpk,y:1,yref:'paper',text:'peak z='+zpk,showarrow:true,arrowhead:2,arrowcolor:'#c084fc',font:{size:11,color:'#c084fc'},ax:20,ay:-30}];
      if(gprOk){
        shapes.push({type:'line',x0:gOpt,x1:gOpt,y0:0,y1:1,yref:'paper',line:{color:'#86efac',width:1.5,dash:'dot'}});
        annots.push({x:gOpt,y:0.84,yref:'paper',text:'GP opt='+parseFloat(gOpt).toFixed(1),showarrow:true,arrowhead:2,arrowcolor:'#86efac',font:{size:11,color:'#86efac'},ax:-25,ay:-30});
      }
      Plotly.newPlot('mr2-plot',traces,{
        paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
        xaxis:{title:'z (image index)',showgrid:true,gridcolor:'rgba(128,128,128,.2)',zeroline:false},
        yaxis:{title:'Peak 2 {MR_MODE} Intensity [pad={MR2_PAD}px]',showgrid:true,gridcolor:'rgba(128,128,128,.2)',zeroline:false},
        shapes:shapes, annotations:annots,
        margin:{t:20,r:20,b:65,l:90}, font:{family:'IBM Plex Sans,Inter,sans-serif',size:12}, hovermode:'x unified'
      },{responsive:true,displaylogo:false,scrollZoom:true,displayModeBar:true,
         modeBarButtonsToAdd:['toggleSpikelines'],
         toImageButtonOptions:{format:'png',filename:'peak2_{MR_PFX}',scale:2}});
    }())</script>
    <!-- ── Peak 2 stats grid ── -->
    <div class="stat-grid">
      <div class="stat-box"><div class="stat-lbl">Peak z</div><div class="stat-val">{MR2_S_PKZN}</div></div>
      <div class="stat-box"><div class="stat-lbl">Centroid z</div><div class="stat-val">{MR2_S_CNTZ}</div></div>
      <div class="stat-box"><div class="stat-lbl">FWHM (Δz)</div><div class="stat-val">{MR2_S_FWHM}</div></div>
      <div class="stat-box"><div class="stat-lbl">SNR</div><div class="stat-val {MR2_S_SNR_CLS}">{MR2_S_SNR}</div></div>
      <div class="stat-box"><div class="stat-lbl">Contrast</div><div class="stat-val {MR2_S_CNTR_CLS}">{MR2_S_CNTR}</div></div>
      <div class="stat-box"><div class="stat-lbl">Mean I</div><div class="stat-val">{MR2_S_MEAN}</div></div>
      <div class="stat-box"><div class="stat-lbl">Std I</div><div class="stat-val">{MR2_S_STD}</div></div>
      <div class="stat-box"><div class="stat-lbl">CV</div><div class="stat-val">{MR2_S_CV}%</div></div>
      <div class="stat-box"><div class="stat-lbl">∫ I dz</div><div class="stat-val">{MR2_S_AREA}</div></div>
    </div>
    <details style="margin-top:10px;">
      <summary style="cursor:pointer;font-size:12px;color:var(--text-muted);padding:4px 0;">
        📊 Peak 2 data table &nbsp;(x={MR2_X0}:{MR2_X1}, y={MR2_Y0}:{MR2_Y1})
      </summary>
      <div style="max-height:240px;overflow-y:auto;margin-top:6px;">
        <table><thead><tr>
          <th>z index</th><th>{MODE_CAP} Intensity</th><th>Filename</th>
        </tr></thead><tbody>{MR2_ROWS}</tbody></table>
      </div>
    </details>
    <div style="margin-top:10px;">
      <form action="/download_csv" method="post" style="display:inline;">
        <input type="hidden" name="z_json"  value="{MR2_ZV_ESC}">
        <input type="hidden" name="i_json"  value="{MR2_IV_ESC}">
        <input type="hidden" name="mode"    value="peak2_{MR_MODE}">
        <input type="hidden" name="prefix"  value="{MR_PFX}">
        <button type="submit" class="btn btn-sm">⬇ CSV Peak 2</button>
      </form>
    </div>
  </div>

  <!-- ── Peak 3: 3rd Brightest Pixel ─────────────────────────────────────── -->
  <div class="card">
    <div class="card-hdr">
      <div class="card-dot" style="background:#22d3ee;box-shadow:0 0 8px #22d3ee;"></div>
      <span class="card-title" style="color:#22d3ee;">
        🥉 3rd Brightest Pixel ROI · pad = {MR3_PAD} px
      </span>
      <span style="margin-left:auto;font-size:11px;color:var(--text-muted);">
        peak at z=<b>{MR3_PKZS}</b> · I=<b>{MR3_PKIS}</b>
      </span>
      <button class="btn btn-sm" onclick="ctrlSetPeak(2)"
              style="margin-left:12px;border-color:#22d3ee;color:#22d3ee;"
              title="Set detector navigator ROI to this peak">📍 Navigate</button>
    </div>
    <div class="meta-chips" style="margin-bottom:14px;">
      <div class="meta-chip">Pixel &nbsp;<b>x={MR3_GX}, y={MR3_GY}</b></div>
      <div class="meta-chip">ROI x &nbsp;<b>{MR3_X0} : {MR3_X1}</b></div>
      <div class="meta-chip">ROI y &nbsp;<b>{MR3_Y0} : {MR3_Y1}</b></div>
      <div class="meta-chip">mode &nbsp;<b>{MR_MODE}</b></div>
    </div>
    <div id="mr3-plot" style="width:100%;height:380px;"></div>
    <script>(function(){
      var zv={MR3_ZV}, iv={MR3_IV}, zpk={MR3_PKZN};
      var gprOk={MR3_GPR_OK}, gZ={MR3_GPR_ZV}, gI={MR3_GPR_IV}, gUp={MR3_GPR_UP}, gLo={MR3_GPR_LO}, gOpt={MR3_GPR_OPTZ};
      var badZ={MR3_BAD_ZV};
      var traces=[{x:zv,y:iv,type:'scatter',mode:'markers+lines',name:'Peak 3 {MR_MODE}',
        line:{color:'#22d3ee',width:2},
        marker:{color:'#22d3ee',size:7,symbol:'circle',line:{color:'#fff',width:1.5}},
        hovertemplate:'z=%{x}<br>I=%{y:.4e}<extra></extra>'}];
      if(gprOk){
        traces.push({x:gZ,y:gUp,type:'scatter',mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'});
        traces.push({x:gZ,y:gLo,type:'scatter',mode:'lines',fill:'tonexty',fillcolor:'rgba(34,211,238,0.13)',line:{width:0},name:'GP ±1σ',hoverinfo:'skip'});
        traces.push({x:gZ,y:gI,type:'scatter',mode:'lines',line:{color:'rgba(34,211,238,0.85)',width:2,dash:'dash'},name:'GP fit',hovertemplate:'z=%{x}<br>GP=%{y:.4e}<extra></extra>'});
      }
      if(badZ.length>0){
        var bI=badZ.map(function(bz){var k=zv.indexOf(bz);return k>=0?iv[k]:null;});
        traces.push({x:badZ,y:bI,type:'scatter',mode:'markers',marker:{color:'#f87171',size:10,symbol:'x-thin',line:{color:'#f87171',width:2}},name:'Outlier',hovertemplate:'z=%{x} ⚠ outlier<extra></extra>'});
      }
      var shapes=[{type:'line',x0:zpk,x1:zpk,y0:0,y1:1,yref:'paper',line:{color:'#67e8f9',width:1.5,dash:'dot'}}];
      var annots=[{x:zpk,y:1,yref:'paper',text:'peak z='+zpk,showarrow:true,arrowhead:2,arrowcolor:'#67e8f9',font:{size:11,color:'#67e8f9'},ax:20,ay:-30}];
      if(gprOk){
        shapes.push({type:'line',x0:gOpt,x1:gOpt,y0:0,y1:1,yref:'paper',line:{color:'#86efac',width:1.5,dash:'dot'}});
        annots.push({x:gOpt,y:0.84,yref:'paper',text:'GP opt='+parseFloat(gOpt).toFixed(1),showarrow:true,arrowhead:2,arrowcolor:'#86efac',font:{size:11,color:'#86efac'},ax:-25,ay:-30});
      }
      Plotly.newPlot('mr3-plot',traces,{
        paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
        xaxis:{title:'z (image index)',showgrid:true,gridcolor:'rgba(128,128,128,.2)',zeroline:false},
        yaxis:{title:'Peak 3 {MR_MODE} Intensity [pad={MR3_PAD}px]',showgrid:true,gridcolor:'rgba(128,128,128,.2)',zeroline:false},
        shapes:shapes, annotations:annots,
        margin:{t:20,r:20,b:65,l:90}, font:{family:'IBM Plex Sans,Inter,sans-serif',size:12}, hovermode:'x unified'
      },{responsive:true,displaylogo:false,scrollZoom:true,displayModeBar:true,
         modeBarButtonsToAdd:['toggleSpikelines'],
         toImageButtonOptions:{format:'png',filename:'peak3_{MR_PFX}',scale:2}});
    }())</script>
    <!-- ── Peak 3 stats grid ── -->
    <div class="stat-grid">
      <div class="stat-box"><div class="stat-lbl">Peak z</div><div class="stat-val">{MR3_S_PKZN}</div></div>
      <div class="stat-box"><div class="stat-lbl">Centroid z</div><div class="stat-val">{MR3_S_CNTZ}</div></div>
      <div class="stat-box"><div class="stat-lbl">FWHM (Δz)</div><div class="stat-val">{MR3_S_FWHM}</div></div>
      <div class="stat-box"><div class="stat-lbl">SNR</div><div class="stat-val {MR3_S_SNR_CLS}">{MR3_S_SNR}</div></div>
      <div class="stat-box"><div class="stat-lbl">Contrast</div><div class="stat-val {MR3_S_CNTR_CLS}">{MR3_S_CNTR}</div></div>
      <div class="stat-box"><div class="stat-lbl">Mean I</div><div class="stat-val">{MR3_S_MEAN}</div></div>
      <div class="stat-box"><div class="stat-lbl">Std I</div><div class="stat-val">{MR3_S_STD}</div></div>
      <div class="stat-box"><div class="stat-lbl">CV</div><div class="stat-val">{MR3_S_CV}%</div></div>
      <div class="stat-box"><div class="stat-lbl">∫ I dz</div><div class="stat-val">{MR3_S_AREA}</div></div>
    </div>
    <details style="margin-top:10px;">
      <summary style="cursor:pointer;font-size:12px;color:var(--text-muted);padding:4px 0;">
        📊 Peak 3 data table &nbsp;(x={MR3_X0}:{MR3_X1}, y={MR3_Y0}:{MR3_Y1})
      </summary>
      <div style="max-height:240px;overflow-y:auto;margin-top:6px;">
        <table><thead><tr>
          <th>z index</th><th>{MODE_CAP} Intensity</th><th>Filename</th>
        </tr></thead><tbody>{MR3_ROWS}</tbody></table>
      </div>
    </details>
    <div style="margin-top:10px;">
      <form action="/download_csv" method="post" style="display:inline;">
        <input type="hidden" name="z_json"  value="{MR3_ZV_ESC}">
        <input type="hidden" name="i_json"  value="{MR3_IV_ESC}">
        <input type="hidden" name="mode"    value="peak3_{MR_MODE}">
        <input type="hidden" name="prefix"  value="{MR_PFX}">
        <button type="submit" class="btn btn-sm">⬇ CSV Peak 3</button>
      </form>
      &nbsp;&nbsp;<a class="dl-btn" href="/">← New scan</a>
    </div>
  </div>

  <!-- ── ML Insights card ──────────────────────────────────────────────────── -->
  <div class="card" style="border:1px solid rgba(134,239,172,.2);margin-top:4px;">
    <div class="card-hdr">
      <div class="card-dot" style="background:#86efac;box-shadow:0 0 8px #86efac;"></div>
      <span class="card-title" style="color:#86efac;">🤖 ML Insights</span>
      <span style="margin-left:auto;font-size:11px;color:var(--text-muted);">{ML_STATUS}</span>
    </div>

    <!-- GP optimal z per peak -->
    <div style="margin-bottom:16px;">
      <div style="font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;">
        Gaussian Process — Optimal sample height (z) per peak
      </div>
      <div class="stat-grid" style="grid-template-columns:repeat(3,1fr);">
        <div class="stat-box" style="border-color:rgba(249,115,22,.3);">
          <div class="stat-lbl" style="color:#f97316;">🥇 Peak 1</div>
          <div class="stat-val">{MR_GPR_OPTZS}</div>
          <div style="font-size:10px;color:var(--text-muted);margin-top:3px;">GP FWHM Δz: {MR_GPR_FWHM}</div>
        </div>
        <div class="stat-box" style="border-color:rgba(168,85,247,.3);">
          <div class="stat-lbl" style="color:#a855f7;">🥈 Peak 2</div>
          <div class="stat-val">{MR2_GPR_OPTZS}</div>
          <div style="font-size:10px;color:var(--text-muted);margin-top:3px;">GP FWHM Δz: {MR2_GPR_FWHM}</div>
        </div>
        <div class="stat-box" style="border-color:rgba(34,211,238,.3);">
          <div class="stat-lbl" style="color:#22d3ee;">🥉 Peak 3</div>
          <div class="stat-val">{MR3_GPR_OPTZS}</div>
          <div style="font-size:10px;color:var(--text-muted);margin-top:3px;">GP FWHM Δz: {MR3_GPR_FWHM}</div>
        </div>
      </div>
    </div>

    <!-- Peak quality ranking table -->
    <div style="margin-bottom:16px;">
      <div style="font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;">
        Peak Quality Ranking &nbsp;<span style="font-weight:400;text-transform:none;">(Sharpness 40% · SNR 30% · Contrast 20% · Stability 10%)</span>
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:12px;">
        <thead>
          <tr style="color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:.04em;">
            <th style="text-align:left;padding:4px 8px;border-bottom:1px solid rgba(255,255,255,.1);">Peak</th>
            <th style="text-align:center;padding:4px 8px;border-bottom:1px solid rgba(255,255,255,.1);">Score</th>
            <th style="text-align:center;padding:4px 8px;border-bottom:1px solid rgba(255,255,255,.1);" title="1/FWHM normalised — narrower peak = higher alignment precision">Sharpness</th>
            <th style="text-align:center;padding:4px 8px;border-bottom:1px solid rgba(255,255,255,.1);" title="(peak − bg) / bg_std">SNR</th>
            <th style="text-align:center;padding:4px 8px;border-bottom:1px solid rgba(255,255,255,.1);" title="peak / background mean">Contrast</th>
            <th style="text-align:center;padding:4px 8px;border-bottom:1px solid rgba(255,255,255,.1);" title="1/CV — lower spread = higher stability">Stability</th>
          </tr>
        </thead>
        <tbody>{ML_RANK_ROWS}</tbody>
      </table>
    </div>

    <!-- Recommended peak + outlier summary -->
    <div style="display:flex;gap:20px;flex-wrap:wrap;">
      <div style="flex:1;min-width:200px;background:rgba(134,239,172,.05);border:1px solid rgba(134,239,172,.15);border-radius:8px;padding:12px 16px;">
        <div style="font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;">
          Recommended for alignment
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:16px;color:#86efac;font-weight:700;">{ML_BEST_PEAK}</div>
        <div style="font-size:11px;color:var(--text-muted);margin-top:4px;">{ML_BEST_REASON}</div>
      </div>
      <div style="flex:1;min-width:200px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:12px 16px;">
        <div style="font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;">
          Outlier frames (MAD Z-score &gt; 3.5σ)
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:16px;color:{ML_BAD_COLOR};font-weight:700;">{ML_N_BAD} frames</div>
        <div style="font-size:11px;color:var(--text-muted);margin-top:4px;word-break:break-all;">{ML_BAD_LIST}</div>
      </div>
    </div>
  </div>

</div>

<!-- ══ Global JS ════════════════════════════════════════════════════════════ -->
<script>
// ── Tab switching ─────────────────────────────────────────────────────────
function switchTab(name) {
  ['roi','px','mr'].forEach(function(t) {
    var btn = document.getElementById('tab-'+t);
    var pan = document.getElementById('panel-'+t);
    if (!btn||!pan) return;
    if (t === name) {
      btn.classList.add('active');
      pan.classList.add('active');
      setTimeout(function(){
        var ids = {roi:'hs-plot', px:'px-plot', mr:'mr-plot'};
        var el  = document.getElementById(ids[t]);
        if (el && el._fullLayout) Plotly.Plots.resize(el);
      }, 60);
    } else {
      btn.classList.remove('active');
      pan.classList.remove('active');
    }
  });
}

// ── Image navigator ───────────────────────────────────────────────────────
var _navAllZ    = {ALL_Z_JSON};
var _navFolder  = {FOLDER_JSON};
// Navigator starts at the Max-Pixel ROI (brightest-pixel pad box)
var _navXMin    = {MR_X0};
var _navXMax    = {MR_X1};
var _navYMin    = {MR_Y0};
var _navYMax    = {MR_Y1};
var _navLogImg  = {NAV_LOGIMG};
var _navCmap    = "{NAV_CMAP}";
var _navVmin    = {CTRL_VMIN_JS};   // display-space vmin (after log if log_img)
var _navVmax    = {CTRL_VMAX_JS};   // display-space vmax
// _orig* = user's original manual ROI — used by ctrlReset() to restore
var _origXMin   = {NAV_XMIN};
var _origXMax   = {NAV_XMAX};
var _origYMin   = {NAV_YMIN};
var _origYMax   = {NAV_YMAX};
var _origVmin   = {CTRL_VMIN_JS};
var _origVmax   = {CTRL_VMAX_JS};
// Top-3 peak ROI boxes (pre-computed server-side)
var _mrPeaks = [
  {x0:{MR_X0},  x1:{MR_X1},  y0:{MR_Y0},  y1:{MR_Y1},  gx:{MR_GX},  gy:{MR_GY}},
  {x0:{MR2_X0}, x1:{MR2_X1}, y0:{MR2_Y0}, y1:{MR2_Y1}, gx:{MR2_GX}, gy:{MR2_GY}},
  {x0:{MR3_X0}, x1:{MR3_X1}, y0:{MR3_Y0}, y1:{MR3_Y1}, gx:{MR3_GX}, gy:{MR3_GY}}
];
function ctrlSetPeak(idx) {
  var pk = _mrPeaks[idx];
  document.getElementById('ctrl-xmin').value = pk.x0;
  document.getElementById('ctrl-xmax').value = pk.x1;
  document.getElementById('ctrl-ymin').value = pk.y0;
  document.getElementById('ctrl-ymax').value = pk.y1;
  document.getElementById('ctrl-info').textContent =
    'Navigator ROI set to Peak '+(idx+1)+' (x='+pk.x0+':'+pk.x1+', y='+pk.y0+':'+pk.y1+')';
  ctrlLiveUpdate();
}
var _navIdx     = _navAllZ.indexOf({Z_PREV_NUM});
if (_navIdx < 0) _navIdx = 0;

(function(){
  var sl = document.getElementById('nav-slider');
  if (sl) { sl.max = _navAllZ.length - 1; sl.value = _navIdx; }
}());

function navGoIdx(idx) {
  idx = Math.max(0, Math.min(_navAllZ.length - 1, idx));
  _navIdx = idx;
  var z = _navAllZ[idx];
  document.getElementById('nav-z-display').textContent = z;
  var inp = document.getElementById('nav-z-input');
  if (inp) inp.value = z;
  var sl = document.getElementById('nav-slider');
  if (sl) sl.value = idx;
  _navLoadZ(z);
}

function navGoZ(z) {
  var idx = _navAllZ.indexOf(z);
  if (idx < 0) {
    // find closest
    idx = 0;
    var best = Math.abs(_navAllZ[0] - z);
    for (var i = 1; i < _navAllZ.length; i++) {
      var d = Math.abs(_navAllZ[i] - z);
      if (d < best) { best = d; idx = i; }
    }
  }
  navGoIdx(idx);
}

function navStep(delta) { navGoIdx(_navIdx + delta); }

function _navLoadZ(z) {
  var usePlotly = document.getElementById('det-plotly-view').style.display !== 'none';
  var fd = new FormData();
  fd.append('folder', _navFolder);
  fd.append('z', z);
  fd.append('x_min', _navXMin);
  fd.append('x_max', _navXMax);
  fd.append('y_min', _navYMin);
  fd.append('y_max', _navYMax);
  fd.append('log_img', _navLogImg ? '1' : '0');
  fd.append('cmap', _navCmap);
  if (_navVmin !== null && _navVmin !== undefined && _navVmin !== '') fd.append('vmin', _navVmin);
  if (_navVmax !== null && _navVmax !== undefined && _navVmax !== '') fd.append('vmax', _navVmax);
  if (usePlotly) fd.append('mode', 'plotly');

  fetch('/preview_ajax', {method:'POST', body:fd})
    .then(function(r){ return r.json(); })
    .then(function(d){
      if (!d.ok) return;
      if (usePlotly && d.zdata) {
        _renderPlotlyHeatmap(d);
      } else if (d.b64) {
        document.getElementById('det-static-img').src = 'data:image/png;base64,'+d.b64;
      }
      if (d.roi_sum  !== undefined) document.getElementById('prev-roi-sum').textContent  = d.roi_sum;
      if (d.roi_mean !== undefined) document.getElementById('prev-roi-mean').textContent = d.roi_mean;
      if (d.roi_max  !== undefined) document.getElementById('prev-roi-max').textContent  = d.roi_max;
      document.getElementById('nav-roi-sum').textContent  = d.roi_sum  || '—';
      document.getElementById('nav-roi-mean').textContent = d.roi_mean || '—';
      document.getElementById('nav-roi-max').textContent  = d.roi_max  || '—';
    });
}

// ── Plotly heatmap view ───────────────────────────────────────────────────
var _plotlyLoaded = false;

function showPngView() {
  document.getElementById('det-png-view').style.display   = 'block';
  document.getElementById('det-plotly-view').style.display= 'none';
  document.getElementById('btn-png-view').classList.add('active');
  document.getElementById('btn-plotly-view').classList.remove('active');
}

function showPlotlyView() {
  document.getElementById('det-png-view').style.display   = 'none';
  document.getElementById('det-plotly-view').style.display= 'block';
  document.getElementById('btn-plotly-view').classList.add('active');
  document.getElementById('btn-png-view').classList.remove('active');
  if (!_plotlyLoaded) {
    document.getElementById('det-plotly-spinner').style.display = 'block';
    document.getElementById('det-plotly').style.display         = 'none';
    var z = _navAllZ[_navIdx];
    var fd = new FormData();
    fd.append('folder', _navFolder);
    fd.append('z', z);
    fd.append('x_min', _navXMin);
    fd.append('x_max', _navXMax);
    fd.append('y_min', _navYMin);
    fd.append('y_max', _navYMax);
    fd.append('log_img', _navLogImg ? '1' : '0');
    fd.append('cmap', _navCmap);
    fd.append('mode', 'plotly');
    fetch('/preview_ajax', {method:'POST', body:fd})
      .then(function(r){ return r.json(); })
      .then(function(d){
        document.getElementById('det-plotly-spinner').style.display = 'none';
        document.getElementById('det-plotly').style.display         = 'block';
        if (d.ok && d.zdata) { _renderPlotlyHeatmap(d); _plotlyLoaded = true; }
      });
  }
}

function _renderPlotlyHeatmap(d) {
  var step = d.step || 1;
  var xscale = [], yscale = [];
  var cols = d.zdata[0].length, rows = d.zdata.length;
  // Increasing axes: xscale[i] = i*step, yscale[j] = j*step
  // → zdata[j][i] sits at original pixel (col i*step, row j*step)
  // → y increases upward from 0 (bottom) = matplotlib origin='lower' convention
  for (var i=0; i<cols; i++) xscale.push(i * step);
  for (var j=0; j<rows; j++) yscale.push(j * step);

  var shapes = [];
  if (d.roi_shape) {
    var rs = d.roi_shape;
    // roi_shape is already in original pixel coordinates — use directly
    shapes.push({
      type:'rect', xref:'x', yref:'y',
      x0: rs.x0, x1: rs.x1,
      y0: rs.y0, y1: rs.y1,
      line:{color:'#FF4444', width:2, dash:'dash'},
      fillcolor:'rgba(255,68,68,0.06)'
    });
  }

  var colMap = {inferno:'Inferno', viridis:'Viridis', hot:'Hot', gray:'Greys'};
  var cs = colMap[_navCmap] || 'Inferno';

  var traceExtras = {};
  if (d.zmin !== null && d.zmin !== undefined) traceExtras.zmin = d.zmin;
  if (d.zmax !== null && d.zmax !== undefined) traceExtras.zmax = d.zmax;

  Plotly.react('det-plotly', [Object.assign({
    type:'heatmap', z:d.zdata, x:xscale, y:yscale,
    colorscale:cs, showscale:true,
    colorbar:{title:{text: d.log ? 'log₁₀(counts)':'counts', side:'right'},
              thickness:15, len:0.9},
    hovertemplate:'x=%{x}<br>y=%{y}<br>val=%{z:.3f}<extra></extra>'
  }, traceExtras)], {
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    xaxis:{title:'x  (col)', showgrid:false, zeroline:false},
    yaxis:{title:'y  (row)', showgrid:false, zeroline:false,
           scaleanchor:'x', scaleratio:1},
    shapes: shapes,
    margin:{t:30, r:80, b:55, l:60},
    font:{family:'IBM Plex Sans,Inter,sans-serif', size:11}
  }, {responsive:true, displaylogo:false, scrollZoom:true, displayModeBar:true,
      toImageButtonOptions:{format:'png', filename:'det_z'+_navAllZ[_navIdx], scale:2}});
}

// ── ROI & Display Controls ────────────────────────────────────────────────
var _ctrlUpdateTimer = null;

function toggleCtrl(){
  var body = document.getElementById('roi-ctrl-body');
  var icon = document.getElementById('ctrl-toggle-icon');
  var hidden = body.style.display === 'none';
  body.style.display = hidden ? 'block' : 'none';
  icon.textContent   = hidden ? '▾' : '▸';
}

function _readCtrl(){
  var xn = parseInt(document.getElementById('ctrl-xmin').value);
  var xx = parseInt(document.getElementById('ctrl-xmax').value);
  var yn = parseInt(document.getElementById('ctrl-ymin').value);
  var yx = parseInt(document.getElementById('ctrl-ymax').value);
  var vn = document.getElementById('ctrl-vmin').value.trim();
  var vx = document.getElementById('ctrl-vmax').value.trim();
  return {
    xmin: isNaN(xn) ? _origXMin : xn,
    xmax: isNaN(xx) ? _origXMax : xx,
    ymin: isNaN(yn) ? _origYMin : yn,
    ymax: isNaN(yx) ? _origYMax : yx,
    vmin: vn === '' ? null : parseFloat(vn),
    vmax: vx === '' ? null : parseFloat(vx)
  };
}

function ctrlLiveUpdate(){
  // Debounce: wait 400 ms after last keystroke before reloading
  clearTimeout(_ctrlUpdateTimer);
  _ctrlUpdateTimer = setTimeout(function(){
    var c = _readCtrl();
    _navXMin = c.xmin; _navXMax = c.xmax;
    _navYMin = c.ymin; _navYMax = c.ymax;
    _navVmin = c.vmin; _navVmax = c.vmax;
    _plotlyLoaded = false; // force Plotly re-render on next view switch
    _navLoadZ(_navAllZ[_navIdx]);
  }, 400);
}

function ctrlReset(){
  document.getElementById('ctrl-xmin').value = _origXMin;
  document.getElementById('ctrl-xmax').value = _origXMax;
  document.getElementById('ctrl-ymin').value = _origYMin;
  document.getElementById('ctrl-ymax').value = _origYMax;
  document.getElementById('ctrl-vmin').value = _origVmin !== null ? _origVmin : '';
  document.getElementById('ctrl-vmax').value = _origVmax !== null ? _origVmax : '';
  _navXMin = _origXMin; _navXMax = _origXMax;
  _navYMin = _origYMin; _navYMax = _origYMax;
  _navVmin = _origVmin; _navVmax = _origVmax;
  _plotlyLoaded = false;
  _navLoadZ(_navAllZ[_navIdx]);
  document.getElementById('ctrl-info').textContent = '↩ Reset to original scan values.';
}

function ctrlAutoROI(pct, method){
  var z = _navAllZ[_navIdx];
  var info = document.getElementById('ctrl-info');
  info.textContent = '⟳ Auto-detecting ROI from z='+z+'…';
  var fd = new FormData();
  fd.append('folder', _navFolder);
  fd.append('z', z);
  fd.append('percentile', pct || 99);
  fd.append('method', method || 'com');
  fetch('/auto_roi_ajax', {method:'POST', body:fd})
    .then(function(r){ return r.json(); })
    .then(function(d){
      if (d.ok) {
        document.getElementById('ctrl-xmin').value = d.x0;
        document.getElementById('ctrl-xmax').value = d.x1;
        document.getElementById('ctrl-ymin').value = d.y0;
        document.getElementById('ctrl-ymax').value = d.y1;
        info.textContent = '✅ '+method.toUpperCase()+' auto-ROI from z='+z+
          ' (threshold='+d.threshold+', n_hot='+d.n_hot+')  →  '+
          'x='+d.x0+':'+d.x1+', y='+d.y0+':'+d.y1;
        ctrlLiveUpdate();
      } else {
        info.textContent = '❌ ' + d.error;
      }
    });
}

function ctrlAutoVminVmax(){
  var z = _navAllZ[_navIdx];
  var info = document.getElementById('ctrl-info');
  info.textContent = '⟳ Computing percentile stats from z='+z+'…';
  var fd = new FormData();
  fd.append('folder', _navFolder);
  fd.append('z', z);
  fd.append('log_img', _navLogImg ? '1' : '0');
  fd.append('mode', 'stats');
  fetch('/preview_ajax', {method:'POST', body:fd})
    .then(function(r){ return r.json(); })
    .then(function(d){
      if (d.ok && d.p1 !== undefined) {
        document.getElementById('ctrl-vmin').value = d.p1.toFixed(3);
        document.getElementById('ctrl-vmax').value = d.p99.toFixed(3);
        info.textContent = '✅ vmin='+d.p1.toFixed(3)+' (1st pct) · vmax='+
          d.p99.toFixed(3)+' (99th pct)  —  '+(d.log ? 'log₁₀ space' : 'linear space');
        ctrlLiveUpdate();
      } else {
        info.textContent = '❌ ' + (d.error || 'failed');
      }
    });
}

// ── Keyboard nav ──────────────────────────────────────────────────────────
document.addEventListener('keydown', function(e){
  var focused = document.activeElement;
  var isInput = focused && (focused.tagName==='INPUT'||focused.tagName==='TEXTAREA'||focused.tagName==='SELECT');
  if (isInput) return;
  if (e.key==='ArrowLeft'  || e.key==='ArrowDown')  navStep(-1);
  if (e.key==='ArrowRight' || e.key==='ArrowUp')    navStep(1);
});
</script>
"""

TABLE_HTML = """
<div class="card">
  <div class="card-hdr">
    <div class="card-dot"></div>
    <span class="card-title">ROI Data Table</span>
  </div>
  <div style="max-height:300px;overflow-y:auto;">
  <table>
    <thead><tr>
      <th>z (image index)</th>
      <th>{MODE_CAP} Intensity (ROI)</th>
      <th>Filename</th>
    </tr></thead>
    <tbody>{ROWS}</tbody>
  </table>
  </div>
</div>
"""

MR_TABLE_HTML = """
<div class="card">
  <div class="card-hdr">
    <div class="card-dot" style="background:#f97316;box-shadow:0 0 8px #f97316;"></div>
    <span class="card-title">Max-ROI Data Table  ·  auto-ROI x={MR_X0}:{MR_X1}, y={MR_Y0}:{MR_Y1}</span>
  </div>
  <div style="max-height:300px;overflow-y:auto;">
  <table>
    <thead><tr>
      <th>z (image index)</th>
      <th>{MR_MODE_CAP} Intensity (auto-ROI)</th>
      <th>Filename</th>
    </tr></thead>
    <tbody>{MR_ROWS}</tbody>
  </table>
  </div>
</div>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    folder = request.args.get("folder", DEFAULT_PATH)
    indices, prefix, file_map = scan_folder(folder) if os.path.isdir(folder) else ([], "", {})

    z_min  = indices[0]  if indices else 0
    z_max  = indices[-1] if indices else 30
    z_prev = z_min

    if indices:
        file_info = (
            f'<div class="alert-info">✅ Found <b>{len(indices)}</b> CBF files &nbsp;|&nbsp; '
            f'Prefix: <span class="path-label">{prefix}</span> &nbsp;|&nbsp; '
            f'Index range: <b>{z_min} – {z_max}</b></div>'
        )
    elif folder and not os.path.isdir(folder):
        file_info = '<div class="alert-warn">⚠ Folder not found or not accessible.</div>'
    else:
        file_info = '<div class="alert-warn">⚠ No .cbf files found in this folder.</div>'

    warn = "" if FABIO_OK else '<div class="alert-error">⚠ <b>fabio</b> is not installed. Run <code>pip install fabio</code> and restart.</div>'

    html = (FORM_HTML
        .replace("{WARN}",        warn)
        .replace("{FOLDER}",      folder)
        .replace("{FOLDER_ENC}",  urlquote(folder, safe=""))
        .replace("{FILE_INFO}",   file_info)
        .replace("{Z_MIN}",       str(z_min))
        .replace("{Z_MAX}",       str(z_max))
        .replace("{Z_PREV}",      str(z_prev))
        .replace("{X_MIN}",       "1106")
        .replace("{X_MAX}",       "1309")
        .replace("{Y_MIN}",       "1511")
        .replace("{Y_MAX}",       "1742")
        .replace("{MAXROI_PAD1}", "50")
        .replace("{MAXROI_PAD2}", "50")
        .replace("{MAXROI_PAD3}", "50")
        .replace("{SEL_SUM}",     "selected")
        .replace("{SEL_MEAN}",    "")
        .replace("{SEL_MAX}",     "")
        .replace("{SEL_MED}",     "")
        .replace("{SEL_INF}",     "selected")
        .replace("{SEL_VIR}",     "")
        .replace("{SEL_HOT}",     "")
        .replace("{SEL_GRY}",     "")
        .replace("{LOG_CHK}",     "checked")
        .replace("{TBL_CHK}",     "")
    )
    return shell(html)


@app.route("/run", methods=["POST"])
def run():
    if not FABIO_OK:
        return shell('<div class="alert-error">fabio not installed.</div>')

    # ── Parse form ────────────────────────────────────────────────────────────
    folder      = request.form.get("folder",     DEFAULT_PATH).strip()
    z_min       = int(request.form.get("z_min",       0))
    z_max       = int(request.form.get("z_max",       30))
    z_prev      = int(request.form.get("z_preview",   z_min))
    x_min       = int(request.form.get("x_min",    1106))
    x_max       = int(request.form.get("x_max",    1309))
    y_min       = int(request.form.get("y_min",    1511))
    y_max       = int(request.form.get("y_max",    1742))
    mode        = request.form.get("mode",    "sum")
    cmap        = request.form.get("cmap",    "inferno")
    log_img     = bool(request.form.get("log_img"))
    show_tbl    = bool(request.form.get("show_table"))
    maxroi_pad  = max(1, int(request.form.get("maxroi_pad",  50)))
    maxroi_pad2 = max(1, int(request.form.get("maxroi_pad2", 50)))
    maxroi_pad3 = max(1, int(request.form.get("maxroi_pad3", 50)))

    # ── Scan folder ───────────────────────────────────────────────────────────
    indices, prefix, file_map = scan_folder(folder)
    if not indices:
        err = f'<div class="alert-error">No .cbf files found in <span class="path-label">{folder}</span></div>'
        return shell(err + f'<br><a href="/?folder={urlquote(folder,safe="")}">← Back</a>')

    selected = sorted(i for i in indices if z_min <= i <= z_max)
    if not selected:
        return shell('<div class="alert-error">No images found in the selected z range.</div><br><a href="/">← Back</a>')

    if z_prev not in file_map:
        z_prev = selected[0]

    # ── Load all images in parallel, cache in memory ─────────────────────────
    # ThreadPoolExecutor parallelises NFS open/read latency.  Images are cached
    # so every pixel is read exactly once across both ROI-integration passes.
    t0 = time.perf_counter()
    z_vals, i_vals, fnames = [], [], []
    px_max_vals, px_x_vals, px_y_vals, px_mean_vals, px_std_vals = [], [], [], [], []
    errors = []
    accum_img = None

    def _load_one(z):
        try:
            return z, load_cbf(file_map[z]), None
        except Exception as exc:
            return z, None, str(exc)

    # Load all selected frames in parallel (8 workers saturates typical NFS)
    _n_workers = min(8, len(selected))
    img_cache  = {}   # z → float32 ndarray  (freed after Pass 2 below)
    with ThreadPoolExecutor(max_workers=_n_workers) as _ex:
        for _z, _img, _err in _ex.map(_load_one, selected):
            if _err:
                errors.append(f"z={_z}: {_err}")
            else:
                img_cache[_z] = _img

    # ── Pass 1: ROI integration + pixel inspector stats (in-memory) ───────────
    for z in selected:
        if z not in img_cache:
            continue
        try:
            img   = img_cache[z]
            fname = os.path.basename(file_map[z])
            val   = roi_value(img, x_min, x_max, y_min, y_max, mode)
            z_vals.append(z)
            i_vals.append(val)
            fnames.append(fname)
            # Global max pixel stats
            flat_max = float(img.max())
            max_pos  = np.unravel_index(img.argmax(), img.shape)  # (row, col) = (y, x)
            px_max_vals.append(flat_max)
            px_x_vals.append(int(max_pos[1]))   # col → x
            px_y_vals.append(int(max_pos[0]))   # row → y
            px_mean_vals.append(float(img.mean()))
            px_std_vals.append(float(img.std()))
            # Accumulate element-wise max (purely in-memory — no extra I/O)
            if accum_img is None:
                accum_img = img.astype(np.float64).copy()
            else:
                np.maximum(accum_img, img, out=accum_img)
        except Exception as exc:
            errors.append(f"z={z}: {exc}")

    if not z_vals:
        return shell('<div class="alert-error">All images failed to load.</div><br><a href="/">← Back</a>')

    # ── Determine Top-3 spatially distinct peak locations ─────────────────────
    h_img, w_img = 0, 0
    if accum_img is not None:
        h_img, w_img = accum_img.shape
    else:
        try:
            _sample = load_cbf(file_map[z_vals[0]])
            h_img, w_img = _sample.shape
        except Exception:
            pass

    def _build_roi(pk, pad=None):
        """Clip pad box around peak to image bounds.  pad defaults to Peak-1 pad."""
        if pad is None:
            pad = maxroi_pad
        px, py = pk['x'], pk['y']
        x0 = max(0, px - pad)
        x1 = min(w_img - 1, px + pad) if w_img > 0 else px + pad
        y0 = max(0, py - pad)
        y1 = min(h_img - 1, py + pad) if h_img > 0 else py + pad
        return x0, x1, y0, y1

    # Use element-wise max image + iterative exact ROI-box masking so that
    # Peak 2 and Peak 3 are guaranteed to be outside Peak 1's (and Peak 2's)
    # actual ROI box — not just within some padding radius.
    mr_peaks = [{'x': 0, 'y': 0, 'value': 0}] * 3
    if accum_img is not None and h_img > 0 and w_img > 0:
        work = accum_img.copy()
        # Mask permanent detector dead zones — columns (x) and rows (y)
        for dc0, dc1 in DETECTOR_DEAD_COLS:
            work[:, max(0, dc0):min(w_img, dc1 + 1)] = 0
        for dr0, dr1 in DETECTOR_DEAD_ROWS:
            work[max(0, dr0):min(h_img, dr1 + 1), :] = 0
        found = []
        for pad_i in [maxroi_pad, maxroi_pad2, maxroi_pad3]:
            if work.max() <= 0:
                break
            row, col = np.unravel_index(work.argmax(), work.shape)
            pk = {'x': int(col), 'y': int(row), 'value': int(accum_img[row, col])}
            found.append(pk)
            # Zero out this peak's exact ROI box (sized by its own pad) so the
            # next search is forced into a completely different spatial region
            rx0, rx1, ry0, ry1 = _build_roi(pk, pad_i)
            work[ry0:ry1 + 1, rx0:rx1 + 1] = 0
        while len(found) < 3:
            found.append({'x': 0, 'y': 0, 'value': 0})
        mr_peaks = found

    # Peak 1 (brightest) — uses its own pad
    mr_gmax_x, mr_gmax_y = mr_peaks[0]['x'], mr_peaks[0]['y']
    mr_x0, mr_x1, mr_y0, mr_y1 = _build_roi(mr_peaks[0], maxroi_pad)
    # Peak 2 (2nd brightest) — uses its own pad
    mr2_gmax_x, mr2_gmax_y, mr2_gmax_v = mr_peaks[1]['x'], mr_peaks[1]['y'], mr_peaks[1]['value']
    mr2_x0, mr2_x1, mr2_y0, mr2_y1 = _build_roi(mr_peaks[1], maxroi_pad2)
    # Peak 3 (3rd brightest) — uses its own pad
    mr3_gmax_x, mr3_gmax_y, mr3_gmax_v = mr_peaks[2]['x'], mr_peaks[2]['y'], mr_peaks[2]['value']
    mr3_x0, mr3_x1, mr3_y0, mr3_y1 = _build_roi(mr_peaks[2], maxroi_pad3)

    # ── Pass 2: integrate all 3 peak ROIs (from in-memory cache — no disk I/O) ──
    mr_i_vals  = []   # peak 1
    mr2_i_vals = []   # peak 2
    mr3_i_vals = []   # peak 3
    for z in z_vals:
        try:
            img = img_cache[z]   # already in RAM — no file open/read
            mr_i_vals.append( roi_value(img, mr_x0,  mr_x1,  mr_y0,  mr_y1,  mode))
            mr2_i_vals.append(roi_value(img, mr2_x0, mr2_x1, mr2_y0, mr2_y1, mode))
            mr3_i_vals.append(roi_value(img, mr3_x0, mr3_x1, mr3_y0, mr3_y1, mode))
        except Exception as exc:
            mr_i_vals.append(0.0);  mr2_i_vals.append(0.0);  mr3_i_vals.append(0.0)
            errors.append(f"MaxROI z={z}: {exc}")

    del img_cache   # free memory now that both passes are done

    elapsed = round(time.perf_counter() - t0, 2)

    # ── Detector preview image ────────────────────────────────────────────────
    ctrl_vmin = None   # display-space (post-log) 1st percentile of preview image
    ctrl_vmax = None   # display-space (post-log) 99th percentile of preview image
    try:
        prev_img    = load_cbf(file_map[z_prev])
        img_shape   = f"{prev_img.shape[1]} × {prev_img.shape[0]}"

        # Compute initial vmin/vmax from full image percentiles in display space
        _disp = np.log10(np.maximum(prev_img, 1)) if log_img else prev_img.astype(float)
        ctrl_vmin = round(float(np.percentile(_disp,  1)), 4)
        ctrl_vmax = round(float(np.percentile(_disp, 99)), 4)

        img_b64     = img_to_b64(prev_img, roi=(x_min, x_max, y_min, y_max),
                                 log_scale=log_img, cmap=cmap,
                                 vmin=ctrl_vmin, vmax=ctrl_vmax)
        roi_patch   = prev_img[y_min:y_max + 1, x_min:x_max + 1]
        roi_sum     = f"{roi_patch.sum():.3e}"
        roi_mean    = f"{roi_patch.mean():.2f}"
        roi_max     = f"{int(roi_patch.max())}"
        gmax_pos    = np.unravel_index(prev_img.argmax(), prev_img.shape)
        prev_gmax   = f"{int(prev_img.max())}"
        prev_gmax_x = str(int(gmax_pos[1]))
        prev_gmax_y = str(int(gmax_pos[0]))
    except Exception as exc:
        img_b64 = ""
        img_shape = "?"
        roi_sum = roi_mean = roi_max = "—"
        prev_gmax = prev_gmax_x = prev_gmax_y = "—"
        errors.append(f"Preview: {exc}")

    # Serialise vmin/vmax for JS (null if not computed) and HTML inputs (empty = auto)
    ctrl_vmin_js   = str(ctrl_vmin) if ctrl_vmin is not None else "null"
    ctrl_vmax_js   = str(ctrl_vmax) if ctrl_vmax is not None else "null"
    ctrl_vmin_html = str(ctrl_vmin) if ctrl_vmin is not None else ""
    ctrl_vmax_html = str(ctrl_vmax) if ctrl_vmax is not None else ""

    # ── ROI peak ──────────────────────────────────────────────────────────────
    peak_idx = int(np.argmax(i_vals))
    peak_z   = z_vals[peak_idx]
    peak_i   = f"{i_vals[peak_idx]:.4e}"

    # ── Pixel inspector peak ──────────────────────────────────────────────────
    px_peak_idx = int(np.argmax(px_max_vals))
    px_peak_z   = z_vals[px_peak_idx]
    px_peak_val = f"{int(px_max_vals[px_peak_idx])}"

    # ── Max-ROI peaks (all 3) ─────────────────────────────────────────────────
    mr_peak_idx  = int(np.argmax(mr_i_vals))  if mr_i_vals  else 0
    mr_peak_z    = z_vals[mr_peak_idx]  if z_vals else 0
    mr_peak_i    = f"{mr_i_vals[mr_peak_idx]:.4e}"  if mr_i_vals  else "—"

    mr2_peak_idx = int(np.argmax(mr2_i_vals)) if mr2_i_vals else 0
    mr2_peak_z   = z_vals[mr2_peak_idx] if z_vals else 0
    mr2_peak_i   = f"{mr2_i_vals[mr2_peak_idx]:.4e}" if mr2_i_vals else "—"

    mr3_peak_idx = int(np.argmax(mr3_i_vals)) if mr3_i_vals else 0
    mr3_peak_z   = z_vals[mr3_peak_idx] if z_vals else 0
    mr3_peak_i   = f"{mr3_i_vals[mr3_peak_idx]:.4e}" if mr3_i_vals else "—"

    # ── Peak statistics (FWHM, SNR, centroid z, contrast, …) ─────────────────
    mr_stats  = compute_peak_stats(z_vals, mr_i_vals)
    mr2_stats = compute_peak_stats(z_vals, mr2_i_vals)
    mr3_stats = compute_peak_stats(z_vals, mr3_i_vals)

    # ── Machine Learning: GPR fit + outlier detection ─────────────────────────
    mr_gpr  = fit_gpr_profile(z_vals, mr_i_vals)
    mr2_gpr = fit_gpr_profile(z_vals, mr2_i_vals)
    mr3_gpr = fit_gpr_profile(z_vals, mr3_i_vals)

    mr_bad  = detect_outlier_frames(z_vals, mr_i_vals)
    mr2_bad = detect_outlier_frames(z_vals, mr2_i_vals)
    mr3_bad = detect_outlier_frames(z_vals, mr3_i_vals)
    # Combined unique bad frames across all 3 ROIs
    all_bad = sorted(set(mr_bad) | set(mr2_bad) | set(mr3_bad))

    # ── ML Insights: composite quality score ─────────────────────────────────
    # Weights: Sharpness (1/FWHM) 40% · SNR 30% · Contrast 20% · Stability 10%
    # Each component is normalised relative to the best among the 3 peaks (0–100).
    def _raw(stats, key):
        try:
            v = float(stats[key])
            # Clamp inf / nan: a dead-zone peak (all-zero ROI) produces
            # bg_std=0 → SNR=inf and bg_mean=0 → contrast=inf.
            # Treat these as 0.0 so _norm3 never sees inf/nan.
            if v != v or abs(v) >= 1e15:
                return 0.0
            return v
        except:
            return 0.0

    def _norm3(vals):
        """Min-max normalise a list of 3 finite floats to [0, 100]."""
        # Safety: clamp any inf/nan that somehow slipped through
        safe = [v if (v == v and abs(v) < 1e15) else 0.0 for v in vals]
        lo, hi = min(safe), max(safe)
        if hi <= lo:
            return [50.0, 50.0, 50.0]
        return [round((v - lo) / (hi - lo) * 100, 1) for v in safe]

    # Raw component values
    snr_raw   = [_raw(s, 'snr')      for s in (mr_stats, mr2_stats, mr3_stats)]
    cntr_raw  = [_raw(s, 'contrast') for s in (mr_stats, mr2_stats, mr3_stats)]
    cv_raw    = [_raw(s, 'cv')       for s in (mr_stats, mr2_stats, mr3_stats)]
    fwhm_raw  = [_raw(s, 'fwhm')     for s in (mr_stats, mr2_stats, mr3_stats)]

    # For sharpness: 1/FWHM — larger is better; protect against 0 / N/A
    # (stats['fwhm'] is a formatted string; _raw returns 0.0 when it is 'N/A')
    sharp_raw = [1.0 / f if f > 0 else 0.0 for f in fwhm_raw]

    # Stability: lower CV is better → invert
    stab_raw  = [1.0 / (cv + 1e-6) for cv in cv_raw]

    # Normalise each component across the 3 peaks
    sharp_n  = _norm3(sharp_raw)   # sharpness (1/FWHM)
    snr_n    = _norm3(snr_raw)
    cntr_n   = _norm3(cntr_raw)
    stab_n   = _norm3(stab_raw)

    # Weighted composite (sharpness leads — best for height alignment)
    W_SHARP, W_SNR, W_CNTR, W_STAB = 0.40, 0.30, 0.20, 0.10
    qs = [round(W_SHARP * sharp_n[i] + W_SNR * snr_n[i]
                + W_CNTR * cntr_n[i] + W_STAB * stab_n[i], 1)
          for i in range(3)]

    # Guard against nan in argmax (shouldn't happen after _raw fix, but just in case)
    _qs_safe = [q if (q == q and abs(q) < 1e15) else 0.0 for q in qs]
    best_pk = int(np.argmax(_qs_safe))
    _pk_labels  = ['🥇 Peak 1', '🥈 Peak 2', '🥉 Peak 3']
    _pk_colors  = ['#f97316',   '#a855f7',   '#22d3ee']

    # Safe integer formatter for score values (guards against any residual nan)
    def _si(v):
        try:
            f = float(v)
            return "N/A" if (f != f or abs(f) >= 1e15) else f"{f:.0f}"
        except Exception:
            return "N/A"

    ml_best_peak   = _pk_labels[best_pk]
    ml_best_reason = (
        f"Score {_si(qs[best_pk])}/100  ·  "
        f"Sharpness {_si(sharp_n[best_pk])}  ·  "
        f"SNR {_si(snr_n[best_pk])}  ·  "
        f"Contrast {_si(cntr_n[best_pk])}  ·  "
        f"Stability {_si(stab_n[best_pk])}"
    )

    # Build ranking table rows (sorted best → worst)
    def _bar(pct, color):
        """Inline mini progress bar."""
        try:
            f = float(pct)
            w = 0.0 if (f != f or abs(f) >= 1e15) else min(100.0, max(0.0, f))
        except Exception:
            w = 0.0
        return (f'<div style="background:rgba(255,255,255,.08);border-radius:3px;'
                f'height:6px;width:60px;display:inline-block;vertical-align:middle;">'
                f'<div style="background:{color};width:{w:.0f}%;height:100%;'
                f'border-radius:3px;"></div></div>')

    _rank_order = sorted(range(3), key=lambda i: -qs[i])
    _rank_medals = ['🏆', '🥈', '🥉']
    _peak_names  = ['Peak 1', 'Peak 2', 'Peak 3']

    rank_rows_html = ""
    for pos, pi in enumerate(_rank_order):
        color   = _pk_colors[pi]
        is_best = (pi == best_pk)
        bg      = "rgba(134,239,172,.06)" if is_best else "transparent"
        medal   = _rank_medals[pos]
        fwhm_disp = mr_stats['fwhm'] if pi == 0 else (mr2_stats['fwhm'] if pi == 1 else mr3_stats['fwhm'])
        rank_rows_html += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:5px 8px;color:{color};font-weight:600;">'
            f'{medal} {_peak_names[pi]}</td>'
            f'<td style="padding:5px 8px;text-align:center;font-family:IBM Plex Mono,monospace;'
            f'font-weight:700;color:{"#86efac" if is_best else "var(--text-main)"};">'
            f'{_si(qs[pi])}</td>'
            f'<td style="padding:5px 8px;text-align:center;">'
            f'{_bar(sharp_n[pi], color)}'
            f'<span style="font-size:10px;color:var(--text-muted);margin-left:4px;">'
            f'{_si(sharp_n[pi])}</span>'
            f'<div style="font-size:10px;color:var(--text-muted);">FWHM={fwhm_disp}</div></td>'
            f'<td style="padding:5px 8px;text-align:center;">'
            f'{_bar(snr_n[pi], color)}'
            f'<span style="font-size:10px;color:var(--text-muted);margin-left:4px;">{_si(snr_n[pi])}</span></td>'
            f'<td style="padding:5px 8px;text-align:center;">'
            f'{_bar(cntr_n[pi], color)}'
            f'<span style="font-size:10px;color:var(--text-muted);margin-left:4px;">{_si(cntr_n[pi])}</span></td>'
            f'<td style="padding:5px 8px;text-align:center;">'
            f'{_bar(stab_n[pi], color)}'
            f'<span style="font-size:10px;color:var(--text-muted);margin-left:4px;">{_si(stab_n[pi])}</span></td>'
            f'</tr>'
        )

    n_bad_total  = len(all_bad)
    ml_bad_color = "#4ade80" if n_bad_total == 0 else "#f87171"
    if n_bad_total == 0:
        ml_bad_list = "No outlier frames detected"
    else:
        shown = all_bad[:12]
        ml_bad_list = "z = " + ", ".join(str(z) for z in shown)
        if len(all_bad) > 12:
            ml_bad_list += f" … (+{len(all_bad)-12} more)"

    ml_status = "scikit-learn ✓  · GPR enabled" if SKLEARN_OK else \
                "scikit-learn not installed · GPR disabled (outlier detection still active)"

    # Helper: serialise a GPR result to JS-ready values (fallback = nulls/empty)
    def _gpr_js(g):
        if g is None:
            return dict(ok="false", zv="[]", iv="[]", up="[]", lo="[]",
                        optz="0", optzs="N/A", fwhm="N/A")
        return dict(
            ok   ="true",
            zv   =json.dumps(g['z_fit']),
            iv   =json.dumps(g['i_fit']),
            up   =json.dumps(g['i_upper']),
            lo   =json.dumps(g['i_lower']),
            optz =str(round(g['opt_z'], 4)),
            optzs=g['opt_z_str'],
            fwhm =g['gpr_fwhm'],
        )

    g1, g2, g3 = _gpr_js(mr_gpr), _gpr_js(mr2_gpr), _gpr_js(mr3_gpr)

    # ── Pixel inspector table rows ────────────────────────────────────────────
    px_rows_html = "\n".join(
        f'<tr>'
        f'<td style="font-family:IBM Plex Mono,monospace;">{z}</td>'
        f'<td style="font-family:IBM Plex Mono,monospace;color:#4ade80;">{int(mv)}</td>'
        f'<td style="font-family:IBM Plex Mono,monospace;">{xp}</td>'
        f'<td style="font-family:IBM Plex Mono,monospace;">{yp}</td>'
        f'<td style="font-family:IBM Plex Mono,monospace;">{mn:.2f}</td>'
        f'<td style="font-family:IBM Plex Mono,monospace;">{sd:.2f}</td>'
        f'<td style="font-size:11px;color:var(--text-muted);">{fn}</td>'
        f'</tr>'
        for z, mv, xp, yp, mn, sd, fn in zip(
            z_vals, px_max_vals, px_x_vals, px_y_vals,
            px_mean_vals, px_std_vals, fnames
        )
    )

    # ── ROI data table (optional) ─────────────────────────────────────────────
    table_block = ""
    if show_tbl:
        rows_html = "\n".join(
            f'<tr><td style="font-family:IBM Plex Mono,monospace;">{z}</td>'
            f'<td style="font-family:IBM Plex Mono,monospace;">{iv:.6e}</td>'
            f'<td style="font-size:11px;color:var(--text-muted);">{fn}</td></tr>'
            for z, iv, fn in zip(z_vals, i_vals, fnames)
        )
        table_block = (TABLE_HTML
            .replace("{MODE_CAP}", mode.title())
            .replace("{ROWS}",     rows_html)
        )

    # ── Max-ROI table rows for all 3 peaks ────────────────────────────────────
    def _mr_rows(i_list, color):
        return "\n".join(
            f'<tr><td style="font-family:IBM Plex Mono,monospace;">{z}</td>'
            f'<td style="font-family:IBM Plex Mono,monospace;color:{color};">{iv:.6e}</td>'
            f'<td style="font-size:11px;color:var(--text-muted);">{fn}</td></tr>'
            for z, iv, fn in zip(z_vals, i_list, fnames)
        )

    mr_rows_html  = _mr_rows(mr_i_vals,  "#f97316")
    mr2_rows_html = _mr_rows(mr2_i_vals, "#a855f7")
    mr3_rows_html = _mr_rows(mr3_i_vals, "#22d3ee")

    # Keep building the legacy MR_TABLE_BLOCK (Peak 1) for the replace chain
    mr_table_block = (MR_TABLE_HTML
        .replace("{MR_MODE_CAP}", mode.title())
        .replace("{MR_X0}",       str(mr_x0))
        .replace("{MR_X1}",       str(mr_x1))
        .replace("{MR_Y0}",       str(mr_y0))
        .replace("{MR_Y1}",       str(mr_y1))
        .replace("{MR_ROWS}",     mr_rows_html)
    )

    # ── Error block ───────────────────────────────────────────────────────────
    if errors:
        table_block += (
            '<div class="card"><div class="card-hdr"><div class="card-dot" '
            'style="background:#f97373;"></div><span class="card-title">Warnings</span></div>'
            '<pre>' + "\n".join(errors) + '</pre></div>'
        )

    # ── Build results page ─────────────────────────────────────────────────────
    # IMPORTANT: Replace longer/more-specific placeholders BEFORE shorter ones
    # to avoid partial-match corruption (e.g. {Z_JSON_ESC} before {Z_JSON}).
    html = (RESULTS_HTML
        # ── Meta / shared ──
        .replace("{PREFIX}",     prefix)
        .replace("{N_IMGS}",     str(len(z_vals)))
        .replace("{ELAPSED}",    str(elapsed))
        .replace("{Z_MIN}",      str(z_min))
        .replace("{Z_MAX}",      str(z_max))
        .replace("{X_MIN}",      str(x_min))
        .replace("{X_MAX}",      str(x_max))
        .replace("{Y_MIN}",      str(y_min))
        .replace("{Y_MAX}",      str(y_max))
        .replace("{MODE_CAP}",   mode.title())
        .replace("{MODE}",       mode)

        # ── Detector preview ──
        .replace("{IMG_B64}",       img_b64)
        .replace("{IMG_SHAPE}",     img_shape)
        .replace("{Z_PREV}",        str(z_prev))
        .replace("{Z_PREV_NUM}",    str(z_prev))
        .replace("{PREV_FILE}",     os.path.basename(file_map.get(z_prev, "")))
        .replace("{ROI_SUM}",       roi_sum)
        .replace("{ROI_MEAN}",      roi_mean)
        .replace("{ROI_MAX_V}",     roi_max)
        .replace("{PREV_GMAX}",     prev_gmax)
        .replace("{PREV_GMAX_X}",   prev_gmax_x)
        .replace("{PREV_GMAX_Y}",   prev_gmax_y)
        .replace("{ALL_Z_JSON}",    json.dumps(z_vals))
        .replace("{FOLDER_JSON}",   json.dumps(folder))
        .replace("{NAV_XMIN}",      str(x_min))
        .replace("{NAV_XMAX}",      str(x_max))
        .replace("{NAV_YMIN}",      str(y_min))
        .replace("{NAV_YMAX}",      str(y_max))
        .replace("{NAV_LOGIMG}",    "true" if log_img else "false")
        .replace("{NAV_CMAP}",      cmap)

        # ── Live ROI & Display Controls vmin/vmax ──
        .replace("{CTRL_VMIN_JS}",   ctrl_vmin_js)
        .replace("{CTRL_VMAX_JS}",   ctrl_vmax_js)
        .replace("{CTRL_VMIN_HTML}", ctrl_vmin_html)
        .replace("{CTRL_VMAX_HTML}", ctrl_vmax_html)

        # ── ROI tab — replace ESC variants FIRST ──
        .replace("{Z_JSON_ESC}",    json.dumps(z_vals).replace('"', "&quot;"))
        .replace("{I_JSON_ESC}",    json.dumps([round(v, 6) for v in i_vals]).replace('"', "&quot;"))
        .replace("{Z_JSON}",        json.dumps(z_vals))
        .replace("{I_JSON}",        json.dumps([round(v, 6) for v in i_vals]))
        .replace("{PEAK_Z_VAL}",    str(peak_z))
        .replace("{PEAK_I}",        peak_i)
        .replace("{TABLE_BLOCK}",   table_block)

        # ── Pixel inspector — replace ESC variants FIRST ──
        .replace("{PX_ZV_ESC}",     json.dumps(z_vals).replace('"', "&quot;"))
        .replace("{PX_MV_ESC}",     json.dumps([int(v) for v in px_max_vals]).replace('"', "&quot;"))
        .replace("{PX_ZV}",         json.dumps(z_vals))
        .replace("{PX_MV}",         json.dumps([int(v) for v in px_max_vals]))
        .replace("{PX_XV}",         json.dumps(px_x_vals))
        .replace("{PX_YV}",         json.dumps(px_y_vals))
        .replace("{PX_TROWS}",      px_rows_html)
        .replace("{PX_PKZN}",       str(px_peak_z))
        .replace("{PX_PKZS}",       str(px_peak_z))
        .replace("{PX_PKVAL}",      px_peak_val)
        .replace("{PX_PFX}",        prefix)

        # ── Max-ROI tab — ESC variants FIRST, longer prefixes BEFORE shorter ──
        # Peak 1
        .replace("{MR_ZV_ESC}",      json.dumps(z_vals).replace('"', "&quot;"))
        .replace("{MR_IV_ESC}",      json.dumps([round(v,6) for v in mr_i_vals]).replace('"', "&quot;"))
        .replace("{MR_ZV}",          json.dumps(z_vals))
        .replace("{MR_IV}",          json.dumps([round(v,6) for v in mr_i_vals]))
        .replace("{MR_PKZN}",        str(mr_peak_z))
        .replace("{MR_PKZS}",        str(mr_peak_z))
        .replace("{MR_PKIS}",        mr_peak_i)
        .replace("{MR_GX}",          str(mr_gmax_x))
        .replace("{MR_GY}",          str(mr_gmax_y))
        .replace("{MR_PAD}",         str(maxroi_pad))
        .replace("{MR2_PAD}",        str(maxroi_pad2))
        .replace("{MR3_PAD}",        str(maxroi_pad3))
        .replace("{MR_X0}",          str(mr_x0))
        .replace("{MR_X1}",          str(mr_x1))
        .replace("{MR_Y0}",          str(mr_y0))
        .replace("{MR_Y1}",          str(mr_y1))
        .replace("{MR_MODE}",        mode)
        .replace("{MR_PFX}",         prefix)
        .replace("{MR_ROWS}",        mr_rows_html)
        .replace("{MR_TABLE_BLOCK}", mr_table_block)

        # Peak 2 — ESC variants first
        .replace("{MR2_ZV_ESC}",     json.dumps(z_vals).replace('"', "&quot;"))
        .replace("{MR2_IV_ESC}",     json.dumps([round(v,6) for v in mr2_i_vals]).replace('"', "&quot;"))
        .replace("{MR2_ZV}",         json.dumps(z_vals))
        .replace("{MR2_IV}",         json.dumps([round(v,6) for v in mr2_i_vals]))
        .replace("{MR2_PKZN}",       str(mr2_peak_z))
        .replace("{MR2_PKZS}",       str(mr2_peak_z))
        .replace("{MR2_PKIS}",       mr2_peak_i)
        .replace("{MR2_GX}",         str(mr2_gmax_x))
        .replace("{MR2_GY}",         str(mr2_gmax_y))
        .replace("{MR2_X0}",         str(mr2_x0))
        .replace("{MR2_X1}",         str(mr2_x1))
        .replace("{MR2_Y0}",         str(mr2_y0))
        .replace("{MR2_Y1}",         str(mr2_y1))
        .replace("{MR2_ROWS}",       mr2_rows_html)

        # Peak 3 — ESC variants first
        .replace("{MR3_ZV_ESC}",     json.dumps(z_vals).replace('"', "&quot;"))
        .replace("{MR3_IV_ESC}",     json.dumps([round(v,6) for v in mr3_i_vals]).replace('"', "&quot;"))
        .replace("{MR3_ZV}",         json.dumps(z_vals))
        .replace("{MR3_IV}",         json.dumps([round(v,6) for v in mr3_i_vals]))
        .replace("{MR3_PKZN}",       str(mr3_peak_z))
        .replace("{MR3_PKZS}",       str(mr3_peak_z))
        .replace("{MR3_PKIS}",       mr3_peak_i)
        .replace("{MR3_GX}",         str(mr3_gmax_x))
        .replace("{MR3_GY}",         str(mr3_gmax_y))
        .replace("{MR3_X0}",         str(mr3_x0))
        .replace("{MR3_X1}",         str(mr3_x1))
        .replace("{MR3_Y0}",         str(mr3_y0))
        .replace("{MR3_Y1}",         str(mr3_y1))
        .replace("{MR3_ROWS}",       mr3_rows_html)

        # ── Peak stats panels ───────────────────────────────────────────────
        # SNR/contrast CSS class: hi (green) if value > threshold, lo (red) if < 3
        # Helper: classify a numeric-string stat value
    )

    def _snr_cls(v_str):
        try:
            v = float(v_str)
            return "hi" if v >= 10 else ("lo" if v < 3 else "")
        except Exception:
            return ""

    def _cntr_cls(v_str):
        try:
            v = float(v_str)
            return "hi" if v >= 5 else ("lo" if v < 2 else "")
        except Exception:
            return ""

    html = (html
        # Peak 1 stats
        .replace("{MR_S_PKZN}",      mr_stats['peak_z'])
        .replace("{MR_S_CNTZ}",      mr_stats['centroid_z'])
        .replace("{MR_S_FWHM}",      mr_stats['fwhm'])
        .replace("{MR_S_SNR_CLS}",   _snr_cls(mr_stats['snr']))
        .replace("{MR_S_SNR}",       mr_stats['snr'])
        .replace("{MR_S_CNTR_CLS}",  _cntr_cls(mr_stats['contrast']))
        .replace("{MR_S_CNTR}",      mr_stats['contrast'])
        .replace("{MR_S_MEAN}",      mr_stats['mean'])
        .replace("{MR_S_STD}",       mr_stats['std'])
        .replace("{MR_S_CV}",        mr_stats['cv'])
        .replace("{MR_S_AREA}",      mr_stats['area'])
        # Peak 2 stats
        .replace("{MR2_S_PKZN}",     mr2_stats['peak_z'])
        .replace("{MR2_S_CNTZ}",     mr2_stats['centroid_z'])
        .replace("{MR2_S_FWHM}",     mr2_stats['fwhm'])
        .replace("{MR2_S_SNR_CLS}",  _snr_cls(mr2_stats['snr']))
        .replace("{MR2_S_SNR}",      mr2_stats['snr'])
        .replace("{MR2_S_CNTR_CLS}", _cntr_cls(mr2_stats['contrast']))
        .replace("{MR2_S_CNTR}",     mr2_stats['contrast'])
        .replace("{MR2_S_MEAN}",     mr2_stats['mean'])
        .replace("{MR2_S_STD}",      mr2_stats['std'])
        .replace("{MR2_S_CV}",       mr2_stats['cv'])
        .replace("{MR2_S_AREA}",     mr2_stats['area'])
        # Peak 3 stats
        .replace("{MR3_S_PKZN}",     mr3_stats['peak_z'])
        .replace("{MR3_S_CNTZ}",     mr3_stats['centroid_z'])
        .replace("{MR3_S_FWHM}",     mr3_stats['fwhm'])
        .replace("{MR3_S_SNR_CLS}",  _snr_cls(mr3_stats['snr']))
        .replace("{MR3_S_SNR}",      mr3_stats['snr'])
        .replace("{MR3_S_CNTR_CLS}", _cntr_cls(mr3_stats['contrast']))
        .replace("{MR3_S_CNTR}",     mr3_stats['contrast'])
        .replace("{MR3_S_MEAN}",     mr3_stats['mean'])
        .replace("{MR3_S_STD}",      mr3_stats['std'])
        .replace("{MR3_S_CV}",       mr3_stats['cv'])
        .replace("{MR3_S_AREA}",     mr3_stats['area'])
    )

    # ── ML placeholders — longer suffixes (OPTZS) before shorter (OPTZ) ──────
    html = (html
        # Peak 1 GPR
        .replace("{MR_GPR_OPTZS}",   g1['optzs'])
        .replace("{MR_GPR_OPTZ}",    g1['optz'])
        .replace("{MR_GPR_FWHM}",    g1['fwhm'])
        .replace("{MR_GPR_OK}",      g1['ok'])
        .replace("{MR_GPR_ZV}",      g1['zv'])
        .replace("{MR_GPR_IV}",      g1['iv'])
        .replace("{MR_GPR_UP}",      g1['up'])
        .replace("{MR_GPR_LO}",      g1['lo'])
        .replace("{MR_BAD_ZV}",      json.dumps(mr_bad))
        # Peak 2 GPR
        .replace("{MR2_GPR_OPTZS}",  g2['optzs'])
        .replace("{MR2_GPR_OPTZ}",   g2['optz'])
        .replace("{MR2_GPR_FWHM}",   g2['fwhm'])
        .replace("{MR2_GPR_OK}",     g2['ok'])
        .replace("{MR2_GPR_ZV}",     g2['zv'])
        .replace("{MR2_GPR_IV}",     g2['iv'])
        .replace("{MR2_GPR_UP}",     g2['up'])
        .replace("{MR2_GPR_LO}",     g2['lo'])
        .replace("{MR2_BAD_ZV}",     json.dumps(mr2_bad))
        # Peak 3 GPR
        .replace("{MR3_GPR_OPTZS}",  g3['optzs'])
        .replace("{MR3_GPR_OPTZ}",   g3['optz'])
        .replace("{MR3_GPR_FWHM}",   g3['fwhm'])
        .replace("{MR3_GPR_OK}",     g3['ok'])
        .replace("{MR3_GPR_ZV}",     g3['zv'])
        .replace("{MR3_GPR_IV}",     g3['iv'])
        .replace("{MR3_GPR_UP}",     g3['up'])
        .replace("{MR3_GPR_LO}",     g3['lo'])
        .replace("{MR3_BAD_ZV}",     json.dumps(mr3_bad))
        # ML Insights card
        .replace("{ML_STATUS}",      ml_status)
        .replace("{ML_RANK_ROWS}",   rank_rows_html)
        .replace("{ML_BEST_PEAK}",   ml_best_peak)
        .replace("{ML_BEST_REASON}", ml_best_reason)
        .replace("{ML_BAD_COLOR}",   ml_bad_color)
        .replace("{ML_N_BAD}",       str(n_bad_total))
        .replace("{ML_BAD_LIST}",    ml_bad_list)
    )
    return shell(html)


@app.route("/download_csv", methods=["POST"])
def download_csv():
    """Stream CSV for download."""
    z_vals = json.loads(request.form.get("z_json", "[]"))
    i_vals = json.loads(request.form.get("i_json", "[]"))
    mode   = request.form.get("mode",   "sum")
    prefix = request.form.get("prefix", "scan")

    lines  = [f"z_index,intensity_{mode}"]
    for z, iv in zip(z_vals, i_vals):
        lines.append(f"{z},{iv}")
    csv_text = "\n".join(lines)

    z0   = z_vals[0] if z_vals else 0
    z1   = z_vals[-1] if z_vals else 0
    fname = f"heightscan_{prefix}_z{z0}-{z1}.csv"
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'}
    )


@app.route("/preview_ajax", methods=["POST"])
def preview_ajax():
    """
    AJAX endpoint.
    mode='png'    (default) → returns base64 PNG + ROI stats
    mode='plotly'           → returns downsampled z-data for Plotly heatmap + ROI stats
    mode='stats'            → returns p1/p99 percentiles in display-space (no image render)
    vmin / vmax params      → display-space contrast limits forwarded to renderers
    """
    if not FABIO_OK:
        return jsonify({"ok": False, "error": "fabio not installed"})
    try:
        folder  = request.form.get("folder", "")
        z_idx   = int(request.form.get("z", 0))
        x_min   = int(request.form.get("x_min", 0))
        x_max   = int(request.form.get("x_max", 100))
        y_min   = int(request.form.get("y_min", 0))
        y_max   = int(request.form.get("y_max", 100))
        log_img = request.form.get("log_img", "1") == "1"
        cmap    = request.form.get("cmap", "inferno")
        mode    = request.form.get("mode", "png")

        # Optional display-space contrast limits
        vmin_s = request.form.get("vmin", "")
        vmax_s = request.form.get("vmax", "")
        vmin   = float(vmin_s) if vmin_s.strip() else None
        vmax   = float(vmax_s) if vmax_s.strip() else None

        _, _, file_map = scan_folder(folder)
        if z_idx not in file_map:
            return jsonify({"ok": False, "error": f"Index {z_idx} not found"})

        img    = load_cbf(file_map[z_idx])
        roi_p  = img[y_min:y_max + 1, x_min:x_max + 1]
        stats  = {
            "roi_sum":  f"{roi_p.sum():.3e}",
            "roi_mean": f"{roi_p.mean():.2f}",
            "roi_max":  f"{int(roi_p.max())}",
            "shape":    f"{img.shape[1]} × {img.shape[0]}",
        }

        if mode == "stats":
            # Compute 1st / 99th percentile in display-space for auto vmin/vmax
            disp = np.log10(np.maximum(img, 1)) if log_img else img.astype(float)
            p1  = round(float(np.percentile(disp,  1)), 4)
            p99 = round(float(np.percentile(disp, 99)), 4)
            return jsonify({"ok": True, "p1": p1, "p99": p99, "log": log_img, **stats})
        elif mode == "plotly":
            pj = img_to_plotly_json(img, roi=(x_min, x_max, y_min, y_max),
                                    log_scale=log_img, max_dim=480,
                                    vmin=vmin, vmax=vmax)
            return jsonify({"ok": True, **stats, **pj})
        else:
            b64 = img_to_b64(img, roi=(x_min, x_max, y_min, y_max),
                             log_scale=log_img, cmap=cmap,
                             vmin=vmin, vmax=vmax)
            return jsonify({"ok": True, "b64": b64, **stats})

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)})


@app.route("/auto_roi_ajax", methods=["POST"])
def auto_roi_ajax():
    """
    Return auto-detected ROI coordinates for a given image.

    POST params:
      folder      — CBF folder path
      z           — z index to use for detection
      percentile  — threshold percentile (default 99)
      pad         — pixel padding around detected region (default 20)
      method      — 'com' (center-of-mass + pad) | 'bbox' (bounding box + pad)
      log_img     — '1' / '0'

    Returns JSON: { ok, x_min, x_max, y_min, y_max, cx, cy, msg }
    """
    if not FABIO_OK:
        return jsonify({"ok": False, "error": "fabio not installed"})
    try:
        folder  = request.form.get("folder", "")
        z_idx   = int(request.form.get("z", 0))
        pct     = float(request.form.get("percentile", 99))
        pad     = int(request.form.get("pad", 20))
        method  = request.form.get("method", "com")   # 'com' or 'bbox'
        log_img = request.form.get("log_img", "1") == "1"

        _, _, file_map = scan_folder(folder)
        if z_idx not in file_map:
            return jsonify({"ok": False, "error": f"Index {z_idx} not found"})

        img = load_cbf(file_map[z_idx])
        h, w = img.shape

        # Work in display space for thresholding
        work = np.log10(np.maximum(img, 1)) if log_img else img.astype(float)
        threshold = np.percentile(work, pct)
        mask = work >= threshold

        if not mask.any():
            # Fall back to full image
            return jsonify({"ok": True, "x_min": 0, "x_max": w - 1,
                            "y_min": 0, "y_max": h - 1,
                            "cx": w // 2, "cy": h // 2,
                            "msg": f"No pixels above p{pct:.0f}; using full image"})

        ys, xs = np.where(mask)

        if method == "bbox":
            # Bounding box of all hot pixels + padding
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(w - 1, int(xs.max()) + pad)
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(h - 1, int(ys.max()) + pad)
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            msg = (f"bbox method (p{pct:.0f}) + {pad}px pad: "
                   f"x=[{x0},{x1}] y=[{y0},{y1}]")
        else:
            # Center-of-mass weighted by display values above threshold
            weights = work[mask]
            cx = int(round(float(np.average(xs, weights=weights))))
            cy = int(round(float(np.average(ys, weights=weights))))
            # Build symmetric box of size determined by spread
            sx = max(pad, int(np.std(xs) * 2))
            sy = max(pad, int(np.std(ys) * 2))
            x0 = max(0, cx - sx)
            x1 = min(w - 1, cx + sx)
            y0 = max(0, cy - sy)
            y1 = min(h - 1, cy + sy)
            msg = (f"COM method (p{pct:.0f}) centre=({cx},{cy}) spread={sx}×{sy}px: "
                   f"x=[{x0},{x1}] y=[{y0},{y1}]")

        return jsonify({"ok": True,
                        "x_min": x0, "x_max": x1,
                        "y_min": y0, "y_max": y1,
                        "cx": cx,   "cy": cy,
                        "msg": msg})

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)})


# ══════════════════════════════════════════════════════════════════════════════
#  File / Folder Browser  —  mirrors /pyfai/browse in app.py
# ══════════════════════════════════════════════════════════════════════════════

BROWSE_HTML = """
<div class="card">
  <div class="card-hdr">
    <div class="card-dot"></div>
    <span class="card-title">📁 Browse — select CBF folder</span>
    <span style="margin-left:auto;font-size:11px;color:var(--text-muted);">
      root: <span class="path-label">{ROOT}</span>
    </span>
  </div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;
              background:rgba(0,200,255,.04);border:1px solid rgba(0,200,255,.15);
              border-radius:5px;padding:7px 12px;margin-bottom:14px;
              overflow-x:auto;white-space:nowrap;">
    {CRUMBS}
  </div>
  {SELECT_BTN}
  {SUBDIRS_BLOCK}
  <hr style="border:none;border-top:1px solid var(--border-subtle);margin:14px 0;">
  <a class="dl-btn" href="/?folder={CURRENT_ENC}">← Back to form (keep current path)</a>
</div>
"""

def _crumbs_html(path):
    parts  = [p for p in path.split("/") if p]
    crumbs = []
    for i in range(len(parts) + 1):
        label = "/" if i == 0 else parts[i - 1]
        nav   = "/" + "/".join(parts[:i])
        href  = f"/browse?path={urlquote(nav, safe='')}"
        crumbs.append(f'<a href="{href}" style="color:var(--accent-glow);text-decoration:none;">{label}</a>')
    return ' <span style="color:var(--text-muted);">/</span> '.join(crumbs)


@app.route("/browse")
def browse():
    raw_path = request.args.get("path", CBF_ROOT)
    path     = os.path.abspath(raw_path)
    if not path.startswith(CBF_ROOT):
        path = CBF_ROOT

    subdirs = []
    if os.path.isdir(path):
        try:
            for item in sorted(os.listdir(path)):
                full = os.path.join(path, item)
                if os.path.isdir(full):
                    subdirs.append(full)
        except PermissionError:
            pass

    crumbs_html = _crumbs_html(path)
    cbf_here    = sorted(glob.glob(os.path.join(path, "*.cbf")))
    n_cbf       = len(cbf_here)

    if n_cbf:
        select_btn = (
            f'<div style="margin-bottom:14px;">'
            f'<a href="/?folder={urlquote(path, safe="")}">'
            f'  <button class="btn-primary" type="button">'
            f'    ✅ Select this folder &nbsp;<span class="badge badge-green">{n_cbf} .cbf files</span>'
            f'  </button>'
            f'</a></div>'
        )
    else:
        select_btn = (
            f'<div style="margin-bottom:10px;font-size:12px;color:var(--text-muted);">'
            f'  No .cbf files here — navigate into a subfolder.</div>'
        )

    if subdirs:
        items = "\n".join(
            f'<li style="margin-bottom:4px;">'
            f'  <a href="/browse?path={urlquote(d, safe="")}"'
            f'     style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;color:var(--accent-glow);">'
            f'    📁 {os.path.basename(d)}</a>'
            f'</li>'
            for d in subdirs
        )
        subdirs_block = (
            f'<div class="card-hdr" style="margin-top:4px;">'
            f'  <div class="card-dot"></div>'
            f'  <span class="card-title">Subfolders ({len(subdirs)})</span>'
            f'</div>'
            f'<ul style="list-style:none;padding:0;margin:0;">{items}</ul>'
        )
    else:
        subdirs_block = '<p style="color:var(--text-muted);font-size:13px;">No subfolders.</p>'

    html = (BROWSE_HTML
        .replace("{ROOT}",          CBF_ROOT)
        .replace("{CRUMBS}",        crumbs_html)
        .replace("{SELECT_BTN}",    select_btn)
        .replace("{SUBDIRS_BLOCK}", subdirs_block)
        .replace("{CURRENT_ENC}",   urlquote(path, safe=""))
    )
    return shell(html)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n  HeightScan App v2.0  →  http://localhost:{PORT}\n")
    print("  Tabs: ROI Line Plot · Pixel Inspector · Max-Pixel ROI\n")
    if not FABIO_OK:
        print("  ⚠  WARNING: fabio not found.  pip install fabio\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)