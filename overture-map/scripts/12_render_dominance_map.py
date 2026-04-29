"""Single overlay map — Overture (blue) vs Foursquare (red).

The story we want the picture to tell, in one glance:
  - Overture is wide  → dim blue covers continental interiors (rural OSM)
  - Foursquare is deep → bright red dots concentrate in cities globally
  - Where both have data  → purple/magenta blend
  - Where neither has data → basemap shows through (transparent)

Approach:
  1. Pivot the 0.5° dominance grid into a 360×720 RGBA raster where the
     red channel = log scaled FSQ count and the blue channel = log scaled
     Ovt count.  This is REAL pixel blending — equal counts produce
     proper purple, not the layer-occlusion that two Plotly heat traces
     produced.
  2. Save the raster as PNG.
  3. Drop the PNG onto an OSM basemap as a single image overlay layer.

Run locally on the Mac:
    .venv/bin/python scripts/12_render_dominance_map.py
"""
from __future__ import annotations
import base64
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
GRID_CSV = OUTPUTS / "source_dominance_grid.csv"
OUT_HTML = OUTPUTS / "source_overlay_map.html"
OUT_PNG = OUTPUTS / "source_overlay_raster.png"

# 0.5° grid → 360 × 720 raster
GRID = 2
NLAT = 180 * GRID  # 360
NLON = 360 * GRID  # 720
LAT_OFFSET = 90 * GRID
LON_OFFSET = 180 * GRID

# Saturate at 10^5 places per cell — cells with more than that become
# fully bright (Manhattan, central Tokyo).  Below 10 → near-invisible.
LOG_MIN = 1.0
LOG_MAX = 5.0


if not GRID_CSV.exists():
    raise SystemExit(
        f"missing {GRID_CSV} — rsync from Leonardo first:\n"
        "  rsync -az uaslam00@login.leonardo.cineca.it:"
        "/leonardo_work/AIFAC_P02_222/overture-map/outputs/"
        "source_dominance_grid.csv overture-map/outputs/"
    )

print(f"[load] {GRID_CSV}")
df = pd.read_csv(GRID_CSV, low_memory=False)
print(f"[load] {len(df):,} cells in the 0.5° grid")


# -------- 1. Build the RGBA raster ----------------------------------

def normalize(arr: np.ndarray) -> np.ndarray:
    """log10(n+1), clipped to [LOG_MIN, LOG_MAX], scaled to [0, 1]."""
    v = np.log10(arr + 1.0)
    return ((v - LOG_MIN) / (LOG_MAX - LOG_MIN)).clip(0.0, 1.0)


ovt_arr = np.zeros((NLAT, NLON), dtype=np.float64)
fsq_arr = np.zeros((NLAT, NLON), dtype=np.float64)
iy = (df["gy"].to_numpy() + LAT_OFFSET).astype(np.int32)
ix = (df["gx"].to_numpy() + LON_OFFSET).astype(np.int32)
valid = (iy >= 0) & (iy < NLAT) & (ix >= 0) & (ix < NLON)
ovt_arr[iy[valid], ix[valid]] = df["n_ovt"].to_numpy()[valid]
fsq_arr[iy[valid], ix[valid]] = df["n_fsq"].to_numpy()[valid]

ovt_n = normalize(ovt_arr)
fsq_n = normalize(fsq_arr)

# Build RGBA: red channel from FSQ, blue channel from Ovt, alpha from max.
# Bias the alpha curve so even small counts become visible.
rgba = np.zeros((NLAT, NLON, 4), dtype=np.uint8)
rgba[..., 0] = (fsq_n * 255).astype(np.uint8)              # red   = FSQ
rgba[..., 1] = (np.minimum(fsq_n, ovt_n) * 60).astype(np.uint8)  # green hint where both
rgba[..., 2] = (ovt_n * 255).astype(np.uint8)              # blue  = Ovt
alpha = np.maximum(ovt_n, fsq_n) ** 0.6                    # γ-correct for visibility
rgba[..., 3] = (alpha * 230).astype(np.uint8)              # cap at 230 so map shows through

# Flip vertically — image origin top-left, geographic origin bottom-left
rgba = np.flipud(rgba)

img = Image.fromarray(rgba, mode="RGBA")
img.save(OUT_PNG)
print(f"[png ] {OUT_PNG}  ({OUT_PNG.stat().st_size / 1e3:.0f} KB)")


# -------- 2. Embed PNG as base64 data URL into Plotly map -----------

with open(OUT_PNG, "rb") as fh:
    img_b64 = base64.b64encode(fh.read()).decode("ascii")
img_uri = f"data:image/png;base64,{img_b64}"


# -------- 3. Plot the basemap with the raster overlay --------------

fig = go.Figure()

# Empty trace just to anchor mapbox layout — the actual data is the raster.
fig.add_trace(go.Scattermapbox(
    lat=[None], lon=[None], mode="markers",
    marker=dict(size=1), showlegend=False, hoverinfo="skip",
))

fig.update_layout(
    title=dict(
        text=(
            "<b>Overture (blue) vs Foursquare (red) — wide vs deep</b><br>"
            "<span style='font-size:13px;color:#aaa'>"
            "blue saturation = Overture density · "
            "red saturation = Foursquare density · "
            "purple = both have data · drag/scroll to zoom"
            "</span>"
        ),
        x=0.5,
        font=dict(size=18, color="#eee"),
    ),
    height=820,
    margin=dict(l=0, r=0, t=80, b=0),
    paper_bgcolor="#0a0a0a",
    font=dict(color="#ddd"),
    showlegend=False,
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=20, lon=10),
        zoom=1.5,
        layers=[dict(
            below="traces",
            sourcetype="image",
            source=img_uri,
            coordinates=[
                [-180,  90],   # top-left
                [ 180,  90],   # top-right
                [ 180, -90],   # bottom-right
                [-180, -90],   # bottom-left
            ],
        )],
    ),
    annotations=[
        dict(
            x=0.01, y=0.05, xref="paper", yref="paper",
            xanchor="left", yanchor="bottom", showarrow=False,
            text=(
                "<span style='background:#0a0a0a;padding:8px 12px;"
                "border:1px solid #555;border-radius:4px;'>"
                "<b>Legend</b><br>"
                "<span style='color:#3b82f6'>■</span> Overture (wide)<br>"
                "<span style='color:#a855f7'>■</span> Both sources<br>"
                "<span style='color:#ef4444'>■</span> Foursquare (deep)<br>"
                "<span style='color:#aaa'>brightness ∝ POI density</span>"
                "</span>"
            ),
            font=dict(size=13, color="#eee"),
        ),
    ],
)

fig.write_html(OUT_HTML, include_plotlyjs="cdn")
print(f"[html] {OUT_HTML}  ({OUT_HTML.stat().st_size / 1e6:.1f} MB)")
print("\nopen in browser — single overlaid raster, true RGB blend.")
