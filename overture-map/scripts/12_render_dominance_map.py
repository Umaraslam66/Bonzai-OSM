"""Single overlay map — Overture (blue) vs Foursquare (red).

The story we want the picture to tell, in one glance:
  - Overture is wide  → dim blue covers continental interiors (rural OSM)
  - Foursquare is deep → bright red dots concentrate in cities globally
  - Where both have data  → purple/magenta blend
  - Where neither has data → basemap shows through (transparent)

Approach:
  1. Pivot the 0.5° dominance grid into a 360×720 RGBA raster where the
     red channel = log scaled FSQ count and the blue channel = log scaled
     Ovt count.  Real pixel blending — equal counts produce proper purple.
  2. Save the raster as PNG.
  3. Drop the PNG onto an OSM basemap as a Folium ImageOverlay (Leaflet,
     robust raster handling).

Run locally on the Mac:
    .venv/bin/python scripts/12_render_dominance_map.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import folium
from folium.raster_layers import ImageOverlay
from branca.element import Template, MacroElement


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

rgba = np.zeros((NLAT, NLON, 4), dtype=np.uint8)
rgba[..., 0] = (fsq_n * 255).astype(np.uint8)              # red   = FSQ
rgba[..., 1] = (np.minimum(fsq_n, ovt_n) * 60).astype(np.uint8)  # green hint where both
rgba[..., 2] = (ovt_n * 255).astype(np.uint8)              # blue  = Ovt
alpha = np.maximum(ovt_n, fsq_n) ** 0.6                    # γ-correct
rgba[..., 3] = (alpha * 220).astype(np.uint8)

# Flip — image origin top-left, geographic origin bottom-left
rgba = np.flipud(rgba)

img = Image.fromarray(rgba, mode="RGBA")
# Upscale 4× with nearest-neighbour so cell borders are crisp at zoom-out
img = img.resize((NLON * 4, NLAT * 4), Image.NEAREST)
img.save(OUT_PNG)
print(f"[png ] {OUT_PNG}  ({OUT_PNG.stat().st_size / 1e3:.0f} KB, "
      f"{img.size[0]}×{img.size[1]})")


# -------- 2. Folium map with ImageOverlay -----------------------------

m = folium.Map(
    location=[20, 10],
    zoom_start=2,
    tiles="cartodbdark_matter",  # dark basemap so red/blue pop
    world_copy_jump=True,
)

ImageOverlay(
    image=str(OUT_PNG),
    bounds=[[-90, -180], [90, 180]],   # [[south, west], [north, east]]
    opacity=0.85,
    interactive=False,
    cross_origin=False,
    zindex=1,
).add_to(m)

# Title + legend HTML overlay
legend_html = """
{% macro html(this, kwargs) %}
<div style="
  position: fixed;
  top: 12px; left: 50%; transform: translateX(-50%);
  background: rgba(10,10,10,0.92); color: #eee;
  padding: 10px 18px; border: 1px solid #444; border-radius: 6px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 14px; z-index: 9999; text-align: center;
">
  <b>Overture (blue) vs Foursquare (red) — wide vs deep</b><br>
  <span style="font-size:12px;color:#aaa">
    blue saturation = Overture density · red = Foursquare · purple = both
  </span>
</div>
<div style="
  position: fixed;
  bottom: 24px; left: 12px;
  background: rgba(10,10,10,0.92); color: #eee;
  padding: 10px 14px; border: 1px solid #444; border-radius: 6px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 13px; z-index: 9999;
">
  <b>Legend</b><br>
  <span style="display:inline-block;width:14px;height:14px;background:#3b82f6;
    border-radius:2px;vertical-align:middle"></span>
  &nbsp;Overture (wide rural)<br>
  <span style="display:inline-block;width:14px;height:14px;background:#a855f7;
    border-radius:2px;vertical-align:middle"></span>
  &nbsp;Both sources<br>
  <span style="display:inline-block;width:14px;height:14px;background:#ef4444;
    border-radius:2px;vertical-align:middle"></span>
  &nbsp;Foursquare (deep urban)<br>
  <span style="color:#aaa;font-size:11px">brightness ∝ POI density</span>
</div>
{% endmacro %}
"""
macro = MacroElement()
macro._template = Template(legend_html)
m.get_root().add_child(macro)

m.save(str(OUT_HTML))
print(f"[html] {OUT_HTML}  ({OUT_HTML.stat().st_size / 1e3:.0f} KB)")
print("\nopen in browser — Folium/Leaflet image overlay on dark CartoDB basemap.")
