"""Single overlay map — Overture (blue) vs Foursquare (red).

Vector approach: every populated 0.5° cell becomes a real GeoJSON
Polygon coloured by source mix.  Vector data reprojects correctly into
Leaflet's Web Mercator (no equirectangular-vs-Mercator stretching),
stays crisp at every zoom level, and is anchored to actual lat/lon
coordinates — so cells fall on the right country, not in the ocean.

Color encoding per cell:
  - red channel  ∝ log10(FSQ count)   (Foursquare strength)
  - blue channel ∝ log10(Ovt count)   (Overture strength)
  - alpha        ∝ max of the two     (POI density saturation)

  → Pure-FSQ cell = red.  Pure-Ovt cell = blue.  Both = purple.
  → Bright = lots of places.  Dim = a handful.

Filter: cells with ≥50 places only — drops single-village noise.

Run locally on the Mac:
    .venv/bin/python scripts/12_render_dominance_map.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import folium
from branca.element import Template, MacroElement


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
GRID_CSV = OUTPUTS / "source_dominance_grid.csv"
OUT_HTML = OUTPUTS / "source_overlay_map.html"

CELL_SIZE = 0.5  # 0.5° grid

# Saturate at 10^5 places per cell — cells with more become fully bright
# (Manhattan, central Tokyo).  Below ~10 → near-invisible.
LOG_MIN = 1.0
LOG_MAX = 5.0
MIN_PLACES = 50  # drop noise cells


if not GRID_CSV.exists():
    raise SystemExit(
        f"missing {GRID_CSV} — rsync from Leonardo first:\n"
        "  rsync -az uaslam00@login.leonardo.cineca.it:"
        "/leonardo_work/AIFAC_P02_222/overture-map/outputs/"
        "source_dominance_grid.csv overture-map/outputs/"
    )

print(f"[load] {GRID_CSV}")
df = pd.read_csv(GRID_CSV, low_memory=False)
print(f"[load] {len(df):,} cells")

df = df[df["n_total"] >= MIN_PLACES].copy()
print(f"[filt] {len(df):,} cells with ≥{MIN_PLACES} places")


# -------- 1. Compute per-cell colour --------------------------------

def normalize(arr: np.ndarray) -> np.ndarray:
    v = np.log10(arr + 1.0)
    return ((v - LOG_MIN) / (LOG_MAX - LOG_MIN)).clip(0.0, 1.0)


ovt_n = normalize(df["n_ovt"].to_numpy())
fsq_n = normalize(df["n_fsq"].to_numpy())

# RGB blend
r = (fsq_n * 255).astype(int)
g = (np.minimum(fsq_n, ovt_n) * 50).astype(int)   # tiny green hint when both
b = (ovt_n * 255).astype(int)
alpha = (np.maximum(ovt_n, fsq_n) ** 0.6) * 0.85   # γ-correct, cap below 1

df["fill"] = [
    f"#{r[i]:02x}{g[i]:02x}{b[i]:02x}" for i in range(len(df))
]
df["alpha"] = alpha


# -------- 2. Build GeoJSON FeatureCollection ------------------------

half = CELL_SIZE / 2
features = []
for _, row in df.iterrows():
    lat = row["lat_center"]
    lon = row["lon_center"]
    feat = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [lon - half, lat - half],
                [lon + half, lat - half],
                [lon + half, lat + half],
                [lon - half, lat + half],
                [lon - half, lat - half],
            ]],
        },
        "properties": {
            "fill": row["fill"],
            "alpha": float(row["alpha"]),
            "n_ovt": int(row["n_ovt"]),
            "n_fsq": int(row["n_fsq"]),
            "country": str(row["country"]) if pd.notna(row["country"]) else "—",
        },
    }
    features.append(feat)

gj = {"type": "FeatureCollection", "features": features}
print(f"[geojson] {len(features):,} polygons "
      f"(~{len(json.dumps(gj)) / 1e6:.1f} MB)")


# -------- 3. Folium map with GeoJson layer --------------------------

m = folium.Map(
    location=[20, 10],
    zoom_start=2,
    tiles="cartodbdark_matter",
    world_copy_jump=True,
    prefer_canvas=True,   # canvas renderer scales better than SVG for many polygons
)

def style_fn(feat):
    return {
        "fillColor": feat["properties"]["fill"],
        "color": feat["properties"]["fill"],
        "weight": 0,
        "fillOpacity": feat["properties"]["alpha"],
    }

def highlight_fn(_):
    return {"weight": 1, "color": "#fff", "fillOpacity": 0.95}

folium.GeoJson(
    gj,
    name="POI dominance",
    style_function=style_fn,
    highlight_function=highlight_fn,
    tooltip=folium.GeoJsonTooltip(
        fields=["country", "n_ovt", "n_fsq"],
        aliases=["country", "Overture", "Foursquare"],
        sticky=True,
    ),
).add_to(m)


# Title + legend overlay
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
    blue saturation = Overture density · red = Foursquare · purple = both ·
    hover for counts
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
print(f"[html] {OUT_HTML}  ({OUT_HTML.stat().st_size / 1e6:.1f} MB)")
print("\nopen in browser — vector polygons, crisp at any zoom.")
