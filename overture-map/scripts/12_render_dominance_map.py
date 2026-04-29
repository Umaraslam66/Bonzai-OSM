"""Side-by-side WebGL heatmaps — Overture vs Foursquare.

Built on deck.gl (via pydeck) — Uber's GPU map renderer.  Each map uses
HeatmapLayer, which does GPU density estimation in screen space, so:
  - low zoom → broad blobs (you see the global pattern)
  - high zoom → tight cells (you see neighborhood-level detail)
  - smooth at every step (no raster blur, no SVG jank)

Free CARTO Dark Matter basemap, no API key.

Run locally on the Mac:
    .venv/bin/python scripts/12_render_dominance_map.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pydeck as pdk


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
GRID_CSV = OUTPUTS / "source_dominance_grid.csv"
OUT_HTML = OUTPUTS / "source_overlay_map.html"
OVT_PAGE = OUTPUTS / "_heatmap_overture.html"
FSQ_PAGE = OUTPUTS / "_heatmap_foursquare.html"

# CARTO dark-matter — free, no token, looks great with red/blue overlays
CARTO_DARK = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

# Color ramps — start fully transparent, fade to opaque source colour.
# RGBA tuples; HeatmapLayer interpolates these along its density gradient.
OVT_RAMP = [
    [10, 30, 80, 0],
    [30, 60, 180, 80],
    [60, 120, 240, 160],
    [100, 180, 255, 220],
    [180, 230, 255, 255],
]
FSQ_RAMP = [
    [80, 10, 30, 0],
    [180, 40, 60, 80],
    [240, 80, 100, 160],
    [255, 160, 120, 220],
    [255, 230, 200, 255],
]


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


def build_deck(df: pd.DataFrame, weight_col: str, ramp: list, title: str) -> pdk.Deck:
    pts = df[df[weight_col] > 0][["lat_center", "lon_center", weight_col]].copy()
    pts.columns = ["lat", "lon", "w"]
    # Log-scale the weights so a 100k-place city doesn't flatten everything else.
    pts["w"] = np.log10(pts["w"] + 1.0)

    layer = pdk.Layer(
        "HeatmapLayer",
        pts.to_dict("records"),
        get_position=["lon", "lat"],
        get_weight="w",
        aggregation="SUM",
        radius_pixels=45,    # heat blob radius — bigger blobs at low zoom
        intensity=1.4,
        threshold=0.04,
        color_range=ramp,
        opacity=0.92,
    )

    view = pdk.ViewState(latitude=25, longitude=10, zoom=1.4)
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_style=CARTO_DARK,
        map_provider=None,   # we provide a self-hosted style URL
        tooltip={"text": "lat {lat}\nlon {lon}\nlog₁₀(count): {w}"},
        parameters={"clearColor": [0.04, 0.04, 0.04, 1.0]},
    )


print("[deck] building Overture heatmap...")
build_deck(df, "n_ovt", OVT_RAMP, "Overture").to_html(
    str(OVT_PAGE), notebook_display=False, iframe_height=820,
)
print("[deck] building Foursquare heatmap...")
build_deck(df, "n_fsq", FSQ_RAMP, "Foursquare").to_html(
    str(FSQ_PAGE), notebook_display=False, iframe_height=820,
)


# Side-by-side wrapper page
wrapper = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Overture vs Foursquare — wide vs deep</title>
  <style>
    html, body {{
      margin: 0; padding: 0; height: 100%;
      background: #0a0a0a; color: #eee;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    .page {{
      display: flex; flex-direction: column; height: 100vh; width: 100vw;
    }}
    .header {{
      flex: 0 0 auto;
      padding: 10px 16px;
      background: rgba(20,20,20,0.96);
      border-bottom: 1px solid #333;
      text-align: center;
    }}
    .header h1 {{ margin: 0; font-size: 16px; }}
    .header p  {{ margin: 4px 0 0; font-size: 12px; color: #aaa; }}
    .panes {{
      flex: 1 1 auto;
      display: flex;
      gap: 2px;
      background: #222;
    }}
    .pane {{
      flex: 1 1 50%; position: relative;
      display: flex; flex-direction: column;
    }}
    .pane-title {{
      flex: 0 0 auto;
      padding: 6px 12px;
      font-size: 13px; font-weight: 600;
      background: rgba(20,20,20,0.96);
      border-bottom: 1px solid #333;
    }}
    .pane.ovt .pane-title {{ color: #6fb1ff; }}
    .pane.fsq .pane-title {{ color: #ff8090; }}
    .pane iframe {{
      flex: 1 1 auto;
      width: 100%;
      border: 0;
      background: #0a0a0a;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <h1>Overture (wide rural) <span style="color:#666">vs</span> Foursquare (deep urban)</h1>
      <p>WebGL heatmaps via deck.gl — pan, zoom, the maps reveal more
      detail at every scale.  Both panes share data from the 0.5° dominance
      grid, weighted by log₁₀(count) so urban density doesn't drown rural coverage.</p>
    </div>
    <div class="panes">
      <div class="pane ovt">
        <div class="pane-title">Overture · {df["n_ovt"].astype(int).sum():,} places · OSM-derived rural breadth</div>
        <iframe src="{OVT_PAGE.name}"></iframe>
      </div>
      <div class="pane fsq">
        <div class="pane-title">Foursquare · {df["n_fsq"].astype(int).sum():,} places · user-generated urban depth</div>
        <iframe src="{FSQ_PAGE.name}"></iframe>
      </div>
    </div>
  </div>
</body>
</html>
"""
OUT_HTML.write_text(wrapper)
print(f"\n[wrap] {OUT_HTML}  ({OUT_HTML.stat().st_size / 1e3:.0f} KB wrapper)")
print(f"[wrap] {OVT_PAGE}  ({OVT_PAGE.stat().st_size / 1e6:.1f} MB)")
print(f"[wrap] {FSQ_PAGE}  ({FSQ_PAGE.stat().st_size / 1e6:.1f} MB)")
print("\nopen source_overlay_map.html in browser.")
