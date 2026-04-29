"""Single map showing Overture (wide) vs Foursquare (deep).

Two density heat layers, one blue (Overture), one red (Foursquare),
both visible at once on an OpenStreetMap base.  Where the two layers
overlap they blend toward purple — so the visual story is:

  - large blue regions with no red       → Overture-only (wide rural coverage)
  - bright red blobs with little blue    → Foursquare-only (deep urban POIs)
  - purple/magenta tinted regions        → both sources have data
  - white/no-color regions               → neither source has data

Single self-contained HTML.  Pan, zoom, hover.  Toggle each source on/off
by clicking its legend entry.

Run locally on the Mac after rsyncing the CSV from Leonardo:
    .venv/bin/python scripts/12_render_dominance_map.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
GRID_CSV = OUTPUTS / "source_dominance_grid.csv"
OUT = OUTPUTS / "source_overlay_map.html"

if not GRID_CSV.exists():
    raise SystemExit(
        f"missing {GRID_CSV} — rsync from Leonardo first:\n"
        "  rsync -az uaslam00@login.leonardo.cineca.it:"
        "/leonardo_work/AIFAC_P02_222/overture-map/outputs/"
        "source_dominance_grid.csv overture-map/outputs/"
    )

print(f"[load] {GRID_CSV}")
df = pd.read_csv(GRID_CSV)
print(f"[load] {len(df):,} cells in the 0.5° grid")

# Filter noise floor — single-village cells add nothing visually.
ovt = df[df["n_ovt"] >= 50].copy()
fsq = df[df["n_fsq"] >= 50].copy()
print(f"[filt] showing {len(ovt):,} Ovt cells and {len(fsq):,} FSQ cells "
      f"with ≥50 places each")

# Use log10 for the heat weight — POI counts span 5 orders of magnitude
ovt_z = np.log10(ovt["n_ovt"]).clip(lower=1.0)
fsq_z = np.log10(fsq["n_fsq"]).clip(lower=1.0)


def transparent_scale(rgb_hex: str) -> list:
    """Build a colorscale that fades from fully transparent to opaque colour.

    Plotly's Densitymapbox interpolates the bottom of the scale onto every
    pixel, so a colorscale that starts at rgba(...,0) keeps unfilled cells
    invisible and lets the basemap show through.
    """
    return [
        [0.00, "rgba(0,0,0,0)"],
        [0.05, "rgba(0,0,0,0)"],
        [0.30, _hex_to_rgba(rgb_hex, 0.30)],
        [0.70, _hex_to_rgba(rgb_hex, 0.65)],
        [1.00, _hex_to_rgba(rgb_hex, 0.85)],
    ]


def _hex_to_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


# Overture → blue (#3b82f6); Foursquare → red (#ef4444)
OVT_COLOR = "#3b82f6"
FSQ_COLOR = "#ef4444"

fig = go.Figure()

fig.add_trace(go.Densitymapbox(
    name="Overture (wide coverage)",
    lat=ovt["lat_center"],
    lon=ovt["lon_center"],
    z=ovt_z,
    radius=12,
    colorscale=transparent_scale(OVT_COLOR),
    showscale=False,
    hovertemplate=(
        "<b>Overture</b><br>"
        "lat %{lat:.1f}, lon %{lon:.1f}<br>"
        "log₁₀ count: %{z:.2f}<extra></extra>"
    ),
    opacity=1.0,
    zmin=1.0,
    zmax=6.0,
))

fig.add_trace(go.Densitymapbox(
    name="Foursquare (deep urban)",
    lat=fsq["lat_center"],
    lon=fsq["lon_center"],
    z=fsq_z,
    radius=12,
    colorscale=transparent_scale(FSQ_COLOR),
    showscale=False,
    hovertemplate=(
        "<b>Foursquare</b><br>"
        "lat %{lat:.1f}, lon %{lon:.1f}<br>"
        "log₁₀ count: %{z:.2f}<extra></extra>"
    ),
    opacity=1.0,
    zmin=1.0,
    zmax=6.0,
))

fig.update_layout(
    title=dict(
        text=(
            "<b>Overture vs Foursquare — wide vs deep</b><br>"
            "<span style='font-size:13px;color:#aaa'>"
            "blue = Overture · red = Foursquare · purple = both<br>"
            "click a legend entry to toggle, scroll to zoom"
            "</span>"
        ),
        x=0.5,
        font=dict(size=18, color="#eee"),
    ),
    height=820,
    margin=dict(l=0, r=0, t=80, b=0),
    paper_bgcolor="#0a0a0a",
    font=dict(color="#ddd"),
    showlegend=True,
    legend=dict(
        x=0.01,
        y=0.99,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(20,20,20,0.92)",
        font=dict(color="#eee", size=14),
        bordercolor="#444",
        borderwidth=1,
    ),
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=20, lon=10),
        zoom=1.5,
    ),
)

fig.write_html(OUT, include_plotlyjs="cdn")
print(f"\n[render] {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
print("\nopen in browser, pan/zoom, toggle layers via legend.")
