"""Render outputs/source_dominance_grid.csv as interactive zoomable maps.

Outputs:
  - outputs/dominance_map.html   colored grid cells, log2(Ovt/FSQ)
  - outputs/density_map.html     heatmap of total POI density
  - outputs/dominance_compare.html  Ovt + FSQ side-by-side toggle

Uses Plotly density_mapbox / scattermapbox so the rendering is WebGL,
basemap is OpenStreetMap tiles (no API key), and zoom/pan/hover work
smoothly even with tens of thousands of cells.

Run locally on the Mac after rsyncing the CSV from Leonardo:
    .venv/bin/python scripts/12_render_dominance_map.py
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
GRID_CSV = OUTPUTS / "source_dominance_grid.csv"

if not GRID_CSV.exists():
    raise SystemExit(
        f"missing {GRID_CSV} — rsync from Leonardo first:\n"
        "  rsync -az uaslam00@login.leonardo.cineca.it:"
        "/leonardo_work/AIFAC_P02_222/overture-map/outputs/"
        "source_dominance_grid.csv overture-map/outputs/"
    )

print(f"[load] {GRID_CSV}")
df = pd.read_csv(GRID_CSV)
print(f"[load] {len(df):,} populated cells")


# ----- shared map config ------------------------------------------------

# Open-street-map style — no API key, real country borders + city names.
MAPBOX_STYLE = "open-street-map"
INITIAL_CENTER = dict(lat=20, lon=10)
INITIAL_ZOOM = 1.5


def common_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, x=0.5, font=dict(size=16, color="#eee")),
        height=780,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="#0a0a0a",
        font=dict(color="#ddd"),
        mapbox=dict(
            style=MAPBOX_STYLE,
            center=INITIAL_CENTER,
            zoom=INITIAL_ZOOM,
        ),
    )


# ----- map 1: dominance ------------------------------------------------

def render_dominance_map(df: pd.DataFrame, out: Path) -> None:
    # Filter to cells with meaningful count — cells with <50 places are
    # noise (single villages, ferry terminals, etc.) and clutter the view.
    d = df[df["n_total"] >= 50].copy()
    print(f"[dom ] showing {len(d):,} cells with ≥50 places "
          f"({100 * len(d) / len(df):.0f}% of populated)")

    d["log2_clipped"] = d["log2_ratio"].clip(-5, 5)
    # Marker size scaled by log10(places) — caps so megacells don't dominate
    d["size"] = d["n_total"].apply(lambda n: max(4, min(28, math.log10(n) * 5)))
    d["hover"] = (
        "Overture: " + d["n_ovt"].map("{:,}".format)
        + "<br>Foursquare: " + d["n_fsq"].map("{:,}".format)
        + "<br>log₂(O/F): " + d["log2_ratio"].map("{:+.2f}".format)
        + "<br>country: " + d["country"].fillna("—")
    )

    fig = go.Figure(
        data=go.Scattermapbox(
            lat=d["lat_center"],
            lon=d["lon_center"],
            mode="markers",
            marker=dict(
                size=d["size"],
                color=d["log2_clipped"],
                colorscale="RdBu_r",
                cmin=-5,
                cmax=5,
                opacity=0.75,
                colorbar=dict(
                    title=dict(text="log₂(O/F)", font=dict(color="#ddd")),
                    tickvals=[-5, -2, 0, 2, 5],
                    ticktext=["FSQ ≥32×", "FSQ 4×", "balanced", "Ovt 4×", "Ovt ≥32×"],
                    tickfont=dict(color="#ddd"),
                    bgcolor="#222",
                    bordercolor="#444",
                    x=0.99,
                    xanchor="right",
                ),
            ),
            text=d["hover"],
            hoverinfo="text",
        )
    )
    fig.update_layout(**common_layout(
        "Overture vs Foursquare — POI source dominance (0.5° grid)"
    ))
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"[dom ] {out}  ({out.stat().st_size / 1e6:.1f} MB)")


# ----- map 2: total density (heatmap) ---------------------------------

def render_density_heatmap(df: pd.DataFrame, out: Path) -> None:
    d = df[df["n_total"] >= 10].copy()
    print(f"[dens] showing {len(d):,} cells with ≥10 places")

    fig = go.Figure(
        data=go.Densitymapbox(
            lat=d["lat_center"],
            lon=d["lon_center"],
            z=np.log10(d["n_total"]),
            radius=10,
            colorscale="Inferno",
            colorbar=dict(
                title=dict(text="log₁₀ places", font=dict(color="#ddd")),
                tickfont=dict(color="#ddd"),
                bgcolor="#222",
                bordercolor="#444",
                x=0.99,
                xanchor="right",
            ),
            opacity=0.85,
        )
    )
    fig.update_layout(**common_layout(
        "POI density — combined Overture + Foursquare (0.5° grid)"
    ))
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"[dens] {out}  ({out.stat().st_size / 1e6:.1f} MB)")


# ----- map 3: side-by-side (toggle layers) -----------------------------

def render_compare_map(df: pd.DataFrame, out: Path) -> None:
    # One map with TWO toggleable trace layers — Overture vs Foursquare.
    # User clicks legend to flip between sources for visual comparison.
    d_ovt = df[df["n_ovt"] >= 10].copy()
    d_fsq = df[df["n_fsq"] >= 10].copy()
    print(f"[cmp ] Ovt cells: {len(d_ovt):,}, FSQ cells: {len(d_fsq):,}")

    def layer(d: pd.DataFrame, col: str, name: str, scale: str) -> go.Densitymapbox:
        return go.Densitymapbox(
            name=name,
            lat=d["lat_center"],
            lon=d["lon_center"],
            z=np.log10(d[col]),
            radius=10,
            colorscale=scale,
            opacity=0.85,
            showscale=False,
            hovertemplate=f"{name}<br>%{{z:.2f}} log₁₀ places<extra></extra>",
        )

    fig = go.Figure()
    fig.add_trace(layer(d_ovt, "n_ovt", "Overture",   "Blues"))
    fig.add_trace(layer(d_fsq, "n_fsq", "Foursquare", "Reds"))
    # Default: show only Overture; user toggles via legend
    fig.data[1].visible = "legendonly"
    fig.update_layout(
        **common_layout("Overture (Blue) vs Foursquare (Red) — toggle in legend"),
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(20,20,20,0.85)",
            font=dict(color="#ddd", size=13),
            bordercolor="#444",
            borderwidth=1,
        ),
    )
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"[cmp ] {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    render_dominance_map(df,    OUTPUTS / "dominance_map.html")
    render_density_heatmap(df,  OUTPUTS / "density_map.html")
    render_compare_map(df,      OUTPUTS / "dominance_compare.html")
    print("\nopen any of the .html files in your browser — pan/zoom/hover.")


if __name__ == "__main__":
    main()
