"""Render outputs/source_dominance_grid.csv as an interactive HTML world map.

Run locally on the Mac after `rsync`-ing the CSV from Leonardo:
    .venv/bin/python scripts/12_render_dominance_map.py

Produces three HTMLs side-by-side under outputs/:
  - source_dominance_map.html        log2(ovt/fsq) coloured per cell
  - source_density_overture.html     Overture density only (log scale)
  - source_density_foursquare.html   Foursquare density only (log scale)

Uses Plotly Scattergeo so the output is a single self-contained file you
open in any browser — pan, zoom, hover.
"""
from __future__ import annotations
import math
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
GRID_CSV = OUTPUTS / "source_dominance_grid.csv"

if not GRID_CSV.exists():
    raise SystemExit(
        f"missing {GRID_CSV} — rsync it from Leonardo first:\n"
        "  rsync -az uaslam00@login.leonardo.cineca.it:"
        "/leonardo_work/AIFAC_P02_222/overture-map/outputs/"
        "source_dominance_grid.csv overture-map/outputs/"
    )

print(f"[load] {GRID_CSV}")
df = pd.read_csv(GRID_CSV)
print(f"[load] {len(df):,} cells")


def render_dominance(df: pd.DataFrame, out: Path) -> None:
    # Colour scale: red (FSQ) → white (balanced) → blue (Overture)
    # log2_ratio range mostly -10..+10; clip to ±5 for colour readability
    df = df.copy()
    df["log2_clipped"] = df["log2_ratio"].clip(-5, 5)
    df["size"] = df["n_total"].apply(lambda n: max(2, min(18, math.log10(n + 1) * 4)))
    df["hover"] = df.apply(
        lambda r: (
            f"({r.lat_center:+.2f}, {r.lon_center:+.2f})<br>"
            f"Overture: {int(r.n_ovt):,}<br>"
            f"Foursquare: {int(r.n_fsq):,}<br>"
            f"log2(O/F): {r.log2_ratio:+.2f}<br>"
            f"country: {r.country if isinstance(r.country, str) else '—'}"
        ),
        axis=1,
    )

    fig = go.Figure(
        data=go.Scattergeo(
            lat=df["lat_center"],
            lon=df["lon_center"],
            mode="markers",
            marker=dict(
                size=df["size"],
                color=df["log2_clipped"],
                colorscale="RdBu",
                cmin=-5,
                cmax=5,
                showscale=True,
                colorbar=dict(
                    title="log2(Ovt / FSQ)",
                    tickvals=[-5, -2, 0, 2, 5],
                    ticktext=["FSQ ≥32×", "FSQ 4×", "balanced", "Ovt 4×", "Ovt ≥32×"],
                ),
                line=dict(width=0),
                opacity=0.75,
            ),
            text=df["hover"],
            hoverinfo="text",
        )
    )
    fig.update_geos(
        projection_type="natural earth",
        showcountries=True,
        countrycolor="#666",
        showland=True,
        landcolor="#1a1a1a",
        showocean=True,
        oceancolor="#0a0a0a",
        showcoastlines=True,
        coastlinecolor="#888",
        bgcolor="#000",
    )
    fig.update_layout(
        title=dict(
            text=(
                "Overture vs Foursquare — POI source dominance "
                "(0.5° grid, log<sub>2</sub> ratio)"
            ),
            x=0.5,
        ),
        height=720,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="#000",
        font=dict(color="#ddd"),
    )
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"[render] {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def render_density(df: pd.DataFrame, col: str, label: str, out: Path) -> None:
    df = df[df[col] > 0].copy()
    df["log_n"] = df[col].apply(lambda n: math.log10(n + 1))
    df["size"] = df["log_n"].clip(0, 6) * 3 + 2
    df["hover"] = df.apply(
        lambda r: (
            f"({r.lat_center:+.2f}, {r.lon_center:+.2f})<br>"
            f"{label}: {int(r[col]):,}<br>"
            f"country: {r.country if isinstance(r.country, str) else '—'}"
        ),
        axis=1,
    )

    fig = go.Figure(
        data=go.Scattergeo(
            lat=df["lat_center"],
            lon=df["lon_center"],
            mode="markers",
            marker=dict(
                size=df["size"],
                color=df["log_n"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=f"log10({label})"),
                line=dict(width=0),
                opacity=0.8,
            ),
            text=df["hover"],
            hoverinfo="text",
        )
    )
    fig.update_geos(
        projection_type="natural earth",
        showcountries=True,
        countrycolor="#666",
        showland=True,
        landcolor="#1a1a1a",
        showocean=True,
        oceancolor="#0a0a0a",
        showcoastlines=True,
        coastlinecolor="#888",
        bgcolor="#000",
    )
    fig.update_layout(
        title=dict(text=f"{label} POI density (0.5° grid, log scale)", x=0.5),
        height=720,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="#000",
        font=dict(color="#ddd"),
    )
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"[render] {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    render_dominance(df, OUTPUTS / "source_dominance_map.html")
    render_density(df, "n_ovt", "Overture", OUTPUTS / "source_density_overture.html")
    render_density(df, "n_fsq", "Foursquare", OUTPUTS / "source_density_foursquare.html")


if __name__ == "__main__":
    main()
