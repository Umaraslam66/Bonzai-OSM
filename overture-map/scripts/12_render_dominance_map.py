"""Render outputs/source_dominance_grid.csv as world maps.

Produces:
  - outputs/source_dominance_map.png       static, log2(Ovt/FSQ) diverging
  - outputs/source_density_overture.png    static, Overture density (log)
  - outputs/source_density_foursquare.png  static, Foursquare density (log)
  - outputs/source_dominance_map.html      interactive Plotly raster heatmap

Uses matplotlib + plotly.imshow (raster, not vector), so 80 k cells
render in a fraction of a second instead of grinding through individual
SVG markers.

Run locally on the Mac after rsyncing the CSV from Leonardo:
    .venv/bin/python scripts/12_render_dominance_map.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
GRID_CSV = OUTPUTS / "source_dominance_grid.csv"

# 0.5° grid → 360 × 720 raster
GRID = 2
NLAT = 180 * GRID
NLON = 360 * GRID
LAT_OFFSET = 90 * GRID
LON_OFFSET = 180 * GRID
EXTENT = [-180.0, 180.0, -90.0, 90.0]


if not GRID_CSV.exists():
    raise SystemExit(
        f"missing {GRID_CSV} — rsync it from Leonardo first:\n"
        "  rsync -az uaslam00@login.leonardo.cineca.it:"
        "/leonardo_work/AIFAC_P02_222/overture-map/outputs/"
        "source_dominance_grid.csv overture-map/outputs/"
    )

print(f"[load] {GRID_CSV}")
df = pd.read_csv(GRID_CSV)
print(f"[load] {len(df):,} populated cells")


def to_raster(df: pd.DataFrame, col: str) -> np.ndarray:
    """Pivot the long CSV into a (NLAT × NLON) raster, NaN where no cell."""
    arr = np.full((NLAT, NLON), np.nan, dtype=np.float64)
    iy = (df["gy"].to_numpy() + LAT_OFFSET).astype(np.int32)
    ix = (df["gx"].to_numpy() + LON_OFFSET).astype(np.int32)
    valid = (iy >= 0) & (iy < NLAT) & (ix >= 0) & (ix < NLON)
    arr[iy[valid], ix[valid]] = df[col].to_numpy()[valid]
    return arr


def render_dominance_png(df: pd.DataFrame, out: Path) -> None:
    n_ovt = to_raster(df, "n_ovt")
    n_fsq = to_raster(df, "n_fsq")
    # log2 ratio with +1 smoothing; mask cells where both empty
    ratio = np.log2((np.nan_to_num(n_ovt) + 1.0) / (np.nan_to_num(n_fsq) + 1.0))
    empty = np.isnan(n_ovt) & np.isnan(n_fsq)
    ratio[empty] = np.nan

    fig, ax = plt.subplots(figsize=(20, 10), dpi=110)
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-5.0, vmax=5.0)
    im = ax.imshow(
        np.flipud(ratio),
        cmap="RdBu_r",
        norm=norm,
        extent=EXTENT,
        interpolation="nearest",
    )
    ax.set_xlabel("longitude", color="#ddd")
    ax.set_ylabel("latitude", color="#ddd")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.set_title(
        "Overture vs Foursquare — POI source dominance "
        "(log₂(Ovt/FSQ), 0.5° grid)",
        color="#ddd",
        fontsize=14,
        pad=12,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_ticks([-5, -2, 0, 2, 5])
    cbar.set_ticklabels(
        ["FSQ ≥32×", "FSQ 4×", "balanced", "Ovt 4×", "Ovt ≥32×"]
    )
    cbar.ax.tick_params(colors="#ddd")
    cbar.outline.set_edgecolor("#444")
    plt.tight_layout()
    plt.savefig(out, facecolor=fig.get_facecolor(), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[png ] {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def render_density_png(df: pd.DataFrame, col: str, label: str, out: Path) -> None:
    arr = to_raster(df, col)
    fig, ax = plt.subplots(figsize=(20, 10), dpi=110)
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")
    norm = LogNorm(vmin=1, vmax=max(np.nanmax(arr), 10))
    im = ax.imshow(
        np.flipud(arr),
        cmap="viridis",
        norm=norm,
        extent=EXTENT,
        interpolation="nearest",
    )
    ax.set_xlabel("longitude", color="#ddd")
    ax.set_ylabel("latitude", color="#ddd")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.set_title(
        f"{label} POI density (log scale, 0.5° grid)",
        color="#ddd",
        fontsize=14,
        pad=12,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(f"{label} places per cell", color="#ddd")
    cbar.ax.tick_params(colors="#ddd")
    cbar.outline.set_edgecolor("#444")
    plt.tight_layout()
    plt.savefig(out, facecolor=fig.get_facecolor(), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[png ] {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def render_dominance_html(df: pd.DataFrame, out: Path) -> None:
    """Interactive Plotly raster heatmap — single image in HTML, fast.

    Uses plotly.express.imshow which renders the whole grid as one
    raster image, not 80 k SVG markers.  Hover gives lat/lon and value.
    """
    import plotly.express as px

    n_ovt = to_raster(df, "n_ovt")
    n_fsq = to_raster(df, "n_fsq")
    ratio = np.log2((np.nan_to_num(n_ovt) + 1.0) / (np.nan_to_num(n_fsq) + 1.0))
    empty = np.isnan(n_ovt) & np.isnan(n_fsq)
    ratio[empty] = np.nan

    fig = px.imshow(
        np.flipud(ratio),
        x=np.linspace(-180, 180, NLON),
        y=np.linspace(90, -90, NLAT),
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        zmin=-5,
        zmax=5,
        origin="upper",
        labels=dict(x="longitude", y="latitude", color="log₂(O/F)"),
        title="Overture vs Foursquare — POI source dominance (0.5° grid)",
    )
    fig.update_layout(
        height=720,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        font=dict(color="#ddd"),
    )
    fig.update_xaxes(gridcolor="#333", zerolinecolor="#444")
    fig.update_yaxes(gridcolor="#333", zerolinecolor="#444", scaleanchor="x", scaleratio=1)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"[html] {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    render_dominance_png(df, OUTPUTS / "source_dominance_map.png")
    render_density_png(df, "n_ovt", "Overture", OUTPUTS / "source_density_overture.png")
    render_density_png(df, "n_fsq", "Foursquare", OUTPUTS / "source_density_foursquare.png")
    render_dominance_html(df, OUTPUTS / "source_dominance_map.html")


if __name__ == "__main__":
    main()
