"""Generate tile bundles into WebDataset shards.

Three modes:
    synthetic        — procedural smoke data (single tile count)
    synth-corpus     — Experiment 0 mixed sparse/dense corpus, train+val split
    overture-region  — real OSM data for a bbox
"""
from __future__ import annotations

import random
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

from bonzai_genai.data.rasteriser import rasterise
from bonzai_genai.data.shard_writer import ShardWriter
from bonzai_genai.data.tile_bundle import TileBundle, TileMetadata
from bonzai_genai.synth.procedural import generate_synthetic_tile
from bonzai_genai.vocab.attributes import load_default_vocab
from bonzai_genai.vocab.tokeniser import Tokeniser

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
console = Console()


@app.command("synthetic")
def cmd_synthetic(
    output: Path = typer.Option(..., "-o", "--output", help="Output directory for shards"),  # noqa: B008
    n: int = typer.Option(100, "-n", help="Number of synthetic tiles"),  # noqa: B008
    shard_size: int = typer.Option(50, "--shard-size"),  # noqa: B008
    seed_base: int = typer.Option(0, "--seed-base"),  # noqa: B008
) -> None:
    """Generate n synthetic procedural tiles into WebDataset shards."""
    vocab = load_default_vocab()
    tokeniser = Tokeniser(vocab)
    writer = ShardWriter(output, shard_size=shard_size)
    with Progress(console=console) as progress:
        task_id = progress.add_task("[green]Generating", total=n)
        for i in range(n):
            geom = generate_synthetic_tile(seed=seed_base + i)
            raster = rasterise(geom)
            tokens = tokeniser.encode(geom)
            meta = TileMetadata(
                tile_id=f"SYN-{i:06d}",
                sw_lat=0.0, sw_lon=0.0,
                country="SYN", koppen="N/A",
                density_bucket="urban", primary_land_use="residential",
            )
            writer.write(TileBundle(raster=raster, tokens=tokens, metadata=meta))
            progress.update(task_id, advance=1)
    writer.close()
    console.print(f"[bold green]Wrote {n} synthetic tiles to {output}")


@app.command("overture-region")
def cmd_overture_region(
    pbf: Path = typer.Option(..., help="Path to .osm.pbf file"),  # noqa: B008
    sw_lat: float = typer.Option(..., help="SW corner latitude"),  # noqa: B008
    sw_lon: float = typer.Option(..., help="SW corner longitude"),  # noqa: B008
    ne_lat: float = typer.Option(..., help="NE corner latitude"),  # noqa: B008
    ne_lon: float = typer.Option(..., help="NE corner longitude"),  # noqa: B008
    output: Path = typer.Option(..., "-o", "--output"),  # noqa: B008
    country: str = typer.Option("SG", "--country"),  # noqa: B008
    koppen: str = typer.Option("Af", "--koppen"),  # noqa: B008
    shard_size: int = typer.Option(100, "--shard-size"),  # noqa: B008
    max_tiles: int = typer.Option(1000, "--max-tiles"),  # noqa: B008
) -> None:
    """Generate tile bundles for every tile in (sw, ne) bbox from an OSM PBF.

    Used for Phase 0a Sweden + Singapore + Sri Lanka validation runs.
    """
    from bonzai_genai.data.sampling import (
        extract_tile_from_features,
        iter_tile_centres,
        load_pbf_features,
    )

    vocab = load_default_vocab()
    tokeniser = Tokeniser(vocab)
    writer = ShardWriter(output, shard_size=shard_size)

    centres = list(iter_tile_centres(sw_lat, sw_lon, ne_lat, ne_lon))[:max_tiles]
    console.print(f"Processing {len(centres)} tiles from {pbf.name}")

    console.print("Scanning PBF (one-time pass)...")
    features = load_pbf_features(pbf, country_bbox=(sw_lon, sw_lat, ne_lon, ne_lat))
    console.print(
        f"  loaded: {len(features.roads)} roads, {len(features.buildings)} buildings, "
        f"{len(features.land)} land polys, {len(features.pois)} POIs"
    )

    n_kept = 0
    n_skipped = 0
    with Progress(console=console) as progress:
        task_id = progress.add_task("[green]Extracting", total=len(centres))
        for i, (lat, lon) in enumerate(centres):
            try:
                geom = extract_tile_from_features(features, lat, lon)
            except Exception as e:
                console.print(f"  [yellow]skip {i}: {e}")
                n_skipped += 1
                progress.update(task_id, advance=1)
                continue
            if len(geom.roads) + len(geom.buildings) < 5:
                # Skip near-empty tiles
                n_skipped += 1
                progress.update(task_id, advance=1)
                continue
            try:
                raster = rasterise(geom)
                tokens = tokeniser.encode(geom)
            except KeyError as e:
                console.print(f"  [yellow]vocab miss tile {i}: {e}")
                n_skipped += 1
                progress.update(task_id, advance=1)
                continue
            except ValueError as e:
                # Tile has >NUM_NODE_REF_TOKENS (8192) unique road nodes.
                # Genuinely extreme density (e.g. central Marina Bay multi-level
                # interchanges); expected to be rare outside that band.
                console.print(f"  [yellow]encode overflow tile {i}: {e}")
                n_skipped += 1
                progress.update(task_id, advance=1)
                continue
            meta = TileMetadata(
                tile_id=f"{country}-{i:06d}",
                sw_lat=lat, sw_lon=lon,
                country=country, koppen=koppen,
                density_bucket="urban",
                primary_land_use="residential",
            )
            writer.write(TileBundle(raster=raster, tokens=tokens, metadata=meta))
            n_kept += 1
            progress.update(task_id, advance=1)
    writer.close()
    console.print(f"[bold green]Kept {n_kept} tiles, skipped {n_skipped}")


@app.command("synth-corpus")
def cmd_synth_corpus(
    output: Path = typer.Option(..., "-o", "--output"),  # noqa: B008
    n_train: int = typer.Option(4500, "--n-train"),  # noqa: B008
    n_val: int = typer.Option(500, "--n-val"),  # noqa: B008
    shard_size: int = typer.Option(500, "--shard-size"),  # noqa: B008
    seed_base: int = typer.Option(0, "--seed-base"),  # noqa: B008
) -> None:
    """Generate Experiment 0 synthetic corpus: mixed sparse/dense density."""
    vocab = load_default_vocab()
    tokeniser = Tokeniser(vocab)
    train_dir = output / "train"
    val_dir = output / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    train_w = ShardWriter(train_dir, shard_size=shard_size)
    val_w = ShardWriter(val_dir, shard_size=shard_size)
    rng = random.Random(seed_base)

    def _emit(writer: ShardWriter, n: int, prefix: str, base: int) -> None:
        with Progress(console=console) as progress:
            task_id = progress.add_task(f"[green]{prefix}", total=n)
            for i in range(n):
                density = "dense" if rng.random() < 0.6 else "sparse"
                geom = generate_synthetic_tile(seed=base + i, density=density)
                raster = rasterise(geom)
                tokens = tokeniser.encode(geom)
                meta = TileMetadata(
                    tile_id=f"{prefix}-{i:06d}",
                    sw_lat=0.0, sw_lon=0.0,
                    country="SYN", koppen="N/A",
                    density_bucket=density,
                    primary_land_use="mixed",
                )
                writer.write(TileBundle(raster=raster, tokens=tokens, metadata=meta))
                progress.update(task_id, advance=1)

    _emit(train_w, n_train, "SYN-T", seed_base)
    _emit(val_w, n_val, "SYN-V", seed_base + n_train)
    train_w.close()
    val_w.close()
    console.print(f"[bold green]Wrote {n_train} train + {n_val} val to {output}")


if __name__ == "__main__":
    app()
