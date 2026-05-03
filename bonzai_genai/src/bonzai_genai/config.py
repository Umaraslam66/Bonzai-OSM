"""Global configuration constants for bonzai_genai.

These values are load-bearing for the entire pipeline; changing them
requires rerunning data prep and retraining models.
"""

# Tile geometry
TILE_SIDE_M: float = 2048.0  # metres
RASTER_PX: int = 512         # pixels per side
METRES_PER_PX: float = TILE_SIDE_M / RASTER_PX  # 4.0 m/px

# Coordinate quantisation for vector tokens
COORD_BINS: int = 512        # one quantisation bin per raster pixel

# Raster channel layout (order matters — used as channel index)
CHANNEL_NAMES: tuple[str, ...] = (
    "all_roads",        # 0 — every road regardless of class
    "major_roads",      # 1 — motorway / trunk / primary
    "mid_roads",        # 2 — secondary / tertiary
    "minor_roads",      # 3 — residential / service
    "buildings",        # 4 — building footprint mask
    "building_density", # 5 — Gaussian-blurred footprints (continuous)
    "water",            # 6 — rivers / lakes / ocean
    "green",            # 7 — parks / forests / fields / agricultural
    "urban",            # 8 — built urban land use
)
NUM_CHANNELS: int = len(CHANNEL_NAMES)

# Building density blur sigma (in raster pixels)
BUILDING_DENSITY_SIGMA_PX: float = 32.0

# Stratification buckets
DENSITY_BUCKETS: tuple[str, ...] = ("rural", "suburban", "urban", "dense_urban")
KOPPEN_GROUPS: tuple[str, ...] = ("tropical", "arid", "temperate", "continental", "polar")
