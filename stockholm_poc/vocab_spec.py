"""
vocab_spec.py
=============

Central definition of the Spatial Foundation Model vocabulary. Both
`tokenize_stockholm.py` and `generate_vocab_dict.py` import from here
so the tokenizer, the model, and the documentation can never drift
out of sync.

Token families
--------------
1. Geometry  (dx/dy)     — relative 1-meter delta, range [-32, +32]
2. Anchor    (X/Y)       — chunk-local 256x256 grid position
3. Semantic  (TAG_K_V)   — <TAG_<KEY>_<VALUE>> with bucketing
4. Attribute  (various)  — road/building/POI qualifiers
5. Structural            — KIND_START / KIND_END / PART_SEP
6. Macro     (CONTEXT)   — region / climate / density prefix

Target ceiling: 4,096 unique tokens. Current spec lands at ~1,100.

Grammar (informal)
------------------
corpus         := context_row (object_row)+
context_row    := CONTEXT_START REGION? CLIMATE? DENSITY? CONTEXT_END
object_row     := KIND_START tag (extra_attr)* X Y geometry? KIND_END
geometry       := (dx dy)*                              (for POI: none)
                  | (dx dy)+ (PART_SEP X Y (dx dy)+)*   (for polygons)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# 1. GEOMETRY — delta x/y tokens
# ---------------------------------------------------------------------------

# dx/dy range in meters. A single token covers one step; longer edges
# are expressed as multiple (dx, dy) token pairs. 33 values per side +
# 0 = 65 per axis.
DELTA_RANGE = 32  # inclusive on both ends

# ---------------------------------------------------------------------------
# 2. ANCHOR — chunk-local X/Y grid
# ---------------------------------------------------------------------------

GRID_SIZE = 256  # 0..255 per axis

# ---------------------------------------------------------------------------
# 3. SEMANTIC — TAG_KEY_VALUE
# ---------------------------------------------------------------------------

# Every accepted OSM value gets its own token. Values outside each list
# collapse to `<TAG_<KEY>_OTHER>`. The `name=*` and `operator=*` tags
# are intentionally excluded (high-cardinality, low-generalization).

TAG_ALLOWED_VALUES: Dict[str, List[str]] = {
    # -----------------------------------------------------------------
    "building": [
        # Top 30+ from TagInfo (covers 99%+ of global buildings)
        "yes", "house", "residential", "detached", "garage",
        "apartments", "shed", "industrial", "hut", "roof",
        "commercial", "retail", "office", "warehouse", "school",
        "church", "hospital", "farm_auxiliary", "kindergarten",
        "university", "college", "cabin", "greenhouse", "barn",
        "service", "hangar", "stadium", "public", "civic",
        "construction", "garages", "static_caravan", "bungalow",
        "terrace", "farm", "semidetached_house", "train_station",
        "chapel", "mosque", "temple",
    ],
    # -----------------------------------------------------------------
    "highway": [
        "residential", "service", "footway", "track", "unclassified",
        "path", "tertiary", "secondary", "primary", "trunk",
        "motorway", "living_street", "pedestrian", "cycleway",
        "steps", "motorway_link", "trunk_link", "primary_link",
        "secondary_link", "tertiary_link", "road", "construction",
        "bridleway", "raceway",
    ],
    # -----------------------------------------------------------------
    "amenity": [
        # Transport & parking
        "parking", "parking_space", "parking_entrance", "bicycle_parking",
        "motorcycle_parking", "fuel", "charging_station", "bus_station",
        "taxi", "ferry_terminal", "bicycle_rental", "car_rental",
        "car_sharing", "boat_rental", "car_wash",
        # Food & beverage
        "restaurant", "cafe", "fast_food", "bar", "pub",
        "ice_cream", "biergarten", "food_court",
        # Education
        "school", "kindergarten", "university", "college",
        "library", "language_school", "music_school", "driving_school",
        "training",
        # Healthcare
        "hospital", "clinic", "pharmacy", "doctors", "dentist",
        "veterinary", "nursing_home",
        # Financial
        "bank", "atm", "bureau_de_change",
        # Civic / public services
        "townhall", "courthouse", "post_office", "post_box", "embassy",
        "public_building", "police", "fire_station", "prison",
        # Culture & entertainment
        "cinema", "theatre", "nightclub", "arts_centre", "casino",
        "community_centre", "events_venue", "studio", "social_centre",
        "social_facility", "planetarium", "exhibition_centre",
        # Worship
        "place_of_worship", "grave_yard", "monastery",
        # Accommodation (amenity-side; most lives under tourism=*)
        "hotel", "hostel", "guest_house", "motel",
        # Street furniture / low-signal-but-common
        "bench", "drinking_water", "toilets", "waste_basket",
        "waste_disposal", "recycling", "shelter", "telephone",
        "clock", "bbq", "fountain", "vending_machine",
        "hunting_stand", "watering_place",
        # Recreation
        "playground", "swimming_pool", "sports_centre",
        "fitness_centre", "gym", "dojo", "dance",
        # Miscellaneous common
        "animal_shelter", "animal_boarding", "veterinary",
        "internet_cafe", "marketplace", "stripclub", "brothel",
        "childcare", "crematorium",
    ],
    # -----------------------------------------------------------------
    "shop": [
        # Top ~60 shop values covering >98% of global shop=* occurrences
        "supermarket", "convenience", "clothes", "hairdresser",
        "car_repair", "bakery", "yes", "beauty", "car", "hardware",
        "butcher", "jewelry", "shoes", "mobile_phone", "florist",
        "bicycle", "tobacco", "stationery", "books", "optician",
        "electronics", "mall", "furniture", "computer", "toys",
        "sports", "chemist", "wine", "pet", "doityourself",
        "laundry", "gift", "department_store", "music",
        "garden_centre", "kiosk", "alcohol", "greengrocer", "tattoo",
        "seafood", "travel_agency", "cosmetics", "frame", "kitchen",
        "bed", "interior_decoration", "car_parts", "outdoor",
        "lighting", "craft", "pastry", "tyres", "cheese",
        "deli", "confectionery", "second_hand", "dry_cleaning",
        "mobile", "houseware", "copyshop", "photo", "hearing_aids",
        "bag", "tea", "coffee", "video_games", "antiques",
        "watches", "motorcycle",
    ],
    # -----------------------------------------------------------------
    "landuse": [
        "farmland", "residential", "grass", "forest", "meadow",
        "orchard", "farmyard", "industrial", "vineyard", "cemetery",
        "commercial", "retail", "allotments", "construction",
        "basin", "recreation_ground", "brownfield", "greenfield",
        "military", "landfill", "religious", "quarry", "education",
        "village_green", "garden", "plant_nursery", "aquaculture",
        "port", "depot", "salt_pond",
    ],
    # -----------------------------------------------------------------
    "natural": [
        # Covers both point and area values; geometry routing decides
        # which kind (POI vs LANDUSE vs NATURAL_LINE) emits the token.
        "tree", "water", "wood", "scrub", "wetland", "grassland",
        "tree_row", "coastline", "bare_rock", "peak", "cliff",
        "rock", "ridge", "valley", "cave_entrance", "stone",
        "hill", "saddle", "glacier", "spring", "heath",
        "sand", "beach", "shingle", "scree", "fell",
        "bay", "strait", "reef", "volcano", "geyser",
        "arete", "sinkhole", "mud",
    ],
    # -----------------------------------------------------------------
    "waterway": [
        "stream", "river", "ditch", "drain", "canal",
        "weir", "dam", "riverbank", "tidal_channel", "lock_gate",
        "waterfall", "fish_pass",
    ],
    # -----------------------------------------------------------------
    "railway": [
        "rail", "abandoned", "tram", "disused", "subway",
        "narrow_gauge", "light_rail", "construction", "preserved",
        "miniature", "monorail", "funicular",
    ],
    # -----------------------------------------------------------------
    "leisure": [
        "pitch", "swimming_pool", "park", "playground",
        "sports_centre", "track", "garden", "fitness_centre",
        "stadium", "golf_course", "marina", "nature_reserve",
        "fishing", "common", "dog_park", "horse_riding",
        "ice_rink", "miniature_golf", "bird_hide",
        "adult_gaming_centre", "slipway", "water_park",
        "picnic_table", "outdoor_seating", "fitness_station",
        "beach_resort", "firepit", "sauna",
    ],
    # -----------------------------------------------------------------
    "public_transport": [
        "stop_position", "platform", "station", "stop_area",
        "pay_telephone",
    ],
    # -----------------------------------------------------------------
    "tourism": [
        "information", "hotel", "attraction", "viewpoint",
        "guest_house", "artwork", "picnic_site", "hostel",
        "museum", "camp_site", "chalet", "apartment",
        "motel", "theme_park", "zoo", "aquarium", "gallery",
        "wilderness_hut", "camp_pitch", "caravan_site",
    ],
    # -----------------------------------------------------------------
    "historic": [
        "memorial", "monument", "archaeological_site",
        "wayside_cross", "wayside_shrine", "castle", "ruins",
        "tomb", "boundary_stone", "city_gate", "tower",
        "manor", "church", "fort",
    ],
}

# Tag keys that can appear on a point (node) as a POI. Precedence:
# first-match wins, so ordering here matters.
POI_TAG_KEYS: Tuple[str, ...] = (
    "amenity", "shop", "natural", "leisure",
    "tourism", "public_transport", "historic",
)

# Tag keys that define a LANDUSE polygon. Precedence: first match wins.
LANDUSE_TAG_KEYS: Tuple[str, ...] = ("landuse", "natural", "leisure")

# Values of `natural` that are legitimately *lines* (the ones we route
# to the NATURAL_LINE kind rather than LANDUSE).
NATURAL_LINE_VALUES: Tuple[str, ...] = (
    "coastline", "tree_row", "cliff", "ridge", "arete",
    "valley", "gully", "earth_bank",
)

# ---------------------------------------------------------------------------
# 4. ATTRIBUTES — per-feature qualifiers
# ---------------------------------------------------------------------------

# Each attribute family has its own set of bucketed values. Missing
# values simply mean "no token emitted", so absence is meaningful too.

SURFACE_VALUES: List[str] = [
    "asphalt", "concrete", "paving_stones", "paved", "sett",
    "cobblestone", "unpaved", "gravel", "compacted", "dirt",
    "ground", "sand", "grass", "wood", "metal", "other",
]

LIT_VALUES: List[str] = ["yes", "no", "24_7", "automatic", "sunset_sunrise"]

LANES_VALUES: List[str] = ["1", "2", "3", "4", "5_plus"]

ACCESS_VALUES: List[str] = [
    "yes", "no", "private", "permissive", "customers",
    "agricultural", "destination", "forestry", "other",
]

ONEWAY_VALUES: List[str] = ["yes", "no", "reverse"]

# Bridge/tunnel are binary — we only emit tokens when truthy, so the
# list below is the set of truthy values accepted by the parser.
BRIDGE_TRUTHY = {"yes", "true", "1", "viaduct", "aqueduct", "movable"}
TUNNEL_TRUTHY = {"yes", "true", "1", "culvert", "building_passage"}

LEVELS_VALUES: List[str] = ["1_2", "3_5", "6_10", "11_plus"]

SPEED_VALUES: List[str] = ["low", "mid", "high"]   # <40, 40-70, >70 kph

HEIGHT_VALUES: List[str] = ["low", "mid", "high", "tall"]  # <10m, 10-25m, 25-75m, 75+m

# ---------------------------------------------------------------------------
# 5. STRUCTURAL — kinds & separators
# ---------------------------------------------------------------------------

KINDS: Tuple[str, ...] = (
    "CONTEXT",
    "BUILDING",
    "ROAD",
    "POI",
    "LANDUSE",
    "WATERWAY",
    "RAILWAY",
    "NATURAL_LINE",
)

# ---------------------------------------------------------------------------
# 6. MACRO CONTEXT — system-prompt-like prefix
# ---------------------------------------------------------------------------

REGIONS: List[str] = [
    "EUROPE", "ASIA", "AFRICA",
    "NORTH_AMERICA", "SOUTH_AMERICA",
    "OCEANIA", "ANTARCTICA",
]

CLIMATES: List[str] = [
    "TROPICAL", "ARID", "TEMPERATE", "CONTINENTAL", "POLAR",
]

DENSITIES: List[str] = [
    "URBAN", "SUBURBAN", "RURAL", "WILDERNESS",
]


# ---------------------------------------------------------------------------
# Token enumeration helpers
# ---------------------------------------------------------------------------


def _sanitize(value: str) -> str:
    """Normalise an OSM value into a token-safe uppercase form. We keep
    digits, replace any run of non-alphanumerics with a single '_', and
    upper-case the rest. This is the same transformation the tokenizer
    applies when emitting <TAG_KEY_VALUE>.
    """
    import re
    v = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_")
    return v.upper() if v else "OTHER"


def enumerate_geometry_tokens() -> List[str]:
    """Return all dx/dy tokens in canonical order."""
    out: List[str] = []
    for axis in ("dx", "dy"):
        for v in range(-DELTA_RANGE, DELTA_RANGE + 1):
            out.append(f"<{axis}_{v}>")
    return out


def enumerate_anchor_tokens() -> List[str]:
    out: List[str] = []
    for axis in ("X", "Y"):
        for v in range(GRID_SIZE):
            out.append(f"<{axis}_{v}>")
    return out


def enumerate_tag_tokens() -> List[str]:
    out: List[str] = []
    for key, values in TAG_ALLOWED_VALUES.items():
        key_upper = key.upper()
        for v in values:
            out.append(f"<TAG_{key_upper}_{_sanitize(v)}>")
        out.append(f"<TAG_{key_upper}_OTHER>")
    return out


def enumerate_attribute_tokens() -> List[str]:
    families = [
        ("SURFACE", SURFACE_VALUES),
        ("LIT", LIT_VALUES),
        ("LANES", LANES_VALUES),
        ("ACCESS", ACCESS_VALUES),
        ("ONEWAY", ONEWAY_VALUES),
        ("LEVELS", LEVELS_VALUES),
        ("SPEED", SPEED_VALUES),
        ("HEIGHT", HEIGHT_VALUES),
    ]
    out: List[str] = []
    for fam, values in families:
        for v in values:
            out.append(f"<{fam}_{v.upper()}>")
    # Bridge/tunnel only have YES variants (absence = not bridge/tunnel).
    out.append("<BRIDGE_YES>")
    out.append("<TUNNEL_YES>")
    return out


def enumerate_structural_tokens() -> List[str]:
    out: List[str] = []
    for kind in KINDS:
        out.append(f"<{kind}_START>")
        out.append(f"<{kind}_END>")
    out.append("<PART_SEP>")
    return out


def enumerate_macro_tokens() -> List[str]:
    out: List[str] = []
    for r in REGIONS:
        out.append(f"<REGION_{r}>")
    for c in CLIMATES:
        out.append(f"<CLIMATE_{c}>")
    for d in DENSITIES:
        out.append(f"<DENSITY_{d}>")
    return out


def enumerate_full_vocabulary() -> List[str]:
    """Every token the tokenizer can possibly emit, in stable order."""
    tokens: List[str] = []
    tokens += enumerate_macro_tokens()
    tokens += enumerate_structural_tokens()
    tokens += enumerate_anchor_tokens()
    tokens += enumerate_geometry_tokens()
    tokens += enumerate_tag_tokens()
    tokens += enumerate_attribute_tokens()
    # Stable ordering + dedup (shouldn't be dupes but be safe).
    seen = set()
    uniq: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def vocabulary_size() -> int:
    return len(enumerate_full_vocabulary())
