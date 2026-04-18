# Context & Objective
You are an expert Machine Learning Data Engineer. We are building a "Spatial Foundation Model" (SFM) designed to understand the structural logic of human infrastructure. Our goal is to train an autoregressive Transformer model to generate procedural, physically logical city infrastructure (roads, buildings) using causal language modeling techniques.

Before we scale to our 86GB global dataset on our EuroHPC cluster (Leonardo), we are doing a Proof of Concept (PoC) on a single city: Stockholm. 

Your objective right now is to build the data extraction and tokenization pipeline that converts raw OpenStreetMap (OSM) `.pbf` data into a 1D sequence of "spatial tokens" that a standard LLM can ingest.

# Task 1: Version Control
Create and checkout a new git branch called `feature/stockholm-poc`. All your work for this prompt should happen here.

# Task 2: HPC Context
We are deploying this on the CINECA Leonardo Booster cluster. Read the documentation in the local directory to understand how SLURM jobs, environment modules, and file paths work on this specific cluster. Write your scripts assuming they will be executed via SLURM on a Leonardo compute node.

# Task 3: Workspace & Data Acquisition
1. Write a setup script (`setup_workspace.sh`) to create a new directory on the Leonardo `WORK` filesystem (e.g., `/leonardo/home/userexternal/.../stockholm_poc`).
2. Do NOT use the global 86GB `.pbf` file we already downloaded. Instead, write a command in your script to download the specific Stockholm `.pbf` extract from BBBike directly into this new folder:
   URL: `https://download.bbbike.org/osm/bbbike/Stockholm/Stockholm.osm.pbf`

# Task 4: The Tokenization Pipeline (Python)
Write a Python script (`tokenize_stockholm.py`) using `pyrosm` (or `osmium`), `shapely`, and `h3` to process the downloaded file. The script must execute the following logic:

1. **Filtering:** Parse the `.pbf` and extract only infrastructure elements: `building=*` (polygons) and `highway=*` (lines/roads).
2. **Simplification:** Use the Ramer-Douglas-Peucker algorithm (via shapely) to simplify complex building footprints to their core structural corners, reducing token bloat.
3. **Spatial Discretization:** Convert all absolute latitude/longitude coordinates into Uber H3 Hexagon string indices (Resolution 11).
4. **Token Translation:** Convert the geometries into a custom sequence format. 
   - We do not want absolute coordinates for every point. We want Anchors and Relative Offsets.
   - Example Building: `['<BUILDING_START>', '<TAG_RESIDENTIAL>', '<H3_8a2a10...>', '<MOVE_N_10M>', '<MOVE_E_15M>', '<MOVE_S_10M>', '<BUILDING_END>']`
   - Example Road: `['<ROAD_START>', '<TAG_PRIMARY>', '<H3_8a2a11...>', '<MOVE_NE_50M>', '<ROAD_END>']`
5. **Serialization (Crucial Step):** To feed this to a Transformer, the 2D map must become a 1D string. Calculate the centroid of every object. Sort all objects using a 2D Space-Filling Curve (like a Z-Order/Morton curve or Hilbert Curve) based on their centroids. 
6. **Output:** Flatten the sorted objects into one massive list of string tokens and save the output as an Apache Parquet file (`stockholm_tokens.parquet`) chunked for HuggingFace `datasets` streaming.

# Constraints
- Ensure your Python code is highly modular and heavily commented so I can review the token logic.
- Ensure the `tokenize_stockholm.py` script accepts CLI arguments for input file and output directory.
- Generate a `slurm_tokenize.sh` script to run this Python job on a single Leonardo node, requesting adequate RAM (at least 128GB) and 1 CPU node.

Take a deep breath and outline your plan step-by-step before writing the code.