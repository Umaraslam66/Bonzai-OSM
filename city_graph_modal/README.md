# City Graph Modal Smoke Test

Graph-native Luxembourg smoke test:

1. parse `data/luxembourg-260419.osm.pbf`
2. build overlapping heterogeneous city-graph chunks
3. train a small graph autoencoder on a Modal A100
4. evaluate reconstruction quality on held-out spatial chunks

## Files

- `prepare_dataset.py`: PBF -> typed city-graph chunks
- `dataset.py`: chunk dataset loader and collator
- `model.py`: small hybrid graph encoder
- `train.py`: masked attribute + edge reconstruction training loop
- `evaluate.py`: checkpoint evaluation on val/test chunks
- `modal_app.py`: Modal upload, prepare, train, and evaluate entrypoint

Evaluation uses deterministic masking and negative-edge sampling, so validation and test metrics are reproducible across runs with the same checkpoint and dataset. Edge metrics are also reported per relation with precision, recall, F1, and support.

## W&B

The Modal app reads `WANDB_API_KEY` from the repo `.env` file and injects it as a Modal secret for training runs.

## Local Python

Install the local requirement set if you want to drive the pipeline from this machine:

```bash
pip install -r city_graph_modal/requirements.txt
```

That local install only provides the Modal client. The parser and training dependencies are installed inside the Modal image defined in [modal_app.py](C:/Users/Neura/Bonzai_osm/city_graph_modal/modal_app.py:56).

If you later want to run `prepare_dataset.py`, `train.py`, or `evaluate.py` directly on your own machine instead of through Modal, use Python 3.11 or 3.12. `osmium` did not build cleanly here under Python 3.13.

## Modal pipeline

Run the full smoke test:

```bash
modal run city_graph_modal/modal_app.py
```

This local entrypoint will:

1. upload the Luxembourg PBF into a Modal Volume
2. prepare the processed city-graph dataset remotely
3. train on a single A100
4. evaluate the best checkpoint on the test split

## Customizing the run

Example:

```bash
modal run city_graph_modal/modal_app.py --epochs 20 --batch-size 12 --learning-rate 2e-4
```

If you only want to upload and prepare the dataset:

```bash
modal run city_graph_modal/modal_app.py --train false --evaluate false
```
