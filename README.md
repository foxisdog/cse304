# CSE304 Data Mining

This project implements and evaluates label propagation algorithms for community detection on graph datasets. It includes synchronous LPA, asynchronous LPA, maximal-independent-set based updates, and modularity-aware variants.

The current experiments cover:

- Zachary's Karate Club network
- Dolphins social network
- American college football network

## Repository Structure

```text
.
├── data/raw/                  # Original graph datasets and dataset README files
├── docs/                      # Poster and presentation artifacts
├── scripts/
│   └── run_experiments.py     # Reproducible experiment entry point
└── src/                       # Reusable project code
    ├── community_detection.py
    ├── datasets.py
    └── evaluation.py
```

Generated outputs, virtual environments, caches, and notebook checkpoints are intentionally excluded from version control.

## Setup

Install `uv`, then create the project environment:

```bash
UV_CACHE_DIR=.uv-cache uv venv .venv
UV_CACHE_DIR=.uv-cache uv pip install -r requirements.txt
```

## Run Experiments

Run all bundled datasets:

```bash
.venv/bin/python scripts/run_experiments.py
```

Run one dataset:

```bash
.venv/bin/python scripts/run_experiments.py --dataset karate --seed 39 --max-iter 100
```

Results are written to `outputs/experiment_results.json`.

## Algorithms

The experiment runner compares these implementations:

- `async_lpa`
- `sync_lpa`
- `sync_lpa_with_mis`
- `modularity_async_lpa`
- `modularity_sync_lpa`
- `modularity_sync_lpa_with_mis`

## Metrics

For datasets with known labels, the script reports normalized mutual information (NMI). For every dataset, it reports modularity and the number of detected communities.

## Data

Dataset files are stored under `data/raw/`. The original dataset README files are preserved with the raw data for citation and provenance.
