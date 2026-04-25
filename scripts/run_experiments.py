from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.community_detection import (
    async_lpa,
    modularity_async_lpa,
    modularity_sync_lpa,
    modularity_sync_lpa_with_mis,
    sync_lpa,
    sync_lpa_with_mis,
)
from src.datasets import load_dolphins, load_football, load_karate
from src.evaluation import (
    football_true_labels_zero_indexed,
    karate_true_labels,
    modularity_score,
    nmi_score,
)


ALGORITHMS = {
    "async_lpa": async_lpa,
    "sync_lpa": sync_lpa,
    "sync_lpa_with_mis": sync_lpa_with_mis,
    "modularity_async_lpa": modularity_async_lpa,
    "modularity_sync_lpa": modularity_sync_lpa,
    "modularity_sync_lpa_with_mis": modularity_sync_lpa_with_mis,
}


def evaluate_dataset(name: str, seed: int, max_iter: int) -> list[dict[str, float | int | str]]:
    if name == "karate":
        graph = load_karate()
        true_labels = karate_true_labels(graph)
    elif name == "football":
        graph = load_football()
        true_labels = football_true_labels_zero_indexed()
    elif name == "dolphins":
        graph = load_dolphins()
        true_labels = None
    else:
        raise ValueError(f"Unknown dataset: {name}")

    rows = []
    for algorithm_name, algorithm in ALGORITHMS.items():
        communities = algorithm(graph, max_iter=max_iter, seed=seed)
        row: dict[str, float | int | str] = {
            "dataset": name,
            "algorithm": algorithm_name,
            "communities": len(communities),
            "modularity": modularity_score(graph, communities),
        }
        if true_labels is not None:
            row["nmi"] = nmi_score(graph, true_labels, communities)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run label propagation experiments.")
    parser.add_argument("--dataset", choices=["karate", "football", "dolphins", "all"], default="all")
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "experiment_results.json",
    )
    args = parser.parse_args()

    dataset_names = ["karate", "football", "dolphins"] if args.dataset == "all" else [args.dataset]
    results = []
    for dataset_name in dataset_names:
        results.extend(evaluate_dataset(dataset_name, seed=args.seed, max_iter=args.max_iter))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
