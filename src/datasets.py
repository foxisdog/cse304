from __future__ import annotations

from pathlib import Path

import networkx as nx
from scipy.io import mmread


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_dolphins(path: Path | None = None) -> nx.Graph:
    data_path = path or RAW_DATA_DIR / "soc-dolphins" / "soc-dolphins.mtx"
    matrix = mmread(data_path)
    return nx.from_scipy_sparse_array(matrix)


def load_football(path: Path | None = None, relabel_to_zero: bool = True) -> nx.Graph:
    data_path = path or RAW_DATA_DIR / "dimacs10-football" / "out.dimacs10-football"
    graph = nx.Graph()

    with data_path.open() as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("%") or stripped.startswith("#"):
                continue
            source, target, *_ = stripped.split()
            graph.add_edge(int(source), int(target))

    if relabel_to_zero:
        mapping = {node: index for index, node in enumerate(sorted(graph.nodes()))}
        graph = nx.relabel_nodes(graph, mapping)

    return graph


def load_karate() -> nx.Graph:
    return nx.karate_club_graph()
