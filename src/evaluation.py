from __future__ import annotations

from typing import Hashable, Sequence

import networkx as nx
from networkx.algorithms.community.quality import modularity
from sklearn.metrics import normalized_mutual_info_score


Node = Hashable


FOOTBALL_CONFERENCES = {
    "Atlantic Coast": list(range(1, 10)),
    "Big East": list(range(10, 18)),
    "Big Ten": list(range(18, 29)),
    "Big Twelve": list(range(29, 41)),
    "Conference USA": list(range(41, 52)),
    "Independents": list(range(52, 57)),
    "Mid-American": list(range(57, 70)),
    "Mountain West": list(range(70, 78)),
    "Pacific Ten": list(range(78, 88)),
    "Southeastern": list(range(88, 100)),
    "Sun Belt": list(range(100, 109)),
    "Western Athletic": list(range(109, 116)),
}


def communities_to_labels(nodes: Sequence[Node], communities: Sequence[Sequence[Node]]) -> list[int]:
    label_by_node: dict[Node, int] = {}
    for label, community in enumerate(communities):
        for node in community:
            label_by_node[node] = label
    return [label_by_node.get(node, -1) for node in nodes]


def nmi_score(graph: nx.Graph, true_labels: Sequence[int], communities: Sequence[Sequence[Node]]) -> float:
    predicted_labels = communities_to_labels(list(graph.nodes()), communities)
    return normalized_mutual_info_score(true_labels, predicted_labels)


def modularity_score(graph: nx.Graph, communities: Sequence[Sequence[Node]]) -> float:
    return modularity(graph, [set(community) for community in communities])


def karate_true_labels(graph: nx.Graph) -> list[int]:
    return [0 if graph.nodes[node]["club"] == "Mr. Hi" else 1 for node in graph.nodes()]


def football_true_labels_zero_indexed() -> list[int]:
    labels = [0] * 115
    for conference_index, teams in enumerate(FOOTBALL_CONFERENCES.values()):
        for team_id in teams:
            labels[team_id - 1] = conference_index
    return labels
