from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Hashable, Iterable

import networkx as nx


Node = Hashable
Labels = dict[Node, int]
Communities = list[list[Node]]


def labels_to_communities(labels: Labels) -> Communities:
    communities: dict[int, list[Node]] = defaultdict(list)
    for node, label in labels.items():
        communities[label].append(node)
    return [sorted(nodes) for nodes in communities.values()]


def majority_label(labels: Labels, neighbors: Iterable[Node], rng: random.Random) -> int | None:
    counts = Counter(labels[neighbor] for neighbor in neighbors)
    if not counts:
        return None

    max_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == max_count]
    return rng.choice(candidates)


def async_lpa(graph: nx.Graph, max_iter: int = 100, seed: int | None = None) -> Communities:
    """Run asynchronous label propagation."""
    rng = random.Random(seed)
    labels = {node: index for index, node in enumerate(graph.nodes())}

    for _ in range(max_iter):
        changed = False
        nodes = list(graph.nodes())
        rng.shuffle(nodes)

        for node in nodes:
            new_label = majority_label(labels, graph.neighbors(node), rng)
            if new_label is None:
                continue
            if new_label != labels[node]:
                labels[node] = new_label
                changed = True

        if not changed:
            break

    return labels_to_communities(labels)


def sync_lpa(graph: nx.Graph, max_iter: int = 100, seed: int | None = None) -> Communities:
    """Run synchronous label propagation."""
    rng = random.Random(seed)
    labels = {node: index for index, node in enumerate(graph.nodes())}

    for _ in range(max_iter):
        changed = False
        new_labels = labels.copy()

        for node in graph.nodes():
            new_label = majority_label(labels, graph.neighbors(node), rng)
            if new_label is None:
                continue
            new_labels[node] = new_label
            if new_label != labels[node]:
                changed = True

        labels = new_labels
        if not changed:
            break

    return labels_to_communities(labels)


def maximal_independent_set(graph: nx.Graph, nodes: Iterable[Node]) -> set[Node]:
    """Build a greedy maximal independent set from the remaining nodes."""
    independent_set: set[Node] = set()
    remaining_nodes = set(nodes)

    while remaining_nodes:
        node = min(
            remaining_nodes,
            key=lambda candidate: sum(1 for neighbor in graph[candidate] if neighbor in remaining_nodes),
        )
        independent_set.add(node)
        remaining_nodes.remove(node)
        remaining_nodes -= set(graph.neighbors(node))

    return independent_set


def sync_lpa_with_mis(graph: nx.Graph, max_iter: int = 100, seed: int | None = None) -> Communities:
    """Run synchronous LPA in batches selected by maximal independent sets."""
    rng = random.Random(seed)
    labels = {node: index for index, node in enumerate(graph.nodes())}
    all_nodes = list(graph.nodes())

    for _ in range(max_iter):
        changed = False
        remaining_nodes = set(all_nodes)

        while remaining_nodes:
            new_labels = labels.copy()
            independent_set = maximal_independent_set(graph, remaining_nodes)

            for node in independent_set:
                new_label = majority_label(labels, graph.neighbors(node), rng)
                if new_label is None:
                    continue
                new_labels[node] = new_label
                if new_label != labels[node]:
                    changed = True

            remaining_nodes -= independent_set
            labels = new_labels

        if not changed:
            break

    return labels_to_communities(labels)


def modularity_gain(
    graph: nx.Graph,
    labels: Labels,
    node: Node,
    old_label: int,
    new_label: int,
    total_degree: int,
    node_degrees: dict[Node, int],
) -> float:
    if old_label == new_label:
        return 0.0

    node_degree = node_degrees[node]
    old_internal_degree = sum(1 for neighbor in graph[node] if labels[neighbor] == old_label)
    new_internal_degree = sum(1 for neighbor in graph[node] if labels[neighbor] == new_label)
    old_total_degree = sum(node_degrees[n] for n, label in labels.items() if label == old_label)
    new_total_degree = sum(node_degrees[n] for n, label in labels.items() if label == new_label)

    return (
        (new_internal_degree - old_internal_degree) / total_degree
        - node_degree * (new_total_degree - old_total_degree) / (total_degree**2)
    )


def _best_modularity_label(graph: nx.Graph, labels: Labels, node: Node, total_degree: int, node_degrees: dict[Node, int]) -> int:
    current_label = labels[node]
    best_label = current_label
    best_gain = 0.0

    for candidate in {labels[neighbor] for neighbor in graph.neighbors(node)}:
        gain = modularity_gain(graph, labels, node, current_label, candidate, total_degree, node_degrees)
        if gain > best_gain:
            best_gain = gain
            best_label = candidate

    return best_label


def modularity_async_lpa(graph: nx.Graph, max_iter: int = 100, seed: int | None = None) -> Communities:
    """Run asynchronous LPA and only apply changes that improve modularity locally."""
    rng = random.Random(seed)
    labels = {node: index for index, node in enumerate(graph.nodes())}
    total_degree = graph.number_of_edges() * 2
    node_degrees = dict(graph.degree())

    if total_degree == 0:
        return labels_to_communities(labels)

    for _ in range(max_iter):
        changed = False
        nodes = list(graph.nodes())
        rng.shuffle(nodes)

        for node in nodes:
            best_label = _best_modularity_label(graph, labels, node, total_degree, node_degrees)
            if best_label != labels[node]:
                labels[node] = best_label
                changed = True

        if not changed:
            break

    return labels_to_communities(labels)


def modularity_sync_lpa(graph: nx.Graph, max_iter: int = 100, seed: int | None = None) -> Communities:
    """Run synchronous LPA and only apply changes that improve modularity locally."""
    labels = {node: index for index, node in enumerate(graph.nodes())}
    total_degree = graph.number_of_edges() * 2
    node_degrees = dict(graph.degree())

    if total_degree == 0:
        return labels_to_communities(labels)

    for _ in range(max_iter):
        changed = False
        new_labels = labels.copy()

        for node in graph.nodes():
            best_label = _best_modularity_label(graph, labels, node, total_degree, node_degrees)
            new_labels[node] = best_label
            if best_label != labels[node]:
                changed = True

        labels = new_labels
        if not changed:
            break

    return labels_to_communities(labels)


def modularity_sync_lpa_with_mis(graph: nx.Graph, max_iter: int = 100, seed: int | None = None) -> Communities:
    """Run modularity-aware synchronous LPA in maximal independent-set batches."""
    labels = {node: index for index, node in enumerate(graph.nodes())}
    total_degree = graph.number_of_edges() * 2
    node_degrees = dict(graph.degree())
    all_nodes = list(graph.nodes())

    if total_degree == 0:
        return labels_to_communities(labels)

    for _ in range(max_iter):
        changed = False
        remaining_nodes = set(all_nodes)

        while remaining_nodes:
            new_labels = labels.copy()
            independent_set = maximal_independent_set(graph, remaining_nodes)

            for node in independent_set:
                best_label = _best_modularity_label(graph, labels, node, total_degree, node_degrees)
                new_labels[node] = best_label
                if best_label != labels[node]:
                    changed = True

            remaining_nodes -= independent_set
            labels = new_labels

        if not changed:
            break

    return labels_to_communities(labels)
