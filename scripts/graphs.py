"""The graph rules lesson 7 derives, in a form both of its notebooks can import.

[07.1](../notebooks/lesson7/07.1-social_graphs.ipynb) writes the mention rule out as a
`groupby` in a cell, and [07.2](../notebooks/lesson7/07.2-graph-properties.ipynb) reads
the numbers it produces. That derivation is the exercise and it stays in the notebooks.
This module is where the finished steps live, so 07.3 can ask a different question of the
same graph without carrying a second copy of the edge rule that then drifts.

    from scripts.graphs import MentionEdges, giant_component, to_graph

    edges = MentionEdges()(chat)
    mentions = giant_component(to_graph(edges, nodes=chat.author.unique()))

`modularity_check` is the piece that turns a clustering into a claim you can defend.
Modularity is a number every graph has, including a graph with no communities in it at
all, so it is never reported on its own here: the same measurement runs on twenty rewired
copies of the graph that keep every node's degree, and what comes back is how many
standard deviations the real graph sits above graphs that are structureless by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from goad_toolkit.datatransforms import TransformBase
from goad_toolkit.visualizer import BasePlot
from networkx.algorithms.community import louvain_communities, modularity


class MentionEdges(TransformBase):
    """Count who addressed whom, keeping only targets that also spoke here.

    `addressed_to` is a regex feature, not a nick detector: `^(\\S+)[:,]\\s` matches
    "yeah, that worked" as happily as "davmor2: try purging it". The `isin` below is the
    single line that separates the two, and it is here rather than in the notebook so no
    graph gets built without it.

    Returns one row per ordered pair, so the direction survives; `to_graph` is where it
    is given up.
    """

    def transform(
        self,
        data: pd.DataFrame,
        sender: str = "author",
        target: str = "addressed_to",
    ) -> pd.DataFrame:
        nicks = set(data[sender])
        named = data[data[target].isin(nicks) & (data[sender] != data[target])]
        return (
            named.groupby([sender, target])
            .size()
            .rename("weight")
            .reset_index()
            .rename(columns={sender: "source", target: "target"})
        )


class HourProfile(TransformBase):
    """One row per author, 24 columns: when in the day that author speaks.

    Each row sums to one, so a person who sent forty messages is comparable with one who
    sent four thousand. `normalise` then divides every hour by the channel's own share of
    traffic in that hour, which is what stops the answer being "everyone sleeps at
    night": after it, the columns say *unusually* active at this hour, and two people who
    are both awake at 03:00 are far more alike than two who are both awake at 20:00.
    """

    def transform(
        self,
        data: pd.DataFrame,
        author: str = "author",
        hour: str = "hh",
        normalise: bool = True,
    ) -> pd.DataFrame:
        counts = (
            data.pivot_table(index=author, columns=hour, aggfunc="size", fill_value=0)
            .reindex(columns=range(24), fill_value=0)
            .astype(float)
        )
        profile = counts.div(counts.sum(axis=1), axis=0)
        if normalise:
            channel = counts.sum(axis=0) / counts.to_numpy().sum()
            profile = profile.div(channel, axis=1)
        return profile.fillna(0.0)


def cosine_edges(profiles: pd.DataFrame, quantile: float = 0.90) -> pd.DataFrame:
    """Connect the pairs whose profiles point in the most similar direction.

    Cosine ignores length, so this is about *shape* — the hours someone is around, not
    how much they say. `quantile` is the whole rule: 0.90 keeps the most similar tenth of
    all pairs, which fixes the edge count and leaves the threshold value to the data.
    It is a dial on the answer in exactly the way 07.1's response window was, so it
    belongs in the caption of anything drawn from it.
    """
    matrix = profiles.to_numpy(dtype=float)
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.maximum(lengths, 1e-12)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, 0.0)

    upper = np.triu_indices_from(similarity, k=1)
    threshold = float(np.quantile(similarity[upper], quantile))
    rows, cols = np.where(np.triu(similarity >= threshold, k=1))
    names = profiles.index.to_numpy()
    return pd.DataFrame(
        {
            "source": names[rows],
            "target": names[cols],
            "weight": similarity[rows, cols],
            "threshold": threshold,
        }
    )


def to_graph(edges: pd.DataFrame, nodes: list | np.ndarray | None = None) -> nx.Graph:
    """Turn an edge frame into an undirected graph, summing both directions.

    `nodes` adds everyone who appears in the data, including the people no edge touches.
    Leaving them out silently changes the denominator of every share you compute
    afterwards.
    """
    graph = nx.Graph()
    if nodes is not None:
        graph.add_nodes_from(nodes)
    for source, target, raw in zip(
        edges["source"], edges["target"], edges["weight"], strict=True
    ):
        weight = float(raw)
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += weight
        else:
            graph.add_edge(source, target, weight=weight)
    return graph


def giant_component(graph: nx.Graph) -> nx.Graph:
    """The largest connected component, as a graph of its own."""
    return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()


def rewired(graph: nx.Graph, seed: int = 0) -> nx.Graph:
    """A graph with the same degree sequence and nothing else of the original in it.

    Repeated double edge swaps keep every node's degree exactly and keep the result a
    simple graph, which the configuration model does not: it produces multi-edges and
    self-loops, and dropping them lowers the edge count enough to raise modularity on its
    own. A dense graph runs out of legal swaps, so the number of swaps steps down until
    one succeeds and the caller is told how deep it had to go.

    Weights are dropped. Swapping preserves the degree sequence, which makes the result
    a null for *which pairs are connected*; there is no equally obvious null for how
    heavy each edge is, and inventing one — scattering the real weights over the rewired
    edges, say — produces a graph whose heaviest edges sit on its sparsest corners and
    beats any real network on weighted modularity. So the comparison this supports is
    about the topology, and `modularity_check` says so by not taking a weight at all.
    """
    edges = graph.number_of_edges()
    plain = nx.Graph()
    plain.add_nodes_from(graph.nodes())
    plain.add_edges_from(graph.edges())

    copy, depth = plain, 0
    for per_edge in (10, 3, 1):
        candidate = nx.Graph(plain)
        try:
            nx.double_edge_swap(
                candidate,
                nswap=per_edge * edges,
                max_tries=500 * per_edge * edges,
                seed=seed,
            )
        except nx.NetworkXAlgorithmError:
            continue
        copy, depth = candidate, per_edge
        break

    copy.graph["swaps_per_edge"] = depth
    return copy


@dataclass
class ModularityCheck:
    """What `modularity_check` found. `z` is the only number worth quoting alone."""

    q: float
    null_mean: float
    null_std: float
    z: float
    k: int
    sizes: list[int]
    swaps_per_edge: int

    def __str__(self) -> str:
        return (
            f"Q={self.q:.3f}  null={self.null_mean:.3f}±{self.null_std:.3f}  "
            f"z={self.z:+.1f}  k={self.k}  sizes={self.sizes}"
        )


def best_partition(
    graph: nx.Graph, weight: str | None = None, seeds: int = 5, resolution: float = 1.0
) -> list[set]:
    """The highest-modularity partition louvain finds over `seeds` restarts.

    Louvain is randomised: it visits nodes in a shuffled order and stops at a local
    optimum, so the partition is a function of the seed as much as of the graph. Taking
    the best of several is the standard fix; how far the seeds disagree is a result in
    its own right, and 07.3 measures it rather than hiding it here.
    """
    partitions = [
        louvain_communities(graph, weight=weight, resolution=resolution, seed=seed)
        for seed in range(seeds)
    ]
    return max(
        partitions,
        key=lambda p: modularity(graph, p, weight=weight, resolution=resolution),
    )


def modularity_check(
    graph: nx.Graph,
    n_null: int = 20,
    seeds: int = 3,
    resolution: float = 1.0,
) -> ModularityCheck:
    """Modularity of the best partition, against rewired graphs of the same degrees.

    The null is clustered with the same algorithm and the same effort, so the comparison
    is between two graphs rather than between a graph and an assumption. Both sides are
    unweighted, for the reason `rewired` gives: the swap is a null for the topology and
    for nothing else.
    """
    partition = best_partition(graph, seeds=seeds, resolution=resolution)
    q = modularity(graph, partition, weight=None, resolution=resolution)

    null_qs, depth = [], 10
    for seed in range(n_null):
        null = rewired(graph, seed=seed)
        depth = min(depth, null.graph["swaps_per_edge"])
        null_partition = best_partition(null, seeds=seeds, resolution=resolution)
        null_qs.append(
            modularity(null, null_partition, weight=None, resolution=resolution)
        )

    null_qs = np.array(null_qs)
    spread = float(null_qs.std())
    return ModularityCheck(
        q=float(q),
        null_mean=float(null_qs.mean()),
        null_std=spread,
        z=float((q - null_qs.mean()) / spread) if spread > 0 else float("nan"),
        k=len(partition),
        sizes=sorted((len(c) for c in partition), reverse=True),
        swaps_per_edge=depth,
    )


def labels(graph: nx.Graph, partition: list[set]) -> pd.Series:
    """A cluster number per node, indexed by node, ordered largest cluster first.

    Comparing two clusterings means comparing two of these on the nodes they share, and
    a Series makes that a `reindex` rather than a bookkeeping exercise in which cluster 0
    of one run is cluster 3 of the other.
    """
    ordered = sorted(partition, key=len, reverse=True)
    return pd.Series(
        {node: index for index, group in enumerate(ordered) for node in group},
        name="cluster",
    ).reindex(list(graph.nodes()))


class GraphPlot(BasePlot):
    """Draw a `networkx` graph on a matplotlib axis, colouring nodes by a label.

    `pos` is a required argument rather than something computed inside, because a layout
    is a choice with a seed in it and no plot should quietly make that choice for you.
    Two panels of the same graph only mean anything side by side when they were handed
    the same positions.
    """

    def build(
        self,
        data: nx.Graph,
        pos: dict,
        color: str | list = "lightgrey",
        node_size: float | list = 60,
        with_labels: bool = False,
        edge_alpha: float = 0.25,
        **kwargs,
    ):
        nx.draw_networkx_edges(
            data, pos, ax=self.ax, alpha=edge_alpha, edge_color="grey", width=0.6
        )
        nx.draw_networkx_nodes(
            data, pos, ax=self.ax, node_color=color, node_size=node_size, **kwargs
        )
        if with_labels:
            nx.draw_networkx_labels(data, pos, ax=self.ax, font_size=8)
        if self.ax is not None:
            self.ax.set_axis_off()
        return self.fig, self.ax
