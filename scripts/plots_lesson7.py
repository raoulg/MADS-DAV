"""Figures for the lesson 7 slides, in the pttrn Paper palette.

Run from the repo root:  uv run python scripts/plots_lesson7.py
Writes PNGs to presentations/img/lesson7/ and prints the numbers the slides quote.
Data: the ubuntu_irc showcase (#ubuntu-uk, 2015) and the karate club and Florentine families
graphs that ship with networkx. Each computation repeats the code and seeds of its notebook,
so the numbers match 07.1–07.3. The Bearman and Moody odds ratios are copied from their
Table 2 (American Journal of Public Health 94(1), 2004).
"""

import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from networkx.algorithms.community import modularity
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score

from scripts.graphs import (
    MentionEdges,
    best_partition,
    giant_component,
    induced,
    labels,
    rewired,
    to_graph,
)
from scripts.pipelines import BuildTimestamp, build_irc_pipeline
from wa_analyzer.data import load_showcase
from wa_analyzer.network_analysis import Config, GraphBuilder
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentations" / "img" / "lesson7"
OUT.mkdir(parents=True, exist_ok=True)

# pttrn tokens, Paper theme (~/code/packages/pttrn-huisstijl/tokens.json): petrol for the main
# series, vermilion for the one thing a figure is about, warm greys for everything else
BG = "#f4f1ec"
TEXT = "#111111"
ACCENT = "#1d5a73"
ACCENT_LIGHT = "#5e93a8"
SECOND = "#c43a14"
THIRD = "#8a6d3b"
GREY = "#2e2a26"
GREY_DIM = "#5a544d"
GREY_FAINT = "#cfc7bb"
INK_BG = "#141210"
INK_TEXT = "#f4f1ec"
INK_MUTED = "#b8b0a5"
INK_PETROL = "#86b9c9"
INK_ACCENT = "#f2643c"
CATEGORIES = [
    ACCENT,
    SECOND,
    THIRD,
    ACCENT_LIGHT,
    "#7a5a8c",
    "#6b7a3a",
    GREY,
    "#c9a66b",
    "#b77a6a",
    "#9c8f7d",
]
CELLS = LinearSegmentedColormap.from_list(
    "cells", [BG, ACCENT_LIGHT, ACCENT, "#0d2c38"]
)
PRESENT = LinearSegmentedColormap.from_list("present", [BG, GREY])

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": TEXT,
        "axes.labelcolor": GREY,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "axes.edgecolor": GREY_FAINT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "font.family": "sans-serif",
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "legend.fontsize": 14,
        "lines.linewidth": 2.2,
    }
)


def save(fig, name, facecolor=BG):
    fig.tight_layout()
    fig.savefig(
        OUT / f"{name}.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.2,
        facecolor=facecolor,
    )
    plt.close(fig)
    print("wrote", name)


def draw(
    ax,
    graph,
    pos,
    color=ACCENT,
    size=80,
    edge_color=GREY_FAINT,
    width=0.8,
    labels_=None,
    font_size=12,
    font_color=TEXT,
    arrows=False,
    edge_width=None,
):
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color=edge_color,
        width=edge_width or width,
        arrows=arrows,
        arrowsize=16,
        node_size=size if np.isscalar(size) else 300,
    )
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_color=color, node_size=size, linewidths=0
    )
    if labels_ is not None:
        nx.draw_networkx_labels(
            graph,
            pos,
            labels=labels_,
            ax=ax,
            font_size=font_size,
            font_color=font_color,
            font_family="sans-serif",
        )
    ax.set_axis_off()


def matrix(ax, graph, order, weight=None, cmap=CELLS, log=False):
    values = nx.to_numpy_array(graph, nodelist=order, weight=weight)
    if log:
        values = np.log1p(values)
    ax.imshow(values, cmap=cmap, interpolation="nearest", vmin=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(GREY_FAINT)


def null_qs(graph, n_null=20, seeds=3):
    """`modularity_check`, keeping the twenty null values instead of their mean and spread."""
    partition = best_partition(graph, seeds=seeds)
    q = modularity(graph, partition, weight=None)
    nulls = []
    for seed in range(n_null):
        null = rewired(graph, seed=seed)
        nulls.append(modularity(null, best_partition(null, seeds=seeds), weight=None))
    nulls = np.array(nulls)
    return q, nulls, (q - nulls.mean()) / nulls.std()


# ---------- data: #ubuntu-uk in 2015, through the course pipeline ----------
pipeline = build_irc_pipeline()
pipeline.add(BuildTimestamp)
irc = pipeline.apply(load_showcase("ubuntu_irc"))
chat = irc[(irc.channel == "#ubuntu-uk") & (irc.date.dt.year == 2015)].copy()
builder = GraphBuilder(
    Config(time_col="timestamp", node_col="author", seconds=600, datafile=None)
)
nearby = builder.build(chat, edge_seconds=600)
named = to_graph(MentionEdges()(chat), nodes=chat.author.unique())
print(f"{len(chat):,} messages, {chat.author.nunique()} authors")

# ---------- 01: G = (V, E), drawn and as a matrix ----------
small = nx.Graph([("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("D", "E")])
fig, axes = plt.subplots(
    1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.2, 1]}
)
pos = {"A": (0, 1), "B": (0, -1), "C": (1.2, 0), "D": (2.5, 0), "E": (3.7, 0)}
draw(
    axes[0],
    small,
    pos,
    color=ACCENT,
    size=1100,
    edge_color=GREY_DIM,
    width=2.4,
    labels_={n: n for n in small},
    font_size=20,
    font_color=BG,
)
axes[0].set_xlim(-0.5, 4.2)
axes[0].set_ylim(-1.5, 1.5)
order = list("ABCDE")
matrix(axes[1], small, order, cmap=PRESENT)
axes[1].set_xticks(range(5), order, fontsize=18)
axes[1].set_yticks(range(5), order, fontsize=18)
axes[1].tick_params(length=0)
axes[1].xaxis.tick_top()
save(fig, "gve")


# ---------- 01: four kinds of graph ----------
def boxed(ax, graph, pos, fill=None, width=None, arrows=True, rad=0.0, font_size=13):
    """Each node a label in a square box, each edge an arrow clipped at the edges of the boxes."""
    boxes = {}
    for node, (x, y) in pos.items():
        face, text_colour = (fill or {}).get(node, (BG, TEXT))
        boxes[node] = ax.text(
            x,
            y,
            node,
            ha="center",
            va="center",
            fontsize=font_size,
            color=text_colour,
            zorder=3,
            bbox={
                "boxstyle": "square,pad=0.45",
                "facecolor": face,
                "edgecolor": ACCENT,
                "linewidth": 1.6,
            },
        )
    for u, v in graph.edges():
        ax.annotate(
            "",
            xy=pos[v],
            xytext=pos[u],
            zorder=2,
            arrowprops={
                "arrowstyle": "-|>,head_length=0.7,head_width=0.35" if arrows else "-",
                "color": GREY_DIM,
                "lw": (width or {}).get((u, v), 1.6),
                "patchA": boxes[u].get_bbox_patch(),
                "patchB": boxes[v].get_bbox_patch(),
                "shrinkA": 3,
                "shrinkB": 3,
                "connectionstyle": f"arc3,rad={rad}",
            },
        )
    xs, ys = zip(*pos.values())
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(min(ys) - 0.4, max(ys) + 0.4)
    ax.set_axis_off()


# policy: the Dutch nitrogen case, things and how they act on each other
policy = nx.DiGraph()
policy.add_edges_from(
    [
        ("livestock", "nitrogen"),
        ("traffic", "nitrogen"),
        ("construction", "nitrogen"),
        ("nitrogen", "nature reserves"),
        ("court ruling 2019", "permits"),
        ("nature reserves", "court ruling 2019"),
        ("permits", "construction"),
        ("construction", "housing"),
        ("construction", "roads"),
    ]
)
pos = {
    "livestock": (0, 2),
    "traffic": (0, 0.9),
    "construction": (1.6, -0.6),
    "nitrogen": (1.5, 1.5),
    "nature reserves": (3.4, 2.0),
    "court ruling 2019": (4.0, 0.7),
    "permits": (3.4, -0.6),
    "housing": (0.5, -1.7),
    "roads": (2.6, -1.7),
}
fig, ax = plt.subplots(figsize=(8, 5.4))
boxed(ax, policy, pos)
save(fig, "ex_policy")

# trains: intercity stops, placed at their real coordinates
stations = {
    "Groningen": (6.56, 53.21),
    "Zwolle": (6.09, 52.51),
    "Amersfoort": (5.37, 52.15),
    "Amsterdam": (4.90, 52.38),
    "Schiphol": (4.76, 52.31),
    "Leiden": (4.48, 52.17),
    "Den Haag": (4.32, 52.08),
    "Rotterdam": (4.47, 51.92),
    "Utrecht": (5.11, 52.09),
    "Arnhem": (5.90, 51.98),
    "Den Bosch": (5.29, 51.69),
    "Eindhoven": (5.48, 51.44),
    "Breda": (4.78, 51.59),
}
offsets = {
    "Amsterdam": (8, 6),
    "Schiphol": (-12, 8),
    "Leiden": (-10, 6),
    "Den Haag": (-10, -2),
    "Rotterdam": (-10, -8),
    "Utrecht": (6, -18),
    "Amersfoort": (8, 2),
}
right_aligned = {"Schiphol", "Leiden", "Den Haag", "Rotterdam"}
trains = nx.Graph(
    [
        ("Groningen", "Zwolle"),
        ("Zwolle", "Amersfoort"),
        ("Amersfoort", "Utrecht"),
        ("Amersfoort", "Amsterdam"),
        ("Amsterdam", "Utrecht"),
        ("Amsterdam", "Schiphol"),
        ("Schiphol", "Leiden"),
        ("Leiden", "Den Haag"),
        ("Den Haag", "Rotterdam"),
        ("Rotterdam", "Utrecht"),
        ("Utrecht", "Arnhem"),
        ("Utrecht", "Den Bosch"),
        ("Den Bosch", "Eindhoven"),
        ("Rotterdam", "Breda"),
        ("Breda", "Eindhoven"),
        ("Schiphol", "Utrecht"),
    ]
)
fig, ax = plt.subplots(figsize=(7, 7))
hub = [SECOND if s == "Utrecht" else ACCENT for s in trains]
draw(ax, trains, stations, color=hub, size=160, edge_color=GREY_DIM, width=2.2)
for name, (x, y) in stations.items():
    ax.annotate(
        name,
        (x, y),
        xytext=offsets.get(name, (8, 4)),
        textcoords="offset points",
        fontsize=13,
        color=SECOND if name == "Utrecht" else GREY,
        ha="right" if name in right_aligned else "left",
    )
ax.set_aspect(1.6)
ax.margins(0.2)
save(fig, "ex_trains")

# money: who pays whom in a month, direction and amount
money = nx.DiGraph()
money.add_weighted_edges_from(
    [
        ("employer", "you", 2400),
        ("you", "landlord", 950),
        ("you", "supermarket", 380),
        ("you", "friend", 25),
        ("friend", "you", 40),
        ("you", "phone company", 20),
        ("landlord", "bank", 700),
    ]
)
pos = {
    "employer": (0, 1.2),
    "you": (1, 0),
    "landlord": (2.2, 0.9),
    "bank": (3.3, 1.5),
    "supermarket": (2.5, -0.3),
    "friend": (-0.1, -1.0),
    "phone company": (1.7, -1.3),
}
fig, ax = plt.subplots(figsize=(8, 5.4))
boxed(
    ax,
    money,
    pos,
    fill={"you": (SECOND, BG)},
    rad=0.12,
    width={(u, v): 1 + money[u][v]["weight"] / 400 for u, v in money.edges()},
)
nx.draw_networkx_edge_labels(
    money,
    pos,
    ax=ax,
    font_size=13,
    font_color=SECOND,
    edge_labels={(u, v): f"€{d['weight']}" for u, v, d in money.edges(data=True)},
    bbox={"facecolor": BG, "edgecolor": "none", "pad": 1},
    connectionstyle="arc3,rad=0.12",
)
save(fig, "ex_money")

# likes: two kinds of node, people and posts
likes = nx.Graph()
people_, posts = ["Ana", "Bo", "Cem", "Dana", "Eli"], ["post 1", "post 2", "post 3"]
likes.add_edges_from(
    [
        ("Ana", "post 1"),
        ("Bo", "post 1"),
        ("Cem", "post 1"),
        ("Cem", "post 2"),
        ("Dana", "post 2"),
        ("Dana", "post 3"),
        ("Eli", "post 3"),
        ("Ana", "post 3"),
    ]
)
pos = {p: (0, -i) for i, p in enumerate(people_)} | {
    p: (1.6, -0.5 - 1.5 * i) for i, p in enumerate(posts)
}
fig, ax = plt.subplots(figsize=(5.4, 5.4))
boxed(
    ax,
    likes,
    pos,
    fill=dict.fromkeys(people_, (ACCENT, BG)) | dict.fromkeys(posts, (SECOND, BG)),
    arrows=False,
)
save(fig, "ex_likes")

# ---------- 02: two rules on the same messages ----------
nearby_edges = {frozenset(e) for e in nearby.edges()}
named_edges = {frozenset(e) for e in named.edges()}
print(
    f"nearby {nearby.number_of_edges():,} edges, named {named.number_of_edges():,} edges"
)
print(
    f"declared found by nearby: {len(nearby_edges & named_edges) / len(named_edges):.1%}, "
    f"nearby undeclared: {len(nearby_edges - named_edges) / len(nearby_edges):.1%}"
)

degrees = pd.concat(
    [
        pd.Series(dict(nearby.degree()), name="nearby"),
        pd.Series(dict(named.degree()), name="named"),
    ],
    axis=1,
).fillna(0)
busy = degrees[degrees.nearby >= 20]
print(
    f"{len(busy)} busy authors, spearman of the two degrees "
    f"{busy.nearby.corr(busy.named, method='spearman'):.2f}"
)
fig, ax = plt.subplots(figsize=(8, 6.4))
ax.plot(
    [0, busy.nearby.max()],
    [0, busy.nearby.max()],
    "--",
    color=GREY_FAINT,
    lw=1.2,
    zorder=0,
)
ax.scatter(busy.nearby, busy.named, s=36, color=ACCENT_LIGHT, alpha=0.7, linewidths=0)
for nick, label in [("lubotu3`", "lubotu3` (a bot)"), ("zmoylan-pi", "zmoylan-pi")]:
    n_near, n_named = busy.loc[nick, ["nearby", "named"]].to_numpy(dtype=int)
    ax.scatter(n_near, n_named, s=90, color=SECOND, zorder=3)
    ax.annotate(
        f"  {label}: {n_near} near, {n_named} named",
        (float(n_near), float(n_named)),
        color=SECOND,
        va="center",
        fontsize=14,
    )
    print(f"{nick}: nearby {n_near}, named {n_named}")
ax.set_xlabel("people they were within 10 minutes of")
ax.set_ylabel("people they addressed by name, or were\naddressed by")
save(fig, "rules")

rows = []
for seconds in (60, 300, 600, 1800):
    edges = {frozenset(e) for e in builder.build(chat, edge_seconds=seconds).edges()}
    rows.append(
        (
            seconds / 60,
            len(edges & named_edges) / len(named_edges),
            len(edges - named_edges) / len(edges),
            len(edges),
        )
    )
dial = pd.DataFrame(rows, columns=["minutes", "found", "undeclared", "edges"])
print(dial.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(8.5, 5.4))
for column, colour, label in [
    ("found", ACCENT, "named pairs the window also connects"),
    ("undeclared", SECOND, "window edges with no name typed"),
]:
    ax.plot(dial.minutes, dial[column] * 100, color=colour, lw=1, alpha=0.5)
    ax.scatter(
        dial.minutes, dial[column] * 100, color=colour, s=70, zorder=3, label=label
    )
    for m, v in zip(dial.minutes, dial[column]):
        ax.annotate(
            f"{v:.0%}",
            (m, v * 100),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            color=colour,
            fontsize=13,
        )
ax.set_xscale("log")
ax.set_xticks(dial.minutes, ["1 min", "5 min", "10 min", "30 min"])
ax.minorticks_off()
ax.set_ylim(0, 110)
ax.set_ylabel("percent")
ax.set_xlabel("window: two people are connected if they spoke within …")
ax.legend(loc="lower right")
save(fig, "window")

# ---------- 02: things as nodes: what is mentioned together within ten minutes ----------
THINGS = {
    "windows": r"\bwindows\b",
    "laptop": r"\blaptop\b",
    "usb": r"\busb\b",
    "ssd": r"\bssd\b",
    "bios": r"\bbios\b|\buefi\b",
    "wifi": r"\bwifi\b|\bwireless\b",
    "sound": r"\bsound\b|\baudio\b",
    "kernel": r"\bkernel\b",
    "nvidia": r"\bnvidia\b",
    "ppa": r"\bppa\b",
    "steam": r"\bsteam\b",
    "phone": r"\bphone\b",
    "android": r"\bandroid\b",
    "car": r"\bcar\b",
    "firefox": r"\bfirefox\b",
    "chrome": r"\bchrome\b|\bchromium\b",
    "bbc": r"\bbbc\b",
    "netflix": r"\bnetflix\b",
    "coffee": r"\bcoffee\b",
    "tea": r"\btea\b",
    "beer": r"\bbeer\b",
    "pub": r"\bpub\b",
    "rain": r"\brain\b|\braining\b",
    "weather": r"\bweather\b",
    "train": r"\btrains?\b",
    "london": r"\blondon\b",
}
text = chat.message.str.lower()
hits = pd.DataFrame(
    {thing: text.str.contains(pattern, regex=True) for thing, pattern in THINGS.items()}
)
present = hits.groupby(chat.timestamp.dt.floor("10min").to_numpy()).any()
together = nx.Graph()
together.add_nodes_from(THINGS)
for a, b in itertools.combinations(THINGS, 2):
    n = int((present[a] & present[b]).sum())
    if n >= 8:
        together.add_edge(a, b, weight=n)
together.remove_nodes_from([n for n in list(together) if together.degree(n) == 0])
print(
    f"things graph: {together.number_of_nodes()} things, {together.number_of_edges()} edges "
    f"(mentioned in the same 10 minutes at least 8 times)"
)
groups = labels(together, best_partition(together, weight="weight"))
order = sorted(
    together, key=lambda n: (groups[n], -together.degree(n, weight="weight"))
)
print(
    "things communities:",
    [sorted(c) for c in best_partition(together, weight="weight")],
)
fig, ax = plt.subplots(figsize=(9, 7.4))
giant = max(nx.connected_components(together), key=len)
pos = nx.spring_layout(induced(together, giant), seed=7, k=1.4, iterations=500)
for i, component in enumerate(
    c for c in nx.connected_components(together) if c != giant
):
    for j, thing in enumerate(sorted(component)):
        pos[thing] = np.array([0.55 + 0.35 * j, -0.45 - 0.2 * i])
mentioned = hits.sum()
draw(
    ax,
    together,
    pos,
    color=[CATEGORIES[groups[n]] for n in together],
    size=[30 + 0.4 * mentioned[n] for n in together],
    edge_color=GREY_FAINT,
    edge_width=[0.4 + d["weight"] / 15 for _, _, d in together.edges(data=True)],
)
knot = [n for n in together if together.degree(n) >= 8]
print(f"the knot: {sorted(knot, key=lambda n: -together.degree(n))}")
for n, (x, y) in pos.items():
    if n not in knot:
        ax.annotate(
            n,
            (x, y),
            xytext=(8, -18) if n == "firefox" else (8, 8),
            textcoords="offset points",
            fontsize=15,
            color=CATEGORIES[groups[n]],
        )
nx.draw_networkx_nodes(
    together,
    pos,
    nodelist=knot,
    ax=ax,
    node_color="none",
    edgecolors=GREY,
    linewidths=2,
    node_size=[30 + 0.4 * mentioned[n] + 260 for n in knot],
)
ax.text(
    0.0,
    0.02,
    "ringed, too tangled to label: "
    + ", ".join(sorted(knot, key=lambda n: -together.degree(n))),
    transform=ax.transAxes,
    fontsize=14,
    color=GREY,
)
save(fig, "things")

fig, ax = plt.subplots(figsize=(7.4, 7.4))
matrix(ax, together, order, weight="weight", log=True)
ax.set_xticks(range(len(order)), order, rotation=90, fontsize=13)
ax.set_yticks(range(len(order)), order, fontsize=13)
ax.tick_params(length=0)
for tick, n in zip(
    list(ax.get_yticklabels()) + list(ax.get_xticklabels()), order + order
):
    tick.set_color(CATEGORIES[groups[n]])
save(fig, "things_matrix")

# ---------- 03: the Florentine families ----------
florence = nx.florentine_families_graph()
families = pd.DataFrame(
    {
        "degree": dict(florence.degree()),
        "betweenness": nx.betweenness_centrality(florence),
        "clustering": nx.clustering(florence),
    }
)
print(families.sort_values("betweenness", ascending=False).round(2).head(6).to_string())
fig, ax = plt.subplots(figsize=(7.5, 6.2))
pos = nx.kamada_kawai_layout(florence)
draw(
    ax,
    florence,
    pos,
    color=[SECOND if f == "Medici" else ACCENT for f in florence],
    size=[160 * families.degree[f] for f in florence],
    edge_color=GREY_DIM,
    width=1.6,
)
for f, (x, y) in pos.items():
    ax.annotate(
        f,
        (x, y),
        xytext=(10, 8),
        textcoords="offset points",
        fontsize=13,
        color=SECOND if f == "Medici" else GREY,
    )
save(fig, "florence")

top = families.sort_values("betweenness", ascending=False).head(6)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=False)
for ax, column, title in zip(
    axes,
    ["degree", "betweenness", "clustering"],
    ["degree", "betweenness", "clustering coefficient"],
):
    ordered = top.sort_values(column)
    ax.barh(
        ordered.index,
        ordered[column],
        color=[SECOND if f == "Medici" else GREY_FAINT for f in ordered.index],
    )
    ax.set_title(title)
    ax.tick_params(axis="y", length=0)
save(fig, "florence_bars")

# ---------- 03: triangles, closed and open ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
ring = {
    f: (np.cos(a), np.sin(a))
    for f, a in zip(range(1, 6), np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 6)[:-1])
}
for ax, friends_know, title in [
    (axes[0], [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 3), (2, 4)], "closed"),
    (axes[1], [(1, 2)], "open"),
]:
    ego = nx.Graph([(0, f) for f in ring] + friends_know)
    pos = {0: (0, 0)} | ring
    nx.draw_networkx_edges(
        ego, pos, edgelist=[(0, f) for f in ring], ax=ax, edge_color=GREY_FAINT, width=2
    )
    nx.draw_networkx_edges(
        ego, pos, edgelist=friends_know, ax=ax, edge_color=SECOND, width=3
    )
    nx.draw_networkx_nodes(
        ego, pos, nodelist=[0], ax=ax, node_color=ACCENT, node_size=900
    )
    nx.draw_networkx_nodes(
        ego, pos, nodelist=list(ring), ax=ax, node_color=ACCENT_LIGHT, node_size=500
    )
    c = nx.clustering(ego, 0)
    ax.set_title(f"{title}: C = {c:.1f}", fontsize=18)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.margins(0.1)
save(fig, "triads")

# Bearman and Moody (2004), Table 2: odds ratios for suicidal thoughts, with 95% intervals
bearman = [
    ("open friendship circle", "girls", 2.198, 1.221, 3.956),
    ("open friendship circle", "boys", 0.747, 0.358, 1.558),
    ("isolated", "girls", 2.010, 1.073, 3.765),
    ("isolated", "boys", 0.665, 0.307, 1.445),
]
fig, ax = plt.subplots(figsize=(10, 4.4))
for i, (_, group, odds, low, high) in enumerate(bearman):
    colour = SECOND if group == "girls" else ACCENT
    y = len(bearman) - 1 - i
    ax.plot([low, high], [y, y], color=colour, lw=3)
    ax.scatter(odds, y, color=colour, s=140, zorder=3)
    ax.annotate(
        f"{odds:.1f}",
        (odds, y),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        color=colour,
        fontsize=14,
    )
ax.axvline(1, color=GREY_DIM, lw=1.2, ls="--")
ax.set_xscale("log")
ax.set_xticks([0.25, 0.5, 1, 2, 4], ["¼", "½", "1", "2", "4"])
ax.minorticks_off()
ax.set_yticks(
    range(len(bearman)),
    [f"{measure}, {group}" for measure, group, *_ in reversed(bearman)],
)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("odds of having seriously thought about suicide, × (1 = no effect)")
save(fig, "bearman")

# ---------- 03: what a centrality adds ----------
core = giant_component(nearby)
core_named = giant_component(named)
scores = pd.DataFrame(
    {
        "degree": dict(core.degree()),
        "weighted_degree": dict(core.degree(weight="weight")),
    }
)
scores["messages"] = chat.author.value_counts().reindex(scores.index)
rho_w = spearmanr(scores.weighted_degree, scores.messages).statistic
r_w = pearsonr(scores.weighted_degree, scores.messages).statistic
print(
    f"weighted degree vs messages: spearman {rho_w:.2f}, pearson {r_w:.2f}, n={len(scores)}"
)
fig, ax = plt.subplots(figsize=(8, 6.4))
ax.scatter(
    scores.messages,
    scores.weighted_degree,
    s=26,
    color=ACCENT_LIGHT,
    alpha=0.7,
    linewidths=0,
)
loud = scores.nlargest(3, "messages")
ax.scatter(loud.messages, loud.weighted_degree, s=70, color=SECOND, zorder=3)
for (nick, row), dy in zip(loud.iterrows(), [14, -2, -16]):
    ax.annotate(
        str(nick),
        (row.messages, row.weighted_degree),
        xytext=(-12, dy),
        textcoords="offset points",
        color=SECOND,
        va="center",
        ha="right",
        fontsize=13,
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("messages sent in 2015")
ax.set_ylabel("weighted degree, nearby rule")
save(fig, "wdegree")

log_w = pearsonr(np.log(scores.weighted_degree), np.log(scores.messages)).statistic
print(f"weighted degree vs messages: pearson on logs {log_w:.2f}")
for k in (2, 10):
    rest = scores.drop(scores.nlargest(k, "messages").index)
    print(
        f"  without the {k} loudest: pearson {pearsonr(rest.weighted_degree, rest.messages).statistic:.2f}, "
        f"spearman {spearmanr(rest.weighted_degree, rest.messages).statistic:.2f}"
    )
rho_d = spearmanr(scores.degree, scores.messages).statistic
r_d = pearsonr(scores.degree, scores.messages).statistic
print(f"degree vs messages: spearman {rho_d:.2f}, pearson {r_d:.2f}")

# Three outcomes of the same check, in rank space, where spearman lives: what the chat gave,
# a made-up crowd whose few loud accounts carry Pearson alone, and plain degree, where people
# leave the diagonal. Rank 1 is the fewest, so "more" is up and to the right.
crowd = np.random.default_rng(7)
fake = pd.DataFrame(
    {
        "x": np.exp(crowd.normal(1, 1, 400)),
        "y": np.exp(crowd.normal(1, 1, 400)),
    }
)
loud_x = np.array([9000, 6000, 4000, 2500, 1600, 1000])
fake.loc[:5, "x"] = loud_x
fake.loc[:5, "y"] = loud_x * np.exp(crowd.normal(0, 0.3, 6))
r_fake = pearsonr(fake.x, fake.y).statistic
rho_fake = spearmanr(fake.x, fake.y).statistic
print(f"synthetic crowd: pearson {r_fake:.2f}, spearman {rho_fake:.2f}")

ranks = scores.rank()
visitors = scores[(scores.messages <= 3) & (scores.degree >= 5)].index
print(f"{len(visitors)} accounts with at most 3 messages and at least 5 neighbours")
print(
    f"lubotu3`: rank {scores.degree.rank(ascending=False, method='min')['lubotu3`']:.0f} by degree, "
    f"{scores.messages.rank(ascending=False, method='min')['lubotu3`']:.0f} by messages, "
    f"{scores.weighted_degree.rank(ascending=False, method='min')['lubotu3`']:.0f} by weighted degree"
)
fig, axes = plt.subplots(1, 3, figsize=(17, 5.9))
panels = [
    (
        ranks.messages,
        ranks.weighted_degree,
        "weighted degree",
        "what we got",
        r_w,
        rho_w,
    ),
    (
        fake.x.rank(),
        fake.y.rank(),
        "",
        "synthetic: Pearson high, ρ low",
        r_fake,
        rho_fake,
    ),
    (
        ranks.messages,
        ranks.degree,
        "degree",
        "plain degree: people leave the line",
        r_d,
        rho_d,
    ),
]
for ax, (x, y, ylabel, title, r, rho) in zip(axes, panels):
    ax.plot([0, len(x)], [0, len(x)], "--", color=GREY_FAINT, lw=1.2, zorder=0)
    ax.scatter(x, y, s=20, color=ACCENT_LIGHT, alpha=0.7, linewidths=0)
    ax.set_title(f"{title}\nPearson r = {r:.2f}, spearman ρ = {rho:.2f}", fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])
axes[0].set_xlabel("rank by messages sent  →")
axes[0].set_ylabel("rank by weighted degree  →")
axes[1].scatter(fake.x.rank()[:6], fake.y.rank()[:6], s=70, color=SECOND, zorder=3)
axes[1].annotate(
    "six loud accounts\nnear one line",
    (fake.x.rank()[:6].min(), 395),
    xytext=(-14, -18),
    textcoords="offset points",
    ha="right",
    va="top",
    color=SECOND,
    fontsize=13,
)
axes[1].set_xlabel("rank by x  →")
axes[1].set_ylabel("rank by y  →")
axes[2].scatter(
    ranks.messages[visitors],
    ranks.degree[visitors],
    s=26,
    color=SECOND,
    alpha=0.8,
    zorder=3,
)
axes[2].scatter(
    ranks.messages["lubotu3`"], ranks.degree["lubotu3`"], s=110, color=SECOND, zorder=4
)
axes[2].annotate(
    "lubotu3` (a bot)",
    (ranks.messages["lubotu3`"], ranks.degree["lubotu3`"]),
    xytext=(0, 14),
    textcoords="offset points",
    ha="center",
    color=SECOND,
    fontsize=13,
)
axes[2].annotate(
    f"{len(visitors)} visitors: 1–3 messages,\n5 or more neighbours",
    (0.02, 0.98),
    xycoords="axes fraction",
    ha="left",
    va="top",
    color=SECOND,
    fontsize=13,
)
axes[2].set_xlabel("rank by messages sent  →")
axes[2].set_ylabel("rank by degree  →")
save(fig, "spearman")

shared = sorted(set(core) & set(core_named))
both = pd.DataFrame(
    {
        "nearby_clustering": pd.Series(nx.clustering(core)).reindex(shared),
        "nearby_degree": pd.Series(dict(core.degree())).reindex(shared),
        "named_clustering": pd.Series(nx.clustering(core_named)).reindex(shared),
        "named_degree": pd.Series(dict(core_named.degree())).reindex(shared),
    }
)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
for ax, rule, title in zip(axes, ["nearby", "named"], ["nearby rule", "mention rule"]):
    degree, clustering = both[f"{rule}_degree"], both[f"{rule}_clustering"]
    alone = degree == 1
    rho_all = spearmanr(degree, clustering).statistic
    rho_rest = spearmanr(degree[~alone], clustering[~alone]).statistic
    print(
        f"{rule}: {alone.sum()} with one neighbour, spearman {rho_all:+.2f} with them, {rho_rest:+.2f} without"
    )
    for label, x in [("pearson", degree), ("pearson on log degree", np.log(degree))]:
        print(
            f"  {label}: {pearsonr(x, clustering).statistic:+.2f} with them, "
            f"{pearsonr(x[~alone], clustering[~alone]).statistic:+.2f} without"
        )
    ax.scatter(
        degree[~alone],
        clustering[~alone],
        s=26,
        color=ACCENT_LIGHT,
        alpha=0.7,
        linewidths=0,
    )
    ax.scatter([1], [0], s=40 + 3 * alone.sum(), color=SECOND, zorder=3)
    ax.annotate(
        f"   {alone.sum()} people with one neighbour",
        (1, 0),
        color=SECOND,
        va="center",
        fontsize=14,
    )
    ax.set_xscale("log")
    ax.set_title(f"{title}: ρ = {rho_all:+.2f}, without the red: {rho_rest:+.2f}")
    ax.set_xlabel("degree")
axes[0].set_ylabel("clustering coefficient")
axes[0].set_ylim(-0.08, 1.05)
save(fig, "signflip")

# The same three measures under every window, each ranked against its own ten-minute ranking
# on the people both giant components contain.
windows = {}
for seconds in (60, 300, 600, 1800):
    giant = giant_component(builder.build(chat, edge_seconds=seconds))
    windows[seconds] = pd.DataFrame(
        {
            "degree": dict(giant.degree()),
            "betweenness": nx.betweenness_centrality(giant),
            "clustering": nx.clustering(giant),
        }
    )
reference = windows[600]
rows = []
for seconds, measures in windows.items():
    common = measures.index.intersection(reference.index)
    a, b = measures.loc[common], reference.loc[common]
    two = (a.degree >= 2) & (b.degree >= 2)
    rows.append(
        {
            "minutes": seconds / 60,
            "people": len(common),
            "two_or_more": int(two.sum()),
            **{
                column: spearmanr(a[column], b[column]).statistic
                for column in ["degree", "betweenness", "clustering"]
            },
            "clustering_two": spearmanr(a.clustering[two], b.clustering[two]).statistic,
        }
    )
sweep = pd.DataFrame(rows)
print(sweep.round(2).to_string(index=False))
ends = windows[60].index.intersection(windows[1800].index)
print(
    "1 min against 30 min: "
    + ", ".join(
        f"{column} {spearmanr(windows[60].loc[ends, column], windows[1800].loc[ends, column]).statistic:.2f}"
        for column in ["degree", "betweenness", "clustering"]
    )
)

fig, ax = plt.subplots(figsize=(8.5, 5.6))
for column, colour, style, label in [
    ("degree", ACCENT, "-", "degree"),
    ("betweenness", THIRD, "-", "betweenness"),
    ("clustering", SECOND, "-", "clustering"),
    ("clustering_two", SECOND, ":", "clustering, 2+ neighbours"),
]:
    ax.plot(
        sweep.minutes,
        sweep[column],
        style,
        color=colour,
        lw=1.6,
        marker="o",
        markersize=8,
        label=label,
    )
for m, v in zip(sweep.minutes, sweep.clustering):
    if m != 10:
        ax.annotate(
            f"{v:.2f}",
            (m, v),
            xytext=(0, -20),
            textcoords="offset points",
            ha="center",
            color=SECOND,
            fontsize=13,
        )
ax.set_xscale("log")
ax.set_xticks(sweep.minutes, ["1 min", "5 min", "10 min", "30 min"])
ax.minorticks_off()
ax.set_ylim(0, 1.05)
ax.set_ylabel("spearman ρ with the 10-minute ranking")
ax.set_xlabel("window (10 min is the reference, so 1 by construction)")
ax.legend(loc="lower right")
save(fig, "window_sweep")

# ---------- 04: one graph, two layouts, and its matrix ----------
hubs = giant_component(to_graph(MentionEdges()(chat)))
hubs = giant_component(induced(hubs, [n for n in hubs if hubs.degree(n) >= 6]))
hub_degrees = np.array([hubs.degree(n) for n in hubs], dtype=float)
print(f"hubs graph: {hubs.number_of_nodes()} people, {hubs.number_of_edges()} edges")
colours = LinearSegmentedColormap.from_list("deg", [ACCENT_LIGHT, ACCENT, SECOND])
fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
for ax, pos, title in [
    (axes[0], nx.spring_layout(hubs, seed=1, k=0.35), "spring layout"),
    (axes[1], nx.circular_layout(hubs), "circular layout"),
]:
    nx.draw_networkx_edges(
        hubs, pos, ax=ax, edge_color=GREY_FAINT, width=0.5, alpha=0.6
    )
    nx.draw_networkx_nodes(
        hubs,
        pos,
        ax=ax,
        node_color=hub_degrees,
        cmap=colours,
        node_size=4 * hub_degrees,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_aspect("equal")
save(fig, "layouts")

by_degree = sorted(hubs, key=hubs.degree, reverse=True)
fig, ax = plt.subplots(figsize=(7, 7))
matrix(ax, hubs, by_degree, weight="weight", log=True)
ax.set_xlabel("people, most connected first →")
ax.set_ylabel("← people, most connected first")
save(fig, "hubmatrix")

# ---------- 04: which view, sparse and dense ----------
west_to_east = sorted(trains, key=lambda s: stations[s][1], reverse=True)
fig, ax = plt.subplots(figsize=(6.4, 6.4))
matrix(ax, trains, west_to_east, cmap=PRESENT)
ax.set_xticks(range(len(west_to_east)), west_to_east, rotation=90, fontsize=12)
ax.set_yticks(range(len(west_to_east)), west_to_east, fontsize=12)
ax.tick_params(length=0)
save(fig, "trains_matrix")

eight = chat.author.value_counts().head(8).index
dense = induced(nearby, eight)
print(
    f"eight busiest: {dense.number_of_edges()} of 28 pairs connected; weights "
    f"{min(d['weight'] for *_, d in dense.edges(data=True))}..{max(d['weight'] for *_, d in dense.edges(data=True))}"
)
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
pos = nx.circular_layout(dense)
draw(
    axes[0],
    dense,
    pos,
    color=ACCENT,
    size=700,
    edge_color=GREY_DIM,
    edge_width=[d["weight"] / 1500 for *_, d in dense.edges(data=True)],
)
for n, (x, y) in pos.items():
    axes[0].text(
        1.28 * x, 1.2 * y, n, ha="center", va="center", fontsize=13, color=GREY
    )
axes[0].set_xlim(-1.6, 1.6)
axes[0].set_ylim(-1.45, 1.45)
weights = nx.to_numpy_array(dense, nodelist=list(eight), weight="weight")
axes[1].imshow(weights, cmap=CELLS, interpolation="nearest")
for i, j in itertools.product(range(8), range(8)):
    if i != j:
        axes[1].text(
            j,
            i,
            f"{weights[i, j] / 1000:.1f}k",
            ha="center",
            va="center",
            fontsize=11,
            color=BG if weights[i, j] > weights.max() / 2 else TEXT,
        )
axes[1].set_xticks(range(8), eight, rotation=90, fontsize=12)
axes[1].set_yticks(range(8), eight, fontsize=12)
axes[1].tick_params(length=0)
save(fig, "dense")

# ---------- 05: the karate club ----------
karate = nx.karate_club_graph()
truth = np.array([0 if karate.nodes[n]["club"] == "Mr. Hi" else 1 for n in karate])
A = nx.to_numpy_array(karate)
L = np.diag(A.sum(axis=1)) - A
values, vectors = np.linalg.eigh(L)
fiedler = vectors[:, np.argsort(values)[1]]
side = (fiedler > 0).astype(int)
print(
    f"karate: Fiedler agrees with the record for {max((side == truth).mean(), 1 - (side == truth).mean()):.1%}"
)
kpos = nx.spring_layout(karate, seed=46)
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
draw(axes[0], karate, kpos, color=GREY_DIM, size=260, edge_color=GREY_FAINT, width=1.2)
axes[0].set_title("who spent time with whom")
draw(
    axes[1],
    karate,
    kpos,
    color=[ACCENT if t == 0 else SECOND for t in truth],
    size=260,
    edge_color=GREY_FAINT,
    width=1.2,
)
axes[1].set_title("which club they joined after the split")
save(fig, "karate")

fig, ax = plt.subplots(figsize=(6.4, 6.4), facecolor=INK_BG)
ax.set_facecolor(INK_BG)
nx.draw_networkx_edges(karate, kpos, ax=ax, edge_color="#5a5148", width=1.2)
nx.draw_networkx_nodes(
    karate,
    kpos,
    ax=ax,
    node_color=[INK_PETROL if t == 0 else INK_ACCENT for t in truth],
    node_size=200,
    linewidths=0,
)
ax.set_axis_off()
save(fig, "cover", facecolor=INK_BG)

flipped = 1 - side if (side == truth).mean() < 0.5 else side
wrong = [n for n in karate if flipped[n] != truth[n]]
shuffled = list(np.random.default_rng(0).permutation(list(karate)))
sides = sorted(karate, key=lambda n: (side[n], -karate.degree(n)))
fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
matrix(axes[0], karate, shuffled, cmap=PRESENT)
axes[0].set_title("members in random order")
matrix(axes[1], karate, sides, cmap=PRESENT)
axes[1].set_title("sorted by the second eigenvector")
cut = int((side == side[sides[0]]).sum()) - 0.5
for ax in [axes[1]]:
    ax.axhline(cut, color=SECOND, lw=1.5)
    ax.axvline(cut, color=SECOND, lw=1.5)
save(fig, "karate_matrix")
print(f"karate: the member on the wrong side is {wrong}")


# ---------- 06: communities in nothing, and in triangles ----------
def introduce(graph, p, seed):
    """Every person introduces each pair of their friends to each other with probability p."""
    rng = np.random.default_rng(seed)
    closed = graph.copy()
    for person in graph.nodes():
        for a, b in itertools.combinations(sorted(graph.neighbors(person)), 2):
            if rng.random() < p:
                closed.add_edge(a, b)
    return closed


counts = np.random.default_rng(0).geometric(0.5, size=100)
counts[0] += counts.sum() % 2
strangers = nx.Graph(nx.configuration_model(counts, seed=0))
strangers.remove_edges_from(nx.selfloop_edges(strangers))
strangers = giant_component(strangers)
friends = introduce(strangers, p=0.8, seed=0)
pos = nx.spring_layout(friends, seed=2)
fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
for ax, graph, title in [
    (axes[0], strangers, "random, no groups"),
    (axes[1], friends, "after introductions"),
]:
    groups = labels(graph, best_partition(graph))
    draw(
        ax,
        graph,
        pos,
        color=[CATEGORIES[groups[n] % len(CATEGORIES)] for n in graph],
        size=90,
        edge_color=GREY_FAINT,
        width=1.0,
    )
    ax.set_title(f"{title}: {graph.number_of_edges()} edges")
save(fig, "introductions")

mentions = giant_component(named)
rows = []
for name, graph in [
    ("karate club", karate),
    ("Florentine families", florence),
    ("random, no groups", strangers),
    ("random + introductions", friends),
    ("irc mentions", mentions),
]:
    q, nulls, z = null_qs(graph)
    rows.append((name, q, nulls, z))
    print(f"{name:24s} Q={q:.3f} null={nulls.mean():.3f}±{nulls.std():.3f} z={z:+.1f}")
fig, ax = plt.subplots(figsize=(11, 5.6))
for i, (name, q, nulls, z) in enumerate(rows[::-1]):
    ax.scatter(
        nulls,
        np.full(len(nulls), i)
        + np.random.default_rng(i).uniform(-0.12, 0.12, len(nulls)),
        s=22,
        color=GREY_FAINT,
        zorder=2,
    )
    ax.scatter(q, i, s=150, color=SECOND, zorder=3)
    ax.annotate(
        f"z = {z:+.1f}",
        (max(q, nulls.max()), i),
        xytext=(16, 0),
        textcoords="offset points",
        va="center",
        fontsize=14,
        color=SECOND,
    )
ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
ax.tick_params(axis="y", length=0)
ax.set_xlabel("modularity Q of the best partition louvain finds")
ax.set_xlim(0.15, 0.85)
ax.scatter([], [], s=150, color=SECOND, label="the graph itself")
ax.scatter([], [], s=22, color=GREY_FAINT, label="20 rewired copies, same degrees")
ax.legend(loc="lower right")
save(fig, "nullmodel")


# ---------- 06: squares or hubs, and the two halves of the year ----------
def by_community(graph, partition):
    groups = labels(graph, partition)
    return sorted(graph, key=lambda n: (groups[n], -graph.degree(n)))


fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
matrix(
    axes[0],
    karate,
    by_community(karate, best_partition(karate, resolution=0.6)),
    cmap=PRESENT,
)
axes[0].set_title("karate club, 34 members")
irc_partition = best_partition(mentions, weight="weight")
matrix(axes[1], mentions, by_community(mentions, irc_partition), cmap=PRESENT)
axes[1].set_title("irc mention graph, 256 people")
for ax, graph, partition in [
    (axes[0], karate, best_partition(karate, resolution=0.6)),
    (axes[1], mentions, irc_partition),
]:
    edges = np.cumsum(sorted((len(c) for c in partition), reverse=True))[:-1] - 0.5
    for e in edges:
        ax.axhline(e, color=SECOND, lw=1)
        ax.axvline(e, color=SECOND, lw=1)
save(fig, "squares")

half = pd.Timestamp("2015-07-01")
first = giant_component(to_graph(MentionEdges()(chat[chat.timestamp < half])))
second = giant_component(to_graph(MentionEdges()(chat[chat.timestamp >= half])))
first_labels = labels(first, best_partition(first, weight="weight"))
second_labels = labels(second, best_partition(second, weight="weight"))
survivors = sorted(set(first) & set(second))
ari = adjusted_rand_score(
    first_labels.reindex(survivors), second_labels.reindex(survivors)
)
table = pd.crosstab(first_labels.reindex(survivors), second_labels.reindex(survivors))
print(f"halves: {len(survivors)} people in both, ARI {ari:.3f}")
print(table)
fig, ax = plt.subplots(figsize=(7.4, 6))
ax.imshow(table.to_numpy(), cmap=CELLS, interpolation="nearest")
for i, j in itertools.product(range(table.shape[0]), range(table.shape[1])):
    value = table.to_numpy()[i, j]
    ax.text(
        j,
        i,
        str(value),
        ha="center",
        va="center",
        fontsize=18,
        color=BG if value > table.to_numpy().max() / 2 else TEXT,
    )
ax.set_xticks(
    range(table.shape[1]), [f"group {k}" for k in range(1, table.shape[1] + 1)]
)
ax.set_yticks(
    range(table.shape[0]), [f"group {k}" for k in range(1, table.shape[0] + 1)]
)
ax.set_xlabel("community in July–December")
ax.set_ylabel("community in January–June")
ax.tick_params(length=0)
save(fig, "halves")

# ---------- formulas: typeset with mathtext, one line each, at 2x for the slides ----------
FORMULAS = {
    "formula_g": r"$G = (V,\, E)$",
    "formula_v": r"$V = \{A, B, C, D, E\}$",
    "formula_e": r"$E = \{\{A,B\},\, \{A,C\},\, \{B,C\},\, \{C,D\},\, \{D,E\}\}$",
    "formula_clustering": r"$C_i \;=\; \dfrac{\mathrm{pairs\ of\ friends\ of\ } i \mathrm{\ who\ know\ each\ other}}{\mathrm{all\ pairs\ of\ friends\ of\ } i}$",
    "formula_spearman": r"$\rho \;=\; \mathrm{corr}\left(\mathrm{rank}(x),\; \mathrm{rank}(y)\right)$",
    "formula_q": r"$Q \;=\; \dfrac{1}{2m} \sum_{ij} \left( A_{ij} - \dfrac{k_i\, k_j}{2m} \right) [\,c_i = c_j\,]$",
    "formula_z": r"$z \;=\; \dfrac{Q - \mathrm{mean}(Q_{\mathrm{rewired}})}{\mathrm{sd}(Q_{\mathrm{rewired}})}$",
}
with plt.rc_context({"mathtext.fontset": "cm"}):
    for name, tex in FORMULAS.items():
        fig = plt.figure(figsize=(1, 1))
        fig.text(0, 0, tex, fontsize=30, color=TEXT)
        fig.savefig(
            OUT / f"{name}.png",
            dpi=144,
            bbox_inches="tight",
            pad_inches=0.04,
            transparent=True,
        )
        plt.close(fig)
        print("wrote", name)
