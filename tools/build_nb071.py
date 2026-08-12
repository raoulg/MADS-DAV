#!/usr/bin/env python3
"""Build notebooks/lesson7/07.1-social_graphs.ipynb (PTT-52).

Written as a builder so the notebook is reproducible and reviewable as source
rather than as a diff of JSON. Run from the repo root:

    uv run python tools/build_nb071.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/lesson7/07.1-social_graphs.ipynb"

cells: list[tuple[str, str]] = []


def md(text: str) -> None:
    cells.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    cells.append(("code", text.strip("\n")))


# ---------------------------------------------------------------- intro
md("""
# 7.1 Social graphs — the edge is the decision

**The question: who talks to whom?**

A graph is two things: nodes and edges. In a chat the nodes are obvious — one per author.
The edges are not, because "talked to" is not a column in your data. You have to define it,
and *the definition is the analysis*. Change it and the same messages give you a different
network, a different central person, and a different sentence in your report.

This lesson defines it twice on the same messages, and measures how far apart the two
answers are:

1. **Who was nearby** — an edge whenever two people speak within a few minutes of each
   other. This is what `wa_analyzer.network_analysis` ships, and what the dashboard draws.
2. **Who was addressed** — an edge only when the sender typed the other person's name.
   That column is `addressed_to`, from lesson 1's regex pipeline.

Then the dashboard, and one loose end from lesson 2: the plot class you wrote there is
about to render inside streamlit without a line of it changing.
""")

code("""
import networkx as nx
import pandas as pd
from goad_toolkit.visualizer import PlotSettings, ScatterPlot

from scripts.pipelines import build_irc_pipeline
from scripts.plots import BarPlot
from wa_analyzer.data import load_own_chat, load_showcase
from wa_analyzer.network_analysis import Config, GraphBuilder, GraphVisualizer
""")

# ---------------------------------------------------------------- 7.1.1
md("""
## 7.1.1 One year of one channel

The IRC showcase again, through lesson 1's pipeline — the same `build_irc_pipeline()` you
have imported since lesson 2, which is where `addressed_to` comes from.

Two narrowings, both for the same reason. A graph of 627,000 messages over five years and
two channels is a hairball: every regular is eventually within ten minutes of every other
regular, so everything connects to everything and the picture says nothing. **A graph needs
a scope before it needs a layout.**
""")

code("""
irc = build_irc_pipeline().apply(load_showcase("ubuntu_irc"))
irc["timestamp"] = (
    irc.date + pd.to_timedelta(irc.hh, unit="h") + pd.to_timedelta(irc.mm, unit="m")
)

chat = irc[(irc.channel == "#ubuntu-uk") & (irc.date.dt.year == 2015)].copy()
print(f"{len(chat):,} messages, {chat.author.nunique()} authors, "
      f"{chat.date.min():%b %Y} to {chat.date.max():%b %Y}")
""")

# ---------------------------------------------------------------- rule 1
md("""
## 7.1.2 Rule one: who was nearby

`GraphAnalyzer.edges` walks the messages in time order with a sliding window and adds an
edge between everyone who spoke inside it. There is nothing clever in it, and that is worth
seeing rather than trusting:

```python
for right_idx in range(len(timestamps)):
    window_start = timestamps[right_idx] - window_size
    while left_idx < right_idx and timestamps[left_idx] < window_start:
        left_idx += 1
    for i in range(left_idx, right_idx):
        if authors[i] != current_author:
            edges[(current_author, authors[i])] += 1
```

The whole rule is that last `if`: **two people who spoke within `seconds` of each other are
connected.** `Config` is where you say how long that window is, and `GraphBuilder` turns
the counted pairs into a `networkx` graph.
""")

code("""
config = Config(time_col="timestamp", node_col="author", seconds=600, datafile=None)
builder = GraphBuilder(config)

nearby = builder.build(chat, edge_seconds=600)
print(f"nearby graph: {nearby.number_of_nodes()} nodes, {nearby.number_of_edges():,} edges")
""")

# ---------------------------------------------------------------- rule 2
md("""
## 7.1.3 Rule two: who was addressed

On IRC people answer each other by name — `davmor2: try purging it first`. Lesson 1 turned
that convention into a column with a three-line `RegexFeature`:

```python
pipeline.add(RegexFeature, name="mentions",
             column="message", pattern=r"^(\\S+)[:,]\\s", feature="addressed_to", mode="extract")
```

So the data already contains a *declared* edge: not "these two were both here", but "this
person aimed this message at that person". Before building anything on it, check what the
regex actually caught.
""")

code("""
print(f"messages starting with a word and a colon: {chat.addressed_to.notna().mean():.1%}")

nicks = set(chat.author)
known = chat.addressed_to.isin(nicks)
print(f"...where that word is also an author here: {known.mean():.1%}")
print(f"\\nthe most common words the regex caught that never spoke:")
print(chat[chat.addressed_to.notna() & ~known].addressed_to.value_counts().head(5).to_string())
""")

md("""
`yeah`, `well`, `oh`, `hmm`, `no`. The pattern `^(\\S+)[:,]\\s` matches *any* word followed by
a comma, and "yeah, that worked" is not a person. 17.0% of messages match the pattern; 12.0%
address someone who actually spoke in this channel — so **a third of the matches are not
names at all.**

This is lesson 1's point arriving with consequences. `addressed_to` was never a nick
detector; it is a regex, and a regex has no idea what a nick is. The check that turns it into
one is the `isin(nicks)` above, and it costs one line — but only because we knew to ask.
""")

code("""
declared = chat[known]
pairs = declared.groupby(["author", "addressed_to"]).size()

named = nx.Graph()
named.add_nodes_from(chat.author.unique())
for (sender, target), weight in pairs.items():
    if sender != target:
        named.add_edge(sender, target, weight=weight)

print(f"named graph:  {named.number_of_nodes()} nodes, {named.number_of_edges():,} edges")
print(f"nearby graph: {nearby.number_of_nodes()} nodes, {nearby.number_of_edges():,} edges")
""")

# ---------------------------------------------------------------- comparison
md("""
## 7.1.4 The same messages, two networks

Same 95,518 messages, same 443 people, and one graph has nearly three times the edges of the
other. That is not a bug in either rule — they are answering different questions. The useful
move is to ask how they disagree, which is a set operation on the edges.
""")

code("""
nearby_edges = {frozenset(edge) for edge in nearby.edges()}
named_edges = {frozenset(edge) for edge in named.edges()}

found = len(nearby_edges & named_edges) / len(named_edges)
extra = len(nearby_edges - named_edges) / len(nearby_edges)
print(f"declared edges the nearby rule also found: {found:.1%}")
print(f"nearby edges with nothing declared:        {extra:.1%} "
      f"({len(nearby_edges - named_edges):,} of {len(nearby_edges):,})")
""")

md("""
Read those two numbers as a pair, because they are the two halves of the same question.

**97.4% — the nearby rule almost never misses a real conversation.** If two people addressed
each other by name at any point in 2015, they were also within ten minutes of each other.
That is reassuring, and it is the easy half.

**65.8% — two thirds of the edges it draws have no declared counterpart.** Some of those are
real conversations where nobody typed a name. Others are two people who happened to be in the
same room at the same time and never exchanged a word. Nothing in the timing data can tell
you which is which.

So the nearby rule is *generous*: it finds everything, plus a great deal more, and it cannot
label the difference. That is fine as long as you say it out loud — and fatal the moment you
write "these two are close" about a specific pair.
""")

code("""
degrees = pd.concat([
    pd.Series(dict(nearby.degree()), name="nearby"),
    pd.Series(dict(named.degree()), name="named"),
], axis=1).fillna(0).reset_index(names="author")

busy = degrees[degrees.nearby >= 20].copy()
busy["per_neighbour"] = (busy.named / busy.nearby).round(2)
busy["gap"] = busy.nearby - busy.named
busy["rank"] = degrees.nearby.rank(ascending=False).astype(int)
print(f"{len(busy)} authors with at least 20 neighbours under the nearby rule")
print(f"spearman between the two degrees: "
      f"{busy.nearby.corr(busy.named, method='spearman'):.2f}\\n")

columns = ["author", "nearby", "named", "per_neighbour", "gap", "rank"]
print("addressed by the smallest share of the people they were near:")
print(busy.nsmallest(3, "per_neighbour")[columns].to_string(index=False))
print(f"\\nand by the fewest of them in absolute terms, out of {len(degrees)} authors:")
print(busy.nlargest(3, "gap")[columns].to_string(index=False))
""")

code("""
flattered = pd.concat([busy.nsmallest(3, "per_neighbour"), busy.nlargest(3, "gap")])
# The other two low-ratio nicks sit on top of each other down in the corner, so they
# stay red without a label rather than turning into a smudge of overlapping text.
labelled = pd.concat([busy.nsmallest(1, "per_neighbour"), busy.nlargest(3, "gap")])

settings = PlotSettings(
    figsize=(7, 6),
    title="The same people, under two definitions of an edge",
    xlabel="people they were within 10 minutes of",
    ylabel="people they addressed by name",
)
scatter = ScatterPlot(settings)
fig, ax = scatter.plot(data=busy, x="nearby", y="named", color="#cccccc")
ax.scatter(flattered.nearby, flattered.named, color="#c44e52", zorder=3)
ax.plot([0, busy.nearby.max()], [0, busy.nearby.max()], "--", color="#dddddd", zorder=0)
for row in labelled.itertuples():
    ax.annotate(f"  {row.author}", (row.nearby, row.named), color="#c44e52", va="center")
""")

md("""
Every point is below the diagonal, which is only to be expected — you cannot address more
people than you were near. What matters is *how far* below, and that it varies.

`lubotu3\\`` is the clearest case: 42 neighbours, **1** person addressed. It is the channel's
bug-tracker bot — it posts a link whenever someone types a bug number, so it is present at
every busy moment and part of no conversation at all. The nearby rule ranks it 36th of 443
for connectedness; the declared rule puts it near the bottom.

`zmoylan-pi` is the more interesting one, because it is a person: 206 neighbours, 51
addressed — the second-most present account in the channel, addressed by name a quarter as
often as it is nearby. Both facts are true, and they describe different social positions.
**A centrality score is a claim about a rule, not about a person.**

Note also what the correlation hides. The two degrees agree at a spearman of 0.90, which
sounds like the rule barely matters; the six red points are inside that 0.90.
""")

# ---------------------------------------------------------------- the slider
md("""
### The window is a dial on the answer

`seconds` looks like a technical detail buried in a config object. It is not: it is the
strength of the claim you are making. Watch both numbers move together.
""")

code("""
rows = []
for seconds in (60, 300, 600, 1800):
    window = builder.build(chat, edge_seconds=seconds)
    edges = {frozenset(edge) for edge in window.edges()}
    rows.append({
        "window": f"{seconds}s",
        "edges": window.number_of_edges(),
        "declared found": len(edges & named_edges) / len(named_edges),
        "undeclared": len(edges - named_edges) / len(edges),
    })
print(pd.DataFrame(rows).to_string(index=False, formatters={
    "declared found": "{:.1%}".format, "undeclared": "{:.1%}".format}))
""")

md("""
One minute finds 93.3% of the declared conversations and half its edges are undeclared.
Thirty minutes finds 98.5% and nearly three quarters are undeclared. There is no setting
where you get one without the other, and no setting the data picks for you.

The dashboard exposes this as a slider labelled "Response Window (seconds)". It reads like a
rendering option, next to node size and edge width. **It is the single most consequential
number in the analysis**, and a reader looking at your screenshot has no way of knowing where
you left it. Put it in the caption.
""")

code("""
nl = irc[(irc.channel == "#ubuntu-nl") & (irc.date.dt.year == 2015)].copy()
nl_known = nl.addressed_to.isin(set(nl.author))
nl_nearby = {frozenset(e) for e in builder.build(nl, edge_seconds=600).edges()}
nl_named = {
    frozenset((sender, target))
    for sender, target in nl[nl_known].groupby(["author", "addressed_to"]).groups
    if sender != target
}
print(f"#ubuntu-nl 2015: {len(nl):,} messages, {nl.author.nunique()} authors")
print(f"declared edges the nearby rule also found: "
      f"{len(nl_nearby & nl_named) / len(nl_named):.1%}")
print(f"nearby edges with nothing declared:        "
      f"{len(nl_nearby - nl_named) / len(nl_nearby):.1%}")
""")

md("""
A different channel, a different language, a fifth of the traffic — 89.9% and 58.3%. Both
numbers are a little lower and the shape is the same, which is what you want from a check
like this. One channel would have been an anecdote.
""")

# ---------------------------------------------------------------- nodes
md("""
### Nodes are a decision too

Everything above argued about edges and quietly accepted `author` as the node column. It is
worth about thirty seconds of doubt.
""")

code("""
variants = chat[chat.author.str.lower() == "moodoo"].author.value_counts()
print(variants.to_string())
print(f"\\nnodes in the graph for this one person: {len(variants)}")
print(f"their degree, split across those nodes: "
      f"{[nearby.degree(nick) for nick in variants.index]}")
""")

md("""
One person, three nodes, and the graph has no idea. Capitalisation is only the mildest form
of it: IRC clients let people change nick mid-session, so the same person also appears with a
trailing underscore whenever they step away. The node list is as much a cleaning problem as
lesson 1's message parsing was.

In your own chat this arrives as the same person appearing under a phone number and a saved
contact name. Whether it is worth fixing depends on the claim: irrelevant if you are counting
messages per day, fatal if you are ranking people by centrality.
""")

# ---------------------------------------------------------------- picture
md("""
## 7.1.5 The picture

Only now, after the edges mean something, is a layout worth drawing. `GraphVisualizer` turns
a `networkx` graph into an interactive plotly figure, and `filter_connections` drops the
long tail of one-conversation nodes so the middle is legible.

The threshold is doing the same job the window did — it is a choice that changes what the
picture says. Nine is not a magic number; it is what leaves roughly fifty nodes.
""")

code("""
viz = GraphVisualizer()
core = viz.filter_connections(named, threshold=9)
print(f"{core.number_of_nodes()} nodes, {core.number_of_edges()} edges")

positions = builder.calculate_layout(core, name="Spring Layout", scale=2.0)
fig = viz(core, builder.node_colors(core), positions, title="Who answered whom, #ubuntu-uk 2015",
          node_scale=0.4, edge_scale=0.5, node_threshold=9)
fig.show()
""")

md("""
Hover a node for its degree, and a line for how many messages it carries.

What a spring layout is doing: it treats edges as springs and nodes as repelling charges,
then settles. So distance on screen is roughly "how strongly connected", and **position is
not data**. Rerunning with a different seed gives a picture that looks different and means
the same thing. Never describe a node as being "on the edge of the network" when what you
mean is that this layout put it there — say its degree instead.
""")

# ---------------------------------------------------------------- dashboard
md("""
## 7.1.6 The dashboard, and a promise from lesson 2

Two dashboards in this repo draw graphs like the one above: `streamlit_app.py` at the root,
which runs on your own processed chat, and `dashboards/dashboard_5.py`, which runs on the
IRC showcase so it works before you have exported anything.

`dashboard_5.py` is worth opening, because of its import block:

```python
from scripts.plots import BarPlot
```

That is the class **you wrote in lesson 2**, in `scripts/plots.py`, unchanged. No streamlit
in it, no `st.` anywhere near it. It renders in the browser because `plot()` returns an
ordinary matplotlib figure and `st.pyplot(fig)` accepts one:

```python
fig, ax = BarPlot(settings).plot(data=top, x="degree", y="author", color="#cccccc")
st.pyplot(fig)
```

This is the whole return on subclassing `BasePlot` instead of writing `plt.subplots` in a
cell. A chart that lives in a notebook cell can only ever be looked at in that notebook; a
chart that is a class can be imported by a dashboard, a report script, or lesson 7. The same
two lines run here:
""")

code("""
top = degrees.nlargest(12, "named").sort_values("named", ascending=False)

bars = PlotSettings(
    figsize=(7, 5),
    title="Most-addressed people in #ubuntu-uk, 2015",
    xlabel="people who addressed them by name",
    ylabel="",
)
fig, ax = BarPlot(bars).plot(data=top, x="named", y="author", color="#cccccc")
""")

md("""
Run the dashboard from the repo root:

```bash
uv run streamlit run dashboards/dashboard_5.py
```

It has one slider — the response window — and shows both graphs' edge counts side by side as
you move it. That is the point of putting it in a dashboard rather than a notebook: the
parameter you most need to feel is the one you should be able to drag.
""")

# ---------------------------------------------------------------- your turn
md("""
## 7.1.7 Your turn

Your own chat has no nick convention, so `addressed_to` will not survive the trip — WhatsApp
has @-mentions in groups and nothing at all in a one-to-one chat. **The nearby rule is what
you have**, which is exactly why the section above spent so long on what it cannot tell you.
""")

code("""
own = load_own_chat()
if own is not None:
    own_config = Config(time_col="timestamp", node_col="author", seconds=600, datafile=None)
    own_graph = GraphBuilder(own_config).build(own, edge_seconds=600)
    own_degrees = pd.Series(dict(own_graph.degree())).sort_values(ascending=False)
    print(f"{own_graph.number_of_nodes()} people, {own_graph.number_of_edges()} edges, "
          f"from {len(own):,} messages")
    print(f"\\n{own_degrees.head(8).to_string()}")
else:
    print("No chat of your own yet — the showcase above still stands.")
""")

md("""
> **Your turn.**
>
> 1. Build the graph at two different windows — 60 seconds and 30 minutes. Does the most
>    central person change? If not, say so; a stable answer is a result.
> 2. A group chat of eight people is nearly complete at any window, so degree tells you
>    almost nothing. Use **edge weight** instead: who exchanges the most messages with whom.
> 3. Name one pair the graph says are connected who you know are not, and explain what the
>    rule did.
""")

md("""
## 7.1.8 What to write down

1. **Your edge rule, as a sentence** — "two people are connected if…", including the window.
2. **What that rule cannot distinguish**, in your data specifically.
3. **One number that is not a picture** — a degree, an edge weight, a count of components.
   A network diagram is very good at looking like a finding.
4. **What you did about the nodes** — merged nicknames, dropped bots, or nothing, deliberately.

---

**Where this goes next.** Nothing, in this course. The tools stop here; what does not stop is
the habit the last seven lessons were actually about. Every one of them had the same shape:
a technique that works, and a decision inside it that the technique cannot make for you —
which category order, which unit of analysis, which window, which baseline, which edge.

You will be asked to defend one of those choices. Not which library you used.
""")


def write() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": kind,
                "id": f"c{i:02d}",
                "metadata": {},
                **({"source": text.splitlines(keepends=True), "outputs": [], "execution_count": None}
                   if kind == "code" else {"source": text.splitlines(keepends=True)}),
            }
            for i, (kind, text) in enumerate(cells)
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} with {len(cells)} cells")


if __name__ == "__main__":
    write()
