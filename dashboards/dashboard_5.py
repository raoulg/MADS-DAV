"""Lesson 7's social graph, with the response window as a slider.

Two things this file exists to show:

- The window is not a rendering option. Drag it and the number of edges changes by a
  factor of two, so the network you screenshot is partly a choice you made with a mouse.
- `BarPlot` is imported from `scripts/plots.py` exactly as lesson 2 wrote it. There is no
  streamlit anywhere in that class; `plot()` returns a matplotlib figure and `st.pyplot`
  takes one. That import is the payoff for subclassing `BasePlot`.

Run it from the repo root:

    uv run streamlit run dashboards/dashboard_5.py
"""

import networkx as nx
import pandas as pd
import streamlit as st
from goad_toolkit.visualizer import PlotSettings

from scripts.pipelines import build_irc_pipeline
from scripts.plots import BarPlot
from wa_analyzer.data import load_showcase
from wa_analyzer.network_analysis import Config, GraphBuilder, GraphVisualizer


@st.cache_data
def load_irc() -> pd.DataFrame:
    """Parse and enrich the IRC showcase once, rather than on every widget interaction.

    Streamlit re-runs the whole script whenever anything is touched, and this is 627,000
    messages through a five-step pipeline. Unlike the penguin dashboards, here the cache
    is doing real work.
    """
    irc = build_irc_pipeline().apply(load_showcase("ubuntu_irc"))
    irc["timestamp"] = (
        irc.date + pd.to_timedelta(irc.hh, unit="h") + pd.to_timedelta(irc.mm, unit="m")
    )
    return irc


def declared_graph(chat: pd.DataFrame) -> nx.Graph:
    """An edge wherever someone addressed a nick that also speaks in this channel."""
    known = chat[chat.addressed_to.isin(set(chat.author))]
    graph = nx.Graph()
    graph.add_nodes_from(chat.author.unique())
    for (sender, target), weight in known.groupby(["author", "addressed_to"]).size().items():
        if sender != target:
            graph.add_edge(sender, target, weight=weight)
    return graph


def main() -> None:
    st.title("Who talks to whom")
    irc = load_irc()

    channel = st.sidebar.selectbox("Channel", sorted(irc.channel.unique()))
    year = st.sidebar.selectbox("Year", sorted(irc.date.dt.year.unique()), index=2)
    seconds = st.sidebar.slider(
        "Response window (seconds)", min_value=60, max_value=1800, value=600, step=60,
        help="Two people are connected if they spoke within this many seconds of each other",
    )
    threshold = st.sidebar.slider("Hide nodes below this degree", 0, 20, 9)

    chat = irc[(irc.channel == channel) & (irc.date.dt.year == year)]
    if chat.empty:
        st.warning(f"No messages in {channel} during {year}.")
        return

    config = Config(time_col="timestamp", node_col="author", seconds=seconds, datafile=None)
    builder = GraphBuilder(config)
    nearby = builder.build(chat, edge_seconds=seconds)
    named = declared_graph(chat)

    nearby_edges = {frozenset(edge) for edge in nearby.edges()}
    named_edges = {frozenset(edge) for edge in named.edges()}

    left, middle, right = st.columns(3)
    left.metric("Messages", f"{len(chat):,}")
    middle.metric("Edges — spoke nearby", f"{len(nearby_edges):,}")
    right.metric(
        "Edges — addressed by name",
        f"{len(named_edges):,}",
        f"{len(nearby_edges - named_edges):,} nearby edges nobody declared",
        delta_color="off",
    )

    rule = st.radio("Draw the graph from", ["addressed by name", "spoke nearby"])
    graph = named if rule == "addressed by name" else nearby

    viz = GraphVisualizer()
    core = viz.filter_connections(graph, threshold=threshold)
    if core.number_of_nodes() < 2:
        st.warning("That degree threshold leaves nothing to draw. Lower it.")
        return

    positions = builder.calculate_layout(core, name="Spring Layout", scale=2.0)
    figure = viz(core, builder.node_colors(core), positions,
                 title=f"{channel}, {year} — {rule}",
                 node_scale=0.4, edge_scale=0.5, node_threshold=threshold)
    st.plotly_chart(figure, use_container_width=True)

    # `BarPlot` comes from scripts/plots.py, written in lesson 2 and untouched since.
    degrees = pd.Series(dict(core.degree()), name="degree")
    top = degrees.nlargest(12).rename_axis("author").reset_index()
    settings = PlotSettings(
        figsize=(7, 5),
        title=f"Most connected, {rule}",
        xlabel="neighbours",
        ylabel="",
    )
    fig, _ = BarPlot(settings).plot(data=top, x="degree", y="author", color="#cccccc")
    st.pyplot(fig)


if __name__ == "__main__":
    main()
