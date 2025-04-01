import datetime
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, List, Optional, Tuple, TypeAlias

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from loguru import logger
from pandas import Timestamp
from plotly.subplots import make_subplots

from wa_analyzer.filehandler import FileHandler
from wa_analyzer.settings import DAY, HOUR, NetworkAnalysisConfig

GraphWindows: TypeAlias = Tuple[Timestamp, nx.Graph]


@dataclass
class Config:
    time_col: str
    node_col: str
    seconds: int
    node_scale: float
    edge_scale: float
    datafile: Path


class GraphAnalyzer:
    """analyzes a pd.DataFrame, and returns nodes/edges"""

    def __init__(self, config: Config):
        self.time_col = config.time_col
        self.node_col = config.node_col

    def edges(self, df: pd.DataFrame, seconds=30) -> dict[tuple[str, str], int]:
        df = df.sort_values(self.time_col).reset_index(drop=True)
        timestamps = df[self.time_col]
        authors = df[self.node_col].values
        window_size = timedelta(seconds=seconds)  # Adjust as needed

        # Initialize a dictionary to store edge weights
        edges = defaultdict(int)

        # Use sliding window approach
        left_idx = 0
        for right_idx in range(len(timestamps)):
            current_ts = timestamps[right_idx]
            current_author = authors[right_idx]
            window_start = current_ts - window_size

            # Move left pointer forward until we're within the window
            while left_idx < right_idx and timestamps[left_idx] < window_start:
                left_idx += 1

            # All authors from left_idx to right_idx-1 are in the window
            # (excluding the current author at right_idx)
            for i in range(left_idx, right_idx):
                window_author = authors[i]
                if window_author != current_author:  # Don't create self-loops
                    edges[(current_author, window_author)] += 1
        return edges

    def nodes(self, df: pd.DataFrame) -> list[str]:
        return list(df[self.node_col].unique())

    def time_windows(
        self, df: pd.DataFrame, window: int, overlap: int
    ) -> list[GraphWindows]:
        if overlap >= window:
            raise ValueError("Time overlap must be smaller than time window")
        start_time = df[self.time_col].min()
        end_time = df[self.time_col].max()

        # Convert window and overlap to timedelta
        window_size = datetime.timedelta(days=window)
        window_overlap = datetime.timedelta(days=overlap)
        step_size = window_size - window_overlap

        if step_size < datetime.timedelta(days=1):
            step_size = datetime.timedelta(days=1)

        current_start = start_time
        time_windows = []
        window_count = 0

        while current_start < end_time:
            current_end = current_start + window_size
            time_windows.append((current_start, current_end))
            current_start = current_end - window_overlap
            window_count += 1

        logger.info(
            f"Created {len(time_windows)} time windows (days: {window_size.days}, overlap: {window_overlap.days})"
        )
        return time_windows


class GraphBuilder:
    """Uses the analyzer to create nodes/edges from
    a pd.DataFrame, and turn the nodes/edges into a nx.Graph

    """

    def __init__(self, config: Config):
        self.node_col = config.node_col
        self.time_col = config.time_col
        self.layout_algorithms = {
            "Spring Layout": nx.spring_layout,
            "Kamada-Kawai": nx.kamada_kawai_layout,
            "Circular Layout": nx.circular_layout,
            "Spectral Layout": nx.spectral_layout,
        }
        self.analyzer = GraphAnalyzer(config)

    def build(self, df: pd.DataFrame, edge_seconds: int) -> nx.Graph:
        """
        seconds (int) : the amount of seconds that will interpret a reaction
        withing that timeframe as adding an edge
        """
        G = nx.Graph()
        nodes = self.analyzer.nodes(df)
        edges = self.analyzer.edges(df, seconds=edge_seconds)
        G = self.add_nodes(G, nodes)
        G = self.add_edges(G, edges)
        return G

    @staticmethod
    def add_nodes(G: nx.Graph, authors: list[str]) -> nx.Graph:
        for author in authors:
            G.add_node(author)
        return G

    @staticmethod
    def node_colors(G: nx.Graph, palette_name="husl") -> dict:
        nodes = list(G.nodes())
        n = len(nodes)
        colors = sns.color_palette(palette_name, n)
        return {node: colors[i] for i, node in enumerate(nodes)}

    @staticmethod
    def add_edges(G: nx.Graph, edges: dict[tuple[str, str], int]) -> nx.Graph:
        for (u, v), count in edges.items():
            G.add_edge(u, v, weight=count)
        return G

    def graph_windows(
        self,
        df: pd.DataFrame,
        window_days: int,
        overlap_days: int,
        edge_seconds: int,
    ) -> list[GraphWindows]:
        graphs_by_window = []

        time_windows = self.analyzer.time_windows(
            df, window=window_days, overlap=overlap_days
        )

        for window_start, window_end in time_windows:
            window_data = df[
                (df[self.time_col] >= window_start) & (df[self.time_col] <= window_end)
            ]

            if not isinstance(window_data, pd.DataFrame):
                raise TypeError()

            G = self.build(df=window_data, edge_seconds=edge_seconds)

            graphs_by_window.append((window_start, G))

        return graphs_by_window

    def calculate_layout(
        self,
        G: nx.Graph,
        name: str,
        scale: float = 1.0,
        k: Optional[float] = None,
        iter: int = 50,
    ):
        layout_kwargs = self._layout_args(name, scale, k, iter)
        layout_func = self.layout_algorithms[name]
        connected_components = list(nx.connected_components(G))
        main_component = max(connected_components, key=len)
        other_components = [
            comp for comp in connected_components if comp != main_component
        ]
        main_pos = layout_func(G.subgraph(main_component), **layout_kwargs)

        pos = main_pos.copy()
        if other_components:
            self._pos_others(G, pos, main_pos, other_components)
        return pos

    def _pos_others(self, G, pos, main_pos, other_components):
        # Position other components around the main one

        # Calculate bounding box of main component
        main_x = [pos[0] for pos in main_pos.values()]
        main_y = [pos[1] for pos in main_pos.values()]
        x_min, x_max = min(main_x), max(main_x)
        y_min, y_max = min(main_y), max(main_y)
        width = x_max - x_min
        height = y_max - y_min

        # Position other components in a circle around main component
        radius = max(width, height) * 1.5
        angle_step = 2 * np.pi / len(other_components)

        for i, component in enumerate(other_components):
            angle = i * angle_step
            center_x = radius * np.cos(angle)
            center_y = radius * np.sin(angle)

            # Layout the component
            component_pos = nx.spring_layout(
                G.subgraph(component), seed=42, k=0.05, iterations=100, scale=0.5
            )

            # Offset to position around main component
            for node, (x, y) in component_pos.items():
                pos[node] = (x + center_x, y + center_y)
        return pos

    @staticmethod
    def _layout_args(
        name: str, scale: float, k: Optional[float] = None, iter: int = 50
    ):
        layout_kwargs: dict[str, Any] = {"scale": scale}

        # Add algorithm-specific parameters
        if name == "Spring Layout":
            layout_kwargs.update(
                {
                    "seed": 42,
                    "k": k,
                    "iterations": iter,
                }
            )
        return layout_kwargs


class GraphVisualizer:
    @staticmethod
    def node_trace(
        G,
        pos,
        scale,
        node_colors,
        fig: Optional[go.Figure] = None,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ):
        is_subplot = fig is not None
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []

        for node in G.nodes():
            if node not in pos:
                continue
            x, y = pos[node]  # type: ignore
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}<br>Degree: {G.degree(node)}")
            # Scale node size with degree and apply multiplier
            node_size.append(
                1 + 1 * scale * G.degree(node)
            )  # Size proportional to degree but controlled by multiplier
            node_color.append(
                f"rgb({int(255 * node_colors[node][0])},"
                f"{int(255 * node_colors[node][1])},"
                f"{int(255 * node_colors[node][2])})"
            )

        trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=list(G.nodes()),
            # text=[node.split()[0] for node in G.nodes()],  # Show first name only
            textposition="top center",
            hovertext=node_text,
            hoverinfo="text",
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                size=node_size,
                color=node_color,
                line_width=2,
            ),
        )
        if is_subplot:
            fig.add_trace(trace, row=row, col=col)
            return fig
        return trace

    @staticmethod
    def edge_trace(
        G: nx.Graph,
        pos: dict,
        scale: float,
        fig: Optional[go.Figure] = None,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ):
        is_subplot = fig is not None
        edge_traces = []
        for edge in G.edges(data=True):
            source, target = edge[0], edge[1]
            if source not in pos or target not in pos:
                continue
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            weight = edge[2].get("weight", 1)

            # Create interpolated points along the edge
            t = np.linspace(0, 1, 20)
            x_interp = x0 * (1 - t) + x1 * t
            y_interp = y0 * (1 - t) + y1 * t
            width = 1 * scale
            # Create directional hover text for each point
            hover_text = [
                f"<b>{source}</b> - <b>{target}</b><br>Interactions: {weight:.2f}"
                for _ in range(len(t))
            ]

            trace = go.Scatter(
                x=x_interp,
                y=y_interp,
                line=dict(width=width, color="#888"),
                mode="lines",
                marker=dict(size=0.1, color="#888", opacity=0),  # Invisible markers
                hoverinfo="text",
                hovertext=hover_text,
                showlegend=False,
            )
            if is_subplot:
                fig.add_trace(trace, row=row, col=col)
            else:
                edge_traces.append(trace)
        if not is_subplot:
            return edge_traces
        return fig

    @staticmethod
    def filter_connections(G: nx.Graph, threshold: int = 1):
        """remove nodes with degree <= 1"""
        filtered_nodes = [node for node in G.nodes() if G.degree[node] > threshold]
        return G.subgraph(filtered_nodes)

    @staticmethod
    def create_figure(node_trace, edge_trace) -> go.Figure:
        fig = go.Figure(
            data=edge_trace + [node_trace],
            layout=go.Layout(
                clickmode="event+select",
                hovermode="closest",
            ),
        )
        return fig

    def create_windows(
        self,
        pos: dict,
        graph_windows: list[GraphWindows],
        node_colors,
        edge_scale: float,
        node_scale: float,
        window_titles: Optional[List[str]] = None,
    ) -> go.Figure:
        num_windows = len(graph_windows)
        num_cols = 3
        num_rows = math.ceil(num_windows / num_cols)
        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            horizontal_spacing=0.05,
            vertical_spacing=0.1,
        )
        fig.update_layout(
            width=900,  # width in pixels
            height=900,  # height in pixels
        )
        for i, (timestamp, G) in enumerate(graph_windows):
            row = (i // 3) + 1
            col = (i % 3) + 1
            fig = self.edge_trace(
                G=G, pos=pos, scale=edge_scale, fig=fig, row=row, col=col
            )
            if not isinstance(fig, go.Figure):
                raise TypeError(f"Got type {type(fig)}, but expected go.Figure")
            title = timestamp.strftime("%d-%m-%Y")
            fig.update_xaxes(
                title_text=title,
                row=row,
                col=col,
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            )
            fig.update_yaxes(
                showgrid=False, zeroline=False, showticklabels=False, row=row, col=col
            )

            fig = self.node_trace(
                G=G,
                pos=pos,
                scale=node_scale,
                node_colors=node_colors,
                fig=fig,
                row=row,
                col=col,
            )
            if not isinstance(fig, go.Figure):
                raise TypeError(f"Got type {type(fig)}, but expected go.Figure")
        if not isinstance(fig, go.Figure):
            raise TypeError()
        return fig

    @staticmethod
    def update_layout(fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        return fig

    def __call__(
        self, G, node_colors, pos, title, node_scale, edge_scale, node_threshold
    ) -> go.Figure:
        G_ = G.copy()
        G_ = self.filter_connections(G_, threshold=node_threshold)
        node_trace = self.node_trace(G_, pos, scale=node_scale, node_colors=node_colors)
        edge_trace = self.edge_trace(G_, pos, scale=edge_scale)
        fig = self.create_figure(node_trace, edge_trace)
        fig = self.update_layout(fig, title)
        return fig


class NetworkAnalysis:
    def __init__(self, config: Config):
        self.config = config
        self.filehandler = FileHandler(config)
        self.graphbuilder = GraphBuilder(config)
        self.visualizer = GraphVisualizer()
        self.df = self.filehandler.load(config.datafile)

    def process(
        self,
        title: str,
        layout: str = "Spring Layout",
        cutoff_days: Optional[int] = None,
        seconds: Optional[int] = None,
        node_scale: Optional[float] = None,
        edge_scale: Optional[float] = None,
        node_threshold: int = 0,
    ) -> go.Figure:
        G, _ = self.make_graph(edge_seconds=seconds, cutoff_days=cutoff_days)
        fig = self.viz_graph(
            G,
            layout=layout,
            title=title,
            node_scale=node_scale,
            edge_scale=edge_scale,
            node_threshold=node_threshold,
        )
        return fig

    def windows(
        self,
        cutoff_days: int,
        edge_seconds: int,
        window_days: int,
        overlap_days: int,
        layout: str = "Spring Layout",
        node_scale: Optional[float] = None,
        edge_scale: Optional[float] = None,
        node_threshold: int = 0,
    ) -> go.Figure:
        G, df = self.make_graph(edge_seconds=edge_seconds, cutoff_days=cutoff_days)
        G = self.visualizer.filter_connections(G, threshold=node_threshold)
        if not node_scale:
            node_scale = self.config.node_scale
        if not edge_scale:
            edge_scale = self.config.edge_scale
        pos = self.graphbuilder.calculate_layout(G, name=layout)
        node_colors = self.graphbuilder.node_colors(G)
        graph_windows = self.graphbuilder.graph_windows(
            df=df,
            window_days=window_days,
            overlap_days=overlap_days,
            edge_seconds=edge_seconds,
        )
        fig = self.visualizer.create_windows(
            pos=pos,
            graph_windows=graph_windows,
            node_colors=node_colors,
            edge_scale=edge_scale,
            node_scale=node_scale,
        )
        return fig

    def make_graph(
        self, edge_seconds: Optional[int] = None, cutoff_days: Optional[int] = None
    ) -> tuple[nx.Graph, pd.DataFrame]:
        df = self.get_df(cutoff_days=cutoff_days)
        if not edge_seconds:
            edge_seconds = self.config.seconds
        G = self.graphbuilder.build(df, edge_seconds=edge_seconds)
        return G, df

    def viz_graph(
        self,
        G: nx.Graph,
        layout: str = "Spring Layout",
        title: str = "Graph",
        node_scale: Optional[float] = None,
        edge_scale: Optional[float] = None,
        node_threshold: int = 0,
    ) -> go.Figure:
        if not node_scale:
            node_scale = self.config.node_scale
        if not edge_scale:
            edge_scale = self.config.edge_scale
        pos = self.graphbuilder.calculate_layout(G, name=layout)
        node_colors = self.graphbuilder.node_colors(G)
        fig = self.visualizer(
            G,
            node_colors,
            pos,
            title=title,
            node_scale=node_scale,
            edge_scale=edge_scale,
            node_threshold=node_threshold,
        )
        return fig

    def get_df(self, cutoff_days: Optional[int] = None) -> pd.DataFrame:
        """
        Get the dataframe.
        """
        df = self.df.copy()
        if cutoff_days:
            df = self._cutoff(df, cutoff_days)
        return df

    def _cutoff(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        """
        Cutoff the dataframe to the last `days` days.
        """
        cutoff = df[self.config.time_col].max() - pd.Timedelta(days=days)
        result = df[df[self.config.time_col] > cutoff]
        if not isinstance(result, pd.DataFrame):
            raise TypeError()
        return result


class WhatsAppNetworkAnalyzer:
    """Analyze WhatsApp chat data as a network of users."""

    def __init__(self, config: NetworkAnalysisConfig):
        """Initialize the network analyzer with configuration."""
        self.config = config
        self.data = None
        self.graph = None
        self.pos = None
        self.time_windows: list = []
        self.graphs_by_window: list = []
        self.node_colors: dict = {}

        # Layout settings
        self.layout_algorithms = {
            "Spring Layout": nx.spring_layout,
            "Kamada-Kawai": nx.kamada_kawai_layout,
            "Circular Layout": nx.circular_layout,
            "Spectral Layout": nx.spectral_layout,
        }
        self.selected_layout = "Spring Layout"
        self.default_node_spacing = 0.15
        self.layout_iterations = 500
        self.layout_scale = 1.5
        self.node_size_multiplier = 0.5  # Default multiplier for node size

    def load_data(self, filepath: Path) -> None:
        """Load preprocessed WhatsApp data."""
        logger.info(f"Loading data from {filepath}")
        self.data = pd.read_csv(filepath)

        # Convert timestamp to datetime and ensure proper timezone handling
        try:
            self.data["timestamp"] = pd.to_datetime(  # type: ignore
                self.data["timestamp"],
                utc=True,  # type: ignore
            ).dt.tz_convert("UTC")
        except Exception as e:
            logger.error(f"Error converting timestamps: {e}")
            logger.info("Attempting alternative timestamp conversion...")

        # Sort by timestamp and reset index
        self.data = self.data.sort_values("timestamp").reset_index(drop=True)  # type: ignore

        # Verify timestamp conversion
        if not pd.api.types.is_datetime64_any_dtype(self.data["timestamp"]):  # type: ignore
            raise ValueError("Timestamp conversion failed - check input data format")  # type: ignore

        logger.info(
            f"Loaded {len(self.data)} messages from {len(self.data['author'].unique())} users"  # type: ignore
        )

    def create_full_graph(self) -> nx.Graph:
        """Create a graph of all interactions."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        G = nx.Graph()

        # Add all users as nodes
        for user in self.data["author"].unique():
            G.add_node(user)

        # Assign random colors to nodes
        self.node_colors = {
            user: np.random.rand(
                3,
            )
            for user in G.nodes()
        }

        # Process all messages to create edges
        self._process_interactions(G, self.data)

        self.graph = G

        # Separate nodes into connected components
        connected_components = list(nx.connected_components(G))
        main_component = max(connected_components, key=len)
        other_components = [
            comp for comp in connected_components if comp != main_component
        ]

        # Generate layout for main component using selected algorithm
        layout_func = self.layout_algorithms[self.selected_layout]

        # Common layout parameters
        layout_kwargs = {"scale": self.layout_scale}

        # Add algorithm-specific parameters
        if self.selected_layout == "Spring Layout":
            layout_kwargs.update(
                {
                    "seed": 42,
                    "k": self.default_node_spacing,
                    "iterations": self.layout_iterations,
                }
            )
        elif self.selected_layout == "Kamada-Kawai":
            layout_kwargs.update({"weight": "weight", "scale": self.layout_scale})
        elif self.selected_layout == "Circular Layout":
            layout_kwargs.update({"scale": self.layout_scale})
        elif self.selected_layout == "Spectral Layout":
            layout_kwargs.update({"weight": "weight", "scale": self.layout_scale})

        main_pos = layout_func(G.subgraph(main_component), **layout_kwargs)

        # Position other components around the main one
        self.pos = main_pos.copy()
        if other_components:
            # Calculate bounding box of main component
            main_x = [pos[0] for pos in main_pos.values()]
            main_y = [pos[1] for pos in main_pos.values()]
            x_min, x_max = min(main_x), max(main_x)
            y_min, y_max = min(main_y), max(main_y)
            width = x_max - x_min
            height = y_max - y_min

            # Position other components in a circle around main component
            radius = max(width, height) * 1.5
            angle_step = 2 * np.pi / len(other_components)

            for i, component in enumerate(other_components):
                angle = i * angle_step
                center_x = radius * np.cos(angle)
                center_y = radius * np.sin(angle)

                # Layout the component
                component_pos = nx.spring_layout(
                    G.subgraph(component), seed=42, k=0.05, iterations=100, scale=0.5
                )

                # Offset to position around main component
                for node, (x, y) in component_pos.items():
                    self.pos[node] = (x + center_x, y + center_y)

        return G

    def create_time_window_graphs(self) -> List[Tuple[datetime.datetime, nx.Graph]]:
        """Create graphs for overlapping time windows."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data["timestamp"]):
            self.data["timestamp"] = pd.to_datetime(self.data["timestamp"], utc=True)

        # Calculate time windows
        start_time = self.data["timestamp"].min()
        end_time = self.data["timestamp"].max()
        total_duration = end_time - start_time

        # Convert window and overlap to timedelta
        window_size = datetime.timedelta(seconds=self.config.time_window)
        overlap = datetime.timedelta(seconds=self.config.time_overlap)

        # Validate window size
        if window_size.total_seconds() <= 0:
            raise ValueError("Time window must be greater than 0 seconds")

        if overlap >= window_size:
            raise ValueError("Time overlap must be smaller than time window")

        # Calculate number of windows that would be created
        step_size = window_size - overlap
        if step_size.total_seconds() <= 0:
            step_size = datetime.timedelta(days=1)  # Minimum step size of 1 day

        # Calculate number of windows
        num_windows = total_duration / step_size

        # Only show warning if we're actually creating too many windows
        max_windows = 100
        if num_windows > max_windows:
            st.warning(
                f"Too many time windows ({num_windows:.0f}) - adjusting settings to create max {max_windows} windows"
            )
            # Adjust window size to create max_windows windows
            new_step_size = total_duration / max_windows
            new_window_size = new_step_size + overlap

            # Update window size and config
            window_size = new_window_size
            self.config.time_window = window_size.total_seconds()

        # Create time windows with overlap
        current_start = start_time
        self.time_windows = []
        window_count = 0

        while current_start < end_time and window_count < max_windows:
            current_end = current_start + window_size
            self.time_windows.append((current_start, current_end))
            current_start = current_end - overlap
            window_count += 1

        logger.info(
            f"Created {len(self.time_windows)} time windows (window size: {window_size}, overlap: {overlap})"
        )

        # Create a graph for each time window
        self.graphs_by_window = []

        for i, (window_start, window_end) in enumerate(self.time_windows):
            window_data = self.data[
                (self.data["timestamp"] >= window_start)
                & (self.data["timestamp"] <= window_end)
            ]

            G = nx.Graph()

            # Add all users as nodes (from the full dataset to maintain consistency)
            for user in self.data["author"].unique():
                G.add_node(user)

            # Process messages in this time window
            self._process_interactions(G, window_data)

            self.graphs_by_window.append((window_start, G))

        return self.graphs_by_window

    def _process_interactions(self, G: nx.Graph, data: pd.DataFrame) -> None:
        """Process interactions to create edges between users."""
        # Track interactions within the response window
        interactions: dict = defaultdict(int)

        # Ensure timestamp is datetime and group by timestamp to process in order
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)

        for timestamp, group in data.groupby("timestamp"):
            # Ensure timestamp is a datetime object
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp, utc=True)
            # Get authors in this timestamp group
            current_authors = group["author"].unique()

            # Find recent messages within response window
            response_cutoff = timestamp - datetime.timedelta(
                seconds=self.config.response_window
            )
            recent_messages = data[
                (data["timestamp"] >= response_cutoff) & (data["timestamp"] < timestamp)
            ]

            # Get unique recent authors
            recent_authors = set(recent_messages["author"].unique())

            # Create edges between current authors and recent authors
            for current_author in current_authors:
                for recent_author in recent_authors:
                    if current_author != recent_author:
                        # Increment the interaction count
                        interactions[(current_author, recent_author)] += 1
                        interactions[(recent_author, current_author)] += 1

        # Add edges with weights based on interaction counts
        for (user1, user2), count in interactions.items():
            weight = count * self.config.edge_weight_multiplier
            if weight >= self.config.min_edge_weight:
                G.add_edge(user1, user2, weight=weight)

    def visualize_graph(
        self,
        G: Optional[nx.Graph] = None,
        title: str = "WhatsApp Interaction Network",
        default_k: float = 0.15,
        default_size: float = 0.5,
        force_layout: bool = False,
        filter_single_connections: bool = False,
        edge_hover_info: bool = True,
        edge_highlight_color: str = "rgba(255, 0, 0, 0.8)",
    ) -> go.Figure:
        """Visualize the network graph with optional layout recalculation."""
        if G is None:
            G = self.graph

        if G is None:
            raise ValueError("No graph available. Create a graph first.")

        # Recalculate layout if forced or if positions don't exist
        if force_layout or self.pos is None:
            self._calculate_layout(G)
            st.session_state.layout_calculated = True

        # Filter nodes based on degree
        if filter_single_connections:
            # Remove nodes with degree <= 1
            filtered_nodes = [node for node in G.nodes() if G.degree(node) > 1]
            st.info(
                f"Filtered out {len(G.nodes()) - len(filtered_nodes)} nodes with only one connection"
            )
        else:
            # Remove completely isolated nodes
            filtered_nodes = [node for node in G.nodes() if G.degree(node) > 0]

        G_filtered = G.subgraph(filtered_nodes)

        # Create edge trace
        edge_trace = []
        for edge in G_filtered.edges(data=True):
            x0, y0 = self.pos[edge[0]]  # type: ignore
            x1, y1 = self.pos[edge[1]]  # type: ignore
            weight = edge[2].get("weight", 1)
            # Scale width based on weight
            width = 1 + 2 * np.log1p(weight)
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=width, color="#888"),
                    mode="lines",
                    customdata=np.array(
                        [[edge[0], edge[1], weight]]
                    ),  # Store edge info for hover
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> ↔ <b>%{customdata[1]}</b><br>"
                        + "Interactions: %{customdata[2]:.2f}<extra></extra>"
                    ),
                )
            )

        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []

        for node in G_filtered.nodes():
            x, y = self.pos[node]  # type: ignore
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}<br>Degree: {G_filtered.degree(node)}")
            # Scale node size with degree and apply multiplier
            node_size.append(
                10 + 5 * self.node_size_multiplier * G_filtered.degree(node)
            )  # Size proportional to degree but controlled by multiplier
            node_color.append(
                f"rgb({int(255 * self.node_colors[node][0])},"
                f"{int(255 * self.node_colors[node][1])},"
                f"{int(255 * self.node_colors[node][2])})"
            )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[
                node.split()[0] for node in G_filtered.nodes()
            ],  # Show first name only
            textposition="top center",
            hovertext=node_text,
            hoverinfo="text",
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                size=node_size,
                color=node_color,
                line_width=2,
            ),
        )

        # Create figure with edge highlighting
        fig = go.Figure(
            data=edge_trace + [node_trace],
            layout=go.Layout(clickmode="event+select", hovermode="closest"),
        )

        # Add edge highlighting on click
        fig.update_traces(
            selected=dict(marker=dict(color=edge_highlight_color, size=12))
        )

        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        # Add Streamlit controls
        col1, col2 = st.columns(2)
        with col1:
            k = st.slider(  # noqa: F841
                "Node spacing (k)",
                min_value=0.05,
                max_value=1.0,
                value=default_k,
                step=0.05,
                help="Optimal distance between nodes",
            )
        with col2:
            size_factor = st.slider(  # noqa: F841
                "Node size multiplier",
                min_value=0.1,
                max_value=2.0,
                value=default_size,
                step=0.1,
                help="Scale factor for node sizes",
            )

        def update_layout(k, size_factor):
            """Update the layout with new spacing and size parameters."""
            # Update layout with new spacing but maintain the current layout algorithm
            layout_func = self.layout_algorithms.get(
                self.selected_layout, nx.spring_layout
            )

            # Set parameters based on layout algorithm
            if self.selected_layout == "Spring Layout":
                self.pos = layout_func(G_filtered, k=k, iterations=500, seed=42)
            elif self.selected_layout == "Kamada-Kawai":
                self.pos = layout_func(
                    G_filtered, weight="weight", scale=self.layout_scale
                )
            elif self.selected_layout == "Circular Layout":
                self.pos = layout_func(G_filtered, scale=self.layout_scale)
            elif self.selected_layout == "Spectral Layout":
                self.pos = layout_func(
                    G_filtered, weight="weight", scale=self.layout_scale
                )
            else:
                # Default to spring layout
                self.pos = nx.spring_layout(G_filtered, k=k, iterations=500, seed=42)

            # Update node positions and sizes
            fig.update_traces(
                x=[self.pos[node][0] for node in G_filtered.nodes()],
                y=[self.pos[node][1] for node in G_filtered.nodes()],
                marker=dict(
                    size=[
                        10
                        + 5
                        * size_factor
                        * self.node_size_multiplier
                        * G_filtered.degree(node)
                        for node in G_filtered.nodes()
                    ]
                ),
                selector={"mode": "markers+text"},
            )

            # Update edge positions
            edge_index = 0
            for edge in G_filtered.edges(data=True):
                if edge_index >= len(fig.data) - 1:  # Last trace is for nodes
                    break
                x0, y0 = self.pos[edge[0]]
                x1, y1 = self.pos[edge[1]]
                weight = edge[2].get("weight", 1)
                width = 1 + 2 * np.log1p(weight)
                fig.data[edge_index].x = [x0, x1, None]
                fig.data[edge_index].y = [y0, y1, None]
                fig.data[edge_index].line.width = width
                edge_index += 1

        # Show the figure in Streamlit
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"network_graph_{title.replace(' ', '_')}",
        )

    def visualize_time_series(
        self,
        output_path: Optional[Path] = None,
        max_windows: int = 9,
        window_titles: Optional[List[str]] = None,
    ) -> None:
        """Visualize the network evolution over time as a static grid of timeframes.

        Args:
            output_path: Optional path to save animation (not implemented yet)
            max_windows: Maximum number of windows to display (default: 9)
        """
        if not self.graphs_by_window or not self.pos:
            raise ValueError(
                "No time window graphs available. Create time window graphs first."
            )

        # Get the last N timeframes (up to max_windows)
        last_windows = self.graphs_by_window[-max_windows:]
        num_windows = len(last_windows)

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=3,
            subplot_titles=(
                window_titles[:num_windows]
                if window_titles
                else [f"Window {i + 1}" for i in range(num_windows)]
            ),
            horizontal_spacing=0.05,
            vertical_spacing=0.1,
        )

        # Plot each timeframe
        for i, (timestamp, G) in enumerate(last_windows):
            row = (i // 3) + 1
            col = (i % 3) + 1

            # Create edge traces
            for edge in G.edges(data=True):
                # Skip edges where nodes don't have positions
                if edge[0] not in self.pos or edge[1] not in self.pos:
                    continue

                x0, y0 = self.pos[edge[0]]
                x1, y1 = self.pos[edge[1]]
                weight = edge[2].get("weight", 1)
                # Scale width based on weight
                width = 1 + 2 * np.log1p(weight)
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        line=dict(width=width, color="#888"),
                        mode="lines",
                        customdata=np.array([[edge[0], edge[1], weight]]),
                        hovertemplate=(
                            "<b>%{customdata[0]}</b> ↔ <b>%{customdata[1]}</b><br>"
                            + "Interactions: %{customdata[2]:.2f}<extra></extra>"
                        ),
                        selectedpoints=[],  # Enable selection
                    ),
                    row=row,
                    col=col,
                )

            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_color = []

            for node in G.nodes():
                # Skip nodes that don't have positions
                if node not in self.pos:
                    continue

                x, y = self.pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"{node}<br>Degree: {G.degree(node)}")
                # Scale node size with degree and apply multiplier
                node_size.append(10 + 5 * self.node_size_multiplier * G.degree(node))
                node_color.append(
                    f"rgb({int(255 * self.node_colors[node][0])},"
                    f"{int(255 * self.node_colors[node][1])},"
                    f"{int(255 * self.node_colors[node][2])})"
                )

            fig.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    text=[node.split()[0] for node in G.nodes()],
                    textposition="top center",
                    hovertext=node_text,
                    hoverinfo="text",
                    marker=dict(
                        showscale=True,
                        colorscale="YlGnBu",
                        size=node_size,
                        color=node_color,
                        line_width=2,
                    ),
                ),
                row=row,
                col=col,
            )
            # Add unique key to each subplot
            fig.update_xaxes(title_text=f"Window {i + 1}", row=row, col=col)

            # Format subplot
            fig.update_xaxes(
                showgrid=False, zeroline=False, showticklabels=False, row=row, col=col
            )
            fig.update_yaxes(
                showgrid=False, zeroline=False, showticklabels=False, row=row, col=col
            )

        # Update layout
        fig.update_layout(
            title_text=f"WhatsApp Network Evolution - Last {num_windows} Time Windows",
            showlegend=False,
            height=900,
            margin=dict(b=20, l=20, r=20, t=100),
        )

        # Show in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        return fig

    def _calculate_layout(self, G: nx.Graph) -> None:
        """Calculate node positions using the selected layout algorithm."""
        # Log the layout calculation
        st.write(f"Calculating layout using {self.selected_layout}...")

        # Separate nodes into connected components
        connected_components = list(nx.connected_components(G))
        if not connected_components:
            st.warning("Graph has no connected components")
            return

        main_component = max(connected_components, key=len)
        other_components = [
            comp for comp in connected_components if comp != main_component
        ]

        # Generate layout for main component using selected algorithm
        layout_func = self.layout_algorithms[self.selected_layout]

        # Common layout parameters
        layout_kwargs = {"scale": self.layout_scale}

        # Add algorithm-specific parameters
        if self.selected_layout == "Spring Layout":
            layout_kwargs.update(
                {
                    "seed": 42,
                    "k": self.default_node_spacing,
                    "iterations": self.layout_iterations,
                }
            )
        elif self.selected_layout == "Kamada-Kawai":
            layout_kwargs.update({"weight": "weight", "scale": self.layout_scale})  # type: ignore
        elif self.selected_layout == "Circular Layout":
            layout_kwargs.update({"scale": self.layout_scale})
        elif self.selected_layout == "Spectral Layout":
            layout_kwargs.update({"weight": "weight", "scale": self.layout_scale})  # type: ignore

        # Create subgraph of main component
        main_subgraph = G.subgraph(main_component)

        try:
            main_pos = layout_func(main_subgraph, **layout_kwargs)
        except Exception as e:
            st.error(f"Error calculating layout: {e}")
            # Fall back to spring layout
            main_pos = nx.spring_layout(
                main_subgraph,
                seed=42,
                k=self.default_node_spacing,
                iterations=self.layout_iterations,
            )

        # Position other components around the main one
        self.pos = main_pos.copy()
        if other_components:
            # Calculate bounding box of main component
            main_x = [pos[0] for pos in main_pos.values()]
            main_y = [pos[1] for pos in main_pos.values()]
            x_min, x_max = min(main_x), max(main_x)
            y_min, y_max = min(main_y), max(main_y)
            width = x_max - x_min
            height = y_max - y_min

            # Position other components in a circle around main component
            radius = max(width, height) * 1.5
            angle_step = 2 * np.pi / len(other_components)

            for i, component in enumerate(other_components):
                angle = i * angle_step
                center_x = radius * np.cos(angle)
                center_y = radius * np.sin(angle)

                # Layout the component
                component_pos = nx.spring_layout(
                    G.subgraph(component), seed=42, k=0.05, iterations=100, scale=0.5
                )

                # Offset to position around main component
                for node, (x, y) in component_pos.items():
                    self.pos[node] = (x + center_x, y + center_y)  # type: ignore

    def export_graph_data(self, output_dir: Optional[Path] = None) -> None:
        """Export graph data in formats compatible with other visualization tools."""
        if self.graph is None:
            raise ValueError("No graph available. Create a graph first.")

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Export node data
            node_data = []
            for node in self.graph.nodes():
                degree = self.graph.degree(node)
                node_data.append(
                    {
                        "id": node,
                        "label": node,
                        "degree": degree,
                        "color": ",".join(
                            map(str, self.node_colors.get(node, (0.5, 0.5, 0.5)))
                        ),
                    }
                )

            node_df = pd.DataFrame(node_data)
            node_df.to_csv(output_dir / "nodes.csv", index=False)
            logger.info(f"Node data exported to {output_dir / 'nodes.csv'}")

            # Export edge data
            edge_data = []
            for u, v, data in self.graph.edges(data=True):
                edge_data.append(
                    {"source": u, "target": v, "weight": data.get("weight", 1)}
                )

            edge_df = pd.DataFrame(edge_data)
            edge_df.to_csv(output_dir / "edges.csv", index=False)
            logger.info(f"Edge data exported to {output_dir / 'edges.csv'}")

            # Export as GraphML for use in tools like Gephi
            nx.write_graphml(self.graph, output_dir / "network.graphml")
            logger.info(f"GraphML exported to {output_dir / 'network.graphml'}")

    def export_graph_metrics(self, output_path: Optional[Path] = None) -> pd.DataFrame:
        """Export network metrics for all time windows."""
        if not self.graphs_by_window:
            raise ValueError(
                "No time window graphs available. Create time window graphs first."
            )

        metrics = []

        # Calculate metrics for each time window
        for timestamp, G in self.graphs_by_window:
            window_metrics = {
                "timestamp": timestamp,
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "density": nx.density(G),
                "avg_clustering": nx.average_clustering(G),
            }

            # Add centrality metrics for each node
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)

            for node in G.nodes():
                window_metrics[f"degree_{node}"] = degree_centrality.get(node, 0)
                window_metrics[f"betweenness_{node}"] = betweenness_centrality.get(
                    node, 0
                )

            metrics.append(window_metrics)

        # Create DataFrame
        metrics_df = pd.DataFrame(metrics)

        if output_path:
            metrics_df.to_csv(output_path, index=False)
            logger.info(f"Metrics exported to {output_path}")

        return metrics_df


def analyze_whatsapp_network(
    data_path: Path,
    response_window: int = HOUR,  # 1 hour in seconds
    time_window: int = DAY * 30 * 2,  # 2 months in seconds
    time_overlap: int = DAY * 10,  # 10 days in seconds
    edge_weight_multiplier: float = 1.0,
    min_edge_weight: float = 0.5,
    output_dir: Optional[Path] = None,
) -> WhatsAppNetworkAnalyzer:
    """Analyze WhatsApp chat data as a network and generate visualizations."""
    # Create configuration
    config = NetworkAnalysisConfig(
        response_window=response_window,
        time_window=time_window,
        time_overlap=time_overlap,
        edge_weight_multiplier=edge_weight_multiplier,
        min_edge_weight=min_edge_weight,
    )
    logger.info(f"Using configuration: {config}")

    # Initialize analyzer
    analyzer = WhatsAppNetworkAnalyzer(config)

    # Load data
    analyzer.load_data(data_path)

    # Create full graph
    analyzer.create_full_graph()

    # Create time window graphs
    analyzer.create_time_window_graphs()

    # Create output directory if specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full graph visualization
        fig = analyzer.visualize_graph(title="Complete WhatsApp Interaction Network")
        if fig:
            fig.write_html(output_dir / "full_network.html")
            logger.info(
                f"Saved network visualization to {output_dir / 'full_network.html'}"
            )

        # Save time series visualization
        analyzer.visualize_time_series()
        # TODO: Implement animation export

        # Export data and metrics
        analyzer.export_graph_data(output_dir)
        analyzer.export_graph_metrics(output_dir / "network_metrics.csv")

    return analyzer
