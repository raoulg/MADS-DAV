import datetime
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypeAlias

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from loguru import logger
from pandas import Timestamp
from plotly.subplots import make_subplots

from wa_analyzer.filehandler import FileHandler

GraphWindows: TypeAlias = Tuple[Timestamp, nx.Graph]


@dataclass
class Config:
    time_col: str
    node_col: str
    seconds: int
    datafile: Path


class GraphAnalyzer:
    """analyzes a pd.DataFrame, and returns nodes/edges"""

    def __init__(self, config: Config):
        self.time_col = config.time_col
        self.node_col = config.node_col

    def edges(self, df: pd.DataFrame, seconds=30) -> dict[tuple[str, str], int]:
        """from a dataframe, build the edges. The edges are tuples, we count the number of edges.
        The data is stored as a dict with the tuple as key, and the count as value

        df: pd.DataFrame
        seconds: int : the amount of seconds that will interpret a reaction within that timeframe as an edge
        """

        df = df.sort_values(self.time_col).reset_index(drop=True)
        timestamps = df[self.time_col]
        authors = df[self.node_col].values
        window_size = timedelta(seconds=seconds)  # Adjust as needed

        # Initialize a dictionary to store edge weights
        edges: defaultdict[tuple[str, str], int] = defaultdict(int)

        # Use sliding window approach for efficiency
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
        """returns the unique nodes in the dataframe from a column"""
        return list(df[self.node_col].unique())

    def time_windows(
        self, df: pd.DataFrame, window: int, overlap: int
    ) -> list[tuple[Timestamp, Timestamp]]:
        """
        Create overlaping time windows for the given dataframe.
        window: int : the size of the window in days
        overlap: int : the size of the overlap in days
        returns: a list of tuples with the start and end time of the windows
        """
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
        df: pd.DataFrame : the dataframe to build the graph from
        edge_seconds (int) : the amount of seconds that will interpret a reaction
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
            # get the actual data from start and end timestamp
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
            logger.info(f"Positioning {len(other_components)} other components")
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
            if fig is not None:
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
                if fig is not None:
                    if fig is not None:
                        fig.add_trace(trace, row=row, col=col)
            else:
                edge_traces.append(trace)
        if not is_subplot:
            return edge_traces
        return fig

    @staticmethod
    def filter_connections(G: nx.Graph, threshold: int = 1):
        """remove nodes with degree <= 1"""
        logger.info(f"Filtering nodes with degree <= {threshold}")
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
        mainG: nx.Graph,
        pos: dict,
        graph_windows: list[GraphWindows],
        node_colors,
        edge_scale: float,
        node_scale: float,
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
                G=mainG,
                pos=pos,
                scale=node_scale,
                node_colors=node_colors,
                fig=fig,
                row=row,
                col=col,
            )
            # fig.update_traces(marker=dict(showscale=False))
            if not isinstance(fig, go.Figure):
                raise TypeError(f"Got type {type(fig)}, but expected go.Figure")
        if not isinstance(fig, go.Figure):
            raise TypeError()
        self.update_layout(fig, title="WhatsApp Network")
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
            updatemenus=[],
        )
        fig.update_traces(marker=dict(showscale=False))
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
        node_scale: float = 1.0,
        edge_scale: float = 1.0,
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

    def make_graph(
        self, edge_seconds: Optional[int] = None, cutoff_days: Optional[int] = None
    ) -> tuple[nx.Graph, pd.DataFrame]:
        df = self.get_df(cutoff_days=cutoff_days)
        if not edge_seconds:
            edge_seconds = self.config.seconds
        G = self.graphbuilder.build(df, edge_seconds=edge_seconds)
        return G, df

    def windows(
        self,
        cutoff_days: Optional[int],
        edge_seconds: int,
        window_days: int,
        overlap_days: int,
        layout: str = "Spring Layout",
        node_scale: float = 1.0,
        edge_scale: float = 1.0,
        node_threshold: int = 0,
    ) -> go.Figure:
        G, df = self.make_graph(edge_seconds=edge_seconds, cutoff_days=cutoff_days)
        G = self.visualizer.filter_connections(G, threshold=node_threshold)

        pos = self.graphbuilder.calculate_layout(G, name=layout)
        node_colors = self.graphbuilder.node_colors(G)

        graph_windows = self.graphbuilder.graph_windows(
            df=df,
            window_days=window_days,
            overlap_days=overlap_days,
            edge_seconds=edge_seconds,
        )
        fig = self.visualizer.create_windows(
            mainG=G,
            pos=pos,
            graph_windows=graph_windows,
            node_colors=node_colors,
            edge_scale=edge_scale,
            node_scale=node_scale,
        )
        return fig

    def viz_graph(
        self,
        G: nx.Graph,
        layout: str = "Spring Layout",
        title: str = "Graph",
        node_scale: float = 1.0,
        edge_scale: float = 1.0,
        node_threshold: int = 0,
    ) -> go.Figure:
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


class SettingsManager:
    """Class to manage application settings with JSON configuration files."""

    def __init__(
        self,
        default_config_path: str = ".default_config.json",
        current_config_path: str = ".current_values.json",
    ):
        """Initialize the settings manager with paths to configuration files."""
        self.default_config_path = Path(default_config_path)
        self.current_config_path = Path(current_config_path)
        self.settings = self.load_settings()

    def load_settings(self) -> Dict[str, Any]:
        """Load settings from current config file or default if it doesn't exist."""
        # First try to load current settings
        if self.current_config_path.exists():
            try:
                with open(self.current_config_path, "r") as f:
                    logger.info(f"Loading settings from {self.current_config_path}")
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading current settings: {e}")

        # Fall back to default settings
        if self.default_config_path.exists():
            try:
                with open(self.default_config_path, "r") as f:
                    logger.info(
                        f"Loading default settings from {self.default_config_path}"
                    )
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading default settings: {e}")
                raise FileNotFoundError("Default config file is missing or corrupted")
        else:
            logger.error(f"Default config file not found at {self.default_config_path}")
            raise FileNotFoundError(
                f"Default config file not found at {self.default_config_path}"
            )

    def save_settings(self) -> bool:
        """Save current settings to file."""
        try:
            with open(self.current_config_path, "w") as f:
                json.dump(self.settings, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return False

    def update_settings(self, new_values: Dict[str, Any]) -> None:
        """Update settings with new values."""

        # For nested dictionaries, we need to update recursively
        def update_dict_recursively(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_dict_recursively(d[k], v)
                else:
                    d[k] = v

        update_dict_recursively(self.settings, new_values)
        self.save_settings()

    def reset_to_defaults(self) -> None:
        """Reset current settings to defaults."""
        try:
            with open(self.default_config_path, "r") as f:
                self.settings = json.load(f)
            self.save_settings()
            logger.info("Settings reset to defaults")
        except Exception as e:
            logger.error(f"Error resetting to defaults: {e}")
            raise

    def get_settings(self) -> Dict[str, Any]:
        return self.settings
