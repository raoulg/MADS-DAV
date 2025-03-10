import datetime
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import ipywidgets as widgets
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from loguru import logger

from wa_analyzer.settings import NetworkAnalysisConfig


class WhatsAppNetworkAnalyzer:
    """Analyze WhatsApp chat data as a network of users."""

    def __init__(self, config: NetworkAnalysisConfig = None):
        """Initialize the network analyzer with configuration."""
        self.config = config or NetworkAnalysisConfig()
        self.data = None
        self.graph = None
        self.pos = None
        self.time_windows = []
        self.graphs_by_window = []
        self.node_colors = {}

    def load_data(self, filepath: Path) -> None:
        """Load preprocessed WhatsApp data."""
        logger.info(f"Loading data from {filepath}")
        self.data = pd.read_csv(filepath)

        # Convert timestamp to datetime and ensure proper timezone handling
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"], utc=True)
        
        # Sort by timestamp and reset index
        self.data = self.data.sort_values("timestamp").reset_index(drop=True)
        
        # Verify timestamp conversion
        if not pd.api.types.is_datetime64_any_dtype(self.data["timestamp"]):
            raise ValueError("Timestamp conversion failed - check input data format")

        logger.info(
            f"Loaded {len(self.data)} messages from {len(self.data['author'].unique())} users"
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
        other_components = [comp for comp in connected_components if comp != main_component]

        # Generate layout for main component using selected algorithm
        layout_func = layout_algorithms[selected_layout]
        layout_kwargs = {
            'seed': 42,
            'k': default_node_spacing,
            'iterations': 500,
            'scale': 1.5
        }
        
        if selected_layout == 'Kamada-Kawai':
            layout_kwargs.pop('k')  # Kamada-Kawai doesn't use k parameter
            
        main_pos = layout_func(
            G.subgraph(main_component),
            **layout_kwargs
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
                    G.subgraph(component),
                    seed=42,
                    k=0.05,
                    iterations=100,
                    scale=0.5
                )
                
                # Offset to position around main component
                for node, (x, y) in component_pos.items():
                    self.pos[node] = (x + center_x, y + center_y)

        return G

    def create_time_window_graphs(self) -> List[Tuple[datetime.datetime, nx.Graph]]:
        """Create graphs for overlapping time windows."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Calculate time windows
        start_time = self.data["timestamp"].min()
        end_time = self.data["timestamp"].max()

        window_size = datetime.timedelta(seconds=self.config.time_window)
        overlap = datetime.timedelta(seconds=self.config.time_overlap)

        # Create time windows with overlap
        current_start = start_time
        self.time_windows = []

        while current_start < end_time:
            current_end = current_start + window_size
            self.time_windows.append((current_start, current_end))
            current_start = current_end - overlap

        logger.info(f"Created {len(self.time_windows)} time windows")

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
        interactions = defaultdict(int)

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
    ) -> None:
        """Visualize the network graph interactively using Plotly."""
        if G is None:
            G = self.graph

        if G is None:
            raise ValueError("No graph available. Create a graph first.")

        # Remove isolated nodes for better visualization
        non_isolated_nodes = [node for node in G.nodes() if G.degree(node) > 0]
        G_filtered = G.subgraph(non_isolated_nodes)

        # Create edge trace
        edge_trace = []
        for edge in G_filtered.edges():
            x0, y0 = self.pos[edge[0]]
            x1, y1 = self.pos[edge[1]]
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=1, color="#888"),
                    hoverinfo="none",
                    mode="lines",
                )
            )

        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []

        for node in G_filtered.nodes():
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}<br>Degree: {G_filtered.degree(node)}")
            node_size.append(
                10 + 10 * np.log1p(G_filtered.degree(node))
            )  # Smaller default size
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

        # Create figure
        fig = go.Figure(
            data=edge_trace + [node_trace],
            layout=go.Layout(
                title=dict(
                    text=title,
                    font=dict(size=16)
                ),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        # Add interactive controls
        k_slider = widgets.FloatSlider(
            value=default_k,  # Use provided default spacing
            min=0.05,  # Allow closer spacing
            max=1.0,
            step=0.05,  # Finer control
            description="Node spacing:",
            continuous_update=False,
        )

        size_slider = widgets.FloatSlider(
            value=default_size,
            min=0.1,
            max=2.0,
            step=0.1,
            description="Node size:",
            continuous_update=False,
        )

        def update_layout(k, size_factor):
            """Update the layout with new spacing and size parameters."""
            # Update layout with new spacing
            self.pos = nx.spring_layout(G_filtered, k=k, iterations=500, seed=42)

            # Update node positions and sizes
            fig.update_traces(
                x=[self.pos[node][0] for node in G_filtered.nodes()],
                y=[self.pos[node][1] for node in G_filtered.nodes()],
                marker=dict(
                    size=[
                        10 + 10 * size_factor * np.log1p(G_filtered.degree(node))
                        for node in G_filtered.nodes()
                    ]
                ),
                selector={"mode": "markers+text"},
            )

            # Update edge positions
            for i, edge in enumerate(G_filtered.edges()):
                x0, y0 = self.pos[edge[0]]
                x1, y1 = self.pos[edge[1]]
                fig.data[i].x = [x0, x1, None]
                fig.data[i].y = [y0, y1, None]

        # Connect sliders to update function
        widgets.interact(update_layout, k=k_slider, size_factor=size_slider)

        # Show the figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def visualize_time_series(self) -> None:
        """Visualize the network evolution over time as a static grid of the last 9 timeframes."""
        if not self.graphs_by_window or not self.pos:
            raise ValueError(
                "No time window graphs available. Create time window graphs first."
            )

        # Get the last 9 timeframes
        last_windows = self.graphs_by_window[-9:]
        num_windows = len(last_windows)
        
        # Create subplots
        fig = make_subplots(
            rows=3, 
            cols=3,
            subplot_titles=[f"Window {i+1}" for i in range(num_windows)],
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )

        # Plot each timeframe
        for i, (timestamp, G) in enumerate(last_windows):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            # Create edge traces
            for edge in G.edges():
                x0, y0 = self.pos[edge[0]]
                x1, y1 = self.pos[edge[1]]
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        line=dict(width=1, color="#888"),
                        hoverinfo="none",
                        mode="lines",
                    ),
                    row=row,
                    col=col
                )

            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_color = []

            for node in G.nodes():
                x, y = self.pos.get(node, (0, 0))
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"{node}<br>Degree: {G.degree(node)}")
                node_size.append(10 + 20 * np.log1p(G.degree(node)))
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
                col=col
            )

            # Format subplot
            fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=row, col=col)
            fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=row, col=col)

        # Update layout
        fig.update_layout(
            title_text="WhatsApp Network Evolution - Last 9 Time Windows",
            showlegend=False,
            height=900,
            margin=dict(b=20, l=20, r=20, t=100)
        )

        # Show in Streamlit
        st.plotly_chart(fig, use_container_width=True)

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
    response_window: int = 3600,  # 1 hour in seconds
    time_window: int = 60 * 60 * 24 * 30 * 2,  # 2 months in seconds
    time_overlap: int = 60 * 60 * 24 * 30,  # 1 month in seconds
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
        analyzer.visualize_graph(title="Complete WhatsApp Interaction Network")
        fig = analyzer.visualize_graph(title="Complete WhatsApp Interaction Network")
        fig.write_html(output_dir / "full_network.html")

        # Save time series animation
        analyzer.visualize_time_series(output_dir / "network_evolution.gif")

        # Export data and metrics
        analyzer.export_graph_data(output_dir)
        analyzer.export_graph_metrics(output_dir / "network_metrics.csv")

    return analyzer
