import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

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
        
        # Convert timestamp to datetime
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        # Sort by timestamp
        self.data = self.data.sort_values('timestamp')
        
        logger.info(f"Loaded {len(self.data)} messages from {len(self.data['author'].unique())} users")
        
    def create_full_graph(self) -> nx.Graph:
        """Create a graph of all interactions."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        G = nx.Graph()
        
        # Add all users as nodes
        for user in self.data['author'].unique():
            G.add_node(user)
            
        # Assign random colors to nodes
        self.node_colors = {user: np.random.rand(3,) for user in G.nodes()}
        
        # Process all messages to create edges
        self._process_interactions(G, self.data)
        
        self.graph = G
        
        # Generate a better layout with optimized parameters
        self.pos = nx.spring_layout(
            G, 
            seed=42,
            k=0.15,  # Optimal distance between nodes
            iterations=200,  # More iterations for better layout
            scale=2.0,  # Spread out more
            pos=nx.circular_layout(G)  # Start from circular layout
        )
        
        return G
        
    def create_time_window_graphs(self) -> List[Tuple[datetime.datetime, nx.Graph]]:
        """Create graphs for overlapping time windows."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Calculate time windows
        start_time = self.data['timestamp'].min()
        end_time = self.data['timestamp'].max()
        
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
            window_data = self.data[(self.data['timestamp'] >= window_start) & 
                                    (self.data['timestamp'] <= window_end)]
            
            G = nx.Graph()
            
            # Add all users as nodes (from the full dataset to maintain consistency)
            for user in self.data['author'].unique():
                G.add_node(user)
                
            # Process messages in this time window
            self._process_interactions(G, window_data)
            
            self.graphs_by_window.append((window_start, G))
            logger.info(f"Window {i+1}: {window_start} to {window_end} - {len(window_data)} messages")
            
        return self.graphs_by_window
    
    def _process_interactions(self, G: nx.Graph, data: pd.DataFrame) -> None:
        """Process interactions to create edges between users."""
        # Track interactions within the response window
        interactions = defaultdict(int)
        
        # Group by timestamp to process in order
        for timestamp, group in data.groupby('timestamp'):
            # Get authors in this timestamp group
            current_authors = group['author'].unique()
            
            # Find recent messages within response window
            response_cutoff = timestamp - datetime.timedelta(seconds=self.config.response_window)
            recent_messages = data[(data['timestamp'] >= response_cutoff) & 
                                  (data['timestamp'] < timestamp)]
            
            # Get unique recent authors
            recent_authors = set(recent_messages['author'].unique())
            
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
    
    def visualize_graph(self, G: Optional[nx.Graph] = None, title: str = "WhatsApp Interaction Network") -> None:
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
            edge_trace.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'))
        
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
            node_size.append(10 + 20 * np.log1p(G_filtered.degree(node)))
            node_color.append(f"rgb({int(255*self.node_colors[node][0])},"
                            f"{int(255*self.node_colors[node][1])},"
                            f"{int(255*self.node_colors[node][2])})")
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[node.split()[0] for node in G_filtered.nodes()],  # Show first name only
            textposition="top center",
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=node_size,
                color=node_color,
                line_width=2))
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace],
                       layout=go.Layout(
                           title=title,
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        
        # Add interactive controls
        k_slider = widgets.FloatSlider(
            value=0.3,
            min=0.1,
            max=1.0,
            step=0.1,
            description='Node spacing:',
            continuous_update=False
        )
        
        def update_layout(k):
            """Update the layout with new spacing parameter."""
            self.pos = nx.spring_layout(
                G_filtered,
                k=k,
                iterations=500,
                seed=42
            )
            # Update node positions
            fig.update_traces(
                x=[self.pos[node][0] for node in G_filtered.nodes()],
                y=[self.pos[node][1] for node in G_filtered.nodes()],
                selector={'mode': 'markers+text'}
            )
            # Update edge positions
            for i, edge in enumerate(G_filtered.edges()):
                x0, y0 = self.pos[edge[0]]
                x1, y1 = self.pos[edge[1]]
                fig.data[i].x = [x0, x1, None]
                fig.data[i].y = [y0, y1, None]
        
        # Connect slider to update function
        widgets.interact(update_layout, k=k_slider)
        
        # Show the figure
        fig.show()
        
    def visualize_time_series(self, output_path: Optional[Path] = None) -> None:
        """Visualize the network evolution over time."""
        if not self.graphs_by_window:
            raise ValueError("No time window graphs available. Create time window graphs first.")
            
        # Create a figure for the animation
        fig, ax = plt.subplots(figsize=(12, 10))
        
        def update(i):
            """Update function for animation."""
            ax.clear()
            timestamp, G = self.graphs_by_window[i]
            
            # Calculate node sizes based on degree centrality
            degree_dict = dict(G.degree())
            node_sizes = [300 + (degree_dict[node] * 100) for node in G.nodes()]
            
            # Scale edge weights for better visualization
            edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
            if edge_weights:
                max_weight = max(edge_weights)
                min_weight = min(edge_weights)
                if max_weight > min_weight:
                    edge_widths = [1 + 5 * (w - min_weight) / (max_weight - min_weight) for w in edge_weights]
                else:
                    edge_widths = [1.5 for _ in edge_weights]
            else:
                edge_widths = []
            
            # Draw the network
            nx.draw_networkx_nodes(G, self.pos, 
                                  node_color=[self.node_colors.get(node, (0.5, 0.5, 0.5)) for node in G.nodes()],
                                  node_size=node_sizes, alpha=0.8, ax=ax)
            
            nx.draw_networkx_edges(G, self.pos, width=edge_widths, alpha=0.5, 
                                  edge_color='gray', style='solid', ax=ax)
            nx.draw_networkx_labels(G, self.pos, font_size=10, font_family='sans-serif',
                                   font_weight='bold', font_color='black', ax=ax)
            
            window_start = timestamp.strftime('%Y-%m-%d')
            window_end = (timestamp + datetime.timedelta(seconds=self.config.time_window)).strftime('%Y-%m-%d')
            ax.set_title(f"WhatsApp Interactions: {window_start} to {window_end}")
            ax.axis('off')
            
        # Create the animation with faster frame rate
        ani = FuncAnimation(fig, update, frames=len(self.graphs_by_window), interval=500, repeat=True)
        
        if output_path:
            ani.save(output_path, writer='pillow', fps=1)
            logger.info(f"Animation saved to {output_path}")
        else:
            plt.tight_layout()
            plt.show()
            
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
                node_data.append({
                    'id': node,
                    'label': node,
                    'degree': degree,
                    'color': ','.join(map(str, self.node_colors.get(node, (0.5, 0.5, 0.5))))
                })
            
            node_df = pd.DataFrame(node_data)
            node_df.to_csv(output_dir / "nodes.csv", index=False)
            logger.info(f"Node data exported to {output_dir / 'nodes.csv'}")
            
            # Export edge data
            edge_data = []
            for u, v, data in self.graph.edges(data=True):
                edge_data.append({
                    'source': u,
                    'target': v,
                    'weight': data.get('weight', 1)
                })
            
            edge_df = pd.DataFrame(edge_data)
            edge_df.to_csv(output_dir / "edges.csv", index=False)
            logger.info(f"Edge data exported to {output_dir / 'edges.csv'}")
            
            # Export as GraphML for use in tools like Gephi
            nx.write_graphml(self.graph, output_dir / "network.graphml")
            logger.info(f"GraphML exported to {output_dir / 'network.graphml'}")
    
    def export_graph_metrics(self, output_path: Optional[Path] = None) -> pd.DataFrame:
        """Export network metrics for all time windows."""
        if not self.graphs_by_window:
            raise ValueError("No time window graphs available. Create time window graphs first.")
            
        metrics = []
        
        # Calculate metrics for each time window
        for timestamp, G in self.graphs_by_window:
            window_metrics = {
                'timestamp': timestamp,
                'node_count': G.number_of_nodes(),
                'edge_count': G.number_of_edges(),
                'density': nx.density(G),
                'avg_clustering': nx.average_clustering(G),
            }
            
            # Add centrality metrics for each node
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            
            for node in G.nodes():
                window_metrics[f'degree_{node}'] = degree_centrality.get(node, 0)
                window_metrics[f'betweenness_{node}'] = betweenness_centrality.get(node, 0)
                
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
    output_dir: Optional[Path] = None
) -> WhatsAppNetworkAnalyzer:
    """Analyze WhatsApp chat data as a network and generate visualizations."""
    # Create configuration
    config = NetworkAnalysisConfig(
        response_window=response_window,
        time_window=time_window,
        time_overlap=time_overlap,
        edge_weight_multiplier=edge_weight_multiplier,
        min_edge_weight=min_edge_weight
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
        plt.figure(figsize=(12, 10))
        analyzer.visualize_graph(title="Complete WhatsApp Interaction Network")
        plt.savefig(output_dir / "full_network.png")
        plt.close()
        
        # Save time series animation
        analyzer.visualize_time_series(output_dir / "network_evolution.gif")
        
        # Export data and metrics
        analyzer.export_graph_data(output_dir)
        analyzer.export_graph_metrics(output_dir / "network_metrics.csv")
    
    return analyzer
