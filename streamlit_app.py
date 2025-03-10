import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import tomllib
from wa_analyzer.network_analysis import WhatsAppNetworkAnalyzer
from wa_analyzer.settings import NetworkAnalysisConfig

# Page config
st.set_page_config(
    page_title="WhatsApp Network Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    
    # File selection
    processed_dir = Path("data/processed")
    available_files = [f.name for f in processed_dir.glob("*.csv")]
    
    # Get current file from config
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
        current_file = config.get("current_file", available_files[0] if available_files else "")
    
    selected_file = st.selectbox(
        "Select chat file",
        available_files,
        index=available_files.index(current_file) if current_file in available_files else 0
    )
    
    # Update config if file changed
    if selected_file != current_file:
        with open("config.toml", "rb") as f:
            config = tomllib.load(f)
        config["current_file"] = selected_file
        with open("config.toml", "w") as f:
            for key, value in config.items():
                f.write(f"{key} = \"{value}\"\n")
        st.rerun()
    
    # Analysis parameters
    st.subheader("Analysis Parameters")
    # Persistent settings storage
    if 'slider_settings' not in st.session_state:
        st.session_state.slider_settings = {
            'response_window': {'min': 300, 'max': 3600},
            'time_window': {'min': 1, 'max': 90},
            'time_overlap': {'min': 0, 'max': 30},
            'edge_weight': {'min': 0.1, 'max': 5.0},
            'min_edge_weight': {'min': 0.1, 'max': 2.0},
            'node_spacing': {'min': 0.05, 'max': 1.0},
            'node_size': {'min': 0.1, 'max': 2.0}
        }

    # Layout algorithm selection
    layout_algorithms = {
        'Spring Layout': nx.spring_layout,
        'Kamada-Kawai': nx.kamada_kawai_layout,
        'Circular Layout': nx.circular_layout,
        'Spectral Layout': nx.spectral_layout
    }
    selected_layout = st.selectbox(
        "Layout Algorithm",
        list(layout_algorithms.keys()),
        index=0
    )

    def create_slider_with_controls(label, key, default_value, step, help_text):
        col1, col2 = st.columns([3, 1])
        with col1:
            # Min/max controls
            st.markdown(f"**{label} Range**")
            min_col, max_col = st.columns(2)
            with min_col:
                new_min = st.number_input(
                    f"Min {label}",
                    value=st.session_state.slider_settings[key]['min'],
                    step=step,
                    key=f"{key}_min"
                )
            with max_col:
                new_max = st.number_input(
                    f"Max {label}",
                    value=st.session_state.slider_settings[key]['max'],
                    step=step,
                    key=f"{key}_max"
                )
            
            # Update stored values
            st.session_state.slider_settings[key]['min'] = new_min
            st.session_state.slider_settings[key]['max'] = new_max
            
            # Create slider
            return st.slider(
                label,
                min_value=new_min,
                max_value=new_max,
                value=default_value,
                step=step,
                help=help_text
            )

    # Response window settings
    st.markdown("**Response Window Settings**")
    response_window = create_slider_with_controls(
        "Response Window (seconds)",
        'response_window',
        default_value=1800,
        step=60,
        help_text="Time window to consider messages as responses"
    )
    
    # Time window settings
    st.markdown("**Time Window Settings**")
    time_window = create_slider_with_controls(
        "Time Window (days)",
        'time_window',
        default_value=60,
        step=1,
        help_text="Size of each analysis window"
    ) * 86400  # Convert to seconds
    
    time_overlap = create_slider_with_controls(
        "Time Overlap (days)",
        'time_overlap',
        default_value=15,
        step=1,
        help_text="Overlap between time windows"
    ) * 86400  # Convert to seconds
    
    # Edge weight settings
    st.markdown("**Edge Weight Settings**")
    edge_weight = create_slider_with_controls(
        "Edge Weight Multiplier",
        'edge_weight',
        default_value=1.0,
        step=0.1,
        help_text="Multiplier for edge weights"
    )
    
    min_edge_weight = create_slider_with_controls(
        "Minimum Edge Weight",
        'min_edge_weight',
        default_value=0.5,
        step=0.1,
        help_text="Minimum weight for edges to be included"
    )
    
    # Visualization parameters
    st.subheader("Visualization Parameters")
    default_node_spacing = create_slider_with_controls(
        "Default Node Spacing",
        'node_spacing',
        default_value=0.15,
        step=0.05,
        help_text="Spacing between nodes in the visualization"
    )
    
    default_node_size = create_slider_with_controls(
        "Default Node Size",
        'node_size',
        default_value=0.5,
        step=0.1,
        help_text="Base size of nodes in the visualization"
    )
    
    # Recalculate button
    if st.button("Run Analysis"):
        st.session_state.run_analysis = True

# Main content
st.title("WhatsApp Network Analyzer")

if selected_file:
    if 'run_analysis' in st.session_state and st.session_state.run_analysis:
        # Load data from selected file
        data = pd.read_csv(processed_dir / selected_file)
        
        # Create config
        config = NetworkAnalysisConfig(
            response_window=response_window,
            time_window=time_window,
            time_overlap=time_overlap,
            edge_weight_multiplier=edge_weight,
            min_edge_weight=min_edge_weight
        )
        
        # Initialize analyzer with layout settings
        analyzer = WhatsAppNetworkAnalyzer(config)
        analyzer.data = data
        analyzer.selected_layout = selected_layout
        analyzer.default_node_spacing = default_node_spacing
        
        # Create graphs
        with st.spinner("Creating network graphs..."):
            analyzer.create_full_graph()
            analyzer.create_time_window_graphs()
        
        # Visualization tabs
        tab1, tab2 = st.tabs(["Network Graph", "Time Series"])
        
        with tab1:
            st.subheader("Interactive Network Graph")
            analyzer.visualize_graph(
                title="WhatsApp Interaction Network",
                default_k=default_node_spacing,
                default_size=default_node_size
            )
            
        with tab2:
            st.subheader("Network Evolution Over Time")
            analyzer.visualize_time_series()
            
        st.success("Analysis complete!")
