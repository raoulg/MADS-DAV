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
        with open("config.toml", "r") as f:
            config = tomllib.load(f)
        config["current_file"] = selected_file
        with open("config.toml", "w") as f:
            for key, value in config.items():
                f.write(f"{key} = \"{value}\"\n")
        st.rerun()
    
    # Analysis parameters
    st.subheader("Analysis Parameters")
    response_window = st.slider(
        "Response Window (seconds)",
        min_value=60,
        max_value=86400,
        value=3600,
        step=60,
        help="Time window to consider messages as responses"
    )
    
    time_window = st.slider(
        "Time Window (days)",
        min_value=1,
        max_value=90,
        value=60,
        step=1,
        help="Size of each analysis window"
    ) * 86400  # Convert to seconds
    
    time_overlap = st.slider(
        "Time Overlap (days)",
        min_value=0,
        max_value=30,
        value=15,
        step=1,
        help="Overlap between time windows"
    ) * 86400  # Convert to seconds
    
    edge_weight = st.slider(
        "Edge Weight Multiplier",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1
    )
    
    min_edge_weight = st.slider(
        "Minimum Edge Weight",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1
    )
    
    # Visualization parameters
    st.subheader("Visualization Parameters")
    default_node_spacing = st.slider(
        "Default Node Spacing",
        min_value=0.05,
        max_value=1.0,
        value=0.15,
        step=0.05
    )
    
    default_node_size = st.slider(
        "Default Node Size",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1
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
        
        # Initialize analyzer
        analyzer = WhatsAppNetworkAnalyzer(config)
        analyzer.data = data
        
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
