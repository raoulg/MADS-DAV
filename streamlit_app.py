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
            'node_size': {'min': 0.1, 'max': 2.0},
            'node_size_multiplier': {'min': 0.1, 'max': 2.0},
            'layout_iterations': {'min': 50, 'max': 1000},
            'layout_scale': {'min': 0.5, 'max': 3.0}
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
        # Min/max controls
        st.markdown(f"**{label} Range**")
        
        # Use columns for min/max controls
        col1, col2 = st.columns(2)
        with col1:
            new_min = st.number_input(
                f"Min {label}",
                value=st.session_state.slider_settings[key]['min'],
                step=step,
                key=f"{key}_min"
            )
        with col2:
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
    
    # Layout controls
    with st.expander("Layout Parameters"):
        default_node_spacing = create_slider_with_controls(
            "Node Spacing (k)",
            'node_spacing',
            default_value=0.15,
            step=0.01,
            help_text="Optimal distance between nodes (k parameter)"
        )
        
        layout_iterations = create_slider_with_controls(
            "Layout Iterations",
            'layout_iterations',
            default_value=500,
            step=50,
            help_text="Number of iterations for layout algorithm"
        )
        
        layout_scale = create_slider_with_controls(
            "Layout Scale",
            'layout_scale',
            default_value=1.5,
            step=0.1,
            help_text="Scale factor for node positions"
        )
    
    # Node appearance
    with st.expander("Node Appearance"):
        default_node_size = create_slider_with_controls(
            "Node Size",
            'node_size',
            default_value=0.5,
            step=0.1,
            help_text="Base size of nodes in the visualization"
        )
        
        node_size_multiplier = create_slider_with_controls(
            "Node Size Multiplier",
            'node_size_multiplier',
            default_value=0.5,
            step=0.1,
            help_text="Multiplier for node size based on degree (lower values = smaller nodes)"
        )
    
    # Time cutoff for visualization
    with st.expander("Time Cutoff Settings"):
        use_time_cutoff = st.checkbox("Use Time Cutoff", value=False)
        if use_time_cutoff:
            time_cutoff_days = st.slider(
                "Show only last X days",
                min_value=1,
                max_value=365,
                value=60,
                step=1,
                help="Only show data from the last X days"
            )
        else:
            time_cutoff_days = None
    
    # Analysis buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Analysis"):
            st.session_state.run_analysis = True
            st.session_state.force_layout = True
    with col2:
        if st.button("Recalculate Layout"):
            st.session_state.force_layout = True
            st.session_state.prev_layout = None  # Force layout recalculation

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
        
        # Apply time cutoff if enabled
        if use_time_cutoff and time_cutoff_days:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
                data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
                
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=time_cutoff_days)
            data = data[data['timestamp'] >= cutoff_date]
            st.info(f"Showing data from the last {time_cutoff_days} days only ({len(data)} messages)")
        
        # Initialize analyzer with layout settings
        analyzer = WhatsAppNetworkAnalyzer(config)
        analyzer.data = data
        analyzer.selected_layout = selected_layout
        analyzer.default_node_spacing = default_node_spacing
        analyzer.layout_iterations = layout_iterations
        analyzer.layout_scale = layout_scale
        analyzer.node_size_multiplier = node_size_multiplier
        
        # Track layout changes
        if 'prev_layout' not in st.session_state:
            st.session_state.prev_layout = selected_layout
            
        # Force layout recalculation if layout algorithm changed
        if st.session_state.prev_layout != selected_layout:
            st.session_state.force_layout = True
            analyzer.pos = None
            st.session_state.prev_layout = selected_layout
            
        # Force layout recalculation if needed
        if 'force_layout' in st.session_state and st.session_state.force_layout:
            analyzer.pos = None
            st.session_state.force_layout = False
        
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
                default_size=default_node_size,
                force_layout=('force_layout' in st.session_state and st.session_state.force_layout)
            )
            
        with tab2:
            st.subheader("Network Evolution Over Time")
            analyzer.visualize_time_series()
            
        st.success("Analysis complete!")
