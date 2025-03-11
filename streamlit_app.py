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

def get_default_settings():
    """Return default settings dictionary"""
    return {
        'slider_settings': {
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
        },
        'current_values': {
            'response_window': 1800,
            'time_window': 60,
            'time_overlap': 15,
            'edge_weight': 1.0,
            'min_edge_weight': 0.5,
            'node_spacing': 0.15,
            'node_size': 0.5,
            'node_size_multiplier': 0.5,
            'layout_iterations': 500,
            'layout_scale': 1.5,
            'selected_layout': 'Spring Layout',
            'filter_single_connections': False,
            'use_time_cutoff': False,
            'time_cutoff_days': 60
        }
    }

# Initialize session state
if 'settings' not in st.session_state:
    settings_file = Path("streamlit_settings.toml")
    if settings_file.exists():
        try:
            with open(settings_file, "rb") as f:
                st.session_state.settings = tomllib.load(f)
        except tomllib.TOMLDecodeError:
            st.session_state.settings = get_default_settings()
    else:
        st.session_state.settings = get_default_settings()

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
            def write_dict(d, indent=0):
                for key, value in d.items():
                    if isinstance(value, dict):
                        f.write(" " * indent + f"[{key}]\n")
                        write_dict(value, indent + 2)
                    else:
                        f.write(" " * indent + f"{key} = {repr(value)}\n")
            
            write_dict(config)
        st.rerun()
    
    
    def save_settings():
        """Save current settings to file"""
        import tomllib
        # tomllib doesn't have a dump function, so we'll write manually
        with open("streamlit_settings.toml", "w") as f:
            def write_dict(d, indent=0):
                for key, value in d.items():
                    if isinstance(value, dict):
                        f.write(" " * indent + f"[{key}]\n")
                        write_dict(value, indent + 2)
                    else:
                        if isinstance(value, str):
                            value = f'"{value}"'
                        f.write(" " * indent + f"{key} = {value}\n")
            
            write_dict(st.session_state.settings)
        logger.info("Settings saved to streamlit_settings.toml")

    # Reset button
    if st.button("Reset All Settings"):
        st.session_state.settings = get_default_settings()
        st.rerun()

    def create_slider_with_controls(label, key, default_value, step, help_text, min_value=None, max_value=None):
        """Helper function to create sliders with min/max controls"""
        st.markdown(f"**{label} Range**")
        col1, col2 = st.columns(2)
        with col1:
            new_min = st.number_input(
                f"Min {label}",
                value=st.session_state.settings['slider_settings'][key]['min'],
                step=step,
                key=f"{key}_min"
            )
        with col2:
            new_max = st.number_input(
                f"Max {label}",
                value=st.session_state.settings['slider_settings'][key]['max'],
                step=step,
                key=f"{key}_max"
            )
        st.session_state.settings['slider_settings'][key]['min'] = new_min
        st.session_state.settings['slider_settings'][key]['max'] = new_max
        
        # Get current value from session state or use default
        current_value = st.session_state.settings['current_values'].get(key, default_value)
        
        # Create slider and store value in session state
        slider_kwargs = {
            "label": label,
            "min_value": new_min,
            "max_value": new_max,
            "value": current_value,
            "step": step,
            "help": help_text,
            "key": f"{key}_slider"
        }
        
        # Apply min/max constraints if provided
        if min_value is not None:
            slider_kwargs["min_value"] = max(min_value, new_min)
        if max_value is not None:
            slider_kwargs["max_value"] = min(max_value, new_max)
        
        value = st.slider(**slider_kwargs)
        st.session_state.settings['current_values'][key] = value
        save_settings()
        return value

    # Data Selection & Time Settings
    with st.expander("📅 Data Selection & Time Settings", expanded=True):
        # Time cutoff settings
        use_time_cutoff = st.checkbox(
            "Use Time Cutoff", 
            value=st.session_state.settings['current_values'].get('use_time_cutoff', False),
            key="use_time_cutoff",
            on_change=lambda: (
                st.session_state.settings['current_values'].__setitem__('use_time_cutoff', st.session_state.use_time_cutoff),
                save_settings()
            )
        )
        
        if use_time_cutoff:
            time_cutoff_days = st.slider(
                "Show only last X days",
                min_value=1,
                max_value=365,
                value=st.session_state.settings['current_values'].get('time_cutoff_days', 60),
                step=1,
                help="Only show data from the last X days",
                key="time_cutoff_days",
                on_change=lambda: (
                    st.session_state.settings['current_values'].__setitem__('time_cutoff_days', st.session_state.time_cutoff_days),
                    save_settings()
                )
            )
        else:
            time_cutoff_days = None

        # Time window settings
        st.markdown("**Time Window Settings**")
        time_window_days = create_slider_with_controls(
            "Time Window (days)",
            'time_window',
            default_value=30,  # Default to 30 days instead of 60
            step=1,
            help_text="Size of each analysis window (minimum 7 days)",
            min_value=7  # Minimum 1 week
        )
        time_window = time_window_days * 86400  # Convert to seconds
        
        # Calculate default overlap as 25% of window size
        default_overlap = max(7, int(time_window_days * 0.25))  # Minimum 7 days overlap
        
        time_overlap_days = create_slider_with_controls(
            "Time Overlap (days)",
            'time_overlap',
            default_value=default_overlap,
            step=1,
            help_text="Overlap between time windows (25% of window size recommended)",
            max_value=time_window_days - 1  # Ensure overlap is smaller than window
        )
        time_overlap = time_overlap_days * 86400  # Convert to seconds
        
        # Show warning if overlap is too large
        if time_overlap_days >= time_window_days * 0.5:
            st.warning("Overlap is more than 50% of window size - consider reducing overlap for better results")

    # Network Analysis Parameters
    with st.expander("🔍 Network Analysis Parameters", expanded=True):
        # Response window settings
        response_window = create_slider_with_controls(
            "Response Window (seconds)",
            'response_window',
            default_value=1800,
            step=60,
            help_text="Time window to consider messages as responses"
        )
        
        # Edge weight settings
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

    # Layout Settings
    with st.expander("⚙️ Layout Settings", expanded=True):
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
            index=list(layout_algorithms.keys()).index(
                st.session_state.settings['current_values'].get('selected_layout', 'Spring Layout')
            ),
            help="Choose the algorithm for node positioning",
            key="selected_layout",
            on_change=lambda: (
                st.session_state.settings['current_values'].__setitem__('selected_layout', st.session_state.selected_layout),
                save_settings()
            )
        )

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

    # Node Appearance Settings
    with st.expander("🔘 Node Appearance", expanded=True):
            filter_single_connections = st.checkbox(
                "Filter nodes with only one connection",
                value=st.session_state.settings['current_values'].get('filter_single_connections', False),
                help="Remove nodes that only have one connection to simplify the graph",
                key="filter_single_connections",
                on_change=lambda: (
                    st.session_state.settings['current_values'].__setitem__('filter_single_connections', st.session_state.filter_single_connections),
                    save_settings()
                )
            )
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

    # Get current settings from session state
    response_window = st.session_state.settings['current_values']['response_window']
    time_window = st.session_state.settings['current_values']['time_window']
    time_overlap = st.session_state.settings['current_values']['time_overlap']
    edge_weight = st.session_state.settings['current_values']['edge_weight']
    min_edge_weight = st.session_state.settings['current_values']['min_edge_weight']
    use_time_cutoff = st.session_state.settings['current_values']['use_time_cutoff']
    time_cutoff_days = st.session_state.settings['current_values']['time_cutoff_days']
    selected_layout = st.session_state.settings['current_values']['selected_layout']
    default_node_spacing = st.session_state.settings['current_values']['node_spacing']
    layout_iterations = st.session_state.settings['current_values']['layout_iterations']
    layout_scale = st.session_state.settings['current_values']['layout_scale']
    node_size_multiplier = st.session_state.settings['current_values']['node_size_multiplier']
    default_node_size = st.session_state.settings['current_values']['node_size']
    filter_single_connections = st.session_state.settings['current_values']['filter_single_connections']

# Main content
st.title("WhatsApp Network Analyzer")

# Initialize analyzer in session state
if "analyzer" not in st.session_state:
    st.session_state.analyzer = None

if selected_file:
    if st.session_state.analyzer is None or ('run_analysis' in st.session_state and st.session_state.run_analysis):
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
                
            cutoff_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=time_cutoff_days)
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
            
            # Limit time windows to a reasonable number
            max_windows = 100  # Maximum number of time windows to calculate
            if time_window > 0:
                total_days = (data['timestamp'].max() - data['timestamp'].min()).days
                if total_days / (time_window_days - time_overlap_days) > max_windows:
                    st.warning(f"Too many time windows - adjusting settings to create max {max_windows} windows")
                    # Adjust time window size to create max_windows windows
                    time_window_days = total_days / max_windows + time_overlap_days
                    time_window = time_window_days * 86400
                    st.session_state.settings['current_values']['time_window'] = time_window_days
                    save_settings()
            
            analyzer.create_time_window_graphs()
        
        # Visualization tabs
        tab1, tab2 = st.tabs(["Network Graph", "Time Series"])
        
        with tab1:
            st.subheader("Interactive Network Graph")
            analyzer.visualize_graph(
                title="WhatsApp Interaction Network",
                default_k=default_node_spacing,
                default_size=default_node_size,
                force_layout=('force_layout' in st.session_state and st.session_state.force_layout),
                filter_single_connections=filter_single_connections
            )
            
        with tab2:
            st.subheader("Network Evolution Over Time")
            analyzer.visualize_time_series()
            
        st.success("Analysis complete!")
