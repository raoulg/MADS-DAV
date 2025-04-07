from pathlib import Path

import pandas as pd
import streamlit as st

from wa_analyzer.network_analysis import (Config, NetworkAnalysis,
                                          SettingsManager)
from wa_analyzer.settings import NetworkAnalysisConfig

# Page config
st.set_page_config(
    page_title="WhatsApp Network Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Initialize session state and settings
if "settingsmanager" not in st.session_state:
    st.session_state.settingsmanager = SettingsManager()
    st.session_state.settingsmanager.save_settings()

# Sidebar for settings
with st.sidebar:
    st.title("Settings")

    # File selection
    processed_dir = Path("data/processed")
    available_files = [f.name for f in processed_dir.glob("*.csv")]
    current_file = st.session_state.settingsmanager.settings["current_values"].get(
        "selected_file", None
    )

    selected_file = st.selectbox(
        "Select chat file",
        available_files,
        index=(
            available_files.index(current_file)
            if current_file in available_files
            else 0
        ),
    )

    if selected_file != current_file:
        st.session_state.settingsmanager.update_settings(
            {"current_values": {"selected_file": selected_file}}
        )
        st.session_state.settingsmanager.save_settings()  # Save immediately after update

    # Reset button
    if st.button("Reset All Settings"):
        st.session_state.settingsmanager.reset_to_defaults()
        st.rerun()

    def create_slider_with_controls(
        label, key, default_value, step, help_text, min_value=None, max_value=None
    ):
        """Helper function to create sliders with min/max controls"""
        st.markdown(f"**{label} Range**")
        col1, col2 = st.columns(2)

        # Get settings
        current_min = st.session_state.settingsmanager.settings["slider_settings"][key][
            "min"
        ]
        current_max = st.session_state.settingsmanager.settings["slider_settings"][key][
            "max"
        ]

        with col1:
            new_min = st.number_input(
                f"Min {label}",
                value=current_min,
                step=step,
                key=f"{key}_min",
            )
        with col2:
            new_max = st.number_input(
                f"Max {label}",
                value=current_max,
                step=step,
                key=f"{key}_max",
            )

        # Only update if values have changed
        if new_min != current_min or new_max != current_max:
            st.session_state.settingsmanager.update_settings(
                {"slider_settings": {key: {"min": new_min, "max": new_max}}}
            )
            st.session_state.settingsmanager.save_settings()

        # Get current value from session state or use default
        current_value = st.session_state.settingsmanager.settings["current_values"].get(
            key, default_value
        )

        # Create a session state key for this slider if not exists
        slider_state_key = f"{key}_value"
        if slider_state_key not in st.session_state:
            st.session_state[slider_state_key] = current_value

        # Create slider with session state value
        slider_kwargs = {
            "label": label,
            "min_value": max(min_value, new_min) if min_value is not None else new_min,
            "max_value": min(max_value, new_max) if max_value is not None else new_max,
            "value": st.session_state[slider_state_key],
            "step": step,
            "help": help_text,
            "key": f"{key}_slider",
            "on_change": lambda: handle_slider_change(key, slider_state_key),
        }

        value = st.slider(**slider_kwargs)

        # Update the session state with the new value
        st.session_state[slider_state_key] = value

        return value

    def handle_slider_change(key, slider_state_key):
        """Function to handle slider changes"""
        new_value = st.session_state[f"{key}_slider"]
        st.session_state[slider_state_key] = new_value
        st.session_state.settingsmanager.update_settings(
            {"current_values": {key: new_value}}
        )
        st.session_state.settingsmanager.save_settings()

    # Data Selection & Time Settings
    with st.expander("📅 Data Selection & Time Settings", expanded=True):
        # Time cutoff settings
        # Create a session state key for checkbox if not exists
        if "use_time_cutoff_value" not in st.session_state:
            st.session_state.use_time_cutoff_value = (
                st.session_state.settingsmanager.settings["current_values"].get(
                    "use_time_cutoff", False
                )
            )

        use_time_cutoff = st.checkbox(
            "Use Time Cutoff",
            value=st.session_state.use_time_cutoff_value,
            key="use_time_cutoff",
            on_change=lambda: handle_checkbox_change("use_time_cutoff"),
        )

        def handle_checkbox_change(key):
            """Function to handle checkbox changes"""
            new_value = st.session_state[key]
            st.session_state[f"{key}_value"] = new_value
            st.session_state.settingsmanager.update_settings(
                {"current_values": {key: new_value}}
            )
            st.session_state.settingsmanager.save_settings()

        time_cutoff_days = None
        if use_time_cutoff:
            # Create a session state key for time cutoff slider if not exists
            if "time_cutoff_days_value" not in st.session_state:
                st.session_state.time_cutoff_days_value = (
                    st.session_state.settingsmanager.settings["current_values"].get(
                        "time_cutoff_days", 60
                    )
                )

            time_cutoff_days = st.slider(
                "Show only last X days",
                min_value=1,
                max_value=365,
                value=st.session_state.time_cutoff_days_value,
                step=1,
                help="Only show data from the last X days",
                key="time_cutoff_days_slider",
                on_change=lambda: handle_slider_change(
                    "time_cutoff_days", "time_cutoff_days_value"
                ),
            )

            # Update the session state with the new value
            st.session_state.time_cutoff_days_value = time_cutoff_days
        else:
            time_cutoff_days = None

        # Time window settings
        st.markdown("**Time Window Settings**")
        time_window = create_slider_with_controls(
            "Time Window (days)",
            "time_window",
            default_value=30,  # Default to 30 days instead of 60
            step=1,
            help_text="Size of each analysis window (minimum 7 days)",
            min_value=7,  # Minimum 1 week
        )

        time_overlap = create_slider_with_controls(
            "Time Overlap (days)",
            "time_overlap",
            default_value=10,
            step=1,
            help_text="Overlap between time windows",
            max_value=time_window // 2,  # Ensure overlap is smaller than window
        )

    # Network Analysis Parameters
    with st.expander("🔍 Network Analysis Parameters", expanded=True):
        # Response window settings
        response_window = create_slider_with_controls(
            "Response Window (seconds)",
            "response_window",
            default_value=1800,
            step=60,
            help_text="Time window to consider messages as responses",
        )

        # Edge weight settings
        edge_weight = create_slider_with_controls(
            "Edge Weight Multiplier",
            "edge_weight",
            default_value=1.0,
            step=0.1,
            help_text="Multiplier for edge weights",
        )

        min_edge_weight = create_slider_with_controls(
            "Minimum Edge Weight",
            "min_edge_weight",
            default_value=0.5,
            step=0.1,
            help_text="Minimum weight for edges to be included",
        )

    # Layout Settings
    with st.expander("⚙️ Layout Settings", expanded=True):
        # Layout algorithm selection
        layout_algorithms = [
            "Spring Layout",
            "Kamada-Kawai",
            "Circular Layout",
            "Spectral Layout",
        ]

        # Create a session state key for layout selection if not exists
        if "selected_layout_value" not in st.session_state:
            st.session_state.selected_layout_value = (
                st.session_state.settingsmanager.settings["current_values"].get(
                    "selected_layout", "Spring Layout"
                )
            )

        selected_layout = st.selectbox(
            "Layout Algorithm",
            layout_algorithms,
            index=layout_algorithms.index(st.session_state.selected_layout_value),
            help="Choose the algorithm for node positioning",
            key="selected_layout_select",
            on_change=lambda: handle_select_change("selected_layout"),
        )

        def handle_select_change(key):
            """Function to handle selectbox changes"""
            new_value = st.session_state[f"{key}_select"]
            st.session_state[f"{key}_value"] = new_value
            st.session_state.settingsmanager.update_settings(
                {"current_values": {key: new_value}}
            )
            st.session_state.settingsmanager.save_settings()

    with st.expander("🔘 Node Appearance", expanded=True):
        # Create a session state key for checkbox if not exists
        if "filter_single_connections_value" not in st.session_state:
            st.session_state.filter_single_connections_value = (
                st.session_state.settingsmanager.settings["current_values"].get(
                    "filter_single_connections", False
                )
            )

        filter_single_connections = st.checkbox(
            "Filter nodes with only one connection",
            value=st.session_state.filter_single_connections_value,
            help="Remove nodes that only have one connection to simplify the graph",
            key="filter_single_connections",
            on_change=lambda: handle_checkbox_change("filter_single_connections"),
        )

        node_size_multiplier = create_slider_with_controls(
            "Node Size Multiplier",
            "node_size_multiplier",
            default_value=0.5,
            step=0.1,
            help_text="Multiplier for node size based on degree (lower values = smaller nodes)",
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


# Main content
st.title("WhatsApp Network Analyzer")


if selected_file:
    datafile = processed_dir / selected_file
    config = Config(
        time_col="timestamp",
        node_col="author",
        seconds=response_window,
        datafile=datafile,
    )
    na = NetworkAnalysis(config)

    data = pd.read_csv(processed_dir / selected_file)

    # Create config
    config = NetworkAnalysisConfig(
        response_window=response_window,
        time_window=time_window,
        time_overlap=time_overlap,
        edge_weight_multiplier=edge_weight,
        min_edge_weight=min_edge_weight,
    )

    # Track layout changes
    if "prev_layout" not in st.session_state:
        st.session_state.prev_layout = selected_layout

    # Visualization tabs
    tab1, tab2 = st.tabs(["Network Graph", "Time Series"])

    with tab1:
        st.subheader("Interactive Network Graph")
        fig = na.process(
            "Network Analysis",
            layout=selected_layout,
            cutoff_days=time_cutoff_days,
            node_threshold=1 if not filter_single_connections else 2,
            node_scale=node_size_multiplier,
            edge_scale=edge_weight,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="network_graph",
        )

    with tab2:
        st.subheader("Network Evolution Over Time")
        fig = na.windows(
            cutoff_days=time_cutoff_days,
            edge_seconds=response_window,
            window_days=time_window,
            overlap_days=time_overlap,
            node_threshold=1 if not filter_single_connections else 2,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="timeseries_graph",
        )

    st.success("Analysis complete!")
