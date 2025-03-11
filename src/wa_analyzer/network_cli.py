import tomllib
from pathlib import Path
import subprocess
import sys

import click
from loguru import logger

from wa_analyzer.network_analysis import analyze_whatsapp_network


@click.command()
@click.option(
    "--data-file",
    "-f",
    type=click.Path(exists=True, path_type=Path),
    help="CSV file with preprocessed WhatsApp messages (overrides config)",
)
@click.option(
    "--response-window",
    "-r",
    type=int,
    default=3600,
    help="Time window in seconds to consider messages as responses (default: 3600 = 1 hour)",
)
@click.option(
    "--time-window",
    "-t",
    type=int,
    default=60 * 60 * 24 * 30 * 2,
    help="Size of each time window in seconds (default: 2 months)",
)
@click.option(
    "--time-overlap",
    "-o",
    type=int,
    default=60 * 60 * 24 * 30,
    help="Overlap between time windows in seconds (default: 1 month)",
)
@click.option(
    "--edge-weight",
    "-w",
    type=float,
    default=1.0,
    help="Multiplier for edge weights (default: 1.0)",
)
@click.option(
    "--min-edge-weight",
    "-m",
    type=float,
    default=0.5,
    help="Minimum edge weight to include in the graph (default: 0.5)",
)
@click.option(
    "--output-dir",
    "-d",
    type=click.Path(path_type=Path),
    help="Directory to save output files (optional)",
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    help="Show interactive visualizations (default: True)",
)
@click.option(
    "--streamlit/--no-streamlit",
    default=False,
    help="Launch Streamlit web interface (default: False)",
)
def main(
    data_file,
    response_window,
    time_window,
    time_overlap,
    edge_weight,
    min_edge_weight,
    output_dir,
    interactive,
    streamlit,
):
    """Analyze WhatsApp chat data as a network of users."""

    # Load config file to get processed data path
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
        processed_dir = Path(config["processed"])
        current_file = config.get("current")

    # Use provided data file or get from config
    if data_file:
        input_file = data_file
        logger.info(f"Using provided data file: {input_file}")
    elif current_file:
        input_file = processed_dir / current_file
        logger.info(f"Using current file from config: {input_file}")
    else:
        # Find the most recent file in the processed directory
        processed_files = list(processed_dir.glob("whatsapp-*.csv"))
        if not processed_files:
            logger.error(f"No processed files found in {processed_dir}")
            return
        input_file = sorted(processed_files)[-1]  # Get the most recent file
        logger.info(f"Using most recent processed file: {input_file}")

    if not input_file.exists():
        logger.error(f"Data file {input_file} not found")
        return

    # Create output directory if specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = processed_dir / "network_analysis"

    # Run analysis
    analyzer = analyze_whatsapp_network(
        data_path=input_file,
        response_window=response_window,
        time_window=time_window,
        time_overlap=time_overlap,
        edge_weight_multiplier=edge_weight,
        min_edge_weight=min_edge_weight,
        output_dir=output_dir,
    )

    # Show appropriate interface
    if streamlit:
        logger.info("Launching Streamlit interface")
        # Launch Streamlit directly with the correct command
        streamlit_cmd = [
            sys.executable,  # Use the same Python interpreter
            "-m", "streamlit", 
            "run", 
            str(Path(__file__).parent.parent.parent / "streamlit_app.py")
        ]
        logger.info(f"Running: {' '.join(streamlit_cmd)}")
        subprocess.run(streamlit_cmd)
        return None
    elif interactive:
        logger.info("Displaying interactive visualizations")
        analyzer.visualize_graph()
        analyzer.visualize_time_series()
        logger.success("Analysis complete!")
        return analyzer
    else:
        logger.success("Analysis complete! Data exported to {}".format(output_dir))
        return analyzer


if __name__ == "__main__":
    # Check if running via Streamlit
    if __name__ == "__main__":
        if "--streamlit" in __import__("sys").argv:
            # Launch Streamlit app directly
            import subprocess
            import sys
            subprocess.run(["streamlit", "run", "streamlit_app.py"])
        else:
            # Run as CLI
            main()
