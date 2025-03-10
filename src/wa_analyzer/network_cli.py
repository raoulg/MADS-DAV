import click
from pathlib import Path
from loguru import logger

from wa_analyzer.network_analysis import analyze_whatsapp_network


@click.command()
@click.argument('data_file', type=click.Path(exists=True, path_type=Path))
@click.option('--response-window', '-r', type=int, default=3600,
              help='Time window in seconds to consider messages as responses (default: 3600 = 1 hour)')
@click.option('--time-window', '-t', type=int, default=60*60*24*30*2,
              help='Size of each time window in seconds (default: 2 months)')
@click.option('--time-overlap', '-o', type=int, default=60*60*24*30,
              help='Overlap between time windows in seconds (default: 1 month)')
@click.option('--edge-weight', '-w', type=float, default=1.0,
              help='Multiplier for edge weights (default: 1.0)')
@click.option('--min-edge-weight', '-m', type=float, default=0.5,
              help='Minimum edge weight to include in the graph (default: 0.5)')
@click.option('--output-dir', '-d', type=click.Path(path_type=Path),
              help='Directory to save output files (optional)')
@click.option('--interactive/--no-interactive', default=True,
              help='Show interactive visualizations (default: True)')
def main(data_file, response_window, time_window, time_overlap, 
         edge_weight, min_edge_weight, output_dir, interactive):
    """Analyze WhatsApp chat data as a network of users.
    
    DATA_FILE should be a CSV file with preprocessed WhatsApp messages.
    """
    logger.info(f"Analyzing WhatsApp network from {data_file}")
    
    # Create output directory if specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    analyzer = analyze_whatsapp_network(
        data_path=data_file,
        response_window=response_window,
        time_window=time_window,
        time_overlap=time_overlap,
        edge_weight_multiplier=edge_weight,
        min_edge_weight=min_edge_weight,
        output_dir=output_dir
    )
    
    # Show interactive visualizations if requested
    if interactive:
        logger.info("Displaying interactive visualizations")
        analyzer.visualize_graph()
        analyzer.visualize_time_series()
    
    logger.success("Analysis complete!")


if __name__ == "__main__":
    main()
