#!/bin/bash

# Function to display the help section
show_help() {
    echo "Usage: ./run_dashboard.sh [OPTION]"
    echo "Run a Python dashboard script."
    echo ""
    echo "Without any OPTION, it defaults to running 'dashboard.py'."
    echo "Options:"
    echo "  1, 2, 3, etc.   Run 'dashboard1.py', 'dashboard2.py', 'dashboard3.py', etc."
    echo "  --help, -h      Display this help and exit."
    echo ""
    echo "Example:"
    echo "  ./run_dashboard.sh 2   # Runs 'dashboard2.py'"
}

# Default script to run if no argument given
SCRIPT="dashboard.py"

# Check if help is requested
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# Check if an argument was provided
if [ $# -gt 0 ]; then
    # Update SCRIPT based on the argument provided
    SCRIPT="dashboard$1.py"
fi

# Inform the user which script will be run
echo "Running $SCRIPT..."

# Execute the Python script
pdm run streamlit run $SCRIPT
