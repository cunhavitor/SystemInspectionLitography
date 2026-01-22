#!/bin/bash

# Ensure we are in the script's directory (project root)
cd "$(dirname "$0")"

# Logs directory
mkdir -p logs

# Log startup
echo "Starting Inspection Camera at $(date)" >> logs/startup.log

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found!"
    read -p "Press Enter to exit..."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# set PYTHONPATH to include src
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Run the application
python3 main.py

# Keep terminal open if it crashes (optional, good for debugging)
if [ $? -ne 0 ]; then
    echo "Application crashed with error code $?"
    read -p "Press Enter to close..."
fi
