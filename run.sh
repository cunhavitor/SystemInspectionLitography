#!/bin/bash

# Activate virtual environment
source ./venv/bin/activate

# Run with standard python (picamera2 handles camera directly)
echo "Starting Inspection Vision Camera..."
exec python3 main.py "$@"
