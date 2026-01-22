#!/bin/bash

# Ensure we are in the script's directory
cd "$(dirname "$0")"

echo "========================================"
echo "   Inspection System Updater"
echo "========================================"

# Check for internet connection (simple ping)
if ! ping -c 1 8.8.8.8 &> /dev/null; then
    echo "❌ Error: No internet connection."
    read -p "Press Enter to exit..."
    exit 1
fi

# Pull latest changes
echo "⬇️  Pulling latest version from Git..."
if git pull; then
    echo "✅ Code updated successfully."
else
    echo "❌ Error updating code. Check git status."
    read -p "Press Enter to exit..."
    exit 1
fi

# Update dependencies
echo "📦 Checking dependencies..."
if [ -d "venv" ]; then
    source venv/bin/activate
    pip install -r requirements.txt
    echo "✅ Dependencies checked."
else
    echo "⚠️  Virtual environment not found. Skipping dependency update."
fi

echo "========================================"
echo "   Update Complete!"
echo "========================================"
read -p "Press Enter to close..."
