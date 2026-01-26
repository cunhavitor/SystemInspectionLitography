#!/bin/bash
# Quick fix: Use bpo reference for both SKUs (temporary solution)

echo "========================================"
echo "Quick Fix for High Scores Issue"
echo "========================================"
echo ""
echo "This script will temporarily use the bpo-rr125 reference"
echo "for BOTH SKUs to test if the dark bpAz reference is causing"
echo "the high anomaly scores."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Step 1: Backing up current bpAz reference..."
if [ -f "models/can_reference/aligned_can_reference448_bpAz-rr125.png" ]; then
    cp models/can_reference/aligned_can_reference448_bpAz-rr125.png \
       models/can_reference/aligned_can_reference448_bpAz-rr125.png.dark_backup
    echo "✅ Backup created: aligned_can_reference448_bpAz-rr125.png.dark_backup"
else
    echo "❌ bpAz reference not found!"
    exit 1
fi

echo ""
echo "Step 2: Copying bpo reference to bpAz..."
cp models/can_reference/aligned_can_reference448_bpo-rr125.png \
   models/can_reference/aligned_can_reference448_bpAz-rr125.png
echo "✅ bpAz now uses bpo reference"

echo ""
echo "Step 3: Verifying..."
if diff models/can_reference/aligned_can_reference448_bpo-rr125.png \
        models/can_reference/aligned_can_reference448_bpAz-rr125.png > /dev/null 2>&1; then
    echo "✅ Files are identical - fix applied successfully"
else
    echo "❌ Files are different - something went wrong"
    exit 1
fi

echo ""
echo "========================================"
echo "✅ Quick Fix Applied!"
echo "========================================"
echo ""
echo "NEXT STEPS:"
echo "1. Create a new OP with 'Bom Petisco Azeite - rr125'"
echo "2. Run inspection with actual Azeite cans"
echo "3. Check if anomaly scores are now LOWER"
echo ""
echo "EXPECTED RESULTS:"
echo "- If scores are LOW (< 2.0): Problem was the dark reference ✅"
echo "- If scores still HIGH: Problem is elsewhere (model/other) ⚠️"
echo ""
echo "TO RESTORE ORIGINAL bpAz REFERENCE:"
echo "  cp models/can_reference/aligned_can_reference448_bpAz-rr125.png.dark_backup \\"
echo "     models/can_reference/aligned_can_reference448_bpAz-rr125.png"
echo ""
echo "For permanent fix, recapture bpAz reference with proper lighting."
echo "See: docs/REFERENCE_IMAGE_PROBLEM_ANALYSIS.md"
echo ""
