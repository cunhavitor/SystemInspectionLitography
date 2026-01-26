#!/bin/bash
# Quick verification script for SKU-specific reference images

echo "========================================"
echo "SKU Reference Images - Status Check"
echo "========================================"
echo ""

# Check Oleo reference
echo "1. Checking Bom Petisco Oleo - rr125..."
if [ -f "models/can_reference/aligned_can_reference448_bpo-rr125.png" ]; then
    size=$(ls -lh models/can_reference/aligned_can_reference448_bpo-rr125.png | awk '{print $5}')
    echo "   ✅ Reference image exists (${size})"
else
    echo "   ❌ Reference image MISSING!"
fi

# Check Azeite reference
echo ""
echo "2. Checking Bom Petisco Azeite - rr125..."
if [ -f "models/can_reference/aligned_can_reference448_bpAz-rr125.png" ]; then
    size=$(ls -lh models/can_reference/aligned_can_reference448_bpAz-rr125.png | awk '{print $5}')
    echo "   ✅ Reference image exists (${size})"
    echo "   🎉 All reference images are ready!"
else
    echo "   ⚠️  Reference image NOT FOUND"
    echo ""
    echo "   To create this reference image:"
    echo "   1. Place a good quality Azeite can in camera view"
    echo "   2. Run: python3 capture_reference.py"
    echo "   3. Press 's' to capture"
    echo "   4. Rename: mv aligned_can_reference448.png \\"
    echo "              models/can_reference/aligned_can_reference448_bpAz-rr125.png"
    echo ""
    echo "   OR use update_reference.py if available"
fi

echo ""
echo "========================================"
echo "Models Check"
echo "========================================"
echo ""

# Check models
for sku in "bpo_rr125_patchcore_v2" "bpAz_rr125_patchcore_v2"; do
    if [ -d "models/$sku" ]; then
        echo "✅ models/$sku"
    else
        echo "❌ models/$sku NOT FOUND"
    fi
done

echo ""
echo "========================================"
echo "Code Verification"
echo "========================================"
echo ""

# Syntax check
python3 -m py_compile src/gui/inspection_window.py 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Python syntax OK"
else
    echo "❌ Python syntax errors!"
    python3 -m py_compile src/gui/inspection_window.py
fi

# Check for SKU mapping
if grep -q "sku_reference_map" src/gui/inspection_window.py; then
    echo "✅ SKU reference mapping found in code"
else
    echo "❌ SKU reference mapping NOT found in code"
fi

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo ""

oleo_ok=false
azeite_ok=false

[ -f "models/can_reference/aligned_can_reference448_bpo-rr125.png" ] && oleo_ok=true
[ -f "models/can_reference/aligned_can_reference448_bpAz-rr125.png" ] && azeite_ok=true

if $oleo_ok && $azeite_ok; then
    echo "🎉 System is FULLY configured"
    echo "   You can now use both SKUs with proper alignment"
elif $oleo_ok; then
    echo "⚠️  System is PARTIALLY configured"
    echo "   - Oleo: Ready ✅"
    echo "   - Azeite: Reference image needed ⚠️"
else
    echo "❌ System needs configuration"
    echo "   Both reference images are missing"
fi

echo ""
