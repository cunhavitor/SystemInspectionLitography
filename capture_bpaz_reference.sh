#!/bin/bash
# Helper script to capture and verify bpAz-rr125 reference image

echo "=========================================="
echo "bpAz-rr125 Reference Capture Helper"
echo "=========================================="
echo ""

# Check if capture script exists
if [ ! -f "capture_reference.py" ]; then
    echo "❌ capture_reference.py not found in current directory"
    echo ""
    echo "Please create this script or capture manually through the application."
    echo "See: docs/CAPTURE_BPAZ_REFERENCE_GUIDE.md for instructions"
    exit 1
fi

echo "PREPARATION CHECKLIST:"
echo "----------------------"
echo "[ ] Good quality Bom Petisco Azeite can placed"
echo "[ ] Same lighting as bpo reference capture"
echo "[ ] Camera positioned correctly"
echo "[ ] Can is centered and visible"
echo ""
read -p "Ready to capture? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled. Prepare and run again when ready."
    exit 0
fi

echo ""
echo "Step 1: Capturing reference image..."
echo "--------------------------------------"
echo "Instructions:"
echo "  - Preview window will open"
echo "  - Adjust can position if needed"
echo "  - Press 's' to save when ready"
echo "  - Press 'q' to quit without saving"
echo ""
read -p "Press Enter to start capture..." 

python3 capture_reference.py

# Check if image was created
if [ ! -f "aligned_can_reference448.png" ]; then
    echo ""
    echo "❌ No image captured. Try again or check capture_reference.py"
    exit 1
fi

echo ""
echo "✅ Image captured!"
echo ""

# Analyze the captured image
echo "Step 2: Quality verification..."
echo "--------------------------------------"

python3 << 'EOF'
import cv2
import numpy as np
import sys

img = cv2.imread('aligned_can_reference448.png')
if img is None:
    print("❌ Cannot load captured image")
    sys.exit(1)

brightness = np.mean(img)
dark_pct = (np.sum(img < 30) / img.size) * 100
white_pct = (np.sum(img > 240) / img.size) * 100

print()
print(f"Image Analysis:")
print(f"  Dimensions: {img.shape}")
print(f"  Brightness: {brightness:.2f} (target: 115-135)")
print(f"  Dark pixels: {dark_pct:.1f}% (target: <20%)")
print(f"  White pixels: {white_pct:.1f}% (should be <40%)")
print()

# Color check
center = img[150:300, 150:300]
b_mean = np.mean(center[:,:,0])
g_mean = np.mean(center[:,:,1])
r_mean = np.mean(center[:,:,2])

print(f"Label Color Analysis:")
print(f"  Blue:  {b_mean:.2f}")
print(f"  Green: {g_mean:.2f}")
print(f"  Red:   {r_mean:.2f}")

if g_mean > b_mean and g_mean > r_mean:
    print("  ✅ Green dominant (correct for Azeite)")
else:
    print("  ⚠️  Not green dominant - verify this is Azeite can")

print()

# Overall assessment
quality_good = True

if img.shape != (448, 448, 3):
    print("❌ FAIL: Wrong dimensions")
    quality_good = False

if brightness < 110 or brightness > 140:
    print("❌ FAIL: Brightness out of range")
    quality_good = False
elif brightness < 115 or brightness > 135:
    print("⚠️  WARNING: Brightness acceptable but not ideal")

if dark_pct > 30:
    print("❌ FAIL: Too many dark pixels")
    quality_good = False
elif dark_pct > 20:
    print("⚠️  WARNING: Dark pixels slightly high")

if white_pct > 40:
    print("❌ FAIL: Too many white pixels (blank/missing can?)")
    quality_good = False

if quality_good:
    print("=" * 60)
    print("✅ QUALITY CHECK PASSED")
    print("=" * 60)
    sys.exit(0)
else:
    print("=" * 60)
    print("❌ QUALITY CHECK FAILED")
    print("=" * 60)
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    # Quality check passed
    echo ""
    echo "Step 3: Deploy reference image..."
    echo "--------------------------------------"
    
    # Backup current bpAz reference
    if [ -f "models/can_reference/aligned_can_reference448_bpAz-rr125.png" ]; then
        timestamp=$(date +%Y%m%d_%H%M%S)
        backup_name="models/can_reference/aligned_can_reference448_bpAz-rr125.png.backup_${timestamp}"
        cp models/can_reference/aligned_can_reference448_bpAz-rr125.png "$backup_name"
        echo "✅ Backed up old reference: $backup_name"
    fi
    
    # Deploy new reference
    cp aligned_can_reference448.png \
       models/can_reference/aligned_can_reference448_bpAz-rr125.png
    
    echo "✅ New bpAz reference deployed!"
    echo ""
    
    # Clean up temp file
    rm aligned_can_reference448.png
    
    echo "=========================================="
    echo "✅ SUCCESS!"
    echo "=========================================="
    echo ""
    echo "The new bpAz-rr125 reference is ready."
    echo ""
    echo "NEXT STEPS:"
    echo "1. Start the application"
    echo "2. Create OP with 'Bom Petisco Azeite - rr125'"
    echo "3. Test inspection with Azeite cans"
    echo "4. Verify alignment is now correct (should be >99%)"
    echo "5. Confirm false positives reduced to <1%"
    echo ""
    
else
    # Quality check failed
    echo ""
    echo "=========================================="
    echo "❌ QUALITY CHECK FAILED"
    echo "=========================================="
    echo ""
    echo "The captured image doesn't meet quality standards."
    echo ""
    echo "SUGGESTIONS:"
    echo "1. Check lighting (should match bpo reference capture)"
    echo "2. Verify can is properly positioned"
    echo "3. Ensure good exposure (not too dark/bright)"
    echo "4. Run this script again to recapture"
    echo ""
    echo "Captured image saved as: aligned_can_reference448.png"
    echo "You can inspect it manually before trying again."
    echo ""
fi
