#!/bin/bash
# Quick test script to verify bpAz_rr125 SKU integration

echo "=== SKU Integration Test ==="
echo ""

echo "1. Checking model files for bpAz_rr125..."
if [ -d "models/bpAz_rr125_patchcore_v2" ]; then
    echo "✓ Model directory exists"
    ls -lh models/bpAz_rr125_patchcore_v2/
else
    echo "✗ Model directory NOT found!"
    exit 1
fi

echo ""
echo "2. Verifying required model files..."
required_files=("model.xml" "model.bin" "threshold_map.npy")
for file in "${required_files[@]}"; do
    if [ -f "models/bpAz_rr125_patchcore_v2/$file" ]; then
        echo "✓ $file exists"
    else
        echo "✗ $file NOT found!"
        exit 1
    fi
done

echo ""
echo "3. Checking Python syntax..."
python3 -m py_compile src/gui/inspection_window.py
if [ $? -eq 0 ]; then
    echo "✓ inspection_window.py syntax OK"
else
    echo "✗ Syntax error in inspection_window.py"
    exit 1
fi

python3 -m py_compile src/inference/patchcore_inference_v2.py
if [ $? -eq 0 ]; then
    echo "✓ patchcore_inference_v2.py syntax OK"
else
    echo "✗ Syntax error in patchcore_inference_v2.py"
    exit 1
fi

echo ""
echo "4. Verifying SKU in code..."
if grep -q "bpAz_rr125" src/gui/inspection_window.py; then
    echo "✓ bpAz_rr125 SKU found in inspection_window.py"
else
    echo "✗ bpAz_rr125 SKU NOT found in code!"
    exit 1
fi

echo ""
echo "=== All Tests Passed! ==="
echo ""
echo "Next steps:"
echo "1. Run the application: ./run_app.sh"
echo "2. Create a new OP and select 'bpAz_rr125' SKU"
echo "3. Verify model loads successfully"
echo "4. Test inspection with bpAz_rr125 product"
