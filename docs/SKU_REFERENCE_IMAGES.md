# SKU-Specific Reference Images for Alignment

## Overview
Each SKU now has its own reference image for can alignment. This ensures accurate alignment for different product types (Oleo vs Azeite).

---

## Changes Made

### 1. Renamed Existing Reference Image
**Old name:** `aligned_can_reference448.png`  
**New name:** `aligned_can_reference448_bpo-rr125.png`

```bash
# Executed command:
cd models/can_reference
mv aligned_can_reference448.png aligned_can_reference448_bpo-rr125.png
```

### 2. Updated Code to Support SKU-Specific References

**File:** `src/gui/inspection_window.py`

**Removed:** Hard-coded single reference image loading at initialization (lines 849-860)
```python
# OLD: Single reference for all SKUs
ref_path = 'models/can_reference/aligned_can_reference448.png'
self.aligner = CanAligner(ref_path, target_size=(448, 448))
```

**Added:** SKU-specific reference mapping in `load_model_for_sku()` (lines 933-937)
```python
# NEW: Map each SKU to its own reference image
sku_reference_map = {
    'Bom Petisco Oleo - rr125': 'models/can_reference/aligned_can_reference448_bpo-rr125.png',
    'Bom Petisco Azeite - rr125': 'models/can_reference/aligned_can_reference448_bpAz-rr125.png'
}
```

**Added:** Reference image loading logic (lines 961-972)
```python
# Load SKU-specific reference image for alignment
if ref_path and os.path.exists(ref_path):
    try:
        self.aligner = CanAligner(ref_path, target_size=(448, 448))
        print(f"✓ Aligner loaded with SKU-specific reference: {ref_path}")
    except Exception as e:
        print(f"WARNING: Failed to load aligner for SKU '{sku}': {e}")
        self.aligner = None
else:
    print(f"WARNING: Reference image not found at {ref_path}")
    print(f"  Alignment will be skipped for SKU '{sku}'")
    self.aligner = None
```

---

## Current Status

### ✅ Completed
- **Bom Petisco Oleo - rr125**: Reference image ready
  - Path: `models/can_reference/aligned_can_reference448_bpo-rr125.png`
  - Size: 320K
  - Status: ✅ EXISTS

### ⚠️ Pending
- **Bom Petisco Azeite - rr125**: Reference image NEEDED
  - Expected path: `models/can_reference/aligned_can_reference448_bpAz-rr125.png`
  - Status: ❌ DOES NOT EXIST
  - Action required: Capture reference image for Azeite product

---

## How to Create Reference Image for bpAz-rr125

### Method 1: Using Existing Capture Tool
```bash
# Run the capture reference script
python3 capture_reference.py

# This will:
# 1. Open camera
# 2. Show live preview
# 3. Press 's' to capture
# 4. Save as reference image

# Then rename to correct name:
mv aligned_can_reference448.png models/can_reference/aligned_can_reference448_bpAz-rr125.png
```

### Method 2: Manual Capture Process
1. Place a good quality Bom Petisco Azeite can in the camera view
2. Ensure proper lighting and positioning
3. Capture image with the application
4. Use image processing to:
   - Detect corners
   - Rectify sheet
   - Crop the can
   - Resize to 448x448
   - Apply CLAHE normalization
5. Save as: `models/can_reference/aligned_can_reference448_bpAz-rr125.png`

### Method 3: Copy and Verify
If the cans are identical in shape (only label differs):
```bash
# Copy the Oleo reference as a starting point
cp models/can_reference/aligned_can_reference448_bpo-rr125.png \
   models/can_reference/aligned_can_reference448_bpAz-rr125.png

# Then test with Azeite product
# If alignment is poor, capture a new reference image
```

---

## How It Works

### Workflow
```
1. User creates new OP and selects SKU
    ↓
2. load_model_for_sku() is called
    ↓
3. System maps SKU to model directory
    ↓
4. System maps SKU to reference image path
    ↓
5. PatchCore model is loaded
    ↓
6. User's threshold is applied
    ↓
7. CanAligner is loaded with SKU-specific reference
    ↓
8. ✅ Ready for inspection with correct model AND reference
```

### Example Console Output
```
Loading model for SKU 'Bom Petisco Azeite - rr125' from 'models/bpAz_rr125_patchcore_v2'...
✓ PatchCore model loaded successfully for SKU 'Bom Petisco Azeite - rr125'
  Default threshold: 10.0 → User threshold: 1.8
✓ Aligner loaded with SKU-specific reference: models/can_reference/aligned_can_reference448_bpAz-rr125.png
```

### If Reference Image is Missing
```
Loading model for SKU 'Bom Petisco Azeite - rr125' from 'models/bpAz_rr125_patchcore_v2'...
✓ PatchCore model loaded successfully for SKU 'Bom Petisco Azeite - rr125'
  Default threshold: 10.0 → User threshold: 1.8
WARNING: Reference image not found at models/can_reference/aligned_can_reference448_bpAz-rr125.png
  Alignment will be skipped for SKU 'Bom Petisco Azeite - rr125'
```
**Impact:** Inspection will still work but without alignment (may affect accuracy)

---

## Directory Structure

```
models/
├── can_reference/
│   ├── aligned_can_reference448_bpo-rr125.png     ✅ EXISTS
│   ├── aligned_can_reference448_bpAz-rr125.png    ❌ NEEDED
│   ├── aligned_can_reference448_old2.png          (backup)
│   └── aligned_can_reference_1024.png             (old size)
├── bpo_rr125_patchcore_v2/                        ✅ EXISTS
│   ├── model.xml
│   ├── model.bin
│   └── threshold_map.npy
└── bpAz_rr125_patchcore_v2/                       ✅ EXISTS
    ├── model.xml
    ├── model.bin
    └── threshold_map.npy
```

---

## Benefits

1. ✅ **Accurate Alignment**: Each product uses its own reference for precise alignment
2. ✅ **Better Detection**: Properly aligned images improve anomaly detection accuracy
3. ✅ **Automatic**: System loads correct reference when SKU is selected
4. ✅ **Flexible**: Easy to add new SKUs by adding reference images
5. ✅ **Graceful Fallback**: If reference missing, inspection continues without alignment

---

## Testing Checklist

### For Bom Petisco Oleo - rr125
- [x] Reference image exists
- [x] Model loads successfully
- [x] Aligner loads successfully
- [ ] Test inspection with real Oleo product
- [ ] Verify alignment quality

### For Bom Petisco Azeite - rr125
- [ ] Create reference image for Azeite product
- [ ] Place image in correct location
- [ ] Verify aligner loads successfully
- [ ] Test inspection with real Azeite product
- [ ] Verify alignment quality

---

## Next Steps

1. **REQUIRED:** Create reference image for bpAz-rr125
   - Capture a good quality Azeite can
   - Save to: `models/can_reference/aligned_can_reference448_bpAz-rr125.png`

2. **Recommended:** Test both SKUs
   - Create OP with Oleo → Verify alignment works
   - Create OP with Azeite → Verify alignment works (after adding reference)

3. **Optional:** Document reference capture process
   - Add instructions to team documentation
   - Include quality criteria for reference images

---

## Troubleshooting

### Problem: "WARNING: Reference image not found"
**Solution:** Create the missing reference image at the specified path

### Problem: Alignment quality is poor
**Solutions:**
- Recapture reference image with better lighting
- Ensure can is properly positioned
- Check that reference matches actual product label

### Problem: Different alignment for same SKU
**Solutions:**
- Verify correct reference image is loaded
- Check console output for loaded reference path
- Ensure reference image hasn't been modified

---

## Status: ⚠️ PARTIALLY COMPLETE

✅ Code implementation complete  
✅ Oleo reference image ready  
❌ Azeite reference image needed  

**Action Required:** Add `aligned_can_reference448_bpAz-rr125.png` to enable full functionality.
