# SKU Integration - bpAz_rr125

## Summary
Successfully integrated the new SKU `bpAz_rr125` into the inspection system. The system now supports dynamic model loading based on the selected SKU during production order creation.

## Changes Made

### 1. SKU Selection (inspection_window.py)
- **Line 448-449**: Added `bpAz_rr125` to the SKU dropdown in `NewOPDialog`
- Users can now select between:
  - `bpo_rr125` (existing)
  - `bpAz_rr125` (new)

### 2. Dynamic Model Loading (inspection_window.py)
- **Lines 918-956**: Created `load_model_for_sku()` method
  - Maps SKU names to model directories
  - `bpo_rr125` → `models/bpo_rr125_patchcore_v2`
  - `bpAz_rr125` → `models/bpAz_rr125_patchcore_v2`
  - Validates model directory exists
  - Loads the appropriate PatchCore model
  - Shows user-friendly error messages if model not found

### 3. OP Creation Integration (inspection_window.py)
- **Lines 1614-1622**: Updated `on_new_op()` method
  - Loads the correct model before creating production order
  - Prevents OP creation if model loading fails
  - Shows confirmation message with loaded model name

### 4. State Restoration (inspection_window.py)
- **Lines 2504-2506**: Updated `load_op_state()` method
  - Automatically loads correct model when restoring saved OP
  - Ensures model consistency across application restarts

## Model Files Verified
```
models/bpAz_rr125_patchcore_v2/
├── model.bin (28M)
├── model.onnx (64K)
├── model.onnx.data (55M)
├── model.xml (55K)
└── threshold_map.npy (785K)
```

## How It Works

1. **User creates new OP**: Selects SKU from dropdown
2. **System validates**: Checks if model directory exists
3. **Model loads**: `PatchCoreInferencer` initialized with correct model path
4. **Inspection runs**: Uses the loaded model for anomaly detection
5. **State persists**: SKU saved with OP data for restoration

## Fallback Behavior
- If unknown SKU is provided, system defaults to `bpo_rr125` model
- If model directory doesn't exist, shows warning and prevents OP creation
- Default SKU for manual inspection (without OP) remains `bpo_rr125`

## Testing Recommendations
1. Create new OP with `bpAz_rr125` SKU
2. Verify model loads successfully
3. Run inspection with bpAz_rr125 product
4. Check that defect images are saved correctly
5. Restart application and verify state restoration

## Notes
- Both models must have the same input/output structure (448x448, same preprocessing)
- Threshold values are model-specific (loaded from each model's configuration)
- Reference images for alignment are currently shared (may need SKU-specific references in future)
