# Threshold Loading Fix - Documentation

## Problem Identified
When switching between SKUs (e.g., from "Bom Petisco Oleo" to "Bom Petisco Azeite"), the system was loading a new PatchCore model but **NOT** applying the user's saved threshold (Limiar Max).

### What Was Happening (BEFORE):
```
User has saved threshold: 1.8 (in config/detection_params.json)
    ↓
User creates new OP with "Bom Petisco Azeite - rr125"
    ↓
load_model_for_sku() creates new PatchCoreInferencer
    ↓
Model loads with DEFAULT threshold: 10.0 ❌
    ↓
User's threshold (1.8) is IGNORED ❌
```

### Result:
- New model uses threshold = 10.0 (too high!)
- All products marked as OK incorrectly
- User's Limiar Max setting was lost

---

## Solution Implemented

### What Happens Now (AFTER):
```
User has saved threshold: 1.8 (in config/detection_params.json)
    ↓
User creates new OP with "Bom Petisco Azeite - rr125"
    ↓
load_model_for_sku() creates new PatchCoreInferencer
    ↓
Model loads with default: 10.0
    ↓
System applies user's threshold: 1.8 ✅
    ↓
inferencer.threshold = 1.8 ✅
```

### Code Changes
**File:** `src/gui/inspection_window.py`  
**Method:** `load_model_for_sku()` (lines 945-965)

```python
# After loading the model
self.inferencer = PatchCoreInferencer(model_dir=model_dir)

# NEW: Apply user's saved threshold
if hasattr(self, 'threshold_spinbox'):
    user_threshold = self.threshold_spinbox.value()
    self.inferencer.threshold = user_threshold
    print(f"Default threshold: 10.0 → User threshold: {user_threshold}")
```

---

## How It Works

### Scenario 1: Switching SKUs During Operation
```
1. User has threshold = 1.8 configured
2. Currently using "Bom Petisco Oleo - rr125"
3. User creates new OP for "Bom Petisco Azeite - rr125"
4. load_model_for_sku() is called
5. New model is loaded
6. threshold_spinbox.value() returns 1.8
7. New model's threshold is set to 1.8 ✅
```

### Scenario 2: Application Startup
```
1. Application starts
2. load_op_state() restores previous OP
3. load_model_for_sku() is called
4. Model is loaded (threshold_spinbox doesn't exist yet)
5. Later, _setup_ui() creates threshold_spinbox
6. load_config() loads threshold = 1.8
7. Lines 1364-1365 apply the threshold:
   self.threshold_spinbox.setValue(saved_threshold)
   self.inferencer.threshold = saved_threshold ✅
```

---

## Current Configuration

**Saved Threshold:** `1.8` (config/detection_params.json)

**Available SKUs:**
- Bom Petisco Oleo - rr125 → models/bpo_rr125_patchcore_v2
- Bom Petisco Azeite - rr125 → models/bpAz_rr125_patchcore_v2

---

## Verification

### Test Case 1: Create New OP with Different SKU
```
Expected Console Output:
Loading model for SKU 'Bom Petisco Azeite - rr125' from 'models/bpAz_rr125_patchcore_v2'...
✓ PatchCore model loaded successfully for SKU 'Bom Petisco Azeite - rr125'
  Default threshold: 10.0 → User threshold: 1.8
```

### Test Case 2: Application Startup
```
Expected Console Output:
Restoring OP state for SKU: Bom Petisco Oleo - rr125
Loading model for SKU 'Bom Petisco Oleo - rr125' from 'models/bpo_rr125_patchcore_v2'...
✓ PatchCore model loaded successfully for SKU 'Bom Petisco Oleo - rr125'
  Threshold: 10.0 (will be updated after UI setup)
[Later during UI setup]
✓ Applied saved threshold: 1.8
```

---

## Benefits

1. ✅ **Consistency:** User's threshold is preserved when switching between products
2. ✅ **No Re-calibration:** Don't need to re-enter threshold after changing SKU
3. ✅ **Correct Detection:** Anomaly detection uses the user's configured threshold immediately
4. ✅ **Visual Feedback:** Graph and bounding boxes reflect correct threshold
5. ✅ **Persistence:** Threshold survives application restarts

---

## Related Files

- `config/detection_params.json` - Stores user's threshold
- `src/gui/inspection_window.py` - Main window logic
  - `load_model_for_sku()` - Loads model and applies threshold (FIXED)
  - `load_config()` - Loads saved threshold from JSON
  - `save_config()` - Saves threshold when user changes it
  - `update_threshold()` - Updates model when spinbox changes

---

## Status: ✅ FIXED

The threshold is now correctly applied when:
- ✅ Creating a new OP with a different SKU
- ✅ Switching models via load_model_for_sku()
- ✅ Restoring state on application startup
- ✅ User manually adjusts the Limiar Max spinbox
