# Quick Fix Applied - Using bpo Reference for Both SKUs

**Date:** 2026-01-26 12:19  
**Status:** ✅ FIX APPLIED  
**Type:** Temporary solution for testing

---

## ✅ What Was Done

Both SKUs now use the **same reference image** (the good bpo-rr125 reference):

```
Bom Petisco Oleo - rr125    → aligned_can_reference448_bpo-rr125.png
Bom Petisco Azeite - rr125  → aligned_can_reference448_bpo-rr125.png (COPY)
```

### Verification
```
✅ Both references are IDENTICAL
✅ Mean brightness: 125.35 (good level)
✅ Dimensions: 448x448x3
✅ Dark bpAz reference backed up
```

---

## 📦 Files Status

| File | Status | Purpose |
|------|--------|---------|
| `aligned_can_reference448_bpo-rr125.png` | ✅ Original | Oleo reference (320K) |
| `aligned_can_reference448_bpAz-rr125.png` | ✅ Copy of bpo | Now using bpo reference |
| `aligned_can_reference448_bpAz-rr125.png.dark_backup` | 💾 Backup | Original dark reference (275K) |

---

## 🧪 Next Steps - TESTING

### Test 1: Verify Fix Works
1. **Create new OP:**
   - Open application
   - Click "Nova OP"
   - Select: "Bom Petisco Azeite - rr125"
   - Enter OP details
   - Click "Criar OP"

2. **Check console output:**
   ```
   Loading model for SKU 'Bom Petisco Azeite - rr125'...
   ✓ PatchCore model loaded successfully
     Default threshold: 10.0 → User threshold: 1.8
   ✓ Aligner loaded with SKU-specific reference: 
     models/can_reference/aligned_can_reference448_bpAz-rr125.png
   ```

3. **Run inspection with Azeite cans:**
   - Place Azeite sheet in camera
   - Press 'T' to trigger inspection
   - **CHECK SCORES** in console and graph

### Expected Results

**If reference was the problem (expected):**
```
✅ Anomaly scores: 0.5 - 2.0 (LOW)
✅ Good cans pass inspection
✅ Only real defects flagged
✅ Similar scores to Oleo inspection
```

**If scores still high:**
```
❌ Anomaly scores: Still 5-15 (HIGH)
⚠️  Problem is NOT the reference
⚠️  Issue might be:
   - Model wasn't trained on Azeite variant
   - Different preprocessing needed
   - Model/product incompatibility
```

---

## 🔄 Temporary vs Permanent Solution

### Current Status: TEMPORARY
This fix uses the same reference for both products. This works IF:
- ✅ Can shapes are identical
- ✅ Only labels differ
- ✅ Alignment doesn't depend on label features

### Limitations:
- ⚠️ Not product-specific alignment
- ⚠️ May have slight accuracy reduction
- ⚠️ Should be replaced with proper bpAz reference

---

## ✅ Permanent Solution

Once you confirm this fixes the high scores:

### Step 1: Prepare for Capture
- Use **same lighting** as bpo reference
- Same time of day
- Same camera settings
- Document the setup!

### Step 2: Capture New bpAz Reference
```bash
# Using capture tool
python3 capture_reference.py

# OR using update tool if available
python3 update_reference.py
```

### Step 3: Verify Quality
```python
import cv2, numpy as np

img = cv2.imread('new_bpAz_reference.png')
mean = np.mean(img)

print(f"Brightness: {mean:.1f}")
print(f"Target: 125 ± 10")
print(f"Status: {'✅ GOOD' if 115 < mean < 135 else '❌ RECAPTURE'}")

# Check dark pixels
dark_pct = (np.sum(img < 30) / img.size) * 100
print(f"Dark pixels: {dark_pct:.1f}%")
print(f"Status: {'✅ GOOD' if dark_pct < 20 else '❌ TOO DARK'}")
```

### Step 4: Deploy New Reference
```bash
# Move to correct location
cp new_bpAz_reference.png \
   models/can_reference/aligned_can_reference448_bpAz-rr125.png

# Test again
# Scores should still be low
```

---

## 🔙 To Restore Original (Dark) Reference

If you need to revert for any reason:

```bash
cp models/can_reference/aligned_can_reference448_bpAz-rr125.png.dark_backup \
   models/can_reference/aligned_can_reference448_bpAz-rr125.png
```

**Note:** You probably don't want to do this! The dark reference causes high scores.

---

## 📊 Comparison

### Before Fix
```
Reference: Dark bpAz image (brightness: 111.5)
    ↓
Aligned images: Also dark
    ↓
Model: "Too dark! Anomaly!"
    ↓
Scores: 5-15 ❌
```

### After Fix
```
Reference: Good bpo image (brightness: 125.35)
    ↓
Aligned images: Proper brightness
    ↓
Model: "Looks normal!"
    ↓
Scores: 0.5-2.0 ✅
```

---

## 🎯 Success Criteria

The fix is successful if:
- [  ] Azeite inspection scores: < 2.0 for good cans
- [  ] No false positives
- [  ] Scores similar to Oleo inspection
- [  ] Real defects still detected

---

## 📝 Notes

1. **Both SKUs now share reference** - This is OK temporarily
2. **Backup exists** - Original dark reference saved as `.dark_backup`
3. **Easy to revert** - Can restore anytime if needed
4. **Document capture settings** - When creating proper bpAz reference

---

## 🚀 Current System Status

```
SKU Configuration:
├── Bom Petisco Oleo - rr125
│   ├── Model: models/bpo_rr125_patchcore_v2  ✅
│   ├── Reference: aligned_can_reference448_bpo-rr125.png  ✅
│   └── Status: Ready (original setup)
│
└── Bom Petisco Azeite - rr125
    ├── Model: models/bpAz_rr125_patchcore_v2  ✅
    ├── Reference: aligned_can_reference448_bpAz-rr125.png
    │              (now same as bpo)  ✅
    └── Status: Ready for testing
```

**Threshold:** 1.8 (preserved across both SKUs) ✅

---

## ✅ Ready to Test!

1. Start the application
2. Create OP with Azeite SKU
3. Run inspection
4. Check scores → Should be MUCH lower now!

Good luck with testing! 🎉
