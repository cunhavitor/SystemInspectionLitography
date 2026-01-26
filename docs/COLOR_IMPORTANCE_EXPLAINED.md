# Are Colors Important in Reference Images?

**Short Answer:** YES, very important! 🎨

**Date:** 2026-01-26  
**Context:** Using bpo (yellow) reference for bpAz (green) cans

---

## 🎨 Color Analysis

### Your Products Have VERY Different Colors!

| Component | bpo (Oleo) | bpAz (Azeite) | Difference |
|-----------|------------|---------------|------------|
| **Blue Channel** | 48.9 | 207.2 | **+158.3** ❌ HUGE! |
| **Green Channel** | 149.9 | 212.8 | **+62.9** ❌ LARGE! |
| **Red Channel** | 167.4 | 210.0 | **+42.6** ❌ LARGE! |

**Label Colors:**
- **bpo (Oleo):** Yellow-dominant label 🟡
- **bpAz (Azeite):** Green-dominant label 🟢

This is a **MAJOR color difference**! Think yellow vs green labels.

---

## 🔍 Why Colors Matter

### 1. For Alignment (CanAligner) ✅
```
Color Impact: MINIMAL
Alignment method: Uses GEOMETRIC features (edges, corners, shapes)
Result: Using bpo reference for bpAz cans → Alignment WORKS ✅
```

**Good News:** Alignment doesn't care much about colors!

### 2. For Anomaly Detection (PatchCore Model) ⚠️
```
Color Impact: SIGNIFICANT!
Model learns: BOTH shape AND color patterns
Result: Color mismatch → Model thinks it's an anomaly ⚠️
```

**Challenge:** The model sees unexpected colors as defects!

---

## 🎯 What PatchCore Model Learned

The model was trained on images with:
- ✅ Specific brightness levels (~125)
- ✅ Specific color patterns (likely yellow Oleo labels)
- ✅ What "normal" vs "defective" looks like

**When inspecting Azeite cans:**

### Scenario A: Using bpo (yellow) Reference (Current Fix)
```
1. Camera captures GREEN Azeite can
2. Alignment to YELLOW reference happens
3. Model sees the image and thinks:
   - "Expected yellow label (from training)"
   - "But this has different colors..."
   - "Color anomaly detected!"
4. Score: Moderate (2-5) instead of low (0.5-2.0)
```

**Better than dark reference, but not perfect!**

### Scenario B: Using Proper bpAz (Green) Reference
```
1. Camera captures GREEN Azeite can
2. Alignment to GREEN reference happens
3. Model sees the image:
   - IF trained on Azeite: "Looks good!" → Score: LOW ✅
   - IF only trained on Oleo: "Wrong color" → Score: MEDIUM ⚠️
```

**Depends on what model was trained on!**

---

## 🔬 Two Separate Problems in Your Case

### Problem 1: BRIGHTNESS (Primary - FIXED ✅)
```
Old bpAz reference: Mean brightness = 111.5 (TOO DARK)
Current fix (bpo): Mean brightness = 125.35 ✅

Impact: HUGE - Dark images → Very high scores (10-15)
Status: FIXED by using bpo reference
```

### Problem 2: COLOR MISMATCH (Secondary - Still Present ⚠️)
```
bpo reference: Yellow label 🟡
bpAz cans: Green label 🟢

Impact: MODERATE - Color difference → Medium scores (2-5)
Status: Not fully addressed yet
```

---

## 📊 Expected Test Results

### What You'll Likely See After Current Fix:

**If Model Was Trained on BOTH Yellow and Green Cans:**
```
✅ Scores: 0.5 - 2.0 (EXCELLENT)
✅ Good cans pass
✅ Only real defects flagged
Result: Problem fully solved! 🎉
```

**If Model Was Trained ONLY on Yellow (Oleo) Cans:**
```
⚠️  Scores: 2.0 - 5.0 (BETTER, but not ideal)
⚠️  Some false positives
⚠️  Green color seen as "different"
Result: Better than before (10-15), but not perfect
```

**If Scores Still 10-15:**
```
❌ Color AND brightness both problematic
❌ Model might have other issues
Result: Need deeper investigation
```

---

## ✅ Action Plan Based on Test Results

### Test Now (with bpo reference for both):

#### If Scores Are 0.5-2.0: 🎉
```
✅ Brightness was THE problem
✅ Color difference doesn't matter much
✅ Keep using this setup (works well enough)
✅ Eventually recapture bpAz with good lighting for consistency
```

#### If Scores Are 2.0-5.0: ⚠️
```
✅ Brightness problem solved
⚠️  Color mismatch causing some elevation
📋 Action: Recapture bpAz reference with:
   - Good lighting (brightness ~125)
   - Actual GREEN Azeite can
   - Should reduce scores further
```

#### If Scores Still 5.0+: ❌
```
❌ Multiple issues present
Possible causes:
   1. Model wasn't trained on Azeite variant
   2. Model needs retraining with both products
   3. Different preprocessing needed
📋 Action: May need separate model for Azeite
```

---

## 🎨 The Color Science

### How PatchCore Sees Colors

```python
# Model learns pixel distributions like:
Normal Yellow Can:
  - Red:   140-170
  - Green: 140-160  
  - Blue:  40-60

When it sees Green Can:
  - Red:   200-220  ← Different!
  - Green: 200-220  ← Different!
  - Blue:  200-220  ← Different!

Result: "This doesn't match my training! → Anomaly!"
```

### Why This Happens

PatchCore uses **patch-based features** that include:
- ✅ Texture patterns
- ✅ **Color information**
- ✅ Edge features
- ✅ Spatial relationships

**Color shifts = Detected as anomalies**

---

## 🔄 Solutions Ranked by Priority

### 1. Test Current Setup (NOW) ⭐⭐⭐
```bash
# Already done!
# Both SKUs use bpo reference
# This fixes brightness issue
```

### 2. Recapture bpAz Reference (SOON) ⭐⭐
```bash
# If scores are 2-5
# Capture GREEN can with GOOD lighting
# Brightness ~125, actual Azeite colors
```

### 3. Verify Model Training Data (IF NEEDED) ⭐
```
# Check what model was trained on
# If only Oleo: Explains color sensitivity
# Might need Azeite training examples
```

### 4. Train Separate Model (LAST RESORT)
```
# Only if color mismatch persists
# Train dedicated Azeite model
# Or retrain unified model with both variants
```

---

## 📝 Key Takeaways

1. **YES, colors ARE important** 🎨
   - PatchCore learns color patterns
   - Yellow vs Green is a BIG difference
   - Color shifts trigger anomaly detection

2. **But BRIGHTNESS was your main problem** 💡
   - Dark reference (111.5) → High scores
   - Fixed with bpo reference (125.35)
   - Should see major improvement

3. **Color might cause MODERATE elevation** ⚠️
   - Using yellow ref for green cans
   - Scores might be 2-5 instead of 0.5-2
   - Still usable, just not perfect

4. **Test results will tell us everything** 🧪
   - Low scores (< 2): Brightness was it! ✅
   - Medium scores (2-5): Color also matters ⚠️
   - High scores (> 5): Deeper issues ❌

---

## 🎯 Bottom Line

**Colors ARE important**, but in your case:
- **Primary issue:** Brightness (dark reference) ← FIXED ✅
- **Secondary issue:** Color (yellow vs green) ← May need addressing

**Current solution addresses the main problem.**  
**Test and see if the color difference causes issues!**

If scores drop from 10-15 to 2-4, your fix is working!  
If they drop to 0.5-2, it's perfect!  
If still high, we'll investigate further.

---

## 🚀 Next Step

**Run the test and check the scores!** 📊

The results will tell us if color is a problem or not.
