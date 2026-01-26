# CRITICAL: Restoring Original Setup

**Date:** 2026-01-26  
**Status:** ❌ Original bpAz reference was lost  
**Action Required:** You need to recreate it

---

## ⚠️ What Happened

The original bpAz reference (that was working for you) was overwritten by our temporary fix today. 

**Current state:**
- Both bpo-rr125 and bpAz-rr125 use the SAME reference (bpo copy)
- Your original bpAz reference is LOST
- No backup was found

---

## ✅ CRITICAL INSIGHT from You

**"The process of preparing an image for inspection has to be the same as the dataset, because I trained the model with the dataset."**

This is the KEY! 

### What This Means:

1. **Your model was trained** with images preprocessed a certain way
2. **Your reference image** must match that EXACT preprocessing
3. **The "dark" reference** we called problematic was actually CORRECT for your use case

### Why It Worked Before:

```
Your Training Data:
  → Captured images
  → Preprocessed with specific pipeline
  → Model trained on THIS distribution

Your Original bpAz Reference:
  → Same preprocessing pipeline
  → Same brightness/color distribution
  → Matched training data → Good results ✅
```

### Why Our "Fix" Made It Worse:

```
Our "Fixed" Reference (bpo copy):
  → Different brightness (125 vs 111)
  → Different preprocessing
  → DOESN'T match training data
  → Model sees unexpected distribution → Bad results ❌
```

---

## 🎯 What You Need to Do

### Step 1: Recreate Your Reference Image

**CRITICAL**: Use the EXACT same preprocessing pipeline as your training data!

```python
# Your reference creation process should be:
# (Fill in with YOUR actual preprocessing steps)

def create_reference_like_training_data(image):
    """
    This must match EXACTLY what you did for training data.
    Common steps might include:
    """
    # 1. Resize? (what size?)
    # 2. CLAHE? (what parameters?)
    # 3. Normalization? (what method?)
    # 4. Color space? (RGB, BGR, Grayscale?)
    # 5. Any other preprocessing?
    
    # Replace with YOUR actual steps:
    processed = image  # Your preprocessing here
    
    return processed
```

### Step 2: Key Questions to Answer

Before recreating, you need to know:

1. **What preprocessing did you use for training data?**
   - CLAHE parameters (if used)?
   - Resize dimensions?
   - Normalization method?
   - Color space?

2. **What was the brightness distribution of training images?**
   - Mean brightness: ~111? ~125? Other?
   - This explains why "dark" reference worked!

3. **What reference creation tool did you use before?**
   - Custom script?
   - Manual process?
   - Application tool?

---

## 📋 Recreating Your Reference

### Method: Match Your Training Pipeline

```bash
# 1. Capture a good Azeite can
# 2. Process it EXACTLY as you processed training images
# 3. Save as reference

# Example (replace with YOUR actual pipeline):
python3 << 'EOF'
import cv2
import numpy as np

# Load captured can
img = cv2.imread('captured_azeite_can.png')

# TODO: Apply YOUR preprocessing pipeline here
# (The same one used for creating training dataset)

# Example (replace with your actual steps):
# processed = cv2.resize(img, (448, 448))
# processed = apply_clahe(processed, clipLimit=2.0, tileGridSize=(8,8))
# processed = normalize(processed, your_method)

# Save as reference
cv2.imwrite('models/can_reference/aligned_can_reference448_bpAz-rr125.png', processed)

print(f"Reference created with brightness: {np.mean(processed):.2f}")
EOF
```

---

## 🔍 Understanding Your Original Setup

### Your Original bpAz Reference Properties:
```
Brightness: ~111.5 (what we called "too dark")
Dark pixels: 32.5%
Color: Green label

BUT this was CORRECT for YOUR model because:
→ Your training data had similar distribution
→ Model learned on this brightness range
→ Reference matched training → Good results ✅
```

### Why We Were Wrong:
```
We assumed:
  ❌ Brightness should be ~125 (standard)
  ❌ Dark pixels should be <20%
  ❌ Higher brightness = better

But YOUR model needs:
  ✅ Brightness matching training data (~111)
  ✅ Preprocessing matching training pipeline
  ✅ Distribution matching what model learned
```

---

## ⚠️ Lesson Learned

**Different models need different preprocessing!**

Your model was trained with a specific pipeline that resulted in:
- Lower brightness (~111)
- More dark pixels (32%)
- Specific color distribution

This is VALID! The reference must match this, not some "standard" values.

---

## 🚀 Action Items

### Immediate (You):
1. **Recreate bpAz reference** using YOUR training pipeline
2. **Document the preprocessing steps** (so this doesn't happen again)
3. **Test with Azeite cans** to verify it works like before

### Important (Us):
1. We should NOT have changed your reference
2. We should have asked about training data preprocessing first
3. Preprocessing must match training data - this is fundamental!

---

## 📝 Preprocessing Documentation Template

Please document your preprocessing pipeline:

```
# My Training Data Preprocessing Pipeline

## Input
- Raw captured image
- Size: _____ x _____
- Format: _____

## Steps
1. Step 1: _____________________
   - Parameters: _____
   
2. Step 2: _____________________
   - Parameters: _____
   
3. Step 3: _____________________
   - Parameters: _____

## Output  
- Final size: 448 x 448
- Brightness range: _____ to _____
- Color space: _____
- Mean brightness: ~_____

## Tools/Scripts Used
- Script: _____________________
- Function: ___________________
```

---

## 🔄 Current System State

```
SKU Configuration:
├── Bom Petisco Oleo - rr125
│   ├── Model: models/bpo_rr125_patchcore_v2  ✅
│   ├── Reference: aligned_can_reference448_bpo-rr125.png  ✅
│   └── Status: Working
│
└── Bom Petisco Azeite - rr125
    ├── Model: models/bpAz_rr125_patchcore_v2  ✅
    ├── Reference: aligned_can_reference448_bpAz-rr125.png  ❌ WRONG
    │              (currently using bpo copy - doesn't match training)
    └── Status: NEEDS RECREATION
```

---

## ✅ Going Forward

**The key principle:**
> Reference image preprocessing MUST match training data preprocessing

This is more important than:
- "Standard" brightness values
- "Best practices" recommendations  
- Any generic preprocessing advice

**Your model, your training data, your preprocessing pipeline = Your rules!**

---

## 🆘 If You Need Help

To recreate your reference properly, we need to know:

1. What preprocessing pipeline did you use for training?
2. What tools/scripts did you use?
3. Can you share a training image example to reverse-engineer the preprocessing?

Once we know YOUR pipeline, we can help recreate the reference correctly.

---

## 🔍 Sorry for the Confusion

We made assumptions about "correct" preprocessing without asking about your training data first. This was our mistake.

The preprocessing that creates your reference MUST match what you used during training, regardless of whether it seems "dark" or "non-standard" to us.
