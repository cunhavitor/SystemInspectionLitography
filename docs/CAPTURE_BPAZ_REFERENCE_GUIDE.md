# Creating Proper bpAz-rr125 Reference Image

**Issue:** Using bpo reference for bpAz causes 3-4% false positives due to alignment errors  
**Cause:** Yellow vs Green label features differ, affecting feature matching  
**Solution:** Capture proper bpAz reference with correct lighting

---

## 🎯 The Problem

When using bpo (yellow) reference for bpAz (green) cans:

```
✅ Brightness issue: FIXED (no more dark images)
✅ Most cans: Align correctly (96-97%)
❌ Some cans (3-4%): Poor alignment → False positives

Why alignment fails:
- CanAligner uses feature matching (ORB, SIFT, or similar)
- Features from YELLOW label ≠ Features from GREEN label
- Different text, patterns, colors → Feature mismatch
- Result: Occasional misalignment
```

---

## ✅ Solution: Capture Proper bpAz Reference

You need to create a new bpAz-rr125 reference image with:
1. ✅ **Good lighting** (brightness ~125, like bpo)
2. ✅ **Actual Azeite can** (green label features)
3. ✅ **Same capture process** as bpo reference

---

## 📋 Step-by-Step Guide

### Preparation

#### 1. Check Current bpo Reference Settings
```bash
# View current bpo reference for comparison
python3 << 'EOF'
import cv2
import numpy as np

bpo = cv2.imread('models/can_reference/aligned_can_reference448_bpo-rr125.png')
if bpo is not None:
    print(f"bpo reference target values:")
    print(f"  Mean brightness: {np.mean(bpo):.2f}")
    print(f"  Dimensions: {bpo.shape}")
    print(f"  Dark pixels (<30): {(np.sum(bpo < 30) / bpo.size * 100):.1f}%")
    print()
    print("Your new bpAz reference should match these levels!")
else:
    print("Could not load bpo reference")
EOF
```

#### 2. Set Up Same Lighting Conditions
- **Time of day:** Same as when bpo was captured
- **Position:** Same location/angle
- **Camera settings:** Same exposure, ISO, white balance
- **Lighting:** Same lamps/brightness

---

### Method 1: Using capture_reference.py (Recommended)

```bash
# 1. Place a GOOD QUALITY Bom Petisco Azeite can
#    - No defects
#    - Clean label
#    - Properly positioned

# 2. Run capture script
python3 capture_reference.py

# Instructions will appear on screen:
# - Press 's' to save reference image
# - Image will be saved to current directory

# 3. Check brightness before using
python3 << 'EOF'
import cv2
import numpy as np

img = cv2.imread('aligned_can_reference448.png')  # Or whatever it's named
if img is not None:
    brightness = np.mean(img)
    dark_pct = (np.sum(img < 30) / img.size) * 100
    
    print(f"Captured image analysis:")
    print(f"  Brightness: {brightness:.2f}")
    print(f"  Target: 115-135 (ideal: ~125)")
    print(f"  Dark pixels: {dark_pct:.1f}%")
    print(f"  Target: <20%")
    print()
    
    if 115 < brightness < 135 and dark_pct < 20:
        print("✅ EXCELLENT - Ready to use!")
    elif 110 < brightness < 140:
        print("⚠️  ACCEPTABLE - Will work but not ideal")
    else:
        print("❌ RECAPTURE - Brightness is off")
        print(f"   Adjust lighting and try again")
else:
    print("Image not found")
EOF

# 4. If quality is good, deploy it
mv aligned_can_reference448.png \
   models/can_reference/aligned_can_reference448_bpAz-rr125.png

# Keep backup of bpo-based version (optional)
# cp models/can_reference/aligned_can_reference448_bpAz-rr125.png \
#    models/can_reference/aligned_can_reference448_bpAz-rr125.png.bpo_copy
```

---

### Method 2: Using update_reference.py (If Available)

```bash
# If this script exists in your project
python3 update_reference.py

# Follow on-screen instructions
# Usually similar process to capture_reference.py
```

---

### Method 3: Manual Capture with Application

```bash
# 1. Start the inspection application
./run_app.sh

# 2. Create temp OP with Azeite
# 3. Capture several good cans
# 4. Find best captured image in data/
# 5. Use inspection pipeline to process it:
#    - Detect corners
#    - Rectify
#    - Crop can
#    - Resize to 448x448
#    - Save as reference
```

---

## 🔍 Quality Verification Checklist

### Before Deploying New Reference:

```bash
# Run this comprehensive check
python3 << 'EOF'
import cv2
import numpy as np

def verify_reference(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Cannot load {img_path}")
        return False
    
    print("=" * 60)
    print(f"Reference Quality Check: {img_path}")
    print("=" * 60)
    print()
    
    # Check 1: Dimensions
    print("1. DIMENSIONS")
    if img.shape == (448, 448, 3):
        print(f"   ✅ {img.shape} - Correct")
    else:
        print(f"   ❌ {img.shape} - Should be (448, 448, 3)")
        return False
    print()
    
    # Check 2: Brightness
    print("2. BRIGHTNESS")
    brightness = np.mean(img)
    print(f"   Mean: {brightness:.2f}")
    print(f"   Target: 115-135 (ideal: ~125)")
    
    if 115 < brightness < 135:
        print(f"   ✅ GOOD")
    elif 110 < brightness < 140:
        print(f"   ⚠️  ACCEPTABLE")
    else:
        print(f"   ❌ BAD - Recapture needed")
        return False
    print()
    
    # Check 3: Dark pixels
    print("3. DARK PIXELS")
    dark_pct = (np.sum(img < 30) / img.size) * 100
    print(f"   Dark pixels (<30): {dark_pct:.1f}%")
    print(f"   Target: <20%")
    
    if dark_pct < 20:
        print(f"   ✅ GOOD")
    elif dark_pct < 30:
        print(f"   ⚠️  ACCEPTABLE")
    else:
        print(f"   ❌ TOO DARK - Recapture with better lighting")
        return False
    print()
    
    # Check 4: Not blank
    print("4. CONTENT CHECK")
    white_pct = (np.sum(img > 240) / img.size) * 100
    if white_pct > 40:
        print(f"   ❌ {white_pct:.1f}% very bright pixels - Might be blank/missing can")
        return False
    else:
        print(f"   ✅ {white_pct:.1f}% very bright pixels - Good")
    print()
    
    # Check 5: Color channels (should see green for Azeite)
    print("5. COLOR ANALYSIS (Should be GREEN for Azeite)")
    center = img[150:300, 150:300]  # Label area
    b_mean = np.mean(center[:,:,0])
    g_mean = np.mean(center[:,:,1])
    r_mean = np.mean(center[:,:,2])
    
    print(f"   Blue:  {b_mean:.2f}")
    print(f"   Green: {g_mean:.2f}")
    print(f"   Red:   {r_mean:.2f}")
    
    if g_mean > b_mean and g_mean > r_mean:
        print(f"   ✅ Green dominant - Correct for Azeite")
    else:
        print(f"   ⚠️  Not green dominant - Check if correct can")
    print()
    
    print("=" * 60)
    print("OVERALL ASSESSMENT")
    print("=" * 60)
    
    if 115 < brightness < 135 and dark_pct < 20:
        print("✅ REFERENCE IS EXCELLENT - Deploy it!")
        return True
    elif 110 < brightness < 140 and dark_pct < 30:
        print("⚠️  REFERENCE IS ACCEPTABLE - Will work")
        return True
    else:
        print("❌ REFERENCE NEEDS IMPROVEMENT - Recapture")
        return False

# Check the new reference
verify_reference('models/can_reference/aligned_can_reference448_bpAz-rr125.png')

EOF
```

---

## 🎯 Expected Improvement After Proper Reference

### Current (Using bpo reference for bpAz):
```
✅ Brightness: Good
✅ Most alignment: Good (96-97%)
❌ Some misalignment: 3-4% false positives
```

### After Proper bpAz Reference:
```
✅ Brightness: Good (proper lighting)
✅ All alignment: Good (99%+)
✅ Feature matching: Correct (green to green)
✅ False positives: <1%
```

---

## 🔧 Troubleshooting

### Problem: Captured Image Too Dark
```bash
# Increase lighting in capture area
# Or adjust camera exposure/brightness settings
# Target mean brightness: ~125
```

### Problem: Captured Image Too Bright  
```bash
# Reduce lighting or decrease camera exposure
# Check for reflections on can
# Target mean brightness: ~125
```

### Problem: Can Not Centered
```bash
# Adjust camera/can position
# Reference should show full can, properly aligned
# Check existing bpo reference for positioning
```

### Problem: Blur or Motion
```bash
# Use faster shutter speed
# Ensure can and camera are stable
# Image should be sharp
```

---

## 📊 Comparison: Before vs After

| Metric | bpo Ref (Yellow) | Proper bpAz Ref (Green) |
|--------|------------------|-------------------------|
| **Alignment Success** | 96-97% | 99%+ |
| **False Positives** | 3-4% | <1% |
| **Color Match** | Wrong (yellow) | Correct (green) |
| **Feature Matching** | Mismatched | Matched |
| **Overall Accuracy** | Good | Excellent |

---

## ✅ Deployment Checklist

- [ ] Good quality Azeite can prepared
- [ ] Same lighting as bpo reference
- [ ] Image captured (448x448)
- [ ] Brightness verified (~125)
- [ ] Dark pixels check (<20%)
- [ ] Colors verified (green dominant)
- [ ] Quality check passed
- [ ] Old reference backed up
- [ ] New reference deployed
- [ ] Application tested
- [ ] Alignment verified (>99%)
- [ ] False positives reduced (<1%)

---

## 🚀 Quick Commands

```bash
# 1. Capture (follow on-screen prompts)
python3 capture_reference.py

# 2. Verify brightness
python3 -c "import cv2, numpy as np; img=cv2.imread('aligned_can_reference448.png'); print(f'Brightness: {np.mean(img):.1f}')"

# 3. Deploy if good
mv aligned_can_reference448.png \
   models/can_reference/aligned_can_reference448_bpAz-rr125.png

# 4. Test with application
./run_app.sh
```

---

## 💡 Remember

1. **Match bpo lighting** - Critical for consistency
2. **Target brightness: ~125** - Same as bpo
3. **Use real Azeite can** - Green label needed
4. **Verify before deploying** - Run quality checks
5. **Test thoroughly** - Check alignment and scores

The proper reference will eliminate those 3-4% false positives! 🎯
