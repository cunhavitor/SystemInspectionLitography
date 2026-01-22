
import onnx
import numpy as np
from onnx import numpy_helper

model_path = "models/padim448/model.onnx"

print(f"Loading {model_path}...")
model = onnx.load(model_path)

print("Checking initializers for NaNs...")
found_nan = False
for initializer in model.graph.initializer:
    try:
        weight = numpy_helper.to_array(initializer)
        if np.isnan(weight).any():
            print(f"⚠ NaN found in {initializer.name} (Shape: {weight.shape})")
            found_nan = True
        if np.isinf(weight).any():
            print(f"⚠ Inf found in {initializer.name} (Shape: {weight.shape})")
            found_nan = True
    except Exception as e:
        print(f"Error checking {initializer.name}: {e}")

if not found_nan:
    print("✓ No NaNs or Infs found in initializers.")
else:
    print("✗ Corruption detected in model weights.")
