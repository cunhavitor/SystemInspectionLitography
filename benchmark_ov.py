
import openvino as ov
import numpy as np
import time
import cv2
import os

# Mock the setup
model_path = "models/can_reference/patchcore448/patchcore_model.xml"

# Function to patch (copy-paste logic for standalone test)
def patch_memory_bank(model, max_vectors=40000):
    large_dim = None
    for op in model.get_ops():
        if op.get_type_name() == "Constant":
            shape = list(op.get_output_tensor(0).get_shape())
            for dim in shape:
                if dim > 40000:
                    large_dim = dim
                    break
            if large_dim:
                break
    
    if not large_dim or large_dim <= max_vectors:
        return

    print(f"Subsampling from {large_dim} to {max_vectors}...")
    indices = np.random.choice(large_dim, max_vectors, replace=False)
    indices.sort()
    
    for op in model.get_ops():
        if op.get_type_name() == "Constant":
            try:
                data = op.data
                shape = data.shape
                if large_dim in shape:
                    axis = shape.index(large_dim)
                    if axis == 0: new_data = data[indices]
                    elif axis == 1: new_data = data[:, indices]
                    else: continue
                    new_const = ov.opset1.constant(new_data, dtype=op.get_element_type())
                    op.output(0).replace(new_const.output(0))
            except: pass

core = ov.Core()
print("Reading model...")
model = core.read_model(model_path)
patch_memory_bank(model, max_vectors=40000)

print("Compiling model (Default)...")
t0 = time.time()
compiled_model = core.compile_model(model, "CPU")
print(f"Compilation took {time.time()-t0:.2f}s")

# Warmup
req = compiled_model.create_infer_request()
input_data = np.random.rand(1, 3, 512, 512).astype(np.float32)
req.infer([input_data])

print("Benchmarking Inference (10 runs)...")
times = []
for _ in range(10):
    t0 = time.time()
    req.infer([input_data])
    times.append(time.time() - t0)

avg_time = sum(times) / len(times)
print(f"Average Inference Time: {avg_time*1000:.2f} ms ({1/avg_time:.1f} FPS)")
