import sys
import os

print(f"Current Working Directory: {os.getcwd()}")
print(f"Python Executable: {sys.executable}")
print(f"Sys Path: {sys.path}")

try:
    import src
    print(f"src module file: {src.__file__}")
except ImportError:
    print("Could not import src")

try:
    import src.inference.patchcore_inference_v2 as pi
    print(f"PatchCoreInferencer file: {pi.__file__}")
except ImportError as e:
    print(f"Could not import patchcore_inference_v2: {e}")

try:
    from src.inference.patchcore_inference_v2 import PatchCoreInferencer
    p = PatchCoreInferencer(model_dir="models/patchcore448_v2")
    if hasattr(p, 'bias_map') and p.bias_map is not None:
         print("Bias Map Loaded OK")
    else:
         print("Bias Map NOT loaded")
except Exception as e:
    print(f"Error instantiating inferencer: {e}")
