
import onnxruntime as ort
import numpy as np

model_path = "models/padim448/model.onnx"

try:
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    print("--- Inputs ---")
    for inputs in session.get_inputs():
        print(f"Name: {inputs.name}, Type: {inputs.type}, Shape: {inputs.shape}")

    print("\n--- Outputs ---")
    for outputs in session.get_outputs():
        print(f"Name: {outputs.name}, Type: {outputs.type}, Shape: {outputs.shape}")

except Exception as e:
    print(f"Error loading model: {e}")
