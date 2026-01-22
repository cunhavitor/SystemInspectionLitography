
import onnx
import sys

model_path = "models/padim448/model.onnx"

try:
    print(f"Loading model: {model_path}...")
    model = onnx.load(model_path)
    
    print("Checking model...")
    onnx.checker.check_model(model)
    print("✓ Model Check Passed")
    
    # Check external data
    # print("External Data Info:")
    # for tensor in onnx.external_data_helper.load_external_data_for_model(model, base_dir="models/padim448/"):
    #     pass
        
except Exception as e:
    print(f"Error: {e}")
