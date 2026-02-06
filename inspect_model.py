
import openvino.runtime as ov
import sys

MODEL_PATH = "models/bpo_rr125_patchcore_resnet50/model.xml"

def main():
    print(f"Inspecting model: {MODEL_PATH}")
    core = ov.Core()
    try:
        model = core.read_model(model=MODEL_PATH)
    except Exception as e:
        print(f"Error reading model: {e}")
        return

    print("--- INPUTS ---")
    for input_node in model.inputs:
        print(f"Name: {input_node.any_name}, Type: {input_node.element_type}, Shape: {input_node.shape}")

    print("\n--- OUTPUTS ---")
    for i, output_node in enumerate(model.outputs):
        print(f"[{i}] Name: {output_node.any_name}, Type: {output_node.element_type}, Shape: {output_node.shape}")

if __name__ == "__main__":
    main()
