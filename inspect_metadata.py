import torch
import os

path = '/home/cunhav/projects/InspectionVisionCamera/models/can_reference/patchcore/metadata_patchcore.pt'
if not os.path.exists(path):
    print(f"File not found: {path}")
else:
    try:
        data = torch.load(path, map_location='cpu')
        print("Keys:", data.keys())
        if 'memory_bank' in data:
            print("Memory Bank Shape:", data['memory_bank'].shape)
        
        # Look for potential projection matrix
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"Tensor '{k}': {v.shape}")
            elif isinstance(v, dict):
                print(f"Dict '{k}': keys {v.keys()}")
                
    except Exception as e:
        print(f"Error loading: {e}")
