import sys
sys.path.append('src')
from inference.efficientad_inference import EfficientAdInferencer
model_dir = "models/bpo_rr125_efficientAD_M"
inferencer = EfficientAdInferencer(model_dir)
print("Mask shape:", inferencer.can_mask.shape)
