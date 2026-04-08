import timm
import torch
import os

MODELS = [
    ("convnextv2_tiny",  "convnextv2_tiny.fcmae_ft_in22k_in1k"),
    ("convnextv2_base",  "convnextv2_base.fcmae_ft_in22k_in1k"),
    ("efficientnetv2_s", "tf_efficientnetv2_s.in21k_ft_in1k"),
    ("swin_tiny",        "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"),
    ("maxvit_tiny",      "maxvit_tiny_tf_224.in1k"),
    ("clip_vit_b16",     "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k"),
]

os.makedirs("weights", exist_ok=True)
for key, name in MODELS:
    print(f"downloading {key}...")
    m = timm.create_model(name, pretrained=True)
    torch.save(m.state_dict(), f"weights/{key}.pth")
    print(f"  saved weights/{key}.pth")
