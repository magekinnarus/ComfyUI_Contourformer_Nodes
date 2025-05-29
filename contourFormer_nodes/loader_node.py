# loader_node.py
import torch
import os
import sys

# Ensure ContourFormer's root and src are in sys.path
# This assumes utilities.py has handled the sys.path.insert for CONTOURFORMER_ROOT
# If not, you'd need to replicate the sys.path.insert logic here or in a common parent module.
# For simplicity, we rely on __init__.py setting up the path.

# Import ContourFormer's core components
try:
    from core import YAMLConfig
    # You might need to import specific model builders, e.g., from models.build import build_model
    # The actual structure of ContourFormer's src/models might vary.
    # We'll use the approach from draw.py where cfg.model is already an object.
except ImportError as e:
    print(f"Error importing ContourFormer core components: {e}")
    print("Please ensure the 'ContourFormer' repository is correctly placed inside 'ComfyUI/custom_nodes/ComfyUI_ContourFormer/'")
    print("And that its 'src' directory contains 'core.py' or similar necessary modules.")
    # Define dummy classes if imports fail to allow ComfyUI to load the node gracefully
    class YAMLConfig:
        def __init__(self, *args, **kwargs):
            print("Warning: YAMLConfig dummy class used. ContourFormer core not found.")
            self.model = DummyModelLoaderComponent()
            self.postprocessor = DummyPostprocessorComponent()
        @property
        def yaml_cfg(self): return {"HGNetv2": {"pretrained": False}}

    class DummyModelLoaderComponent(torch.nn.Module):
        eval_spatial_size = (512, 512) # Default eval size for dummy
        def load_state_dict(self, state_dict): pass
        def deploy(self): return self

    class DummyPostprocessorComponent(torch.nn.Module):
        def deploy(self, *args, **kwargs): return self

# Define the ContourFormerInferenceWrapper as before
class ContourFormerInferenceWrapper(torch.nn.Module):
    def __init__(self, cfg_model, cfg_postprocessor):
        super().__init__()
        self.model = cfg_model.deploy()
        self.postprocessor = cfg_postprocessor.deploy()
        
    def forward(self, images, orig_target_sizes):
        input_sizes = torch.tensor([[images.shape[-1], images.shape[-2]]], device=images.device)
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes, input_sizes)
        return outputs


class ContourFormerModelLoader:
    CATEGORY = "ContourFormer"
    
    @classmethod
    def INPUT_TYPES(s):
        # Determine CONTOURFORMER_ROOT dynamically for file listing
        # This is a bit redundant with sys.path.insert but safer for node loading
        current_dir = os.path.dirname(os.path.abspath(__file__))
        contourformer_root = os.path.join(current_dir, '..', 'ContourFormer')

        config_dir = os.path.join(contourformer_root, 'configs', 'contourformer')
        weight_dir = os.path.join(contourformer_root, 'weight')

        config_files = [f for f in os.listdir(config_dir) if f.endswith('.yml')] if os.path.exists(config_dir) else []
        weight_files = [f for f in os.listdir(weight_dir) if f.endswith('.pth')] if os.path.exists(weight_dir) else []

        if not config_files:
            print(f"Warning: No config files found in {config_dir}. Please check your ContourFormer setup.")
            config_files = ["(No configs found - check setup)"]
        if not weight_files:
            print(f"Warning: No weight files found in {weight_dir}. Please download and place them.")
            weight_files = ["(No weights found - check setup)"]

        return {
            "required": {
                "config_name": (config_files, ),
                "checkpoint_name": (weight_files, ),
                "device": (["cuda", "cpu"],),
            }
        }

    RETURN_TYPES = ("CONTOURFORMER_MODEL", "CONTROFORMER_CONFIG")
    FUNCTION = "load_model_and_config"

    def load_model_and_config(self, config_name, checkpoint_name, device):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        contourformer_root = os.path.join(current_dir, '..', 'ContourFormer')

        config_path = os.path.join(contourformer_root, 'configs', 'contourformer', config_name)
        checkpoint_path = os.path.join(contourformer_root, 'weight', checkpoint_name)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}. Please check your ContourFormer setup.")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Please download it and place it in '{os.path.join(contourformer_root, 'weight')}' folder.")

        cfg = YAMLConfig(config_path)

        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        cfg.model.load_state_dict(state)
        model = ContourFormerInferenceWrapper(cfg.model, cfg.postprocessor).to(device)
        model.eval()

        return (model, cfg)