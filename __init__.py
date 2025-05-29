# __init__.py
import os
import sys

# Add the ContourFormer root to sys.path so its modules can be imported
CONTOURFORMER_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ContourFormer')
sys.path.insert(0, CONTOURFORMER_ROOT)
sys.path.insert(0, os.path.join(CONTOURFORMER_ROOT, 'src'))

# Import individual node classes from their respective files
from .contourformer_nodes.loader_node import ContourFormerModelLoader
from .contourformer_nodes.predictor_node import ContourFormerPredictor
from .contourformer_nodes.processor_node import ContourFormerMaskProcessor

# Define the node mappings
NODE_CLASS_MAPPINGS = {
    "ContourFormerModelLoader": ContourFormerModelLoader,
    "ContourFormerPredictor": ContourFormerPredictor,
    "ContourFormerMaskProcessor": ContourFormerMaskProcessor,
}

# Define the display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ContourFormerModelLoader": "ContourFormer Model Loader",
    "ContourFormerPredictor": "ContourFormer Inference",
    "ContourFormerMaskProcessor": "ContourFormer Mask Processor",
}

# Optional: You can also define a global category for your nodes here
# For example:
# __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']