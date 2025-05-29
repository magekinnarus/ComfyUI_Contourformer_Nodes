# predictor_node.py
import torch
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as T
import random
from itertools import cycle

# Import utility functions
from .utilities import convert_polygons_to_mask

# You may need to define class_names_map globally or pass it.
# For demonstration, hardcoding common SBD/COCO classes.
CLASS_NAMES_MAP = {
    # SBD dataset classes (example, confirm actual mapping from ContourFormer docs)
    # These are common IDs for SBD (Pascal VOC)
    1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle",
    6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow",
    11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person",
    16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor",
    # If using COCO, this map would be much larger (80 classes)
}


class ContourFormerPredictor:
    CATEGORY = "ContourFormer"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",), # ComfyUI IMAGE is (B, H, W, C) float32 [0,1]
                "contourformer_model": ("CONTOURFORMER_MODEL",),
                "contourformer_config": ("CONTROFORMER_CONFIG",), # Access eval_spatial_size
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "FLOAT", "INT")
    RETURN_NAMES = ("MASKS_BATCH", "PREVIEW_IMAGE", "CLASS_LABELS_STR", "SCORES_BATCH", "CLASS_IDS_BATCH")
    FUNCTION = "run_inference"

    def run_inference(self, image: torch.Tensor, contourformer_model, contourformer_config, threshold: float):
        device = next(contourformer_model.parameters()).device

        # Image Preprocessing (adapted from draw.py)
        img_np = (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        original_h, original_w = img_np.shape[0], img_np.shape[1]
        im_pil = Image.fromarray(img_np)
        
        orig_size = torch.tensor([original_w, original_h])[None].to(device) # Note: draw.py uses w, h

        default_height, default_width = contourformer_config.model.encoder.eval_spatial_size

        scaleX = default_width / original_w
        scaleY = default_height / original_h

        scale = min(scaleX, scaleY) # Use min for scaling (as implied by draw.py)

        new_H = int(scale * original_h)
        new_W = int(scale * original_w)

        val_h = (default_height - new_H) // 2
        val_w = (default_width - new_W) // 2

        transforms = T.Compose([
            T.Resize((new_H, new_W)),
            T.Pad(padding=(val_w, val_h, val_w, val_h)),
            T.ToTensor(), # Puts to C, H, W
        ])

        im_data = transforms(im_pil)[None].to(device) # Add batch dim: 1, C, H, W

        # Run Inference
        with torch.no_grad():
            outputs = contourformer_model(im_data, orig_size)
            labels, boxes, coords, scores = outputs

        # Filter predictions by threshold
        scr = scores[0]
        mask_score_thresh = scr > threshold
        lab = labels[0][mask_score_thresh]
        coord = coords[0][mask_score_thresh]
        scrs = scores[0][mask_score_thresh]

        # Convert polygons (coords) to masks
        output_masks = []
        class_labels_list = []
        
        if coord.shape[0] == 0: # No objects detected
            # Return empty tensors of appropriate shape
            empty_mask = torch.zeros((1, original_h, original_w), dtype=torch.float32)
            return (empty_mask, image, "No objects detected.", torch.tensor([]), torch.tensor([]))

        for j in range(coord.shape[0]):
            poly_np = coord[j].cpu().numpy().reshape((-1, 2))
            mask_np = convert_polygons_to_mask(
                [poly_np], image_shape=(original_h, original_w)
            )
            output_masks.append(torch.from_numpy(mask_np).float())
            class_id = lab[j].item()
            class_labels_list.append(f"{CLASS_NAMES_MAP.get(class_id, f'Class_{class_id}')}: {scrs[j].item():.2f}")

        # Stack filtered masks into a batch tensor (N, H, W)
        output_masks_batch = torch.stack(output_masks)
        scores_batch = scrs.cpu()
        class_ids_batch = lab.cpu()

        # Generate preview image with all detected contours drawn
        preview_image_np = img_np.copy()
        colors = ["#1F77B4", "#FF7F0E", "#2EA02C", "#D62827", "#9467BD", 
                  "#8C564B", "#E377C2", "#7E7E7E", "#BCBD20", "#1ABECF"]
        
        random.shuffle(colors)
        color_cycle = cycle(colors)

        for j in range(coord.shape[0]):
            poly_np_draw = coord[j].cpu().numpy().reshape((-1, 2))
            color_hex = next(color_cycle)
            color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            
            cv2.polylines(preview_image_np, [poly_np_draw.astype(np.int32)], True, color_rgb, 2)
            
            mask_fill_np = convert_polygons_to_mask(
                [poly_np_draw], image_shape=(original_h, original_w)
            )
            colored_mask_fill = np.zeros_like(preview_image_np)
            colored_mask_fill[mask_fill_np > 0.5] = color_rgb
            preview_image_np = cv2.addWeighted(preview_image_np, 1, colored_mask_fill, 0.5, 0)
            
            if poly_np_draw.size > 0:
                x, y, w_box, h_box = cv2.boundingRect(poly_np_draw.astype(np.int32))
                label_text = class_labels_list[j].split(':')[0]
                cv2.putText(preview_image_np, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_rgb, 2, cv2.LINE_AA)

        preview_image_tensor = torch.from_numpy(preview_image_np).float() / 255.0
        preview_image_tensor = preview_image_tensor.unsqueeze(0)

        return (output_masks_batch, preview_image_tensor, ", ".join(class_labels_list), scores_batch, class_ids_batch)