# processor_node.py
import torch
import numpy as np
import cv2

class ContourFormerMaskProcessor:
    CATEGORY = "ContourFormer/Utils"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks_batch": ("MASK",), # Batch of masks from ContourFormerPredictor (N, H, W)
                "original_image": ("IMAGE",), # For background removal/inpainting (1, H, W, C)
                "scores_batch": ("FLOAT",), # Scores corresponding to masks_batch
                "class_ids_batch": ("INT",), # Class IDs corresponding to masks_batch
                "selection_method": (["Select by Index", "Select by Class ID", "Combine All"],),
                "target_indices": ("STRING", {"default": "0", "multiline": False, "placeholder": "Comma-separated indices (e.g., 0,2,5)"}), # For "Select by Index"
                "target_class_ids": ("STRING", {"default": "0", "multiline": False, "placeholder": "Comma-separated class IDs (e.g., 7,15)"}), # For "Select by Class ID"
                "mode": (["Object Mask", "Background Removal", "Inpaint Object", "Inpaint Background"],),
                "background_color": ("STRING", {"default": "transparent", "placeholder": "transparent, black, white, #RRGGBB"}),
                "dilation_amount": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("PROCESSED_MASK", "PROCESSED_IMAGE", "SELECTED_INFO", "PREVIEW_MASK_OVERLAY")
    FUNCTION = "process_masks"

    def process_masks(self, masks_batch: torch.Tensor, original_image: torch.Tensor, scores_batch: torch.Tensor, class_ids_batch: torch.Tensor,
                      selection_method: str, target_indices: str, target_class_ids: str,
                      mode: str, background_color: str, dilation_amount: int):
        
        selected_masks = []
        selected_info_list = []
        
        if masks_batch.shape[0] == 0:
            return (torch.zeros_like(original_image[:,:,:,0].squeeze(0)), original_image, "No masks available.", original_image)

        # Parse target indices/class IDs
        parsed_indices = []
        try:
            if target_indices:
                parsed_indices = [int(x.strip()) for x in target_indices.split(',') if x.strip().isdigit()]
        except ValueError:
            selected_info_list.append("Warning: Invalid format for target_indices. Ignoring.")

        parsed_class_ids = []
        try:
            if target_class_ids:
                parsed_class_ids = [int(x.strip()) for x in target_class_ids.split(',') if x.strip().isdigit()]
        except ValueError:
            selected_info_list.append("Warning: Invalid format for target_class_ids. Ignoring.")

        # --- Mask Selection Logic ---
        if selection_method == "Select by Index":
            for idx in parsed_indices:
                if 0 <= idx < masks_batch.shape[0]:
                    selected_masks.append(masks_batch[idx])
                    selected_score = scores_batch[idx].item()
                    selected_class_id = class_ids_batch[idx].item()
                    selected_info_list.append(f"Index: {idx}, Class ID: {selected_class_id}, Score: {selected_score:.2f}")
                else:
                    selected_info_list.append(f"Invalid index {idx}. Skipped.")
        
        elif selection_method == "Select by Class ID":
            for target_cid in parsed_class_ids:
                found_at_least_one = False
                for i, cid in enumerate(class_ids_batch):
                    if cid.item() == target_cid:
                        selected_masks.append(masks_batch[i])
                        selected_score = scores_batch[i].item()
                        selected_info_list.append(f"Class ID: {target_cid}, Index: {i}, Score: {selected_score:.2f}")
                        found_at_least_one = True
                if not found_at_least_one:
                    selected_info_list.append(f"No mask found for Class ID: {target_cid}.")
        
        elif selection_method == "Combine All":
            if masks_batch.shape[0] > 0:
                # Combine all masks using logical OR
                selected_masks.append(torch.max(masks_batch, dim=0).values) # Max along batch dim
                selected_info_list.append("All masks combined.")
            else:
                selected_info_list.append("No masks to combine.")

        # Consolidate selected masks
        if not selected_masks:
            # If no masks were selected, return an empty mask of the correct dimensions
            H, W = original_image.shape[1], original_image.shape[2]
            processed_mask = torch.zeros((1, H, W), dtype=torch.float32, device=original_image.device)
            selected_info = "No masks selected."
        else:
            # Combine all selected masks into a single mask (logical OR)
            combined_selected_mask = torch.max(torch.stack(selected_masks), dim=0).values
            
            # Apply dilation if requested
            if dilation_amount > 0:
                kernel = np.ones((dilation_amount * 2 + 1, dilation_amount * 2 + 1), np.uint8)
                mask_np_dilated = cv2.dilate(combined_selected_mask.cpu().numpy(), kernel, iterations=1)
                combined_selected_mask = torch.from_numpy(mask_np_dilated).float().to(original_image.device)
            
            processed_mask = combined_selected_mask.unsqueeze(0) # Ensure (1, H, W) for output
            selected_info = "; ".join(selected_info_list)
        
        processed_image = original_image.clone()

        # --- Apply Mode Specific Operations ---
        if processed_mask is not None and processed_mask.sum() > 0: # Only process if mask is not empty
            mask_np_float = processed_mask.squeeze(0).cpu().numpy() # H, W, float [0,1]
            mask_np_binary = (mask_np_float * 255).astype(np.uint8) # H, W, binary 0 or 255

            if mode == "Background Removal":
                img_np = (original_image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

                if background_color == "transparent":
                    if img_np.shape[2] == 3: # If RGB, add alpha channel
                        rgba_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2RGBA)
                    else: # Already RGBA
                        rgba_image = img_np.copy()
                    
                    alpha_channel = rgba_image[:, :, 3]
                    alpha_channel[mask_np_binary == 0] = 0 # Set alpha to 0 for background
                    rgba_image[:, :, 3] = alpha_channel
                    processed_image_np = rgba_image
                else:
                    bg_color_rgb = [0, 0, 0] # Default black
                    if background_color == "white":
                        bg_color_rgb = [255, 255, 255]
                    elif background_color.startswith('#') and len(background_color) == 7:
                        try:
                            # Convert hex to BGR for OpenCV
                            bg_color_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))[::-1]
                        except ValueError:
                            print(f"Warning: Invalid hex color {background_color}. Using black.")
                            bg_color_rgb = [0, 0, 0]
                    
                    background_img = np.full_like(img_np, bg_color_rgb)
                    processed_image_np = np.where(mask_np_binary[:, :, None] == 255, img_np, background_img)
                
                processed_image = torch.from_numpy(processed_image_np).float() / 255.0
                if processed_image.ndim == 3:
                    processed_image = processed_image.unsqueeze(0) # Add batch dim
            
            elif mode == "Inpaint Object":
                # Mask where 1 (or 255) indicates the area to be inpainted.
                # So, if we want to inpaint the object, the selected_mask is the inpaint mask.
                # processed_mask is already (1, H, W)
                processed_mask = (processed_mask > 0.5).float() # Ensure binary 0 or 1

            elif mode == "Inpaint Background":
                # Invert the selected_mask to get the background as the inpaint area.
                processed_mask = (1.0 - processed_mask > 0.5).float() # Ensure binary 0 or 1
            
            # If mode is "Object Mask", processed_mask is already the selected mask (1,H,W)
        
        # Create preview overlay image
        preview_overlay_np = (original_image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        
        if processed_mask is not None and processed_mask.sum() > 0:
            mask_overlay_np = (processed_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            # Use red overlay for selected mask
            red_overlay = np.zeros_like(preview_overlay_np)
            red_overlay[mask_overlay_np > 0] = [255, 0, 0]
            preview_overlay_np = cv2.addWeighted(preview_overlay_np, 1, red_overlay, 0.5, 0)
        
        preview_overlay_image = torch.from_numpy(preview_overlay_np).float() / 255.0
        if preview_overlay_image.ndim == 3:
            preview_overlay_image = preview_overlay_image.unsqueeze(0)

        return (processed_mask, processed_image, selected_info, preview_overlay_image)