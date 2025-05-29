# utilities.py
import numpy as np
import cv2
import torch

def convert_polygons_to_mask(polygons: np.ndarray, image_shape):
    """
    Converts a list of polygons (N_points x 2) to a binary mask.
    Args:
        polygons (list of np.ndarray): List of polygon vertices.
                                      Each polygon is an (N_points, 2) numpy array.
        image_shape (tuple): (height, width) of the target mask.
    Returns:
        np.ndarray: A binary mask (H, W) with float values [0.0, 1.0].
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    # cv2.fillPoly expects a list of polygons, where each polygon is an array of points
    cv2.fillPoly(mask, [poly.astype(np.int32) for poly in polygons], 255)
    return mask / 255.0 # Convert to float [0, 1]

# You can add other utility functions here as needed