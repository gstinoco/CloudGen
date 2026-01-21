"""
Contour Detection Module - Optimized for Single-Region Interaction

This module provides streamlined functionality for detecting and extracting contours 
from images, optimized for single-click region detection. It replaces complex, 
redundant algorithms with a highly efficient, adaptive Flood Fill approach.

Core Functionality:
1. Fast Single-Region Detection: Optimized cv2.floodFill implementation.
2. Adaptive Tolerance: Automatically adjusts parameters based on local image texture.
3. Interactive Refinement: Support for adding/subtracting regions via seeds.
4. Brush Tools: Manual correction capabilities.
5. GrabCut (Optional): Retained for complex foreground extraction if needed.

Changes & Optimizations:
- Replaced slow Python-based `region_growing` with C++ optimized `cv2.floodFill`.
- Removed redundant `watershed_segmentation` to simplify the API.
- Unified single-click and multi-seed logic under one robust core function.
- Enhanced performance for real-time interaction (O(N) complexity).

Author: Gerardo Tinoco-Guerrero
Date: May, 2025
Last Modification: January 21st, 2026
"""

import numpy as np
import logging
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_region_at_point(image, click_x, click_y, tolerance=30):
    """
    Detects a single connected region starting from a seed point using optimized Flood Fill.
    
    This is the primary function for single-click interaction. It analyzes the local
    texture around the click point to adaptively set the flood fill tolerance, ensuring
    robust detection across both uniform and textured regions.
    
    Args:
        image (numpy.ndarray): Input image (BGR or grayscale).
        click_x (int): X coordinate of the seed point.
        click_y (int): Y coordinate of the seed point.
        tolerance (int, optional): Base tolerance for pixel similarity. Default: 30.
    
    Returns:
        numpy.ndarray: Binary mask (uint8) where 255 is the detected region.
    """
    try:
        # 1. Preprocessing
        if len(image.shape) == 3:
            # Work in a color space that separates luma/chroma might be better, 
            # but for consistency with previous code, we'll check if we need grayscale 
            # or if we apply floodFill on color (which is supported and often better).
            # We will use the input image directly for floodFill to leverage color info.
            work_image = image.copy()
            # For statistics, grayscale is easier
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            work_image = image.copy()
            gray = image.copy()

        h, w = work_image.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8) # FloodFill needs mask +2 pixels
        
        # Clamp coordinates
        click_x = max(0, min(click_x, w - 1))
        click_y = max(0, min(click_y, h - 1))

        # 2. Adaptive Tolerance Calculation
        # Analyze a small window around the click to estimate noise/texture
        window_size = 9
        x1 = max(0, click_x - window_size)
        y1 = max(0, click_y - window_size)
        x2 = min(w, click_x + window_size + 1)
        y2 = min(h, click_y + window_size + 1)
        
        local_region = gray[y1:y2, x1:x2]
        if local_region.size > 0:
            local_std = np.std(local_region)
        else:
            local_std = 0
            
        # Adjust tolerance:
        # - Low std (uniform): Use base tolerance (or slightly higher to catch gradients)
        # - High std (texture): Increase tolerance to bridge noise
        # We use a lower/upper diff approach.
        
        if local_std < 5:
            # Very uniform: strict lower bound, slightly loose upper
            lo_diff = (max(tolerance, 10),) * 3
            up_diff = (max(tolerance, 10),) * 3
        elif local_std < 15:
            # Moderate: add std to tolerance
            adj = int(tolerance + local_std * 0.5)
            lo_diff = (adj,) * 3
            up_diff = (adj,) * 3
        else:
            # Textured: more permissive
            adj = int(tolerance + local_std * 1.0)
            lo_diff = (adj,) * 3
            up_diff = (adj,) * 3

        # 3. Apply Flood Fill
        # flags: 4 or 8 connectivity | FLOODFILL_FIXED_RANGE (compare to seed) or not
        # FLOODFILL_FIXED_RANGE is usually better for segmenting a specific object color.
        flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
        
        cv2.floodFill(work_image, mask, (click_x, click_y), (255, 255, 255), 
                      lo_diff, up_diff, flags)
        
        # Crop mask to original size (remove the +2 padding)
        result_mask = mask[1:-1, 1:-1]
        
        # 4. Post-processing (Morphology)
        # Close small holes and smooth boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel)
        
        return result_mask

    except Exception as e:
        logger.error(f"Error in detect_region_at_point: {str(e)}")
        # Return empty mask on failure
        return np.zeros(image.shape[:2], dtype=np.uint8)


def interactive_segmentation_with_seeds(image, positive_seeds, negative_seeds, tolerance=30):
    """
    Interactive segmentation using positive and negative seed points.
    
    Now powered by the optimized `detect_region_at_point` function.
    
    Args:
        image (numpy.ndarray): Input image.
        positive_seeds (list): List of (x, y) foreground points.
        negative_seeds (list): List of (x, y) background points.
        tolerance (int): Tolerance for region growing.
    
    Returns:
        numpy.ndarray: Combined binary mask.
    """
    h, w = image.shape[:2]
    final_mask = np.zeros((h, w), dtype=np.uint8)
    
    if not positive_seeds:
        return final_mask
        
    # Process positive seeds (Union)
    for px, py in positive_seeds:
        region_mask = detect_region_at_point(image, px, py, tolerance)
        final_mask = cv2.bitwise_or(final_mask, region_mask)
        
    # Process negative seeds (Subtraction)
    for nx, ny in negative_seeds:
        # For negative seeds, we also detect the region they belong to
        # and subtract it.
        neg_region_mask = detect_region_at_point(image, nx, ny, tolerance)
        
        # Optional: Also strictly remove a small radius around the click
        # to ensure the user's explicit "not here" click is respected
        cv2.circle(neg_region_mask, (nx, ny), 5, 255, -1)
        
        final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(neg_region_mask))
        
    return final_mask


def grabcut_interactive(image, rect=None, mask=None, iterations=5):
    """
    Interactive GrabCut segmentation using graph-cut optimization.
    
    Retained for advanced users needing foreground extraction in complex scenes.
    """
    if rect is None and mask is None:
        return None
    
    h, w = image.shape[:2]
    
    # Initialize mask
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        x, y, width, height = rect
        mask[y:y+height, x:x+width] = cv2.GC_PR_FGD
    
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        if rect is not None:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        else:
            cv2.grabCut(image, mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
        
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        return mask2 * 255
        
    except Exception as e:
        logger.error(f"Error in GrabCut segmentation: {str(e)}")
        return None


def refine_mask_with_brush(current_mask, brush_strokes):
    """
    Refine a segmentation mask using interactive brush strokes.
    """
    if current_mask is None:
        return None
    
    refined_mask = current_mask.copy()
    
    for stroke in brush_strokes:
        points = stroke.get('points', [])
        mode = stroke.get('mode', 'add')
        size = stroke.get('size', 10)
        
        if not points:
            continue
        
        stroke_mask = np.zeros_like(refined_mask)
        
        if len(points) == 1:
            pt = (int(points[0][0]), int(points[0][1]))
            cv2.circle(stroke_mask, pt, size // 2, 255, -1)
        else:
            for i in range(len(points) - 1):
                pt1 = (int(points[i][0]), int(points[i][1]))
                pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))
                cv2.line(stroke_mask, pt1, pt2, 255, size)
        
        if mode == 'add':
            refined_mask = cv2.bitwise_or(refined_mask, stroke_mask)
        elif mode == 'remove':
            refined_mask = cv2.bitwise_and(refined_mask, cv2.bitwise_not(stroke_mask))
    
    return refined_mask
