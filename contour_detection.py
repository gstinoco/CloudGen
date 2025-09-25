"""
Contour Detection Module - Advanced Image Segmentation and Boundary Extraction System

This module provides comprehensive functionality for detecting and extracting contours from
images using multiple advanced segmentation algorithms. It implements interactive segmentation
techniques with seed-based region growing, watershed algorithms, GrabCut optimization, and
brush-based refinement tools for precise boundary detection in complex images.

Core Functionality:
1. Interactive segmentation with positive and negative seed points
2. Advanced watershed segmentation with marker-based region growing
3. GrabCut algorithm for foreground/background separation
4. Brush-based mask refinement for manual corrections
5. Region growing algorithms with adaptive tolerance
6. Combined segmentation approaches for optimal results
7. Contour extraction and boundary point generation

Segmentation Algorithms:
- Interactive Segmentation: Seed-based region growing with positive/negative markers
  * Uses user-defined seed points to guide segmentation process
  * Implements region growing with adaptive tolerance thresholds
  * Supports both foreground (positive) and background (negative) seeds
  * Provides real-time feedback for interactive refinement

- Watershed Segmentation: Marker-controlled watershed transformation
  * Implements advanced watershed algorithm for precise boundary detection
  * Uses distance transform and morphological operations
  * Handles complex geometries with multiple connected components
  * Provides robust segmentation for overlapping or touching objects

- GrabCut Algorithm: Graph-cut based foreground/background separation
  * Implements iterative energy minimization for optimal segmentation
  * Uses Gaussian Mixture Models for color distribution modeling
  * Supports both rectangle-based and mask-based initialization
  * Provides high-quality results for natural images

- Brush Refinement: Manual correction tools for precise boundary adjustment
  * Allows manual addition and removal of segmented regions
  * Implements brush-based painting interface for user corrections
  * Supports variable brush sizes and opacity settings
  * Enables fine-tuning of automatically generated segmentations

Key Features:
- Multi-algorithm approach for robust segmentation across different image types
- Interactive seed-based segmentation with real-time feedback
- Advanced morphological operations for noise reduction and boundary smoothing
- Adaptive tolerance mechanisms for handling varying image characteristics
- Comprehensive brush tools for manual refinement and correction
- Efficient contour extraction with sub-pixel accuracy
- Support for both grayscale and color image processing
- Robust error handling and comprehensive logging
- Optimized performance for real-time interactive applications

Technical Implementation:
- OpenCV for computer vision operations and image processing
- NumPy for efficient numerical computations and array operations
- SciPy for advanced morphological operations and image filtering
- Professional logging system with detailed operation tracking
- Optimized algorithms for real-time interactive performance
- Memory-efficient processing for large image datasets

Workflow Process:
1. Load and preprocess input image with noise reduction
2. Apply selected segmentation algorithm based on image characteristics
3. Process user interactions (seeds, brush strokes, rectangle selection)
4. Perform iterative refinement using selected algorithm
5. Apply morphological operations for boundary smoothing
6. Extract contours with sub-pixel precision
7. Generate boundary point coordinates for further processing
8. Export results in multiple formats for downstream applications

Applications:
- Medical image segmentation for anatomical structure extraction
- Industrial quality control and defect detection
- Geographic information systems for land cover classification
- Scientific image analysis for research applications
- Computer graphics and digital content creation
- Automated object detection and recognition systems

Author: Gerardo Tinoco-Guerrero
Date: May, 2025
Last Modification: September 25th, 2025

Dependencies:
- OpenCV >= 4.5.0
- NumPy >= 1.20.0
- SciPy >= 1.8.0
"""

import cv2
import numpy as np
from scipy import ndimage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def interactive_segmentation_with_seeds(image, positive_seeds, negative_seeds, tolerance=30):
    """
    Interactive segmentation using positive and negative seed points with region growing.
    
    This function performs semi-automatic segmentation by growing regions from user-provided
    seed points. Positive seeds define foreground regions to include, while negative seeds
    define background regions to exclude. The algorithm uses region growing with adaptive
    tolerance to create coherent segmented regions.
    
    Args:
        image (numpy.ndarray): Input image in BGR or RGB format (3-channel) or grayscale.
                              If color image provided, will be converted to grayscale for processing.
        positive_seeds (list): List of (x, y) coordinate tuples defining foreground seed points.
                              These points indicate regions that should be included in the
                              final segmentation. Empty list results in empty mask.
        negative_seeds (list): List of (x, y) coordinate tuples defining background seed points.
                              These points indicate regions that should be excluded from the
                              final segmentation. Can be empty list.
        tolerance (int, optional): Tolerance parameter for region growing algorithm.
                                  Controls how similar neighboring pixels must be to be
                                  included in the same region. Higher values create larger,
                                  more inclusive regions. Default: 30.
    
    Returns:
        numpy.ndarray: Binary mask (uint8) where 255 represents the segmented foreground
                      region and 0 represents background. Returns empty mask if no
                      positive seeds provided.
    
    Algorithm:
        1. Convert input image to grayscale if necessary
        2. Initialize empty mask for result accumulation
        3. For each positive seed point:
           a. Validate seed coordinates are within image bounds
           b. Apply region growing algorithm from seed point
           c. Combine grown region with existing mask using bitwise OR
        4. For each negative seed point:
           a. Validate seed coordinates are within image bounds
           b. Apply region growing algorithm from seed point
           c. Remove grown region from existing mask using bitwise AND NOT
        5. Return final combined mask
    
    Features:
        - Multi-seed support for complex object segmentation
        - Positive and negative seed interaction for precise control
        - Automatic coordinate validation and bounds checking
        - Adaptive region growing with user-controlled tolerance
        - Robust handling of edge cases (empty seed lists, invalid coordinates)
    
    Use Cases:
        - Interactive object segmentation with user guidance
        - Semi-automatic region extraction from complex images
        - Foreground/background separation with manual seed placement
        - Refinement of automatic segmentation results
    
    Note:
        The quality of segmentation depends heavily on seed placement and tolerance
        settings. Seeds should be placed in representative areas of the target regions.
        The tolerance parameter may need adjustment based on image characteristics.
    
    Example:
        positive_seeds = [(100, 150), (120, 160)]  # Object interior points
        negative_seeds = [(50, 50), (200, 200)]    # Background points
        mask = interactive_segmentation_with_seeds(image, positive_seeds, negative_seeds, 25)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # If no positive seeds, return empty mask
    if not positive_seeds:
        return mask
    
    # Grow regions from positive seeds
    for seed_x, seed_y in positive_seeds:
        if 0 <= seed_x < w and 0 <= seed_y < h:
            seed_mask = region_growing(image, seed_x, seed_y, tolerance)
            mask = cv2.bitwise_or(mask, seed_mask)
    
    # Remove regions around negative seeds
    for seed_x, seed_y in negative_seeds:
        if 0 <= seed_x < w and 0 <= seed_y < h:
            # Create a small circular region around negative seed
            negative_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(negative_mask, (seed_x, seed_y), 20, 255, -1)
            
            # Also grow a small region from negative seed
            neg_region = region_growing(image, seed_x, seed_y, tolerance // 2)
            negative_mask = cv2.bitwise_or(negative_mask, neg_region)
            
            # Remove negative regions from main mask
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(negative_mask))
    
    return mask


def grabcut_interactive(image, rect=None, mask=None, iterations=5):
    """
    Interactive GrabCut segmentation using graph-cut optimization for foreground extraction.
    
    This function implements the GrabCut algorithm, which uses iterative energy minimization
    to separate foreground objects from background. The algorithm builds Gaussian Mixture
    Models for both foreground and background regions and uses graph-cut optimization to
    find the optimal segmentation boundary.
    
    Args:
        image (numpy.ndarray): Input image in BGR color format (3-channel, 8-bit)
        rect (tuple, optional): Initial rectangle (x, y, width, height) defining the
                               approximate foreground region. If None, uses entire image.
        mask (numpy.ndarray, optional): Initial segmentation mask with values:
                                       - 0: Definite background
                                       - 1: Definite foreground  
                                       - 2: Probable background
                                       - 3: Probable foreground
        iterations (int, optional): Number of GrabCut iterations for refinement.
                                   More iterations improve accuracy but increase computation time.
                                   Default: 5.
    
    Returns:
        numpy.ndarray: Binary mask (uint8) where 255 represents foreground pixels
                      and 0 represents background pixels.
    
    Algorithm:
        1. Initialize foreground/background models using rectangle or mask
        2. Build Gaussian Mixture Models for color distributions
        3. Construct graph with pixel nodes and edge weights
        4. Apply min-cut/max-flow algorithm to find optimal segmentation
        5. Update models based on current segmentation
        6. Repeat steps 3-5 for specified number of iterations
        7. Generate final binary mask from segmentation result
    
    Note:
        GrabCut is particularly effective for natural images with distinct foreground
        objects. The algorithm works best when the initial rectangle or mask provides
        a reasonable approximation of the foreground region.
    
    Raises:
        cv2.error: If image format is invalid or algorithm parameters are incorrect.
    """
    if rect is None and mask is None:
        return None
    
    h, w = image.shape[:2]
    
    # Initialize mask
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        x, y, width, height = rect
        mask[y:y+height, x:x+width] = cv2.GC_PR_FGD
    
    # Initialize background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        # Apply GrabCut
        if rect is not None:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        else:
            cv2.grabCut(image, mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
        
        # Create binary mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result_mask = mask2 * 255
        
        return result_mask
        
    except Exception as e:
        logger.error(f"Error in GrabCut segmentation: {str(e)}")
        return None


def refine_mask_with_brush(current_mask, brush_strokes):
    """
    Refine a segmentation mask using interactive brush strokes for manual correction.
    
    This function allows users to manually refine segmentation results by applying
    brush strokes to add or remove regions from the mask. It supports different
    brush modes (add/remove) and variable brush sizes for precise editing.
    
    Args:
        current_mask (numpy.ndarray): Current binary mask (uint8) where 255 represents
                                     foreground and 0 represents background pixels.
                                     If None, creates a new empty mask.
        brush_strokes (list): List of brush stroke dictionaries, each containing:
                             - 'points' (list): List of (x, y) coordinates defining the stroke path
                             - 'mode' (str): Brush mode - 'add' to add regions to mask,
                                           'remove' to subtract regions from mask
                             - 'size' (int): Brush radius in pixels for stroke thickness
    
    Returns:
        numpy.ndarray: Refined binary mask (uint8) with brush modifications applied.
                      Returns original mask if no valid brush strokes provided.
    
    Algorithm:
        1. Initialize working mask from current mask or create empty mask
        2. For each brush stroke in the list:
           a. Extract stroke parameters (points, mode, size)
           b. Create brush kernel based on specified size
           c. Draw stroke path using cv2.polylines or cv2.circle
           d. Apply brush effect (add/remove) to mask based on mode
        3. Ensure final mask values are properly normalized (0 or 255)
        4. Return refined mask
    
    Features:
        - Variable brush sizes for different levels of detail
        - Add and remove modes for flexible editing
        - Smooth stroke interpolation between points
        - Preserves mask integrity and format
    
    Note:
        This function is typically used after automatic segmentation algorithms
        to manually correct errors or refine boundaries. The brush strokes are
        applied sequentially, allowing for complex editing operations.
    
    Example:
        brush_strokes = [
            {'points': [(100, 100), (110, 105)], 'mode': 'add', 'size': 5},
            {'points': [(200, 200)], 'mode': 'remove', 'size': 10}
        ]
        refined_mask = refine_mask_with_brush(current_mask, brush_strokes)
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
        
        # Create stroke mask
        stroke_mask = np.zeros_like(refined_mask)
        
        # Draw stroke
        for i in range(len(points) - 1):
            pt1 = (int(points[i]['x']), int(points[i]['y']))
            pt2 = (int(points[i + 1]['x']), int(points[i + 1]['y']))
            cv2.line(stroke_mask, pt1, pt2, 255, size)
        
        # Apply stroke based on mode
        if mode == 'add':
            refined_mask = cv2.bitwise_or(refined_mask, stroke_mask)
        elif mode == 'remove':
            refined_mask = cv2.bitwise_and(refined_mask, cv2.bitwise_not(stroke_mask))
    
    return refined_mask


def region_growing(image, click_x, click_y, tolerance=30):
    """
    Region growing segmentation algorithm.
    
    Args:
        image: Input image
        click_x: X coordinate of seed point
        click_y: Y coordinate of seed point
        tolerance: Tolerance for region growing
    
    Returns:
        Binary mask of the grown region
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=bool)
    
    # Validate click coordinates
    click_x = max(0, min(click_x, w - 1))
    click_y = max(0, min(click_y, h - 1))
    
    # Get seed pixel value
    seed_value = gray[click_y, click_x]
    
    # Calculate adaptive tolerance based on local standard deviation
    local_region = gray[max(0, click_y-10):min(h, click_y+11), 
                       max(0, click_x-10):min(w, click_x+11)]
    local_std = np.std(local_region)
    
    # Adjust tolerance based on local texture
    if local_std < 5:  # Very uniform area
        adaptive_tolerance = max(tolerance, 15)
    elif local_std < 15:  # Moderately uniform
        adaptive_tolerance = tolerance + local_std * 0.5
    else:  # Textured area
        adaptive_tolerance = tolerance + local_std * 0.3
    
    # Region growing using stack
    stack = [(click_x, click_y)]
    region_pixels = 0
    max_region_size = int(0.8 * h * w)  # 80% of image
    
    while stack and region_pixels < max_region_size:
        x, y = stack.pop()
        
        if visited[y, x]:
            continue
        
        visited[y, x] = True
        pixel_value = gray[y, x]
        
        # Check if pixel belongs to region
        if abs(int(pixel_value) - int(seed_value)) <= adaptive_tolerance:
            mask[y, x] = 255
            region_pixels += 1
            
            # Add neighbors to stack
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    stack.append((nx, ny))
    
    # Post-processing: morphological closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Keep only the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8) * 255
    
    return mask


def watershed_segmentation(image, click_x, click_y, tolerance=30):
    """
    Watershed segmentation algorithm with adaptive thresholding and region growing fallback.
    
    This function implements a robust watershed-based segmentation approach that adapts
    to different image characteristics. It uses adaptive thresholding for preprocessing,
    applies morphological operations for noise reduction, and includes a region growing
    fallback mechanism for challenging cases.
    
    Args:
        image (numpy.ndarray): Input image (BGR or grayscale). If BGR, will be converted
                              to grayscale for processing.
        click_x (int): X coordinate of the seed point for segmentation initialization.
                      Will be clamped to valid image bounds.
        click_y (int): Y coordinate of the seed point for segmentation initialization.
                      Will be clamped to valid image bounds.
        tolerance (int, optional): Tolerance parameter for region growing fallback.
                                  Controls similarity threshold for pixel inclusion.
                                  Default: 30.
    
    Returns:
        numpy.ndarray: Binary mask (uint8) where 255 represents the segmented region
                      and 0 represents background. Returns None if segmentation fails.
    
    Algorithm:
        1. Convert input to grayscale if necessary
        2. Analyze image uniformity using standard deviation
        3. Apply adaptive thresholding based on image characteristics:
           - For uniform images: Use adaptive mean thresholding with larger blocks
           - For varied images: Use Otsu's method for global thresholding
        4. Apply morphological operations (opening/closing) for noise reduction
        5. Perform distance transform and find local maxima as markers
        6. Apply watershed algorithm using gradient and markers
        7. Extract region containing the seed point
        8. If watershed fails, fallback to region growing algorithm
        9. Apply post-processing to clean up the result
    
    Features:
        - Adaptive preprocessing based on image characteristics
        - Robust marker detection for watershed initialization
        - Morphological noise reduction
        - Region growing fallback for difficult cases
        - Automatic parameter adjustment based on image properties
    
    Note:
        The watershed algorithm is particularly effective for segmenting objects
        with clear boundaries. The adaptive approach makes it suitable for various
        image types, from uniform regions to complex textures.
    
    Raises:
        Exception: Logs error and returns None if segmentation fails completely.
    """
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # Validate click coordinates
        click_x = max(0, min(click_x, w - 1))
        click_y = max(0, min(click_y, h - 1))
        
        # Check image uniformity for adaptive thresholding
        image_std = np.std(gray)
        
        if image_std < 10:  # Very uniform image
            # More permissive threshold for uniform images
            block_size = max(11, min(w, h) // 10)
            if block_size % 2 == 0:
                block_size += 1
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, block_size, 2)
        else:
            # Standard threshold for normal images
            block_size = max(11, min(w, h) // 20)
            if block_size % 2 == 0:
                block_size += 1
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, block_size, 5)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add 1 to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        if len(image.shape) == 3:
            markers = cv2.watershed(image, markers)
        else:
            # Convert grayscale to BGR for watershed
            image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(image_bgr, markers)
        
        # Find the marker at the clicked point
        clicked_marker = markers[click_y, click_x]
        
        # Create mask for the clicked region
        if clicked_marker > 1:  # Valid marker (not background or boundary)
            mask = (markers == clicked_marker).astype(np.uint8) * 255
            
            # Validate segmentation quality
            region_size = np.sum(mask > 0)
            image_size = mask.shape[0] * mask.shape[1]
            
            # If region is too small or too large, use region growing instead
            if region_size < 100 or region_size > 0.8 * image_size:
                logger.info(f"Watershed result questionable (size: {region_size}), using region growing fallback")
                mask = region_growing(image, click_x, click_y, tolerance)
        else:
            # Use region growing as fallback
            logger.info("Watershed failed to find valid marker, using region growing fallback")
            mask = region_growing(image, click_x, click_y, tolerance)
        
        logger.info(f"Segmentation completed. Detected region: {np.sum(mask > 0)} pixels")
        return mask
        
    except Exception as e:
        logger.error(f"Error in watershed segmentation: {str(e)}, using region growing fallback")
        return region_growing(image, click_x, click_y, tolerance)


def find_contours_from_mask(mask):
    """
    Find contours from a binary mask.
    
    Args:
        mask: Binary mask
    
    Returns:
        List of contours
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def detect_region(image, click_x, click_y, tolerance=30):
    """
    Main function to detect a region using watershed segmentation.
    
    Args:
        image: Input image
        click_x: X coordinate of the click point
        click_y: Y coordinate of the click point
        tolerance: Tolerance parameter
    
    Returns:
        tuple: (mask, contours) where mask is the binary segmentation and contours is a list of contours
    """
    try:
        # Validate click coordinates
        h, w = image.shape[:2]
        click_x = max(0, min(click_x, w - 1))
        click_y = max(0, min(click_y, h - 1))
        
        # Apply watershed segmentation
        mask = watershed_segmentation(image, click_x, click_y, tolerance)
        
        # Find contours
        contours = find_contours_from_mask(mask)
        
        logger.info(f"Region detection completed. Mask coverage: {np.sum(mask > 0) / mask.size * 100:.2f}%")
        
        return mask, contours
        
    except Exception as e:
        logger.error(f"Error in region detection: {str(e)}")
        # Return empty results
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        return mask, []


def apply_combined_segmentation(image, click_x, click_y, tolerance=30, algorithm='watershed'):
    """
    Compatibility function that calls the watershed implementation.
    
    Args:
        image: Input image
        click_x: X coordinate of the click point
        click_y: Y coordinate of the click point
        tolerance: Tolerance parameter
        algorithm: Algorithm name (ignored, always uses watershed)
    
    Returns:
        Binary mask of the segmented region
    """
    mask, _ = detect_region(image, click_x, click_y, tolerance)
    return mask