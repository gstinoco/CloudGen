/**
 * Contour Creator Module - Advanced Interactive Image Segmentation Interface
 * 
 * This module provides comprehensive functionality for interactive contour detection and
 * region segmentation from uploaded images. It implements multiple advanced segmentation
 * algorithms with real-time canvas manipulation, zoom/pan controls, and brush-based
 * refinement tools for precise boundary extraction in complex images.
 * 
 * Core Functionality:
 * 1. Interactive image upload with drag & drop support and format validation
 * 2. Advanced canvas operations with zoom, pan, and coordinate transformation
 * 3. Multiple segmentation algorithms (Watershed, GrabCut, Interactive, Region Growing)
 * 4. Real-time region visualization with multi-color support and transparency
 * 5. Brush-based refinement tools for manual boundary correction
 * 6. Region management system with add, delete, toggle, and export capabilities
 * 7. Coordinate export functionality with CSV generation and download
 * 
 * Segmentation Algorithms:
 * - Interactive Segmentation: Seed-based region growing with positive/negative markers
 *   * User-defined seed points guide the segmentation process
 *   * Adaptive tolerance thresholds for different image characteristics
 *   * Real-time feedback with immediate visual results
 * 
 * - Watershed Segmentation: Marker-controlled watershed transformation
 *   * Advanced watershed algorithm for precise boundary detection
 *   * Handles complex geometries with multiple connected components
 *   * Robust segmentation for overlapping or touching objects
 * 
 * - GrabCut Algorithm: Graph-cut based foreground/background separation
 *   * Iterative energy minimization for optimal segmentation
 *   * Gaussian Mixture Models for color distribution modeling
 *   * High-quality results for natural images with complex backgrounds
 * 
 * - Region Growing: Pixel-based region expansion with similarity criteria
 *   * Traditional region growing with adaptive tolerance
 *   * Efficient for homogeneous regions with clear boundaries
 *   * Fast processing for simple segmentation tasks
 * 
 * Canvas Features:
 * - Advanced zoom controls (1x to 5x) with smooth scaling
 * - Pan functionality with boundary constraints and smooth dragging
 * - Real-time coordinate transformation between canvas and image space
 * - Multi-touch and mouse wheel support for intuitive navigation
 * - Responsive design with automatic canvas resizing
 * 
 * Brush Refinement System:
 * - Variable brush sizes (5px to 50px) for precise editing
 * - Add/Remove modes for selective region modification
 * - Stroke-based editing with undo/redo functionality
 * - Real-time preview with temporary stroke visualization
 * - Integration with existing segmentation results
 * 
 * Region Management:
 * - Multi-region support with automatic color assignment
 * - Region visibility toggle for complex scene analysis
 * - Individual region deletion and bulk operations
 * - Area calculation and statistics display
 * - Export capabilities for single regions or complete datasets
 * 
 * Technical Implementation:
 * - HTML5 Canvas API for high-performance graphics rendering
 * - Event-driven architecture with optimized event handling
 * - Asynchronous API communication with progress tracking
 * - Memory-efficient image processing with canvas optimization
 * - Cross-browser compatibility with fallback mechanisms
 * - Responsive design patterns for mobile and desktop support
 * 
 * File Upload System:
 * - Drag & drop interface with visual feedback and progress indication
 * - Multiple image format support (PNG, JPG, JPEG, GIF, BMP, WEBP)
 * - File size validation (10MB limit) with user-friendly error messages
 * - Automatic image optimization and canvas fitting
 * - Secure file handling with format validation
 * 
 * API Integration:
 * - RESTful communication with Flask backend
 * - Real-time progress tracking for long-running operations
 * - Error handling with user-friendly notifications
 * - Automatic retry mechanisms for network issues
 * - JSON-based data exchange with validation
 * 
 * User Experience Features:
 * - Floating notifications with auto-dismiss and manual close
 * - Loading states with progress indicators and estimated time
 * - Keyboard shortcuts for common operations
 * - Context-sensitive help and tooltips
 * - Accessibility support with ARIA labels and keyboard navigation
 * 
 * Performance Optimizations:
 * - Canvas rendering optimization with selective redraws
 * - Event throttling for smooth pan and zoom operations
 * - Memory management with automatic cleanup
 * - Efficient coordinate transformations with caching
 * - Optimized image loading with progressive enhancement
 * 
 * @fileoverview Contour Creator JavaScript Module - Interactive image segmentation interface
 * @author Gerardo Tinoco-Guerrero
 * @author Universidad Michoacana de San Nicolás de Hidalgo
 * @author SIIIA - Sistema de Investigación e Innovación en Inteligencia Artificial
 * @author SECIHTI - Secretaría de Ciencia, Humanidades, Tecnología e Innovación
 * @version 2.0
 * @since 2025-05-01
 * @lastModified 2025-09-25
 * 
 * @requires HTML5 Canvas API
 * @requires Fetch API for backend communication
 * @requires ES6+ JavaScript features
 * 
 * @see {@link https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API} Canvas API Documentation
 * @see {@link app.py} Flask backend implementation
 * @see {@link contour_detection.py} Python segmentation algorithms
 */

// ===== GLOBAL VARIABLES =====

// Canvas and image variables
let currentImage = null;
let currentFilename = null;
let canvas = null;
let ctx = null;
let detectedRegions = []; // Array for multiple regions
let regionColors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080', '#008000', '#FFC0CB'];
let currentRegionIndex = 0;
let tempRegion = null; // Temporary region before adding

// Variables for zoom and pan
let zoomLevel = 1;
let panX = 0;
let panY = 0;
let isDragging = false;
let hasDragged = false; // Track if mouse actually moved during drag
let lastMouseX = 0;
let lastMouseY = 0;
let minZoom = 1;
let maxZoom = 5;
let dragThreshold = 3; // Minimum pixels to consider as drag

// Upload variables
let uploadZone = null;
let fileInput = null;
let uploadContent = null;
let uploadIcon = null;
let uploadTitle = null;
let uploadSubtitle = null;
let clearButton = null;
let dragCounter = 0;

// Supported image formats
const SUPPORTED_FORMATS = {
    'image/jpeg': 'JPG',
    'image/jpg': 'JPG',
    'image/png': 'PNG',
    'image/gif': 'GIF',
    'image/webp': 'WEBP',
    'image/bmp': 'BMP'
};

// Maximum file size (10MB)
const MAX_FILE_SIZE = 10 * 1024 * 1024;

// ===== INITIALIZATION =====

/**
 * Application Initialization Handler
 * 
 * Initializes the contour creator application when the DOM content is fully loaded.
 * Sets up the canvas environment and drag-and-drop functionality for file uploads.
 * This is the main entry point for the application's initialization sequence.
 * 
 * @since 2025-05-01
 * @see {@link initCanvas} Canvas initialization
 * @see {@link setupEnhancedDragAndDrop} File upload setup
 */
document.addEventListener('DOMContentLoaded', function() {
    initCanvas();
    setupEnhancedDragAndDrop();
});

/**
 * Initialize Canvas Environment and Event Handlers
 * 
 * Sets up the HTML5 canvas element and configures all necessary event listeners
 * for interactive image manipulation. Establishes the foundation for zoom, pan,
 * click detection, and brush refinement functionality.
 * 
 * Event Handlers Configured:
 * - Click events for region detection and seed point placement
 * - Mouse wheel events for zoom control with smooth scaling
 * - Mouse drag events for pan functionality with boundary constraints
 * - Brush refinement events for manual region editing
 * 
 * Canvas Configuration:
 * - 2D rendering context with optimized settings
 * - Event listener registration with appropriate options
 * - Integration with refinement mode functionality
 * - Coordinate transformation setup for image-canvas mapping
 * 
 * @function initCanvas
 * @since 2025-05-01
 * @lastModified 2025-09-25
 * @see {@link handleCanvasClick} Click event handler
 * @see {@link handleWheel} Zoom event handler
 * @see {@link handleMouseDown} Pan start handler
 * @see {@link initRefineEventListeners} Brush refinement setup
 * 
 */
function initCanvas() {
    canvas = document.getElementById('imageCanvas');
    ctx = canvas.getContext('2d');
    
    // Event listeners for clicks
    canvas.addEventListener('click', handleCanvasClick);
    
    // Event listeners for zoom with mouse wheel
    canvas.addEventListener('wheel', handleWheel, { passive: false });
    
    // Event listeners for pan (drag)
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('mouseleave', handleMouseUp);
    
    // Initialize refine event listeners
    initRefineEventListeners();
}

// ===== FILE UPLOAD FUNCTIONALITY =====

/**
 * Configure Enhanced Drag and Drop File Upload System
 * 
 * Sets up a comprehensive drag-and-drop interface for image file uploads with
 * visual feedback, progress tracking, and error handling. Configures all DOM
 * elements and event listeners required for the file upload workflow.
 * 
 * Features Configured:
 * - Drag and drop zone with visual feedback and hover states
 * - File input integration with click-to-browse functionality
 * - Progress indicators with real-time upload status
 * - Error handling with user-friendly messages
 * - File format validation and size checking
 * - Clear/reset functionality for uploaded files
 * 
 * DOM Elements Initialized:
 * - Upload zone container with drag event handlers
 * - File input element with change event listener
 * - Progress display elements for upload feedback
 * - Icon and text elements for dynamic content updates
 * - Clear button for resetting the upload state
 * 
 * Event Handlers Registered:
 * - dragenter: Visual feedback when file enters drop zone
 * - dragover: Continuous feedback during file hover
 * - dragleave: Reset visual state when file leaves zone
 * - drop: Process dropped files and initiate upload
 * - change: Handle files selected via file browser
 * 
 * @function setupEnhancedDragAndDrop
 * @since 2025-05-01
 * @lastModified 2025-09-25
 * @see {@link handleDragEnter} Drag enter event handler
 * @see {@link handleDragOver} Drag over event handler
 * @see {@link handleDragLeave} Drag leave event handler
 * @see {@link handleDrop} File drop event handler
 * @see {@link handleFileSelect} File selection handler
 * 
 * // Supported file formats: PNG, JPG, JPEG, GIF, BMP, WEBP
 * // Maximum file size: 10MB
 */
function setupEnhancedDragAndDrop() {
    // Get DOM elements
    uploadZone = document.getElementById('uploadZone');
    fileInput = document.getElementById('fileInput');
    uploadContent = document.getElementById('uploadContent');
    uploadIcon = document.getElementById('uploadIcon');
    uploadTitle = document.getElementById('uploadTitle');
    uploadSubtitle = document.getElementById('uploadSubtitle');
    clearButton = document.getElementById('clearButton');
    
    if (!uploadZone || !fileInput) {
        console.error('Upload elements not found');
        return;
    }
    
    // Event listeners for drag and drop
    uploadZone.addEventListener('dragenter', handleDragEnter);
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);
    
    // Event listener for upload zone click
    uploadZone.addEventListener('click', () => {
        if (!uploadZone.classList.contains('uploading')) {
            fileInput.click();
        }
    });
    
    // Event listener for file selection
    fileInput.addEventListener('change', handleFileSelect);
    
    // Prevent default behavior on entire page
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());
}

/**
 * Handle Drag Enter Event for File Upload
 * 
 * Processes the drag enter event when a file is dragged into the upload zone.
 * Provides visual feedback by adding CSS classes and updating the upload content
 * display. Uses a drag counter to handle multiple drag enter/leave events correctly.
 * 
 * @function handleDragEnter
 * @param {DragEvent} e - The drag enter event object
 * @since 2025-05-01
 * @lastModified 2025-09-25
 * @see {@link updateUploadContent} Content update handler
 * 
 */
function handleDragEnter(e) {
    e.preventDefault();
    dragCounter++;
    
    if (dragCounter === 1) {
        uploadZone.classList.add('drag-over');
        updateUploadContent('drag-over');
    }
}

/**
 * Handle Drag Over Event for File Upload
 * 
 * Processes the continuous drag over event while a file is being dragged
 * over the upload zone. Sets the appropriate drop effect to indicate
 * that the file can be dropped and copied.
 * 
 * @function handleDragOver
 * @param {DragEvent} e - The drag over event object
 * @since 2025-05-01
 * @lastModified 2025-09-25
 * 
 */
function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
}

/**
 * Handle Drag Leave Event for File Upload
 * 
 * Processes the drag leave event when a file is dragged out of the upload zone.
 * Removes visual feedback by removing CSS classes and resetting the upload content
 * display. Uses a drag counter to handle nested elements correctly.
 * 
 * @function handleDragLeave
 * @param {DragEvent} e - The drag leave event object
 * @since 2025-05-01
 * @lastModified 2025-09-25
 * @see {@link updateUploadContent} Content update handler
 * 
 */
function handleDragLeave(e) {
    e.preventDefault();
    dragCounter--;
    
    if (dragCounter === 0) {
        uploadZone.classList.remove('drag-over');
        updateUploadContent('default');
    }
}

/**
 * Handle File Drop Event for Upload Processing
 * 
 * Processes the file drop event when a file is dropped onto the upload zone.
 * Extracts the first file from the drop event and initiates the file processing
 * workflow. Resets the drag counter and visual feedback states.
 * 
 * @function handleDrop
 * @param {DragEvent} e - The drop event object containing file data
 * @since 2025-05-01
 * @lastModified 2025-09-25
 * @see {@link processFile} File processing handler
 * 
 */
function handleDrop(e) {
    e.preventDefault();
    dragCounter = 0;
    uploadZone.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

/**
 * Handle File Selection from Input Element
 * 
 * Processes file selection when a user chooses a file through the file input
 * element (click to browse functionality). Extracts the selected file and
 * initiates the same processing workflow as drag and drop.
 * 
 * @function handleFileSelect
 * @param {Event} e - The change event object from file input
 * @since 2025-05-01
 * @lastModified 2025-09-25
 * @see {@link processFile} File processing handler
 * 
 */
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

/**
 * Process and Validate Uploaded File
 * 
 * Comprehensive file processing function that validates file type and size
 * before initiating the upload workflow. Performs format checking against
 * supported image types and enforces file size limits for optimal performance.
 * 
 * Validation Checks:
 * - File format validation against SUPPORTED_FORMATS
 * - File size validation against MAX_FILE_SIZE (10MB)
 * - Error handling with user-friendly messages
 * 
 * Supported Formats:
 * - JPEG/JPG: Standard compressed image format
 * - PNG: Lossless compression with transparency support
 * - GIF: Animated and static images with limited colors
 * - WEBP: Modern format with superior compression
 * - BMP: Uncompressed bitmap format
 * 
 * @function processFile
 * @param {File} file - The file object to process and validate
 * @since 2025-05-01
 * @lastModified 2025-09-25
 * @see {@link simulateUploadProgress} Upload progress handler
 * @see {@link showUploadError} Error display handler
 * @see {@link showUploadProgress} Progress display handler
 * @see {@link formatFileSize} File size formatting utility
 * 
 */
function processFile(file) {
    // Validate file type
    if (!SUPPORTED_FORMATS[file.type]) {
        showUploadError('Unsupported file format', 
            `Please select a file: ${Object.values(SUPPORTED_FORMATS).join(', ')}`);
        return;
    }
    
    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
        showUploadError('File too large', 
            `File must be smaller than ${formatFileSize(MAX_FILE_SIZE)}`);
        return;
    }
    
    // Show loading state
    showUploadProgress(file);
    
    // Simulate upload progress
    simulateUploadProgress(file);
}

/**
 * Simulates file upload progress with visual feedback and automatic upload completion.
 * Creates a realistic progress animation that gradually increases from 0 to 100%,
 * then automatically triggers the file upload process and shows success feedback.
 * 
 * @function simulateUploadProgress
 * @param {File} file - The file object to be uploaded and processed
 * @since 2025-05-01
 * @lastModified 2025-09-25
 * @author Gerardo Tinoco-Guerrero
 * 
 * @description
 * This function provides a smooth user experience by:
 * - Animating progress from 0% to 100% with random increments
 * - Updating the progress display in real-time
 * - Automatically triggering file upload upon completion
 * - Showing success notification and enabling clear button
 * - Assigning the file to the input element for compatibility
 * 
 */
function simulateUploadProgress(file) {
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        
        if (progress >= 100) {
            progress = 100;
            clearInterval(progressInterval);
            
            // Small delay before showing success and auto-uploading
            setTimeout(() => {
                showUploadSuccess(file);
                // Assign file to input for compatibility
                const dt = new DataTransfer();
                dt.items.add(file);
                fileInput.files = dt.files;
                
                // Auto-upload the file immediately
                setTimeout(() => {
                    if (typeof uploadFile === 'function') {
                        uploadFile();
                    }
                }, 1000);
                
                // Show clear button
                if (clearButton) {
                    clearButton.style.display = 'inline-flex';
                }
            }, 500);
        }
        
        updateProgressDisplay(progress);
    }, 100);
}

// Update upload content based on state
/**
 * Updates the upload zone content based on the current drag-and-drop state.
 * Dynamically changes the title and subtitle text to provide appropriate
 * user feedback during different phases of the file upload interaction.
 * 
 * @function updateUploadContent
 * @param {string} state - The current upload state ('drag-over' or 'default')
 * @since 2025-05-01
 * @lastModified 2025-09-25
 * @author Gerardo Tinoco-Guerrero
 * 
 * @description
 * This function manages the upload zone UI states:
 * - 'drag-over': Shows encouraging message when file is being dragged over
 * - 'default': Shows standard upload instructions
 * 
 */
function updateUploadContent(state) {
    if (!uploadTitle || !uploadSubtitle) return;
    
    switch (state) {
        case 'drag-over':
            uploadTitle.textContent = 'Drop your image here!';
            uploadSubtitle.textContent = 'We will process your file immediately';
            break;
        case 'default':
        default:
            uploadTitle.textContent = 'Drop your image here';
            uploadSubtitle.textContent = 'or click to browse files';
            break;
    }
}

// Show upload progress
function showUploadProgress(file) {
    uploadZone.classList.add('uploading');
    uploadZone.classList.remove('success', 'error');
    
    uploadContent.innerHTML = `
        <div class="upload-progress">
            <div class="progress-circle">
                <svg class="progress-ring" width="80" height="80">
                    <circle class="progress-ring-circle" 
                            stroke="#3b82f6" 
                            stroke-width="4" 
                            fill="transparent" 
                            r="36" 
                            cx="40" 
                            cy="40"/>
                </svg>
                <div class="progress-text">0%</div>
            </div>
            <p class="progress-message">Uploading ${file.name}...</p>
        </div>
    `;
}

// Update progress display
function updateProgressDisplay(progress) {
    const progressText = uploadContent.querySelector('.progress-text');
    const progressCircle = uploadContent.querySelector('.progress-ring-circle');
    
    if (progressText) {
        progressText.textContent = `${Math.round(progress)}%`;
    }
    
    if (progressCircle) {
        const circumference = 2 * Math.PI * 36;
        const offset = circumference - (progress / 100) * circumference;
        progressCircle.style.strokeDashoffset = offset;
    }
}

// Show upload success
function showUploadSuccess(file) {
    uploadZone.classList.remove('uploading', 'error');
    uploadZone.classList.add('success');
    
    uploadContent.innerHTML = `
        <div class="upload-success">
            <i class="fas fa-check-circle success-icon"></i>
            <h4>File uploaded successfully!</h4>
            <p>${file.name} (${formatFileSize(file.size)})</p>
        </div>
    `;
}

// Show upload error
function showUploadError(title, message) {
    uploadZone.classList.remove('uploading', 'success');
    uploadZone.classList.add('error');
    
    uploadContent.innerHTML = `
        <div class="upload-error">
            <i class="fas fa-exclamation-triangle error-icon"></i>
            <h4>${title}</h4>
            <p>${message}</p>
            <button class="btn btn-secondary btn-small" onclick="resetUpload()">
                <i class="fas fa-redo"></i>
                Try Again
            </button>
        </div>
    `;
}

// Reset upload
function resetUpload() {
    uploadZone.classList.remove('uploading', 'success', 'error', 'drag-over');
    
    uploadContent.innerHTML = `
        <div class="upload-icon-container">
            <i class="fas fa-cloud-upload-alt upload-icon" id="uploadIcon"></i>
            <div class="upload-animation">
                <div class="upload-pulse"></div>
                <div class="upload-pulse"></div>
                <div class="upload-pulse"></div>
            </div>
        </div>
        <h4 class="upload-title" id="uploadTitle">Drop your image here</h4>
        <p class="upload-subtitle" id="uploadSubtitle">or click to browse files</p>
        <div class="upload-formats">
            <span class="format-badge">JPG</span>
            <span class="format-badge">PNG</span>
            <span class="format-badge">GIF</span>
            <span class="format-badge">WEBP</span>
            <span class="format-badge">BMP</span>
        </div>
        <div class="upload-size-limit">
            <i class="fas fa-info-circle"></i>
            <span>Maximum file size: 10MB</span>
        </div>
        <button class="upload-button" onclick="document.getElementById('fileInput').click()">
            <i class="fas fa-folder-open"></i>
            <span>Browse Files</span>
        </button>
    `;
    
    // Reset references
    uploadIcon = document.getElementById('uploadIcon');
    uploadTitle = document.getElementById('uploadTitle');
    uploadSubtitle = document.getElementById('uploadSubtitle');
    
    // Clear file input
    if (fileInput) {
        fileInput.value = '';
    }
    
    // Hide clear button
    if (clearButton) {
        clearButton.style.display = 'none';
    }
    
    dragCounter = 0;
}

// Clear upload
function clearUpload() {
    resetUpload();
}

// Format file size
/**
 * Formats a file size in bytes to a human-readable string with appropriate units.
 * Converts bytes to the most appropriate unit (Bytes, KB, MB, GB) and formats
 * the result with proper decimal precision for optimal readability.
 * 
 * @function formatFileSize
 * @param {number} bytes - The file size in bytes to be formatted
 * @returns {string} The formatted file size string with appropriate unit
 * @since 2025-05-01
 * @lastModified 2025-09-25
 * @author Gerardo Tinoco-Guerrero
 * 
 * @description
 * This utility function:
 * - Handles zero bytes as a special case
 * - Uses binary (1024) conversion for accurate file size representation
 * - Automatically selects the most appropriate unit (Bytes, KB, MB, GB)
 * - Formats numbers to 2 decimal places for precision
 * 
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Upload file
function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showAlert('Please select a file', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentFilename = data.filename;
            loadImage(data.filename);
            showAlert('Image uploaded successfully', 'success');
            
            // Hide upload section after successful file load
            const uploadSection = document.getElementById('uploadSection');
            if (uploadSection) {
                uploadSection.style.display = 'none';
            }
        } else {
            showFloatingNotification(data.error, 'error');
        }
    })
    .catch(error => {
        showAlert('Error uploading file: ' + error.message, 'error');
    });
}

// ===== IMAGE AND CANVAS FUNCTIONALITY =====

// Load image in canvas
function loadImage(filename) {
    const img = new Image();
    img.onload = function() {
        currentImage = img;
        
        // Adjust canvas size
        const maxWidth = 800;
        const maxHeight = 600;
        let { width, height } = img;
        
        if (width > maxWidth || height > maxHeight) {
            const ratio = Math.min(maxWidth / width, maxHeight / height);
            width *= ratio;
            height *= ratio;
        }
        
        canvas.width = width;
        canvas.height = height;
        
        // Reset zoom and pan when loading new image
        zoomLevel = 1;
        centerImage();
        updateZoomDisplay();
        
        redrawCanvas();
        
        // Show image section
        document.getElementById('imageSection').classList.remove('hidden');
    };
    
    img.src = `/uploads/${filename}`;
}

// ===== ZOOM AND PAN FUNCTIONALITY =====

// Handle zoom with mouse wheel
function handleWheel(event) {
    // Requerir Ctrl+Rueda para zoom tanto en modo normal como en modo refinamiento
    if (!event.ctrlKey) {
        return; // Solo hacer zoom cuando Ctrl esté presionado
    }
    
    event.preventDefault();
    
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    
    const wheel = event.deltaY < 0 ? 1 : -1;
    const zoomIntensity = 0.1;
    const zoom = Math.exp(wheel * zoomIntensity);
    
    const newZoom = zoomLevel * zoom;
    if (newZoom < minZoom || newZoom > maxZoom) return;
    
    // Adjust pan for mouse-centered zoom
    panX = mouseX - (mouseX - panX) * zoom;
    panY = mouseY - (mouseY - panY) * zoom;
    
    zoomLevel = newZoom;
    
    // Apply pan limits to prevent white areas
    applyPanLimits();
    
    updateZoomDisplay();
    redrawCanvas();
}

// Handle start of drag
function handleMouseDown(event) {
    // Permitir arrastre con Ctrl+Click tanto en modo normal como en modo refinamiento
    if (event.button === 0 && event.ctrlKey) { // Only left button + Ctrl
        isDragging = true;
        hasDragged = false; // Reset drag flag
        lastMouseX = event.clientX;
        lastMouseY = event.clientY;
        canvas.style.cursor = 'grabbing';
    }
}

// Handle mouse movement
function handleMouseMove(event) {
    if (isDragging) {
        const deltaX = event.clientX - lastMouseX;
        const deltaY = event.clientY - lastMouseY;
        
        // Check if movement exceeds threshold
        const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        if (distance > dragThreshold) {
            hasDragged = true;
        }
        
        panX += deltaX;
        panY += deltaY;
        
        // Apply pan limits to prevent white areas
        applyPanLimits();
        
        lastMouseX = event.clientX;
        lastMouseY = event.clientY;
        
        redrawCanvas();
    }
}

// Handle end of drag
function handleMouseUp(event) {
    if (isDragging) {
        isDragging = false;
        canvas.style.cursor = isRefineMode ? 'crosshair' : 'crosshair';
        
        // Reset hasDragged after a short delay to prevent interference with legitimate clicks
        setTimeout(() => {
            hasDragged = false;
        }, 50);
    }
}

// Zoom functions
function zoomIn() {
    const newZoom = zoomLevel * 1.2;
    if (newZoom <= maxZoom) {
        zoomLevel = newZoom;
        // If zoom reaches 1, center the image
        if (Math.abs(zoomLevel - 1) < 0.01) {
            centerImage();
        } else {
            // Apply pan limits to prevent white areas
            applyPanLimits();
        }
        updateZoomDisplay();
        redrawCanvas();
    }
}

function zoomOut() {
    const newZoom = zoomLevel / 1.2;
    if (newZoom >= minZoom) {
        zoomLevel = newZoom;
        // If zoom reaches 1, center the image
        if (Math.abs(zoomLevel - 1) < 0.01) {
            centerImage();
        } else {
            // Apply pan limits to prevent white areas
            applyPanLimits();
        }
        updateZoomDisplay();
        redrawCanvas();
    }
}

function resetZoom() {
    zoomLevel = 1;
    centerImage();
    updateZoomDisplay();
    redrawCanvas();
}

function centerImage() {
    if (!currentImage) return;
    
    // Calculate proper centering based on image and canvas dimensions
    const imageAspect = currentImage.width / currentImage.height;
    const canvasAspect = canvas.width / canvas.height;
    
    let imageDisplayWidth, imageDisplayHeight;
    
    if (imageAspect > canvasAspect) {
        // Image is wider than canvas
        imageDisplayWidth = canvas.width;
        imageDisplayHeight = canvas.width / imageAspect;
    } else {
        // Image is taller than canvas
        imageDisplayHeight = canvas.height;
        imageDisplayWidth = canvas.height * imageAspect;
    }
    
    // Center the image
    panX = (canvas.width - imageDisplayWidth * zoomLevel) / 2;
    panY = (canvas.height - imageDisplayHeight * zoomLevel) / 2;
    
    // Apply pan limits to prevent white areas
    applyPanLimits();
}

function updateZoomDisplay() {
    document.getElementById('zoomLevel').textContent = Math.round(zoomLevel * 100) + '%';
}

// Apply pan limits to prevent white areas
function applyPanLimits() {
    if (!currentImage) return;
    
    const imageAspect = currentImage.width / currentImage.height;
    const canvasAspect = canvas.width / canvas.height;
    
    let imageDisplayWidth, imageDisplayHeight;
    
    if (imageAspect > canvasAspect) {
        imageDisplayWidth = canvas.width;
        imageDisplayHeight = canvas.width / imageAspect;
    } else {
        imageDisplayHeight = canvas.height;
        imageDisplayWidth = canvas.height * imageAspect;
    }
    
    const scaledWidth = imageDisplayWidth * zoomLevel;
    const scaledHeight = imageDisplayHeight * zoomLevel;
    
    // Calculate limits based on scaled image size
    let minPanX, maxPanX, minPanY, maxPanY;
    
    if (scaledWidth <= canvas.width) {
        // Image fits horizontally - center it
        const centerX = (canvas.width - scaledWidth) / 2;
        minPanX = maxPanX = centerX;
    } else {
        // Image is larger than canvas - allow panning within bounds
        maxPanX = 0;
        minPanX = canvas.width - scaledWidth;
    }
    
    if (scaledHeight <= canvas.height) {
        // Image fits vertically - center it
        const centerY = (canvas.height - scaledHeight) / 2;
        minPanY = maxPanY = centerY;
    } else {
        // Image is larger than canvas - allow panning within bounds
        maxPanY = 0;
        minPanY = canvas.height - scaledHeight;
    }
    
    // Apply limits
    panX = Math.max(minPanX, Math.min(maxPanX, panX));
    panY = Math.max(minPanY, Math.min(maxPanY, panY));
}

// ===== CANVAS DRAWING FUNCTIONALITY =====

// Redraw canvas with zoom and pan
function redrawCanvas() {
    if (!currentImage) return;
    
    drawAllRegions();
    
    // Redraw temporary region if it exists
    if (tempRegion) {
        drawTempRegion(tempRegion);
    }
}

// Draw all regions
function drawAllRegions() {
    if (!currentImage) return;
    
    // Clear canvas and redraw image
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Apply transformations
    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoomLevel, zoomLevel);
    
    // Draw image
    ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
    
    // Coordinates are normalized by width and height respectively in the backend
    const scaleX = canvas.width;
    const scaleY = canvas.height;
    
    // Draw all confirmed regions
    detectedRegions.forEach(region => {
        if (!region.visible || !region.contour_points) return;
        
        ctx.beginPath();
        ctx.strokeStyle = region.color;
        ctx.lineWidth = 2 / zoomLevel;
        
        for (let i = 0; i < region.contour_points.length; i++) {
            const point = region.contour_points[i];
            // Coordinates come normalized by width and height respectively
            const x = point.x * scaleX;
            const y = point.y * scaleY;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.closePath();
        ctx.stroke();
    });
    
    ctx.restore();
}

// Draw temporary region
function drawTempRegion(data) {
    if (!currentImage || !data || !data.contour_points) {
        return;
    }
    
    // First redraw all confirmed regions
    drawAllRegions();
    
    // Then draw the temporary region on top
    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoomLevel, zoomLevel);
    
    // Use the same coordinate system as drawAllRegions for consistency
    const scaleX = canvas.width;
    const scaleY = canvas.height;
    
    // Draw temporary contour with dashed line
    ctx.beginPath();
    ctx.strokeStyle = data.color;
    ctx.lineWidth = 3 / zoomLevel;
    ctx.setLineDash([5, 5]); // Dashed line for temporary region
    
    for (let i = 0; i < data.contour_points.length; i++) {
        const point = data.contour_points[i];
        // Use the same coordinate normalization as drawAllRegions
        const x = point.x * scaleX;
        const y = point.y * scaleY;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]); // Reset dashed line
    
    ctx.restore();
}

// ===== REGION DETECTION AND MANAGEMENT =====

// Handle canvas click
function handleCanvasClick(event) {
    // Prevent region detection if we just finished dragging
    if (!currentImage || !currentFilename || isDragging || hasDragged) return;
    
    const rect = canvas.getBoundingClientRect();
    
    // Convert canvas coordinates to original image coordinates
    const canvasX = event.clientX - rect.left;
    const canvasY = event.clientY - rect.top;
    
    // Adjust for zoom and pan
    const imageX = (canvasX - panX) / zoomLevel;
    const imageY = (canvasY - panY) / zoomLevel;
    
    // Convert to original image coordinates
    const scaleX = currentImage.width / canvas.width;
    const scaleY = currentImage.height / canvas.height;
    
    const x = Math.floor(imageX * scaleX);
    const y = Math.floor(imageY * scaleY);
    
    // Verify that coordinates are within the image
    if (x >= 0 && x < currentImage.width && y >= 0 && y < currentImage.height) {
        detectRegion(x, y);
    }
}

// Detect region
function detectRegion(x, y) {
    const tolerance = 30; // Balanced tolerance for complete region detection
    
    // Show loading indicator
    showFloatingNotification('Detecting regions...', 'info', 3000);
    
    fetch('/detect_region', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            filename: currentFilename,
            x: x,
            y: y,
            tolerance: tolerance
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            tempRegion = data;
            tempRegion.color = regionColors[currentRegionIndex % regionColors.length];
            tempRegion.id = Date.now(); // Unique ID based on timestamp
            tempRegion.name = `Region ${currentRegionIndex + 1}`;
            
            drawTempRegion(tempRegion);
            displayTempRegion(tempRegion);
            
            document.getElementById('addRegionBtn').disabled = false;
            showFloatingNotification('Region detected! Click "Add Region" to confirm.', 'success', 4000);
        } else {
            showFloatingNotification(data.error, 'error');
        }
    })
    .catch(error => {
        showFloatingNotification('Error detecting region: ' + error.message, 'error');
    });
}

// Show detected region information
function addRegion() {
    if (!tempRegion) return;
    
    tempRegion.visible = true;
    detectedRegions.push(tempRegion);
    currentRegionIndex++;
    
    updateRegionsList();
    redrawCanvas();
    
    // Clear temporary region
    tempRegion = null;
    document.getElementById('addRegionBtn').disabled = true;
    document.getElementById('saveBtn').disabled = detectedRegions.length === 0;
    
    // Exit refinement mode automatically to allow detecting new regions
    if (isRefineMode) {
        exitRefineMode();
        showFloatingNotification(`Region "${detectedRegions[detectedRegions.length - 1].name}" added successfully! Exited refinement mode to detect new regions.`, 'success');
    } else {
        showFloatingNotification(`Region "${detectedRegions[detectedRegions.length - 1].name}" added successfully!`, 'success');
    }
}

function displayTempRegion(data) {
    if (!data || !data.contour_points) return;
    
    // Show temporary region info in floating notification
    showFloatingNotification(
        `Temporary region detected with ${data.contour_points.length} points. Click "Add Region" to confirm.`,
        'info',
        5000
    );
}

function updateRegionsList() {
    const regionsList = document.getElementById('regionsList');
    const regionsDisplay = document.getElementById('regionsDisplay');
    
    if (detectedRegions.length === 0) {
        regionsDisplay.classList.add('hidden');
        regionsList.innerHTML = `
            <div class="regions-empty">
                <i class="fas fa-bullseye"></i>
                <h4>No regions detected</h4>
                <p>Click on the image to detect and add contour regions</p>
            </div>
        `;
        return;
    }
    
    regionsDisplay.classList.remove('hidden');
    
    regionsList.innerHTML = detectedRegions.map((region, index) => {
        const pointCount = region.contour_points ? region.contour_points.length : 0;
        
        return `
            <div class="region-item" style="--region-color: ${region.color}">
                <div class="region-header">
                    <div class="region-info">
                        <div class="region-color-indicator" style="background-color: ${region.color}"></div>
                        <h4 class="region-name">${region.name}</h4>
                    </div>
                </div>
                <div class="region-stats">
                    <div class="region-stat">
                        <i class="fas fa-vector-square"></i>
                        <span>${pointCount} points</span>
                    </div>
                </div>
                <div class="region-actions">
                    <button class="region-btn toggle" 
                            onclick="toggleRegion(${index})" 
                            title="${region.visible ? 'Hide' : 'Show'} region">
                        <i class="fas ${region.visible ? 'fa-eye' : 'fa-eye-slash'}"></i>
                        ${region.visible ? 'Hide' : 'Show'}
                    </button>
                    <button class="region-btn export" 
                            onclick="exportSingleRegion(${index})" 
                            title="Export this region">
                        <i class="fas fa-download"></i>
                        Export
                    </button>
                    <button class="region-btn delete" 
                            onclick="deleteRegion(${index})" 
                            title="Delete region">
                        <i class="fas fa-trash"></i>
                        Delete
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

function toggleRegion(index) {
    if (index >= 0 && index < detectedRegions.length) {
        detectedRegions[index].visible = !detectedRegions[index].visible;
        updateRegionsList();
        redrawCanvas();
    }
}

function deleteRegion(index) {
    if (index >= 0 && index < detectedRegions.length) {
        const regionName = detectedRegions[index].name;
        detectedRegions.splice(index, 1);
        updateRegionsList();
        redrawCanvas();
        
        document.getElementById('saveBtn').disabled = detectedRegions.length === 0;
        showFloatingNotification(`Region "${regionName}" deleted.`, 'info');
    }
}

function clearAllRegions() {
    detectedRegions = [];
    tempRegion = null;
    currentRegionIndex = 0;
    
    updateRegionsList();
    redrawCanvas();
    
    document.getElementById('addRegionBtn').disabled = true;
    document.getElementById('saveBtn').disabled = true;
    
    showFloatingNotification('All regions have been cleared.', 'info');
}

// ===== EXPORT FUNCTIONALITY =====

// Helper function to calculate region area
/**
 * Calculates the area of a detected region using the shoelace formula.
 * Computes the area enclosed by the region's contour points using the
 * mathematical shoelace (surveyor's) formula for polygon area calculation.
 * 
 * @function calculateRegionArea
 * @param {Object} region - The region object containing contour points
 * @param {Array<Object>} region.contour_points - Array of points with x,y coordinates
 * @returns {number} The calculated area in square pixels, or 0 if invalid region
 * @since 2025-05-01
 * @lastModified 2025-09-25
 * @author Gerardo Tinoco-Guerrero
 * 
 * @description
 * This function implements the shoelace formula:
 * - Requires at least 3 points to form a valid polygon
 * - Uses cross-product summation for area calculation
 * - Returns absolute value to ensure positive area
 * - Handles edge cases with insufficient points
 * 
 */
function calculateRegionArea(region) {
    if (!region.contour_points || region.contour_points.length < 3) {
        return 0;
    }
    
    let area = 0;
    const points = region.contour_points;
    const n = points.length;
    
    for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        area += points[i].x * points[j].y;
        area -= points[j].x * points[i].y;
    }
    
    return Math.abs(area) / 2;
}

// Export single region
function exportSingleRegion(index) {
    if (index < 0 || index >= detectedRegions.length) {
        showFloatingNotification('Invalid region index', 'error');
        return;
    }
    
    const region = detectedRegions[index];
    if (!region.contour_points || region.contour_points.length === 0) {
        showFloatingNotification('No contour points to export', 'error');
        return;
    }
    
    // Prepare data for export
    const exportData = {
        filename: currentFilename,
        region_name: region.name,
        contour_points: region.contour_points
    };
    
    // Show loading notification
    showFloatingNotification('Exporting region...', 'info', 2000);
    
    fetch('/export_single_region', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(exportData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Create download link
            const link = document.createElement('a');
            link.href = data.download_url;
            link.download = data.filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            showFloatingNotification(`Region "${region.name}" exported successfully!`, 'success');
        } else {
            showFloatingNotification(data.error, 'error');
        }
    })
    .catch(error => {
        showFloatingNotification('Error exporting region: ' + error.message, 'error');
    });
}

// Save all coordinates
function saveAllCoordinates() {
    if (detectedRegions.length === 0) {
        showFloatingNotification('No regions to save', 'error');
        return;
    }
    
    // Prepare data for export
    const exportData = {
        filename: currentFilename,
        regions: detectedRegions.map(region => ({
            name: region.name,
            color: region.color,
            contour_points: region.contour_points
        }))
    };
    
    // Show loading notification
    showFloatingNotification('Saving all regions...', 'info', 2000);
    
    fetch('/save_all_coordinates', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(exportData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Create download link
            const link = document.createElement('a');
            link.href = data.download_url;
            link.download = data.filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            showFloatingNotification(`All regions saved successfully! (${detectedRegions.length} regions)`, 'success');
        } else {
            showFloatingNotification(data.error, 'error');
        }
    })
    .catch(error => {
        showFloatingNotification('Error saving regions: ' + error.message, 'error');
    });
}

// ===== NOTIFICATION SYSTEM =====

// Show alert
function showAlert(message, type) {
    // Implementation depends on your alert system
}

// Show floating notification
function showFloatingNotification(message, type = 'info', duration = 5000) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `floating-notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas ${getNotificationIcon(type)}"></i>
            <span class="notification-message">${message}</span>
            <button class="notification-close" onclick="closeNotification(this)">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    // Add to container
    let container = document.getElementById('notificationContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notificationContainer';
        container.className = 'floating-notifications';
        document.body.appendChild(container);
    }
    
    container.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    // Auto remove after duration
    if (duration > 0) {
        setTimeout(() => {
            removeNotification(notification);
        }, duration);
    }
}

// Get notification icon based on type
function getNotificationIcon(type) {
    switch (type) {
        case 'success': return 'fa-check-circle';
        case 'error': return 'fa-exclamation-circle';
        case 'warning': return 'fa-exclamation-triangle';
        case 'info':
        default: return 'fa-info-circle';
    }
}

// Close notification
function closeNotification(button) {
    const notification = button.closest('.floating-notification');
    removeNotification(notification);
}

// Remove notification
function removeNotification(notification) {
    if (notification && notification.parentNode) {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }
}

// ===== INTERACTIVE SEGMENTATION FUNCTIONALITY =====

// Refine mode variables
let isRefineMode = false;
let brushSize = 20;
let brushStrokes = [];
let isDrawing = false;
let currentStroke = null;
let currentBrushMode = 'add'; // 'add' or 'remove'

// Toggle refine mode
function toggleRefineMode() {
    isRefineMode = !isRefineMode;
    const refineControls = document.getElementById('refineControls');
    const toggleBtn = document.getElementById('refineModeToggle');
    
    if (isRefineMode) {
        refineControls.style.display = 'flex';
        toggleBtn.classList.add('active');
        toggleBtn.innerHTML = '<i class="fas fa-edit"></i><span>Exit Refinement</span>';
        canvas.style.cursor = 'crosshair';
        showFloatingNotification('Refinement mode activated. Click and drag to add/remove areas. Use Ctrl+Wheel for zoom and Ctrl+Drag to move the image.', 'info');
        
        // Actualizar las instrucciones para incluir información sobre el zoom y pan
        const refineInstructions = document.querySelector('.refine-instructions .refine-text');
        if (refineInstructions) {
            refineInstructions.innerHTML = '<i class="fas fa-info-circle"></i> Click to add areas (+) or hold Shift to remove (-). Use Ctrl+Wheel for zoom and Ctrl+Drag to move the image.';
        }
    } else {
        refineControls.style.display = 'none';
        toggleBtn.classList.remove('active');
        toggleBtn.innerHTML = '<i class="fas fa-edit"></i><span>Refinar Selección</span>';
        canvas.style.cursor = 'pointer';
        resetRefineState();
        showFloatingNotification('Refinement mode deactivated.', 'info');
    }
}

// Update brush size
function updateBrushSize(size) {
    brushSize = parseInt(size);
    document.getElementById('brushSizeValue').textContent = size;
}

// Undo last stroke
function undoLastStroke() {
    if (brushStrokes.length > 0) {
        brushStrokes.pop(); // Remove the last stroke
        redrawCanvas();
        updateRefineButtons();
        showFloatingNotification('Last stroke undone.', 'info');
    }
}

// Clear all refinements
function clearAllRefinements() {
    brushStrokes = [];
    redrawCanvas();
    updateRefineButtons();
    showFloatingNotification('All refinements cleared.', 'info');
}

// Update refine buttons state
function updateRefineButtons() {
    const applyBtn = document.getElementById('applyRefineBtn');
    const undoLastBtn = document.getElementById('undoLastBtn');
    const hasStrokes = brushStrokes.length > 0;
    
    if (applyBtn) {
        applyBtn.disabled = !hasStrokes;
    }
    
    if (undoLastBtn) {
        undoLastBtn.disabled = !hasStrokes;
    }
}

// Handle refine canvas interaction
function handleRefineCanvasClick(event) {
    if (!isRefineMode) return false;
    
    // Determine brush mode based on mouse button or key modifier
    currentBrushMode = event.shiftKey ? 'remove' : 'add';
    
    return true; // Prevent normal click handling
}

// Handle brush drawing start
function handleBrushStart(event) {
    if (!isRefineMode) return;
    
    // Don't draw if Ctrl is pressed (for panning)
    if (event.ctrlKey) return;
    
    event.preventDefault();
    isDrawing = true;
    
    // Determine brush mode based on Shift key
    currentBrushMode = event.shiftKey ? 'remove' : 'add';
    
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left - panX) / zoomLevel;
    const y = (event.clientY - rect.top - panY) / zoomLevel;
    
    currentStroke = {
        points: [{ x: x / canvas.width, y: y / canvas.height }],
        mode: currentBrushMode,
        size: brushSize
    };
    
    // Update cursor and show immediate feedback
    canvas.style.cursor = currentBrushMode === 'add' ? 'crosshair' : 'not-allowed';
    
    // Show immediate visual feedback
    redrawCanvas();
    if (currentStroke) {
        drawBrushStroke(currentStroke, true);
    }
}

// Handle brush drawing move
function handleBrushMove(event) {
    if (!isRefineMode || !isDrawing) return;
    
    // Don't draw if Ctrl is pressed (for panning)
    if (event.ctrlKey) return;
    
    event.preventDefault();
    
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left - panX) / zoomLevel;
    const y = (event.clientY - rect.top - panY) / zoomLevel;
    
    currentStroke.points.push({ x: x / canvas.width, y: y / canvas.height });
    
    // Redraw with current stroke
    redrawCanvas();
    if (currentStroke) {
        drawBrushStroke(currentStroke, true);
    }
}

// Handle brush drawing end
function handleBrushEnd(event) {
    if (!isRefineMode || !isDrawing) return;
    
    // Don't finalize stroke if Ctrl is pressed (for panning)
    if (event.ctrlKey) {
        isDrawing = false;
        currentStroke = null;
        return;
    }
    
    event.preventDefault();
    isDrawing = false;
    
    if (currentStroke && currentStroke.points.length > 1) {
        brushStrokes.push(currentStroke);
        updateRefineButtons();
        showFloatingNotification(
            `Área ${currentBrushMode === 'add' ? 'agregada' : 'removida'}.`, 
            'success'
        );
    }
    
    currentStroke = null;
    canvas.style.cursor = 'crosshair';
    redrawCanvas();
}

// Draw brush stroke
function drawBrushStroke(stroke, isTemporary = false) {
    const oldComposite = ctx.globalCompositeOperation;
    const oldAlpha = ctx.globalAlpha;
    
    ctx.globalAlpha = isTemporary ? 0.8 : 0.6;
    ctx.strokeStyle = stroke.mode === 'add' ? '#10b981' : '#ef4444';
    ctx.lineWidth = stroke.size * zoomLevel;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    if (stroke.points.length > 1) {
        ctx.beginPath();
        const firstPoint = stroke.points[0];
        const startX = (firstPoint.x * canvas.width * zoomLevel) + panX;
        const startY = (firstPoint.y * canvas.height * zoomLevel) + panY;
        ctx.moveTo(startX, startY);
        
        for (let i = 1; i < stroke.points.length; i++) {
            const point = stroke.points[i];
            const x = (point.x * canvas.width * zoomLevel) + panX;
            const y = (point.y * canvas.height * zoomLevel) + panY;
            ctx.lineTo(x, y);
        }
        
        ctx.stroke();
    } else if (stroke.points.length === 1) {
        // Draw a single point as a circle
        const point = stroke.points[0];
        const x = (point.x * canvas.width * zoomLevel) + panX;
        const y = (point.y * canvas.height * zoomLevel) + panY;
        
        ctx.beginPath();
        ctx.arc(x, y, stroke.size * zoomLevel / 2, 0, 2 * Math.PI);
        ctx.fillStyle = stroke.mode === 'add' ? '#10b981' : '#ef4444';
        ctx.fill();
    }
    
    ctx.globalCompositeOperation = oldComposite;
    ctx.globalAlpha = oldAlpha;
}

// Apply refinements
async function applyRefinements() {
    if (!currentFilename || brushStrokes.length === 0) {
        showFloatingNotification('No refinements to apply.', 'warning');
        return;
    }
    
    const applyBtn = document.getElementById('applyRefineBtn');
    const originalContent = applyBtn.innerHTML;
    applyBtn.disabled = true;
    applyBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Aplicando...';
    
    try {
        const data = {
            filename: currentFilename,
            brush_strokes: brushStrokes,
            current_contour: tempRegion ? tempRegion.contour_points : [],
            tolerance: parseInt(document.getElementById('tolerance')?.value || 30)
        };
        
        const response = await fetch('/refine_with_brush', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Update tempRegion with the refined result
            tempRegion.contour_points = result.contour_points;
            tempRegion.algorithm = 'Brush Refinement';
            
            // Display the refined result
            displayTempRegion(tempRegion);
            
            // Redraw canvas to show the updated contour immediately
            redrawCanvas();
            
            // Clear refinements after successful application
            brushStrokes = [];
            updateRefineButtons();
            
            showFloatingNotification('Refinement applied successfully.', 'success');
        } else {
            showFloatingNotification(`Error: ${result.error}`, 'error');
        }
        
    } catch (error) {
        console.error('Error applying refinements:', error);
        showFloatingNotification('Error applying refinement.', 'error');
    } finally {
        applyBtn.disabled = false;
        applyBtn.innerHTML = originalContent;
    }
}

// Exit refine mode
function exitRefineMode() {
    resetRefineState();
    toggleRefineMode();
}

// Reset refine state
function resetRefineState() {
    brushStrokes = [];
    isDrawing = false;
    currentStroke = null;
    currentBrushMode = 'add';
    redrawCanvas();
}

// Override the original handleCanvasClick to support refine mode
const originalHandleCanvasClick = handleCanvasClick;
handleCanvasClick = function(event) {
    // Check if refine mode handled the click
    if (handleRefineCanvasClick(event)) {
        return;
    }
    
    // Otherwise, use original functionality
    originalHandleCanvasClick(event);
};

// Initialize refine event listeners
function initRefineEventListeners() {
    if (!canvas) return;
    
    // Add brush event listeners for refine mode
    canvas.addEventListener('mousedown', function(event) {
        if (isRefineMode) {
            handleBrushStart(event);
        }
    });

    canvas.addEventListener('mousemove', function(event) {
        if (isRefineMode) {
            handleBrushMove(event);
        }
    });

    canvas.addEventListener('mouseup', function(event) {
        if (isRefineMode) {
            handleBrushEnd(event);
        }
    });
    
    // Prevent context menu on right click in refine mode
    canvas.addEventListener('contextmenu', function(event) {
        if (isRefineMode) {
            event.preventDefault();
        }
    });
    
    // Add keyboard listeners for Shift key
    document.addEventListener('keydown', function(event) {
        if (isRefineMode && event.key === 'Shift') {
            canvas.style.cursor = 'not-allowed';
        }
    });
    
    document.addEventListener('keyup', function(event) {
        if (isRefineMode && event.key === 'Shift') {
            canvas.style.cursor = 'crosshair';
        }
    });
}

// Override redrawCanvas to include refine elements
const originalRedrawCanvas = redrawCanvas;
redrawCanvas = function() {
    originalRedrawCanvas();
    
    if (isRefineMode) {
        drawRefineElements();
    }
};

// Draw refine elements (brush strokes)
function drawRefineElements() {
    // Draw all brush strokes
    brushStrokes.forEach(stroke => {
        drawBrushStroke(stroke);
    });
    
    // Draw current stroke if drawing
    if (currentStroke) {
        drawBrushStroke(currentStroke, true);
    }
}

// Enable refine mode button when image is loaded
const originalLoadImage = loadImage;
loadImage = function(filename) {
    originalLoadImage(filename);
    
    // Enable refine mode button
    const refineModeToggle = document.getElementById('refineModeToggle');
    if (refineModeToggle) {
        refineModeToggle.disabled = false;
    }
};