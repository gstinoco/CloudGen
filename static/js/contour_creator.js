// ContourCreator - Consolidated Module
// Complete functionality for ContourCreator including canvas operations and file upload

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

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    initCanvas();
    setupEnhancedDragAndDrop();
});

// Initialize canvas
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
}

// ===== FILE UPLOAD FUNCTIONALITY =====

// Configure optimized drag and drop
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

// Handle drag enter
function handleDragEnter(e) {
    e.preventDefault();
    dragCounter++;
    
    if (dragCounter === 1) {
        uploadZone.classList.add('drag-over');
        updateUploadContent('drag-over');
    }
}

// Handle drag over
function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
}

// Handle drag leave
function handleDragLeave(e) {
    e.preventDefault();
    dragCounter--;
    
    if (dragCounter === 0) {
        uploadZone.classList.remove('drag-over');
        updateUploadContent('default');
    }
}

// Handle file drop
function handleDrop(e) {
    e.preventDefault();
    dragCounter = 0;
    uploadZone.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

// Handle file selection
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

// Process selected file
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

// Simulate upload progress and auto-upload
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
    if (event.button === 0) { // Only left button
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
    isDragging = false;
    canvas.style.cursor = 'crosshair';
    
    // Reset hasDragged after a short delay to prevent interference with legitimate clicks
    setTimeout(() => {
        hasDragged = false;
    }, 50);
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
    
    // Coordinates are normalized by max(width, height) in the backend
    const maxDim = Math.max(currentImage.width, currentImage.height);
    const scaleX = canvas.width / currentImage.width;
    const scaleY = canvas.height / currentImage.height;
    
    // Draw all confirmed regions
    detectedRegions.forEach(region => {
        if (!region.visible || !region.contour_points) return;
        
        ctx.beginPath();
        ctx.strokeStyle = region.color;
        ctx.lineWidth = 2 / zoomLevel;
        
        for (let i = 0; i < region.contour_points.length; i++) {
            const point = region.contour_points[i];
            // Coordinates come normalized by max(width, height)
            const x = point.x * maxDim * scaleX;
            const y = point.y * maxDim * scaleY;
            
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
    
    // Coordinates are normalized by max(width, height) in the backend
    const maxDim = Math.max(currentImage.width, currentImage.height);
    const scaleX = canvas.width / currentImage.width;
    const scaleY = canvas.height / currentImage.height;
    
    // Draw temporary contour with dashed line
    ctx.beginPath();
    ctx.strokeStyle = data.color;
    ctx.lineWidth = 3 / zoomLevel;
    ctx.setLineDash([5, 5]); // Dashed line for temporary region
    
    for (let i = 0; i < data.contour_points.length; i++) {
        const point = data.contour_points[i];
        // Coordinates come normalized by max(width, height)
        const x = point.x * maxDim * scaleX;
        const y = point.y * maxDim * scaleY;
        
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
    const tolerance = 2; // Fixed tolerance for better edge detection
    
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
    
    showFloatingNotification(`Region "${detectedRegions[detectedRegions.length - 1].name}" added successfully!`, 'success');
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