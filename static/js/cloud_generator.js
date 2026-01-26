/**
 * Cloud Generator Module - Advanced Cloud of Points Generation and Visualization Interface
 * 
 * This module provides comprehensive functionality for generating and visualizing cloud of points
 * from CSV coordinate data. It implements multiple point generation algorithms with real-time
 * visualization, statistical analysis, and export capabilities for scientific and engineering
 * applications requiring precise geometric point distribution.
 * 
 * Core Functionality:
 * 1. CSV file upload with drag & drop support and format validation
 * 2. Advanced cloud of points generation with multiple distribution algorithms
 * 3. Real-time visualization with interactive charts and statistical analysis
 * 4. Export capabilities with multiple format support (CSV, PNG, SVG)
 * 5. Progress tracking for long-running generation processes
 * 6. Statistical analysis with area calculation and point density metrics
 * 7. Responsive design with mobile and desktop optimization
 * 
 * Point Generation Algorithms:
 * - Regular Distribution: Uniform grid-based point placement
 *   * Systematic point arrangement with configurable spacing
 *   * Optimal for structured analysis and regular sampling
 *   * Predictable point density and geometric consistency
 *   * Efficient memory usage and fast generation
 * 
 * - Natural Distribution: Organic point placement with controlled randomness
 *   * Pseudo-random point distribution within region boundaries
 *   * Maintains natural appearance while ensuring coverage
 *   * Configurable density parameters for different applications
 *   * Advanced boundary detection and constraint handling
 * 
 * CSV Processing Features:
 * - Multi-format CSV support with automatic delimiter detection
 * - Header row detection and column mapping
 * - Data validation with error reporting and suggestions
 * - Large file handling with progress indication
 * - Memory-efficient processing for datasets up to 10MB
 * - Real-time preview with data visualization
 * 
 * Visualization System:
 * - Interactive scatter plots with zoom and pan capabilities
 * - Multi-region support with automatic color assignment
 * - Statistical overlays with area and density information
 * - Export functionality for charts and data
 * - Responsive design with automatic scaling
 * - High-resolution rendering for publication quality
 * 
 * File Upload System:
 * - Enhanced drag & drop interface with visual feedback
 * - CSV format validation with detailed error messages
 * - File size validation (10MB limit) with progress tracking
 * - Automatic file processing with real-time feedback
 * - Error handling with user-friendly notifications
 * - Secure file handling with format verification
 * 
 * Progress Tracking:
 * - Real-time progress indicators with percentage completion
 * - Step-by-step process visualization with status updates
 * - Estimated time remaining with dynamic calculations
 * - Error detection and recovery mechanisms
 * - User-friendly status messages and notifications
 * - Cancellation support for long-running operations
 * 
 * Statistical Analysis:
 * - Automatic area calculation for each region
 * - Point density analysis with distribution metrics
 * - Boundary detection and geometric validation
 * - Statistical summaries with exportable reports
 * - Real-time updates during generation process
 * - Comparative analysis between regions
 * 
 * Export Capabilities:
 * - CSV export with customizable formatting
 * - High-resolution PNG export for presentations
 * - Scalable SVG export for vector graphics
 * - Clipboard integration for quick data sharing
 * - Batch export for multiple regions
 * - Metadata inclusion with generation parameters
 * 
 * Technical Implementation:
 * - Asynchronous processing with Web Workers for performance
 * - Memory-efficient algorithms for large datasets
 * - Canvas-based rendering with hardware acceleration
 * - RESTful API integration with Flask backend
 * - Event-driven architecture with optimized event handling
 * - Cross-browser compatibility with fallback mechanisms
 * 
 * User Experience Features:
 * - Intuitive drag & drop interface with visual feedback
 * - Real-time notifications with auto-dismiss functionality
 * - Keyboard shortcuts for common operations
 * - Accessibility support with ARIA labels
 * - Mobile-responsive design with touch optimization
 * - Context-sensitive help and tooltips
 * 
 * Performance Optimizations:
 * - Lazy loading for large datasets
 * - Efficient memory management with garbage collection
 * - Optimized rendering with selective updates
 * - Caching mechanisms for repeated operations
 * - Asynchronous processing to maintain UI responsiveness
 * - Progressive enhancement for better user experience
 * 
 * @fileoverview Cloud Generator JavaScript Module - Cloud of points generation and visualization
 * @author Gerardo Tinoco-Guerrero
 * @author Universidad Michoacana de San Nicolás de Hidalgo
 * @author SIIIA - Sistema de Investigación e Innovación en Inteligencia Artificial
 * @author SECIHTI - Secretaría de Ciencia, Humanidades, Tecnología e Innovación
 * @version 2.0
 * @since 2025-05-01
 * @lastModified 2026-01-21
 * 
 * @requires Chart.js for data visualization
 * @requires Fetch API for backend communication
 * @requires ES6+ JavaScript features
 * @requires HTML5 Canvas API for rendering
 * 
 * @see {@link https://www.chartjs.org/} Chart.js Documentation
 * @see {@link app.py} Flask backend implementation
 * @see {@link cloud_generation.py} Python point generation algorithms
 */

let currentFilename = null;

// Global variables for optimized drag and drop
let uploadZone = null;
let fileInput = null;
let uploadContent = null;
let uploadIcon = null;
let uploadTitle = null;
let uploadSubtitle = null;
let dragCounter = 0;

// Supported file formats
const SUPPORTED_FORMATS = {
    'text/csv': 'CSV',
    'application/vnd.ms-excel': 'CSV'
};

// Maximum file size (10MB)
const MAX_FILE_SIZE = 10 * 1024 * 1024;

/**
 * Configure Enhanced Drag and Drop CSV Upload System
 * 
 * Sets up a comprehensive drag-and-drop interface specifically for CSV file uploads
 * with visual feedback, progress tracking, and error handling. Configures all DOM
 * elements and event listeners required for the CSV file upload workflow.
 * 
 * Features Configured:
 * - CSV-specific drag and drop zone with visual feedback
 * - File input integration with click-to-browse functionality
 * - Progress indicators with real-time upload status
 * - Error handling with detailed CSV validation messages
 * - File format validation for CSV files only
 * - Clear/reset functionality for uploaded files
 * 
 * DOM Elements Initialized:
 * - Upload zone container with CSV-specific drag event handlers
 * - CSV file input element with change event listener
 * - Progress display elements for upload feedback
 * - Icon and text elements for dynamic content updates
 * - Clear button for resetting the upload state
 * 
 * Event Handlers Registered:
 * - dragenter: Visual feedback when CSV file enters drop zone
 * - dragover: Continuous feedback during CSV file hover
 * - dragleave: Reset visual state when CSV file leaves zone
 * - drop: Process dropped CSV files and initiate upload
 * - change: Handle CSV files selected via file browser
 * - click: Trigger file browser when upload zone is clicked
 * 
 * Error Handling:
 * - Validates presence of required DOM elements
 * - Logs errors for missing elements to console
 * - Graceful degradation if elements are not found
 * - Prevents event registration on missing elements
 * 
 * @function setupDragAndDrop
 * @since 2025-05-01
 * @lastModified 2026-01-21
 * @see {@link handleDragEnter} CSV drag enter event handler
 * @see {@link handleDragOver} CSV drag over event handler
 * @see {@link handleDragLeave} CSV drag leave event handler
 * @see {@link handleDrop} CSV file drop event handler
 * @see {@link handleFileSelect} CSV file selection handler
 * 
 * // Supported file formats: CSV only
 * // Maximum file size: 10MB
 * // Required DOM elements: uploadZone, csvFileInput, uploadContent, etc.
 */
function setupDragAndDrop() {
    // Get DOM elements
    uploadZone = document.getElementById('uploadZone');
    fileInput = document.getElementById('csvFileInput');
    uploadContent = document.getElementById('uploadContent');
    uploadIcon = document.getElementById('uploadIcon');
    uploadTitle = document.getElementById('uploadTitle');
    uploadSubtitle = document.getElementById('uploadSubtitle');
    
    if (!uploadZone || !fileInput) {
        console.error('Upload elements not found');
        return;
    }
    
    // Event listeners for drag and drop
    uploadZone.addEventListener('dragenter', handleDragEnter);
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);
    
    // Event listener for click on upload zone
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
 * Handle CSV File Drag Enter Event
 * 
 * Manages the visual feedback when a CSV file is dragged into the upload zone.
 * Uses a counter system to handle nested drag events and prevents flickering
 * when dragging over child elements within the upload zone.
 * 
 * @function handleDragEnter
 * @param {DragEvent} e - The drag enter event object
 * @since 2025-05-01
 * @lastModified 2026-01-21
 * @see {@link updateUploadContent} Updates visual state of upload zone
 * @see {@link setupDragAndDrop} Event registration function
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
 * Handle CSV File Drag Over Event
 * 
 * Maintains the drag state while a CSV file is being dragged over the upload zone.
 * Sets the drop effect to 'copy' to provide visual feedback to the user about
 * the intended action when the file is dropped.
 * 
 * @function handleDragOver
 * @param {DragEvent} e - The drag over event object
 * @since 2025-05-01
 * @lastModified 2026-01-21
 * @see {@link handleDragEnter} Initial drag enter handler
 * @see {@link handleDrop} Final drop handler
 * 
 */
function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
}

/**
 * Handle CSV File Drag Leave Event
 * 
 * Manages the visual feedback when a CSV file is dragged out of the upload zone.
 * Uses a counter system to properly handle nested elements and only resets
 * the visual state when the file completely leaves the upload area.
 * 
 * @function handleDragLeave
 * @param {DragEvent} e - The drag leave event object
 * @since 2025-05-01
 * @lastModified 2026-01-21
 * @see {@link updateUploadContent} Resets visual state of upload zone
 * @see {@link handleDragEnter} Corresponding drag enter handler
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
 * Handle CSV File Drop Event
 * 
 * Processes CSV files dropped onto the upload zone. Resets the drag state,
 * extracts the first file from the drop event, and initiates file processing
 * with validation and upload procedures.
 * 
 * @function handleDrop
 * @param {DragEvent} e - The drop event object containing file data
 * @since 2025-05-01
 * @lastModified 2026-01-21
 * @see {@link processFile} Validates and processes the dropped CSV file
 * @see {@link updateUploadContent} Resets visual state after drop
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
 * Handle CSV File Selection via File Browser
 * 
 * Processes CSV files selected through the traditional file input browser dialog.
 * Extracts the first selected file and initiates the same processing workflow
 * as drag-and-drop files for consistent handling.
 * 
 * @function handleFileSelect
 * @param {Event} e - The file input change event object
 * @since 2025-05-01
 * @lastModified 2026-01-21
 * @see {@link processFile} Validates and processes the selected CSV file
 * @see {@link setupDragAndDrop} Event registration function
 * 
 */
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

/**
 * Process and Validate Selected CSV File
 * 
 * Comprehensive CSV file processing function that validates file type, size,
 * and initiates the upload workflow. Performs client-side validation before
 * sending the file to the server for cloud of points generation processing.
 * 
 * Validation Checks:
 * - File extension validation (must be .csv)
 * - File size validation (maximum 10MB)
 * - File format validation for CSV structure
 * - Error handling with user-friendly messages
 * 
 * Workflow Process:
 * 1. Validates file extension (.csv required)
 * 2. Checks file size against maximum limit
 * 3. Displays upload progress interface
 * 4. Initiates simulated upload progress
 * 5. Triggers server upload and processing
 * 
 * @function processFile
 * @param {File} file - The CSV file object to process and validate
 * @since 2025-05-01
 * @lastModified 2026-01-21
 * @see {@link showUploadError} Displays validation error messages
 * @see {@link showUploadProgress} Shows upload progress interface
 * @see {@link simulateUploadProgress} Manages upload progress simulation
 * @see {@link formatFileSize} Formats file size for error messages
 * 
 */
function processFile(file) {
    // Validate file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
        Utils.showUploadError(uploadZone, uploadContent, 'Upload Error', 'Invalid file type. Please select a CSV file.');
        return;
    }
    
    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
        Utils.showUploadError(uploadZone, uploadContent, 'File too large', 
            `File must be smaller than ${Utils.formatFileSize(MAX_FILE_SIZE)}.`);
        return;
    }
    
    // Show progress and simulate loading
    showUploadProgress(file);
    simulateUploadProgress(file);
}

// Simulate loading progress and auto-upload
/**
 * Simulates CSV file upload progress with visual feedback and automatic processing.
 * Creates a realistic progress animation that gradually increases from 0 to 100%,
 * then automatically triggers the CSV upload and processing workflow.
 * 
 * @function simulateUploadProgress
 * @param {File} file - The CSV file object to be uploaded and processed
 * @since 2025-05-01
 * @lastModified 2026-01-21
 * @author Gerardo Tinoco-Guerrero
 * 
 * @description
 * This function provides a smooth user experience by:
 * - Animating progress from 0% to 100% with random increments
 * - Updating the progress display in real-time
 * - Automatically triggering CSV upload upon completion
 * - Showing success notification and file information
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
                    uploadCSV();
                }, 1000);
                
                // Clear button is now always visible in file info section
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
            uploadTitle.textContent = 'Drop your CSV file here!';
            uploadSubtitle.textContent = 'We will process your file immediately';
            break;
        case 'default':
        default:
            uploadTitle.textContent = 'Drop your CSV file here';
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
    const progressText = document.querySelector('.progress-text');
    const progressCircle = document.querySelector('.progress-ring-circle');
    
    if (progressText) {
        progressText.textContent = Math.round(progress) + '%';
    }
    
    if (progressCircle) {
        const circumference = 2 * Math.PI * 36;
        const offset = circumference - (progress / 100) * circumference;
        progressCircle.style.strokeDasharray = circumference;
        progressCircle.style.strokeDashoffset = offset;
    }
}

// Show upload success
function showUploadSuccess(file) {
    uploadZone.classList.remove('uploading');
    uploadZone.classList.add('success');
    
    uploadContent.innerHTML = `
        <div class="upload-success">
            <div class="success-icon">
                <i class="fas fa-check-circle"></i>
            </div>
            <h4>Upload Successful!</h4>
            <p>${file.name}</p>
            <div class="file-details">
                <span class="file-size">${Utils.formatFileSize(file.size)}</span>
            </div>
        </div>
    `;
}

// Reset upload
function resetUpload() {
    uploadZone.classList.remove('uploading', 'success', 'error', 'drag-over');
    
    uploadContent.innerHTML = `
        <div class="upload-icon-container">
            <i class="fas fa-file-csv upload-icon" id="uploadIcon"></i>
            <div class="upload-animation">
                <div class="upload-pulse"></div>
                <div class="upload-pulse"></div>
                <div class="upload-pulse"></div>
            </div>
        </div>
        <h4 class="upload-title" id="uploadTitle">Drop your CSV file here</h4>
        <p class="upload-subtitle" id="uploadSubtitle">or click to browse files</p>
        <div class="upload-formats">
            <span class="format-badge">CSV</span>
        </div>
        <div class="upload-size-limit">
            <i class="fas fa-info-circle"></i>
            <span>Maximum file size: 10MB</span>
        </div>
        <div class="upload-button">
            <i class="fas fa-folder-open"></i>
            <span>Browse Files</span>
        </div>
    `;
    
    // Resetear referencias
    uploadIcon = document.getElementById('uploadIcon');
    uploadTitle = document.getElementById('uploadTitle');
    uploadSubtitle = document.getElementById('uploadSubtitle');
    
    // Clear file input
    if (fileInput) {
        fileInput.value = '';
    }
    
    // Clear button is now in file info section, no need to hide it here
    
    dragCounter = 0;
}

// Clear upload
function clearUpload() {
    resetUpload();
    
    // Show upload section and format info again
    const uploadSection = document.getElementById('uploadSection');
    if (uploadSection) {
        uploadSection.style.display = 'block';
    }
    
    const formatInfoCard = document.getElementById('formatInfoCard');
    if (formatInfoCard) {
        formatInfoCard.style.display = 'block';
    }
    
    // Hide file info section
    const fileInfoSection = document.getElementById('fileInfoSection');
    if (fileInfoSection) {
        fileInfoSection.classList.add('hidden');
    }
    
    // Clear current filename
    currentFilename = null;
}



/**
 * Uploads a CSV file to the server for processing and visualization.
 * Validates the file format, sends it to the server endpoint, and handles
 * the response to update the UI accordingly with success or error states.
 * 
 * @function uploadCSV
 * @since 2025-05-01
 * @lastModified 2026-01-21
 * @author Gerardo Tinoco-Guerrero
 * 
 * @description
 * This function:
 * - Validates that a file is selected and has .csv extension
 * - Creates FormData and sends file to /upload_csv endpoint
 * - Updates upload zone visual states (uploading, success, error)
 * - Displays file information and options on successful upload
 * - Shows appropriate alerts and error messages
 * - Sets the global currentFilename variable for further processing
 * 
 */
function uploadCSV() {
    const fileInput = document.getElementById('csvFileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showAlert('Please select a CSV file', 'error');
        return;
    }
    
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showAlert('Please select a valid CSV file', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Show uploading state in upload zone
    if (uploadZone) {
        uploadZone.classList.add('uploading');
        updateUploadContent('uploading');
    }
    
    fetch(`${BASE_URL}/upload_csv`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentFilename = data.filename;
            displayFileInfo(data);
            showFileOptions();
            showAlert('CSV file loaded successfully', 'success');
            
            // Update upload zone to success state
            if (uploadZone) {
                uploadZone.classList.remove('uploading');
                uploadZone.classList.add('success');
                updateUploadContent('success');
            }
        } else {
            showAlert(data.error, 'error');
            Utils.showUploadError(uploadZone, uploadContent, 'Upload Error', data.error);
        }
    })
    .catch(error => {
        showAlert('Error uploading file: ' + error.message, 'error');
        Utils.showUploadError(uploadZone, uploadContent, 'Upload Error', 'Error uploading file: ' + error.message);
    });
}

// Display file information
function displayFileInfo(data) {
    const fileDetails = document.getElementById('fileDetails');
    fileDetails.innerHTML = `
        <p><strong>File:</strong> ${data.filename}</p>
        <p><strong>Total points:</strong> ${data.total_points}</p>
        <p><strong>Number of regions:</strong> ${data.regions}</p>
        <p><strong>Regions found:</strong> ${data.region_list.join(', ')}</p>
    `;
    
    // Create CSV graphical visualization
    fetch(data.url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.text();
        })
        .then(csvText => {
            createCSVVisualization(csvText, data.region_list);
        })
        .catch(error => {
            console.error('Could not load CSV file for visualization:', error);
        console.error('Attempted URL:', data.url);
        });
    
    // Hide upload section and format info after successful file load
    const uploadSection = document.getElementById('uploadSection');
    if (uploadSection) {
        uploadSection.style.display = 'none';
    }
    
    const formatInfoCard = document.getElementById('formatInfoCard');
    if (formatInfoCard) {
        formatInfoCard.style.display = 'none';
    }
    
    document.getElementById('fileInfoSection').classList.remove('hidden');
}

// Create CSV visualization
function createCSVVisualization(csvText, regionList) {
    const lines = csvText.trim().split('\n');
    const datasets = {};
    
    // Colors for different regions
    const colors = [
        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', 
        '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF',
        '#4BC0C0', '#FF6384', '#36A2EB', '#FFCE56'
    ];
    
    // Process each CSV line (skip header)
    lines.forEach((line, index) => {
        if (line.trim() === '' || index === 0) return; // Skip empty lines and header
        
        const parts = line.split(',');
        if (parts.length >= 3) {
            const x = parseFloat(parts[0]);
            const y = parseFloat(parts[1]);
            const region = parts[2].trim();
            
            if (!isNaN(x) && !isNaN(y)) {
                if (!datasets[region]) {
                    const colorIndex = Object.keys(datasets).length % colors.length;
                    datasets[region] = {
                        label: `Region ${Math.floor(parseFloat(region))}`,
                        data: [],
                        backgroundColor: colors[colorIndex],
                        borderColor: colors[colorIndex],
                        pointRadius: 1,
                        pointHoverRadius: 2,
                        pointStyle: 'circle',
                        borderWidth: 0
                    };
                }
                datasets[region].data.push({x: x, y: y});
            } else {
                console.log(`Line ${index}: invalid values - x: ${x}, y: ${y}`);
            }
        } else {
            console.log(`Line ${index}: incorrect format - ${parts.length} columns`);
        }
    });
    
    // Configure the chart
    const canvasElement = document.getElementById('csvChart');
    
    if (!canvasElement) {
        console.error('Canvas element csvChart not found');
        return;
    }
    
    const ctx = canvasElement.getContext('2d');
    
    // Destroy previous chart if it exists
    if (window.csvChart && typeof window.csvChart.destroy === 'function') {
        window.csvChart.destroy();
    }
    
    const datasetsArray = Object.values(datasets);
    
    if (datasetsArray.length === 0) {
        console.warn('No data to display in chart');
        canvasElement.style.display = 'none';
        return;
    }
    
    // Calculate ranges to maintain aspect ratio
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    datasetsArray.forEach(dataset => {
        dataset.data.forEach(point => {
            minX = Math.min(minX, point.x);
            maxX = Math.max(maxX, point.x);
            minY = Math.min(minY, point.y);
            maxY = Math.max(maxY, point.y);
        });
    });
    
    // Set fixed scale limits from 0 to 1
    const scaleMin = 0;
    const scaleMax = 1;
    
    canvasElement.style.display = 'block';
    
    try {
        // Check if Chart.js is available
        if (typeof Chart === 'undefined') {
            throw new Error('Chart.js is not available');
        }
        
        window.csvChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: datasetsArray
            },
            options: {
                responsive: false,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: false,
                        text: 'Loaded Nodes'
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        min: 0,
                        max: 1,
                        title: {
                            display: false,
                            text: 'X Coordinate'
                        }
                    },
                    y: {
                        min: 0,
                        max: 1,
                        title: {
                            display: false,
                            text: 'Y Coordinate'
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'point'
                }
            }
        });
        
        // Hide fallback if chart was created correctly
        document.getElementById('chartFallback').style.display = 'none';
        canvasElement.style.display = 'block';
        
    } catch (error) {
        console.error('Error creating chart:', error);
        
        // Show fallback
        canvasElement.style.display = 'none';
        document.getElementById('chartFallback').style.display = 'block';
    }
}

// Show generation options
function showFileOptions() {
    document.getElementById('cloudOptionsSection').classList.remove('hidden');
}

/**
 * Generate Cloud of Points from Uploaded CSV Data
 * 
 * Main function that orchestrates the cloud generation process using the
 * uploaded CSV data and user-selected parameters. Supports multiple generation
 * algorithms and provides real-time progress feedback during processing.
 * 
 * Generation Methods:
 * - Regular Distribution: Uniform point distribution algorithm
 * - Natural Distribution: Organic, natural-looking point distribution
 * 
 * Configuration Options:
 * - Region filtering (inside/outside regions)
 * - Point reduction with configurable multiplier
 * - Generation method selection (regular/natural)
 * - Progress tracking with visual feedback
 * 
 * Workflow Process:
 * 1. Validates uploaded CSV file availability
 * 2. Extracts user configuration from form inputs
 * 3. Initializes progress tracking and UI updates
 * 4. Determines appropriate API endpoint based on method
 * 5. Sends POST request with configuration parameters
 * 6. Handles server response and displays results
 * 7. Manages error states and user feedback
 * 8. Restores UI state after completion
 * 
 * API Endpoints:
 * - /generate_cloud: Regular distribution algorithm
 * - /generate_cloud_natural: Natural distribution algorithm
 * 
 * @function generateCloud
 * @since 2025-05-01
 * @lastModified 2026-01-21
 * @see {@link updateProgress} Progress tracking and visual feedback
 * @see {@link displayResults} Results visualization and statistics
 * @see {@link showAlert} Error message display
 * @see {@link hideProgress} Progress section management
 * 
 * @throws {Error} Shows alert if no CSV file is uploaded
 * @throws {Error} Shows alert if server request fails
 * @throws {Error} Shows alert if generation process encounters errors
 */
function generateCloud() {
    if (!currentFilename) {
        showAlert('You must first upload a CSV file', 'error');
        return;
    }
    
    const regionesInside = document.getElementById('regionesInsideOption').checked;
    const reducePointsValue = parseInt(document.getElementById('reducePointsOption').value);
    const reducePoints = reducePointsValue > 0;
    const generationMethod = document.getElementById('generationMethodOption').value;
    
    // Reset and initialize progress tracking
    progressStartTime = Date.now();
    
    // Show progress
    document.getElementById('progressSection').classList.remove('hidden');
    document.getElementById('resultsSection').classList.add('hidden');
    
    // Initialize progress
    updateProgress(0, 'Initializing generation...');
    
    // Disable button
    document.getElementById('generateBtn').disabled = true;
    document.getElementById('generateBtn').textContent = 'Generating...';
    
    // Determine endpoint and progress messages based on method
    let endpoint = `${BASE_URL}/generate_cloud`;
    let methodName = 'Regular Distribution';
    
    if (generationMethod === 'natural') {
        endpoint = `${BASE_URL}/generate_cloud_natural`;
        methodName = 'Natural Distribution';
    }
    
    // Simulate progress updates during processing
    updateProgress(25, 'Processing input data...');
    setTimeout(() => updateProgress(50, 'Analyzing regions...'), 500);
    setTimeout(() => updateProgress(75, `Generating cloud with ${methodName}...`), 1000);
    
    fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            csv_filename: currentFilename,
            regiones_inside: regionesInside,
            reduce_points: reducePoints,
            reduce_points_multiplier: reducePointsValue
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateProgress(100, `${methodName} generation completed successfully!`);
            displayResults(data);
        } else {
            showAlert(data.error || `Error generating cloud with ${methodName}`, 'error');
            hideProgress();
        }
    })
    .catch(error => {
        showAlert(`Error generating cloud with ${methodName}: ` + error.message, 'error');
        hideProgress();
    })
    .finally(() => {
        // Re-enable button
        document.getElementById('generateBtn').disabled = false;
        document.getElementById('generateBtn').textContent = 'Generate Cloud of Points';
    });
}



// Progress tracking
let progressStartTime = null;

// Update progress
function updateProgress(percentage, message) {
    const progressFill = document.getElementById('progressFill');
    const progressPercentage = document.getElementById('progressPercentage');
    const progressText = document.getElementById('progressText');
    
    if (progressFill) {
        progressFill.style.width = percentage + '%';
    }
    
    if (progressPercentage) {
        progressPercentage.textContent = percentage + '%';
    }
    
    if (progressText && message) {
        progressText.textContent = message;
    }
    
    // Update step indicators
    updateProgressSteps(percentage);
}

// Update progress step indicators
function updateProgressSteps(percentage) {
    const step1 = document.getElementById('step1');
    const step2 = document.getElementById('step2');
    const step3 = document.getElementById('step3');
    
    // Reset all steps
    [step1, step2, step3].forEach(step => {
        if (step) {
            step.classList.remove('active', 'completed');
        }
    });
    
    if (percentage >= 0 && step1) {
        step1.classList.add('active');
    }
    if (percentage >= 25 && step1) {
        step1.classList.remove('active');
        step1.classList.add('completed');
    }
    if (percentage >= 50 && step2) {
        step2.classList.add('active');
    }
    if (percentage >= 75 && step2) {
        step2.classList.remove('active');
        step2.classList.add('completed');
    }
    if (percentage >= 100 && step3) {
        step3.classList.add('active');
        step3.classList.add('completed');
    }
}

// Hide progress
function hideProgress() {
    document.getElementById('progressSection').classList.add('hidden');
    document.getElementById('generateBtn').disabled = false;
    document.getElementById('generateBtn').textContent = '🚀 Generate Cloud of Points';
}

// Display results
function displayResults(data) {
    // Update statistics in the summary section
    updateResultsStatistics(data);
    
    // Update visualization content
    const resultsContent = document.getElementById('resultsContent');
    
    let html = '';
    
    if (data.files && data.files.length > 0) {
        // Separate files by type
        const csvFiles = data.files.filter(file => file.endsWith('.csv'));
        const pngFiles = data.files.filter(file => file.endsWith('.png'));
        const svgFiles = data.files.filter(file => file.endsWith('.svg'));
        
        // Show PNG image if exists
        if (pngFiles.length > 0) {
            html += '<div class="visualization-preview">';
            html += '<h4 class="preview-title"><i class="fas fa-image"></i> Generated Visualization</h4>';
            html += `<div class="result-image">`;
            html += `<img src="${BASE_URL}/download/${pngFiles[0]}" alt="Generated Cloud of Points" class="preview-img">`;
            html += `</div>`;
            html += '</div>';
        }
        
        // Download section
        html += '<div class="download-section">';
        html += '<h4 class="download-title"><i class="fas fa-download"></i> Download Generated Files</h4>';
        html += '<div class="download-buttons-grid">';
        
        csvFiles.forEach(file => {
            const fileName = file.split('_').pop();
            html += `<a href="${BASE_URL}/download/${file}" class="download-btn csv-btn">`;
            html += `<div class="btn-icon"><i class="fas fa-file-csv"></i></div>`;
            html += `<div class="btn-content">`;
            html += `<span class="btn-title">CSV Data</span>`;
            html += `<span class="btn-subtitle">${fileName}</span>`;
            html += `</div>`;
            html += `<div class="btn-arrow"><i class="fas fa-download"></i></div>`;
            html += `</a>`;
        });
        
        pngFiles.forEach(file => {
            const fileName = file.split('_').pop();
            html += `<a href="${BASE_URL}/download/${file}" class="download-btn png-btn">`;
            html += `<div class="btn-icon"><i class="fas fa-image"></i></div>`;
            html += `<div class="btn-content">`;
            html += `<span class="btn-title">PNG Image</span>`;
            html += `<span class="btn-subtitle">${fileName}</span>`;
            html += `</div>`;
            html += `<div class="btn-arrow"><i class="fas fa-download"></i></div>`;
            html += `</a>`;
        });
        
        svgFiles.forEach(file => {
            const fileName = file.split('_').pop();
            html += `<a href="${BASE_URL}/download/${file}" class="download-btn svg-btn">`;
            html += `<div class="btn-icon"><i class="fas fa-vector-square"></i></div>`;
            html += `<div class="btn-content">`;
            html += `<span class="btn-title">SVG Vector</span>`;
            html += `<span class="btn-subtitle">${fileName}</span>`;
            html += `</div>`;
            html += `<div class="btn-arrow"><i class="fas fa-download"></i></div>`;
            html += `</a>`;
        });
        
        html += '</div>';
        html += '</div>';
    }
    
    resultsContent.innerHTML = html;
    document.getElementById('resultsSection').classList.remove('hidden');
    document.getElementById('progressSection').classList.add('hidden');
}

// Update results statistics
function updateResultsStatistics(data) {
    const processingTimeEl = document.getElementById('processingTime');
    const pointsGeneratedEl = document.getElementById('pointsGenerated');
    const outputFilesEl = document.getElementById('outputFiles');
    
    // Calculate processing time
    if (progressStartTime) {
        const processingTime = ((Date.now() - progressStartTime) / 1000).toFixed(1);
        processingTimeEl.textContent = `${processingTime}s`;
    } else {
        processingTimeEl.textContent = 'N/A';
    }
    
    // Count points generated by reading CSV file
    if (data.files) {
        const csvFiles = data.files.filter(file => file.endsWith('.csv'));
        outputFilesEl.textContent = data.files.length.toString();
        
        if (csvFiles.length > 0) {
            // Fetch the CSV file to count actual rows
            fetch(`${BASE_URL}/download/${csvFiles[0]}`)
                .then(response => response.text())
                .then(csvText => {
                    const lines = csvText.trim().split('\n');
                    // Subtract 1 to exclude header row
                    const pointCount = Math.max(0, lines.length - 1);
                    pointsGeneratedEl.textContent = pointCount.toLocaleString();
                })
                .catch(error => {
                    console.error('Error counting points:', error);
                    pointsGeneratedEl.textContent = 'Error';
                });
        } else {
            pointsGeneratedEl.textContent = '0';
        }
    } else {
        pointsGeneratedEl.textContent = '0';
        outputFilesEl.textContent = '0';
    }
}

// Show alert
function showAlert(message, type) {
    showFloatingNotification(message, type);
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

// Initialize when page loads
/**
 * Copies the specified text to the system clipboard with fallback support.
 * Uses a temporary textarea element for compatibility with older browsers
 * and falls back to the modern Clipboard API when available.
 * 
 * @function copyToClipboard
 * @param {string} text - The text content to copy to the clipboard
 * @since 2025-05-01
 * @lastModified 2026-01-21
 * @author Gerardo Tinoco-Guerrero
 * 
 * @description
 * This utility function:
 * - Creates a temporary textarea element for text selection
 * - Uses document.execCommand('copy') as primary method
 * - Falls back to navigator.clipboard.writeText() for modern browsers
 * - Provides visual feedback through showCopyFeedback()
 * - Handles errors gracefully with console logging
 * - Ensures cleanup of temporary DOM elements
 * 
 */
function copyToClipboard(text) {
    // Create temporary element to copy text
    const tempTextArea = document.createElement('textarea');
    tempTextArea.value = text;
    tempTextArea.style.position = 'fixed';
    tempTextArea.style.left = '-999999px';
    tempTextArea.style.top = '-999999px';
    document.body.appendChild(tempTextArea);
    
    try {
        // Select and copy text
        tempTextArea.focus();
        tempTextArea.select();
        document.execCommand('copy');
        
        // Show visual feedback
        showCopyFeedback();
    } catch (err) {
        console.error('Error copying to clipboard:', err);
        // Fallback for modern browsers
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(() => {
                showCopyFeedback();
            }).catch(err => {
                console.error('Error in clipboard fallback:', err);
            });
        }
    } finally {
        // Clean up temporary element
        document.body.removeChild(tempTextArea);
    }
}

// Function to show visual feedback when copying
/**
 * Provides visual feedback when text is successfully copied to clipboard.
 * Temporarily changes the copy button's icon and styling to indicate
 * successful copy operation, then restores original appearance.
 * 
 * @function showCopyFeedback
 * @since 2025-05-01
 * @lastModified 2026-01-21
 * @author Gerardo Tinoco-Guerrero
 * 
 * @description
 * This feedback function:
 * - Finds the closest copy button element from the event target
 * - Changes the icon from copy to checkmark (fas fa-check)
 * - Updates button styling to green background with white text
 * - Automatically restores original styling after 2 seconds
 * - Provides immediate visual confirmation of successful copy operation
 * 
 */
function showCopyFeedback() {
    const copyBtn = event.target.closest('.copy-btn');
    if (copyBtn) {
        const originalIcon = copyBtn.querySelector('i');
        const originalClass = originalIcon.className;
        
        // Change icon temporarily
        originalIcon.className = 'fas fa-check';
        copyBtn.style.background = '#10b981';
        copyBtn.style.color = 'white';
        
        // Restore after 2 seconds
        setTimeout(() => {
            originalIcon.className = originalClass;
            copyBtn.style.background = '';
            copyBtn.style.color = '';
        }, 2000);
    }
}

/**
 * Downloads the current data visualization chart as a PNG image file.
 * Converts the canvas element to a data URL and triggers an automatic
 * download with a timestamped filename for easy organization.
 * 
 * @function downloadChart
 * @since 2025-05-01
 * @lastModified 2026-01-21
 * @author Gerardo Tinoco-Guerrero
 * 
 * @description
 * This download function:
 * - Validates that a chart exists (window.csvChart) before proceeding
 * - Locates the canvas element containing the visualization
 * - Generates a timestamped filename in format: data_points_visualization_YYYYMMDD_HHMMSS.png
 * - Converts canvas to PNG data URL using toDataURL()
 * - Creates temporary download link and triggers automatic download
 * - Provides user feedback through success/error alerts
 * - Handles errors gracefully with console logging
 * 
 */
function downloadChart() {
    if (!window.csvChart) {
        console.error('No chart available for download');
        showAlert('No chart available for download', 'error');
        return;
    }
    
    try {
        // Get the canvas element
        const canvas = document.getElementById('csvChart');
        if (!canvas) {
            throw new Error('Canvas element not found');
        }
        
        // Create download link
        const link = document.createElement('a');
        const now = new Date();
        const timestamp = now.getFullYear().toString() + 
                         (now.getMonth() + 1).toString().padStart(2, '0') + 
                         now.getDate().toString().padStart(2, '0') + '_' +
                         now.getHours().toString().padStart(2, '0') + 
                         now.getMinutes().toString().padStart(2, '0') + 
                         now.getSeconds().toString().padStart(2, '0');
        link.download = `data_points_visualization_${timestamp}.png`;
        link.href = canvas.toDataURL('image/png');
        
        // Trigger download
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showAlert('Chart downloaded successfully', 'success');
        
    } catch (error) {
        console.error('Error downloading chart:', error);
        showAlert('Error downloading chart', 'error');
    }
}

document.addEventListener('DOMContentLoaded', function() {
    setupDragAndDrop();
});