// CloudGenerator JavaScript Module
// Functionality for cloud of points generation

let currentFilename = null;
let currentTaskId = null;
let statusCheckInterval = null;

// Global variables for optimized drag and drop
let uploadZone = null;
let fileInput = null;
let uploadContent = null;
let uploadIcon = null;
let uploadTitle = null;
let uploadSubtitle = null;
let clearButton = null;
let dragCounter = 0;

// Supported file formats
const SUPPORTED_FORMATS = {
    'text/csv': 'CSV',
    'application/vnd.ms-excel': 'CSV'
};

// Maximum file size (10MB)
const MAX_FILE_SIZE = 10 * 1024 * 1024;

// Configure optimized drag and drop
function setupDragAndDrop() {
    // Get DOM elements
    uploadZone = document.getElementById('uploadZone');
    fileInput = document.getElementById('csvFileInput');
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

// Manejar entrada de drag
function handleDragEnter(e) {
    e.preventDefault();
    dragCounter++;
    
    if (dragCounter === 1) {
        uploadZone.classList.add('drag-over');
        updateUploadContent('drag-over');
    }
}

// Manejar drag over
function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
}

// Manejar salida de drag
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
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showUploadError('Invalid file type. Please select a CSV file.');
        return;
    }
    
    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
        showUploadError(`File too large. Maximum size is ${formatFileSize(MAX_FILE_SIZE)}.`);
        return;
    }
    
    // Show progress and simulate loading
    showUploadProgress(file);
    simulateUploadProgress(file);
}

// Simulate loading progress and auto-upload
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
                <span class="file-size">${formatFileSize(file.size)}</span>
            </div>
        </div>
    `;
}

// Show upload error
function showUploadError(message) {
    uploadZone.classList.remove('uploading', 'success');
    uploadZone.classList.add('error');
    
    uploadContent.innerHTML = `
        <div class="upload-error">
            <div class="error-icon">
                <i class="fas fa-exclamation-triangle"></i>
            </div>
            <h4>Upload Error</h4>
            <p>${message}</p>
            <button class="btn btn-secondary btn-small" onclick="resetUpload()">
                <i class="fas fa-redo"></i>
                Try Again
            </button>
        </div>
    `;
}

// Resetear upload
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

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Upload CSV file
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
    
    fetch('/upload_csv', {
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
            showUploadError(data.error);
        }
    })
    .catch(error => {
        showAlert('Error uploading file: ' + error.message, 'error');
        showUploadError('Error uploading file: ' + error.message);
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
    fetch(`/uploads/${data.filename}`)
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
        console.error('Attempted URL:', `/uploads/${data.filename}`);
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
                        pointRadius: 2,
                        pointHoverRadius: 4
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

// Generate cloud of points
function generateCloud() {
    if (!currentFilename) {
        showAlert('You must first upload a CSV file', 'error');
        return;
    }
    
    const regionesInside = document.getElementById('regionesInsideOption').checked;
    const reducePoints = document.getElementById('reducePointsOption').checked;
    
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
    
    fetch('/generate_cloud', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            filename: currentFilename,
            regiones_inside: regionesInside,
            reduce_points: reducePoints
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentTaskId = data.task_id;
            updateProgress(25, 'Processing input data...');
            startStatusCheck();
        } else {
            showAlert(data.error, 'error');
            hideProgress();
        }
    })
    .catch(error => {
        showAlert('Error generating cloud of points: ' + error.message, 'error');
        hideProgress();
    });
}

// Start status check
function startStatusCheck() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }
    
    statusCheckInterval = setInterval(checkCloudStatus, 2000); // Check every 2 seconds
}

// Check cloud of points status
function checkCloudStatus() {
    if (!currentTaskId) return;
    
    fetch(`/cloud_status/${currentTaskId}`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'running') {
                // Provide more granular progress based on time elapsed
                const currentProgress = Math.min(75, Math.max(25, getCurrentProgress()));
                const message = data.message || getProgressMessage(currentProgress);
                updateProgress(currentProgress, message);
            } else if (data.status === 'completed') {
                updateProgress(100, 'Generation completed successfully!');
                displayResults(data);
                stopStatusCheck();
            } else if (data.status === 'failed' || data.status === 'error') {
                showAlert(data.message, 'error');
                hideProgress();
                stopStatusCheck();
            }
        })
        .catch(error => {
            console.error('Error checking status:', error);
        });
}

// Get current progress based on elapsed time
let progressStartTime = null;

function getCurrentProgress() {
    if (!progressStartTime) {
        progressStartTime = Date.now();
        return 25;
    }
    
    const elapsed = Date.now() - progressStartTime;
    const seconds = elapsed / 1000;
    
    // Simulate progress based on time (adjust as needed)
    if (seconds < 5) return 25;
    if (seconds < 10) return 35;
    if (seconds < 15) return 45;
    if (seconds < 20) return 55;
    if (seconds < 25) return 65;
    return 75;
}

// Get progress message based on percentage
function getProgressMessage(percentage) {
    if (percentage < 30) return 'Processing input data...';
    if (percentage < 50) return 'Analyzing regions...';
    if (percentage < 70) return 'Generating cloud of points...';
    if (percentage < 90) return 'Creating visualization...';
    return 'Finalizing results...';
}

// Stop status check
function stopStatusCheck() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
    
    // Re-enable button
    document.getElementById('generateBtn').disabled = false;
    document.getElementById('generateBtn').textContent = '🚀 Generate Cloud of Points';
}

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
            html += `<img src="/download/${pngFiles[0]}" alt="Generated Cloud of Points" class="preview-img">`;
            html += `</div>`;
            html += '</div>';
        }
        
        // Download section
        html += '<div class="download-section">';
        html += '<h4 class="download-title"><i class="fas fa-download"></i> Download Generated Files</h4>';
        html += '<div class="download-buttons-grid">';
        
        csvFiles.forEach(file => {
            const fileName = file.split('_').pop();
            html += `<a href="/download/${file}" class="download-btn csv-btn">`;
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
            html += `<a href="/download/${file}" class="download-btn png-btn">`;
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
            html += `<a href="/download/${file}" class="download-btn svg-btn">`;
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
            fetch(`/download/${csvFiles[0]}`)
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
// Function to copy text to clipboard
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

// Function to download chart as PNG
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