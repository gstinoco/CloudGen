/**
 * NeighborsCalculator Module
 * Handles file upload and neighbor calculation for the NeighborsCalculator tool.
 * Implements drag-and-drop functionality consistent with CloudGenerator and CloudViewer.
 */

let uploadZone = null;
let fileInput = null;
let uploadContent = null;
let uploadTitle = null;
let uploadSubtitle = null;
let dragCounter = 0;
let currentFile = null;

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

document.addEventListener('DOMContentLoaded', () => {
    setupDragAndDrop();
});

function setupDragAndDrop() {
    uploadZone = document.getElementById('uploadZone');
    fileInput = document.getElementById('csvFileInput');
    uploadContent = document.getElementById('uploadContent');
    uploadTitle = document.getElementById('uploadTitle');
    uploadSubtitle = document.getElementById('uploadSubtitle');

    if (!uploadZone || !fileInput) return;

    // Drag and drop events
    uploadZone.addEventListener('dragenter', handleDragEnter);
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);

    // Click to upload
    uploadZone.addEventListener('click', (e) => {
        // Prevent click if clicking on buttons inside
        if (e.target.closest('button')) return;
        
        if (!uploadZone.classList.contains('uploading')) {
            fileInput.click();
        }
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Prevent default drag behaviors
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());
}

function handleDragEnter(e) {
    e.preventDefault();
    dragCounter++;
    
    if (dragCounter === 1) {
        uploadZone.classList.add('drag-over');
        updateUploadContent('drag-over');
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
}

function handleDragLeave(e) {
    e.preventDefault();
    dragCounter--;
    
    if (dragCounter === 0) {
        uploadZone.classList.remove('drag-over');
        updateUploadContent('default');
    }
}

function handleDrop(e) {
    e.preventDefault();
    dragCounter = 0;
    uploadZone.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function updateUploadContent(state) {
    // If elements don't exist (e.g. during upload progress), return
    const title = document.getElementById('uploadTitle');
    const subtitle = document.getElementById('uploadSubtitle');
    
    if (!title || !subtitle) return;
    
    switch (state) {
        case 'drag-over':
            title.textContent = 'Drop your CSV file here!';
            subtitle.textContent = 'We will process your file immediately';
            break;
        case 'default':
        default:
            title.textContent = 'Drop your CSV file here';
            subtitle.textContent = 'or click to browse files';
            break;
    }
}

function processFile(file) {
    // Validation
    if (!file.name.toLowerCase().endsWith('.csv')) {
        Utils.showUploadError(uploadZone, uploadContent, 'Invalid File', 'Please select a CSV file.');
        return;
    }
    if (file.size > MAX_FILE_SIZE) {
        Utils.showUploadError(uploadZone, uploadContent, 'File too large', `File must be smaller than ${Utils.formatFileSize(MAX_FILE_SIZE)}.`);
        return;
    }

    currentFile = file;
    simulateUpload(file);
}

function simulateUpload(file) {
    // Add uploading class to zone
    uploadZone.classList.add('uploading');
    uploadZone.classList.remove('success', 'error');
    
    // Inject progress HTML structure matching CloudGenerator
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
    
    const progressText = uploadContent.querySelector('.progress-text');
    const progressCircle = uploadContent.querySelector('.progress-ring-circle');
    
    // Setup circle animation
    const radius = 36;
    const circumference = 2 * Math.PI * radius;
    progressCircle.style.strokeDasharray = `${circumference} ${circumference}`;
    progressCircle.style.strokeDashoffset = circumference;
    
    let progress = 0;
    const interval = setInterval(() => {
        progress += 5;
        if (progress > 100) progress = 100;
        
        progressText.textContent = `${progress}%`;
        const offset = circumference - (progress / 100) * circumference;
        progressCircle.style.strokeDashoffset = offset;

        if (progress === 100) {
            clearInterval(interval);
            setTimeout(() => {
                showUploadSuccess(file);
                
                // Auto-trigger calculation
                setTimeout(calculateNeighbors, 800);
            }, 500);
        }
    }, 30);
}

function showUploadSuccess(file) {
    uploadZone.classList.remove('uploading');
    uploadZone.classList.add('success');
    
    // Inject success HTML structure matching CloudGenerator
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

function resetUpload() {
    // Reset zone classes
    uploadZone.classList.remove('uploading', 'success', 'error', 'drag-over');
    
    // Restore original HTML structure
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
    
    // Re-bind references if needed (though we use getElementById usually)
    uploadTitle = document.getElementById('uploadTitle');
    uploadSubtitle = document.getElementById('uploadSubtitle');
    
    // Clear file input
    if (fileInput) fileInput.value = '';
    
    currentFile = null;
    dragCounter = 0;
    
    // Reset result if needed
    const resultSection = document.getElementById('resultSection');
    const uploadCard = document.querySelector('.upload-card');
    
    if (resultSection && !resultSection.classList.contains('hidden')) {
        resultSection.classList.add('hidden');
        uploadCard.classList.remove('hidden');
    }
}

function calculateNeighbors() {
    if (!currentFile) return;

    const loadingSection = document.getElementById('loadingSection');
    const resultSection = document.getElementById('resultSection');
    const uploadCard = document.querySelector('.upload-card');

    // Show loading state
    uploadCard.classList.add('hidden');
    loadingSection.classList.remove('hidden');
    resultSection.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', currentFile);

    fetch('/upload_neighbors', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Hide loading
        loadingSection.classList.add('hidden');
        
        if (data.success) {
            // Show result section
            resultSection.classList.remove('hidden');
            
            // Display statistics
            const statsContainer = document.getElementById('statsContainer');
            if (statsContainer && data.stats) {
                statsContainer.innerHTML = `
                    <div class="stat-card">
                        <div class="stat-icon">
                            <i class="fas fa-layer-group"></i>
                        </div>
                        <div class="stat-info">
                            <span class="stat-value count-up" data-target="${data.stats.total_points}">0</span>
                            <span class="stat-label">Total Points</span>
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">
                            <i class="fas fa-map-marked-alt"></i>
                        </div>
                        <div class="stat-info">
                            <span class="stat-value count-up" data-target="${data.stats.total_regions}">0</span>
                            <span class="stat-label">Regions</span>
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">
                            <i class="fas fa-network-wired"></i>
                        </div>
                        <div class="stat-info">
                            <span class="stat-value count-up" data-target="${data.stats.avg_neighbors}" data-decimals="2">0</span>
                            <span class="stat-label">Avg Neighbors</span>
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">
                            <i class="fas fa-chart-bar"></i>
                        </div>
                        <div class="stat-info">
                            <span class="stat-value count-up" data-target="${data.stats.max_neighbors}">0</span>
                            <span class="stat-label">Max Neighbors (k)</span>
                        </div>
                    </div>
                `;
                
                // Trigger count up animation
                setTimeout(animateNumbers, 100);
            }
            
            // Setup download button
            const downloadBtn = document.getElementById('downloadNeighbors');
            if (downloadBtn) {
                downloadBtn.href = data.neighbors_csv_url;
                // Update filename for download attribute if possible, or let server handle it
                downloadBtn.setAttribute('download', 'neighbors.csv');
            }
            
            // Handle Graph Visualization Display
            const graphContainer = document.getElementById('graphContainer');
            const graphImage = document.getElementById('graphImage');
            
            if (graphContainer && graphImage && (data.svg_url || data.png_url)) {
                // Prefer SVG if available for sharpness
                graphImage.src = data.svg_url || data.png_url;
                graphContainer.classList.remove('hidden');
            } else if (graphContainer) {
                graphContainer.classList.add('hidden');
            }
            
            // Scroll to result
            resultSection.scrollIntoView({ behavior: 'smooth' });
        } else {
            // Show error in upload zone
            uploadCard.classList.remove('hidden');
            Utils.showUploadError(uploadZone, uploadContent, 'Calculation Error', data.error || 'An error occurred during neighbor calculation.');
        }
    })
    .catch(error => {
        loadingSection.classList.add('hidden');
        uploadCard.classList.remove('hidden');
        Utils.showUploadError(uploadZone, uploadContent, 'Network Error', error.message);
    });
}

function animateNumbers() {
    const counters = document.querySelectorAll('.count-up');
    
    counters.forEach(counter => {
        const target = parseFloat(counter.getAttribute('data-target'));
        const decimals = parseInt(counter.getAttribute('data-decimals') || 0);
        const duration = 1500; // ms
        const frameDuration = 1000 / 60; // 60fps
        const totalFrames = Math.round(duration / frameDuration);
        
        let frame = 0;
        
        const easeOutQuad = t => t * (2 - t);
        
        const updateCounter = () => {
            frame++;
            const progress = easeOutQuad(frame / totalFrames);
            const current = target * progress;
            
            if (frame < totalFrames) {
                if (decimals > 0) {
                    counter.innerText = current.toFixed(decimals);
                } else {
                    counter.innerText = Math.round(current).toLocaleString();
                }
                requestAnimationFrame(updateCounter);
            } else {
                if (decimals > 0) {
                    counter.innerText = target.toFixed(decimals);
                } else {
                    counter.innerText = Math.round(target).toLocaleString();
                }
            }
        };
        
        updateCounter();
    });
}

// Expose functions to global scope for onclick handlers
window.resetUpload = resetUpload;
window.clearUpload = resetUpload;
