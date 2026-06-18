/**
 * CloudViewer Module
 * Handles file upload and visualization for the CloudViewer tool.
 * Implements drag-and-drop functionality consistent with CloudGenerator.
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
                
                // Auto-trigger visualization
                setTimeout(visualizeData, 800);
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
    
    // Reset visualization result if needed
    const resultSection = document.getElementById('resultSection');
    const uploadCard = document.querySelector('.upload-card');
    
    if (resultSection && !resultSection.classList.contains('hidden')) {
        resultSection.classList.add('hidden');
        uploadCard.classList.remove('hidden');
    }
}

// --- Interactive Canvas State ---
let canvasData = {
    points: [],
    regions: [],
    classifications: [],
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    minX: 0, maxX: 0, minY: 0, maxY: 0,
    currentTool: 'pan', // 'pan', 'move', 'delete'
    selectedPointIdx: -1,
    isDragging: false,
    lastMouseX: 0,
    lastMouseY: 0
};

const REGION_COLORS = [
    '#3399FF', '#4DE64D', '#FFB333', '#CC4DFF', 
    '#E69933', '#FF80CC', '#B3B3B3'
];

function visualizeData() {
    if (!currentFile) return;

    const loadingSection = document.getElementById('loadingSection');
    const resultSection = document.getElementById('resultSection');
    const uploadCard = document.querySelector('.upload-card');

    uploadCard.classList.add('hidden');
    loadingSection.classList.remove('hidden');
    resultSection.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', currentFile);

    fetch('/upload_viewer', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loadingSection.classList.add('hidden');
        
        if (data.success) {
            resultSection.classList.remove('hidden');
            
            // Initialize canvas data
            canvasData.points = data.points;
            canvasData.regions = data.regions;
            canvasData.classifications = data.classifications;
            
            initCanvas();
            
            resultSection.scrollIntoView({ behavior: 'smooth' });
        } else {
            uploadCard.classList.remove('hidden');
            Utils.showUploadError(uploadZone, uploadContent, 'Visualization Error', data.error || 'An error occurred during visualization.');
        }
    })
    .catch(error => {
        loadingSection.classList.add('hidden');
        uploadCard.classList.remove('hidden');
        Utils.showUploadError(uploadZone, uploadContent, 'Network Error', error.message);
    });
}

function initCanvas() {
    const canvas = document.getElementById('cloudCanvas');
    const ctx = canvas.getContext('2d');
    const wrapper = document.querySelector('.canvas-wrapper');
    
    // Set actual canvas resolution to match display size
    canvas.width = wrapper.clientWidth;
    canvas.height = wrapper.clientHeight;
    
    // Calculate bounds
    if (canvasData.points.length > 0) {
        let xs = canvasData.points.map(p => p[0]);
        let ys = canvasData.points.map(p => p[1]);
        canvasData.minX = Math.min(...xs);
        canvasData.maxX = Math.max(...xs);
        canvasData.minY = Math.min(...ys);
        canvasData.maxY = Math.max(...ys);
    }
    
    resetView();
    setupCanvasEvents(canvas);
    setupToolbarEvents();
    
    document.getElementById('downloadModifiedCsv').addEventListener('click', exportCsv);
    document.getElementById('downloadCanvasPng').addEventListener('click', exportPng);
}

function resetView() {
    const canvas = document.getElementById('cloudCanvas');
    const padding = 40;
    const dataWidth = canvasData.maxX - canvasData.minX || 1;
    const dataHeight = canvasData.maxY - canvasData.minY || 1;
    
    const scaleX = (canvas.width - padding * 2) / dataWidth;
    const scaleY = (canvas.height - padding * 2) / dataHeight;
    canvasData.scale = Math.min(scaleX, scaleY);
    
    const cx = (canvasData.minX + canvasData.maxX) / 2;
    const cy = (canvasData.minY + canvasData.maxY) / 2;
    
    canvasData.offsetX = canvas.width / 2 - cx * canvasData.scale;
    canvasData.offsetY = canvas.height / 2 + cy * canvasData.scale; // + because Y is flipped in math vs screen
    
    drawCanvas();
}

function drawCanvas() {
    const canvas = document.getElementById('cloudCanvas');
    const ctx = canvas.getContext('2d');
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = canvasData.offsetX % 50; x < canvas.width; x += 50) {
        ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height);
    }
    for (let y = canvasData.offsetY % 50; y < canvas.height; y += 50) {
        ctx.moveTo(0, y); ctx.lineTo(canvas.width, y);
    }
    ctx.stroke();

    // Draw Points
    const r = 3;
    
    for (let i = 0; i < canvasData.points.length; i++) {
        const pt = canvasData.points[i];
        const screenX = pt[0] * canvasData.scale + canvasData.offsetX;
        const screenY = -pt[1] * canvasData.scale + canvasData.offsetY;
        
        // Skip if outside viewport for performance
        if (screenX < -10 || screenX > canvas.width + 10 || screenY < -10 || screenY > canvas.height + 10) continue;
        
        const regionId = canvasData.regions[i];
        const isBoundary = canvasData.classifications[i] === 'boundary';
        
        // Boundaries are bright white, interiors use region color
        const color = isBoundary ? '#FFFFFF' : (REGION_COLORS[(regionId - 1) % REGION_COLORS.length] || '#A0AEC0');
        
        ctx.beginPath();
        ctx.arc(screenX, screenY, i === canvasData.selectedPointIdx ? r * 2 : r, 0, 2 * Math.PI);
        ctx.fillStyle = i === canvasData.selectedPointIdx ? '#FF3366' : color;
        ctx.fill();
        
        if (i === canvasData.selectedPointIdx || isBoundary) {
            ctx.strokeStyle = i === canvasData.selectedPointIdx ? '#FFFFFF' : 'rgba(0,0,0,0.5)';
            ctx.lineWidth = i === canvasData.selectedPointIdx ? 2 : 1;
            ctx.stroke();
        }
    }
    
    // Update Stats
    document.getElementById('pointCountDisplay').innerText = `Total Points: ${canvasData.points.length}`;
}

function getPointAt(screenX, screenY) {
    const threshold = 6; // Hit radius
    for (let i = 0; i < canvasData.points.length; i++) {
        const pt = canvasData.points[i];
        const px = pt[0] * canvasData.scale + canvasData.offsetX;
        const py = -pt[1] * canvasData.scale + canvasData.offsetY;
        
        const dx = px - screenX;
        const dy = py - screenY;
        if (dx*dx + dy*dy <= threshold*threshold) {
            return i;
        }
    }
    return -1;
}

function setupCanvasEvents(canvas) {
    canvas.addEventListener('mousedown', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        canvasData.lastMouseX = x;
        canvasData.lastMouseY = y;
        canvasData.isDragging = true;
        
        const clickedIdx = getPointAt(x, y);
        
        if (canvasData.currentTool === 'delete') {
            if (clickedIdx !== -1) {
                canvasData.points.splice(clickedIdx, 1);
                canvasData.regions.splice(clickedIdx, 1);
                canvasData.classifications.splice(clickedIdx, 1);
                canvasData.selectedPointIdx = -1;
                drawCanvas();
            }
        } else if (canvasData.currentTool === 'move') {
            canvasData.selectedPointIdx = clickedIdx;
            drawCanvas();
        }
    });

    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const dx = x - canvasData.lastMouseX;
        const dy = y - canvasData.lastMouseY;
        
        // Tooltip updates
        const hoverIdx = getPointAt(x, y);
        const tooltip = document.getElementById('nodeTooltip');
        if (hoverIdx !== -1 && canvasData.currentTool !== 'pan') {
            const pt = canvasData.points[hoverIdx];
            tooltip.innerText = `ID: ${hoverIdx}\nX: ${pt[0].toFixed(3)}, Y: ${pt[1].toFixed(3)}\nReg: ${canvasData.regions[hoverIdx]}`;
            tooltip.style.left = (e.clientX + 10) + 'px';
            tooltip.style.top = (e.clientY + 10) + 'px';
            tooltip.classList.remove('hidden');
            canvas.style.cursor = 'pointer';
        } else {
            tooltip.classList.add('hidden');
            canvas.style.cursor = canvasData.currentTool === 'pan' ? (canvasData.isDragging ? 'grabbing' : 'grab') : 'crosshair';
        }

        // Coordinate display
        const mathX = (x - canvasData.offsetX) / canvasData.scale;
        const mathY = -(y - canvasData.offsetY) / canvasData.scale;
        document.getElementById('coordinatesDisplay').innerText = `X: ${mathX.toFixed(3)}, Y: ${mathY.toFixed(3)}`;

        if (!canvasData.isDragging) return;

        if (canvasData.currentTool === 'pan') {
            canvasData.offsetX += dx;
            canvasData.offsetY += dy;
            drawCanvas();
        } else if (canvasData.currentTool === 'move' && canvasData.selectedPointIdx !== -1) {
            canvasData.points[canvasData.selectedPointIdx][0] = mathX;
            canvasData.points[canvasData.selectedPointIdx][1] = mathY;
            drawCanvas();
        }
        
        canvasData.lastMouseX = x;
        canvasData.lastMouseY = y;
    });

    window.addEventListener('mouseup', () => {
        canvasData.isDragging = false;
        canvasData.selectedPointIdx = -1;
        drawCanvas();
    });

    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const zoomIntensity = 0.1;
        const wheel = e.deltaY < 0 ? 1 : -1;
        const zoomFactor = Math.exp(wheel * zoomIntensity);

        // Adjust offset so zoom is centered on mouse
        canvasData.offsetX = x - (x - canvasData.offsetX) * zoomFactor;
        canvasData.offsetY = y - (y - canvasData.offsetY) * zoomFactor;
        canvasData.scale *= zoomFactor;
        
        drawCanvas();
    });
}

function setupToolbarEvents() {
    const btnPan = document.getElementById('toolPan');
    const btnMove = document.getElementById('toolMove');
    const btnDelete = document.getElementById('toolDelete');
    const btnReset = document.getElementById('btnResetView');
    
    const setTool = (tool, activeBtn) => {
        canvasData.currentTool = tool;
        [btnPan, btnMove, btnDelete].forEach(b => b.classList.remove('active'));
        activeBtn.classList.add('active');
        document.getElementById('cloudCanvas').style.cursor = tool === 'pan' ? 'grab' : 'crosshair';
    };

    btnPan.addEventListener('click', () => setTool('pan', btnPan));
    btnMove.addEventListener('click', () => setTool('move', btnMove));
    btnDelete.addEventListener('click', () => setTool('delete', btnDelete));
    btnReset.addEventListener('click', resetView);
}

function exportCsv(e) {
    e.preventDefault();
    if (canvasData.points.length === 0) return;
    
    let csvContent = "x,y,classification,region\n";
    for (let i = 0; i < canvasData.points.length; i++) {
        const pt = canvasData.points[i];
        csvContent += `${pt[0]},${pt[1]},${canvasData.classifications[i]},${canvasData.regions[i]}\n`;
    }
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `modified_cloud_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function exportPng(e) {
    e.preventDefault();
    const canvas = document.getElementById('cloudCanvas');
    const url = canvas.toDataURL('image/png');
    
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `cloud_snapshot_${Date.now()}.png`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Expose functions to global scope for onclick handlers
window.resetUpload = resetUpload;
window.clearUpload = resetUpload;
