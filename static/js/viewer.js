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
    currentTool: 'pan', // 'pan', 'move', 'delete', 'add'
    selectedPointIdx: -1,
    isDragging: false,
    lastMouseX: 0,
    lastMouseY: 0,

    // Advanced features
    history: [],
    historyIndex: -1,
    printMode: false,
    colorRegions: false,
    pointMoved: false
};

// Helper for displaying very small numbers properly
function formatSci(val) {
    if (val === 0) return "0.000";
    if (Math.abs(val) < 0.01 || Math.abs(val) >= 10000) {
        return val.toExponential(3);
    }
    return val.toFixed(4);
}

function calculateDensityMetrics() {
    if (canvasData.points.length < 2) {
        document.getElementById('minDistDisplay').innerText = '0.000';
        document.getElementById('avgDistDisplay').innerText = '0.000';
        return;
    }

    let minDist = Infinity;
    let sumDist = 0;

    // Calculate nearest neighbor for each point
    for (let i = 0; i < canvasData.points.length; i++) {
        let minLocalDist = Infinity;
        const p1 = canvasData.points[i];

        for (let j = 0; j < canvasData.points.length; j++) {
            if (i === j) continue;
            const p2 = canvasData.points[j];
            const dx = p1[0] - p2[0];
            const dy = p1[1] - p2[1];
            const distSq = dx * dx + dy * dy;
            if (distSq < minLocalDist) {
                minLocalDist = distSq;
            }
        }

        const dist = Math.sqrt(minLocalDist);
        if (dist < minDist) minDist = dist;
        sumDist += dist;
    }

    const avgDist = sumDist / canvasData.points.length;

    document.getElementById('minDistDisplay').innerText = formatSci(minDist);
    document.getElementById('avgDistDisplay').innerText = formatSci(avgDist);
}

function updateUndoRedoButtons() {
    document.getElementById('btnUndo').disabled = canvasData.historyIndex <= 0;
    document.getElementById('btnRedo').disabled = canvasData.historyIndex >= canvasData.history.length - 1;
}

function saveState() {
    // Truncate history if we are not at the end
    if (canvasData.historyIndex < canvasData.history.length - 1) {
        canvasData.history = canvasData.history.slice(0, canvasData.historyIndex + 1);
    }

    // Copy state
    const pointsCopy = canvasData.points.map(p => [...p]);
    const regionsCopy = [...canvasData.regions];
    const classCopy = [...canvasData.classifications];

    canvasData.history.push({
        points: pointsCopy,
        regions: regionsCopy,
        classifications: classCopy
    });

    // Limit history size to 20 to prevent memory issues
    if (canvasData.history.length > 20) {
        canvasData.history.shift();
    } else {
        canvasData.historyIndex++;
    }

    updateUndoRedoButtons();
    calculateDensityMetrics();
}

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

    fetch(`${BASE_URL}/upload_viewer`, {
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

                canvasData.history = [];
                canvasData.historyIndex = -1;
                saveState();

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

        let fillColor;
        let strokeColor = 'rgba(0,0,0,0.5)';

        if (canvasData.printMode) {
            // White background: boundaries black, interiors colored or grey
            if (canvasData.colorRegions) {
                fillColor = isBoundary ? '#000000' : (REGION_COLORS[(regionId - 1) % REGION_COLORS.length] || '#718096');
            } else {
                fillColor = isBoundary ? '#000000' : '#4299E1';
            }
            if (i === canvasData.selectedPointIdx) {
                fillColor = '#FF0000';
                strokeColor = '#000000';
            } else if (isBoundary) {
                strokeColor = '#000000';
            }
        } else {
            // Dark background: boundaries white, interiors colored or grey
            if (canvasData.colorRegions) {
                fillColor = isBoundary ? '#FFFFFF' : (REGION_COLORS[(regionId - 1) % REGION_COLORS.length] || '#A0AEC0');
            } else {
                fillColor = isBoundary ? '#FFFFFF' : '#4299E1';
            }
            if (i === canvasData.selectedPointIdx) {
                fillColor = '#FF3366';
                strokeColor = '#FFFFFF';
            } else if (isBoundary) {
                strokeColor = 'rgba(0,0,0,0.5)';
            }
        }

        ctx.beginPath();
        ctx.arc(screenX, screenY, i === canvasData.selectedPointIdx ? r * 2 : r, 0, 2 * Math.PI);
        ctx.fillStyle = fillColor;
        ctx.fill();

        if (i === canvasData.selectedPointIdx || isBoundary) {
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = i === canvasData.selectedPointIdx ? 2 : 1;
            ctx.stroke();
        }
    }

    // Update Stats
    document.getElementById('pointCountDisplay').innerText = canvasData.points.length;
}

function getPointAt(screenX, screenY) {
    const threshold = 6; // Hit radius
    for (let i = 0; i < canvasData.points.length; i++) {
        const pt = canvasData.points[i];
        const px = pt[0] * canvasData.scale + canvasData.offsetX;
        const py = -pt[1] * canvasData.scale + canvasData.offsetY;

        const dx = px - screenX;
        const dy = py - screenY;
        if (dx * dx + dy * dy <= threshold * threshold) {
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
                saveState();
                drawCanvas();
            }
        } else if (canvasData.currentTool === 'move') {
            canvasData.selectedPointIdx = clickedIdx;
            canvasData.pointMoved = false; // Reset movement tracker
            drawCanvas();
        } else if (canvasData.currentTool === 'add') {
            const mathX = (x - canvasData.offsetX) / canvasData.scale;
            const mathY = -(y - canvasData.offsetY) / canvasData.scale;

            // Find nearest neighbor region
            let nearestRegion = 1; // Default
            let minDist = Infinity;
            for (let i = 0; i < canvasData.points.length; i++) {
                const pt = canvasData.points[i];
                const dx = pt[0] - mathX;
                const dy = pt[1] - mathY;
                const d = dx * dx + dy * dy;
                if (d < minDist) {
                    minDist = d;
                    nearestRegion = canvasData.regions[i];
                }
            }

            canvasData.points.push([mathX, mathY]);
            canvasData.regions.push(nearestRegion);
            canvasData.classifications.push('interior');
            saveState();
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
            tooltip.innerText = `ID: ${hoverIdx}\nX: ${formatSci(pt[0])}, Y: ${formatSci(pt[1])}\nReg: ${canvasData.regions[hoverIdx]}`;
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
        document.getElementById('coordinatesDisplay').innerText = `X: ${formatSci(mathX)}, Y: ${formatSci(mathY)}`;

        if (!canvasData.isDragging) return;

        if (canvasData.currentTool === 'pan') {
            canvasData.offsetX += dx;
            canvasData.offsetY += dy;
            drawCanvas();
        } else if (canvasData.currentTool === 'move' && canvasData.selectedPointIdx !== -1) {
            canvasData.points[canvasData.selectedPointIdx][0] = mathX;
            canvasData.points[canvasData.selectedPointIdx][1] = mathY;
            canvasData.pointMoved = true;
            drawCanvas();
        }

        canvasData.lastMouseX = x;
        canvasData.lastMouseY = y;
    });

    window.addEventListener('mouseup', () => {
        if (canvasData.isDragging && canvasData.currentTool === 'move' && canvasData.pointMoved) {
            saveState();
        }
        canvasData.isDragging = false;
        canvasData.selectedPointIdx = -1;
        canvasData.pointMoved = false;
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

    const btnAdd = document.getElementById('toolAdd');
    const btnUndo = document.getElementById('btnUndo');
    const btnRedo = document.getElementById('btnRedo');
    const togglePrintMode = document.getElementById('togglePrintMode');
    const toggleColorRegions = document.getElementById('toggleColorRegions');
    const printModeBtn = document.getElementById('printModeBtn');
    const colorModeBtn = document.getElementById('colorModeBtn');

    const setTool = (tool, activeBtn) => {
        canvasData.currentTool = tool;
        [btnPan, btnMove, btnDelete, btnAdd].forEach(b => b.classList.remove('active'));
        activeBtn.classList.add('active');
        document.getElementById('cloudCanvas').style.cursor = tool === 'pan' ? 'grab' : 'crosshair';
    };

    btnPan.addEventListener('click', () => setTool('pan', btnPan));
    btnMove.addEventListener('click', () => setTool('move', btnMove));
    btnDelete.addEventListener('click', () => setTool('delete', btnDelete));
    btnAdd.addEventListener('click', () => setTool('add', btnAdd));
    btnReset.addEventListener('click', resetView);

    // Undo/Redo logic
    btnUndo.addEventListener('click', () => {
        if (canvasData.historyIndex > 0) {
            canvasData.historyIndex--;
            const state = canvasData.history[canvasData.historyIndex];
            canvasData.points = state.points.map(p => [...p]);
            canvasData.regions = [...state.regions];
            canvasData.classifications = [...state.classifications];
            updateUndoRedoButtons();
            calculateDensityMetrics();
            drawCanvas();
        }
    });

    btnRedo.addEventListener('click', () => {
        if (canvasData.historyIndex < canvasData.history.length - 1) {
            canvasData.historyIndex++;
            const state = canvasData.history[canvasData.historyIndex];
            canvasData.points = state.points.map(p => [...p]);
            canvasData.regions = [...state.regions];
            canvasData.classifications = [...state.classifications];
            updateUndoRedoButtons();
            calculateDensityMetrics();
            drawCanvas();
        }
    });

    // Toggles logic
    togglePrintMode.addEventListener('change', (e) => {
        canvasData.printMode = e.target.checked;
        const wrapper = document.querySelector('.canvas-wrapper');
        if (canvasData.printMode) {
            wrapper.style.background = '#ffffff';
            printModeBtn.style.background = '#e2e8f0';
        } else {
            wrapper.style.background = '#1a1f24';
            printModeBtn.style.background = 'white';
        }
        drawCanvas();
    });

    toggleColorRegions.addEventListener('change', (e) => {
        canvasData.colorRegions = e.target.checked;
        if (canvasData.colorRegions) {
            colorModeBtn.classList.add('active');
        } else {
            colorModeBtn.classList.remove('active');
        }
        drawCanvas();
    });

    // Keyboard shortcuts for Undo/Redo
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            if (e.key === 'z') {
                e.preventDefault();
                btnUndo.click();
            } else if (e.key === 'y') {
                e.preventDefault();
                btnRedo.click();
            }
        }
    });
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
