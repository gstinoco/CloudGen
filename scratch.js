    regionsList.innerHTML = detectedRegions.map((region, index) => {
        const pointCount = region.contour_points ? region.contour_points.length : 0;
        const color = region.color || '#3b82f6';
        
        return `
            <div class="region-card" style="--rc: ${color}">
                <div class="rc-header">
                    <div class="rc-color-dot" style="background: ${color}; box-shadow: 0 0 10px ${color}80;"></div>
                    <h4 class="rc-title">${region.name}</h4>
                    <span class="rc-points">${pointCount} pts</span>
                </div>
                
                <div class="rc-body">
                    <div class="rc-retention">
                        <div class="rc-retention-header">
                            <span><i class="fas fa-compress-arrows-alt"></i> Node Retention</span>
                            <span id="smoothing-val-${index}" class="rc-retention-val">${region.smoothing !== undefined ? region.smoothing : 100}%</span>
                        </div>
                        <input type="range" class="rc-slider" min="1" max="100" value="${region.smoothing !== undefined ? region.smoothing : 100}" 
                               oninput="document.getElementById('smoothing-val-${index}').textContent = this.value + '%'"
                               onchange="applyRegionSmoothing(${index}, this.value)">
                    </div>
                </div>
                
                <div class="rc-actions">
                    <button class="rc-btn rc-toggle ${region.visible ? 'active' : ''}" onclick="toggleRegion(${index})" title="${region.visible ? 'Hide' : 'Show'} region">
                        <i class="fas ${region.visible ? 'fa-eye' : 'fa-eye-slash'}"></i>
                    </button>
                    <button class="rc-btn rc-export" onclick="exportSingleRegion(${index})" title="Export region">
                        <i class="fas fa-download"></i>
                    </button>
                    <button class="rc-btn rc-delete" onclick="deleteRegion(${index})" title="Delete region">
                        <i class="fas fa-trash-alt"></i>
                    </button>
                </div>
            </div>
        `;
    }).join('');
