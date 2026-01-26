/**
 * @fileoverview Examples Page Interactive Functionality
 * 
 * Handles interactive features for the Examples page, including image lightboxes.
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Examples.js loaded');

    // 1. Initialize Lightbox
    // Use ID selector for specificity
    let lightbox = document.getElementById('lightbox');
    
    // If not present, create it
    if (!lightbox) {
        console.log('Creating lightbox element');
        lightbox = document.createElement('div');
        lightbox.id = 'lightbox';
        lightbox.className = 'lightbox';
        
        lightbox.innerHTML = `
            <div class="lightbox-content">
                <button class="lightbox-close" aria-label="Close">&times;</button>
                <img class="lightbox-image" src="" alt="Full size view">
                <div class="lightbox-caption"></div>
            </div>
        `;
        
        // Append to body to ensure it's not trapped in overflow containers
        document.body.appendChild(lightbox);
    }
    
    // Cache elements
    const lightboxImg = lightbox.querySelector('.lightbox-image');
    const lightboxCaption = lightbox.querySelector('.lightbox-caption');
    const closeBtn = lightbox.querySelector('.lightbox-close');
    
    // 2. Open Lightbox Function
    function openLightbox(imgElement) {
        console.log('Opening lightbox for:', imgElement.src);
        
        if (!imgElement.src) {
            console.error('Image has no source');
            return;
        }

        // Set content
        lightboxImg.src = imgElement.src;
        lightboxImg.alt = imgElement.alt || 'Full size view';
        
        // Get caption safely
        let captionText = '';
        try {
            const card = imgElement.closest('.example-card');
            const title = card ? card.querySelector('.example-title').innerText : '';
            // Try to find the label sibling or find it within the container
            let label = '';
            const container = imgElement.closest('.example-image-container');
            if (container) {
                const labelEl = container.querySelector('.image-label');
                if (labelEl) label = labelEl.innerText;
            }
            
            captionText = title ? `${title} - ${label}` : label;
        } catch (e) {
            console.warn('Could not extract caption', e);
        }
        lightboxCaption.textContent = captionText;
        
        // Activate
        // 1. Set display to flex (via style to ensure it's in layout)
        lightbox.style.display = 'flex';
        
        // 2. Force reflow
        void lightbox.offsetWidth;
        
        // 3. Add active class for transitions
        lightbox.classList.add('active');
        
        // 4. Lock body scroll
        document.body.style.overflow = 'hidden';
    }
    
    // 3. Close Lightbox Function
    function closeLightbox() {
        console.log('Closing lightbox');
        
        // Remove active class
        lightbox.classList.remove('active');
        
        // Unlock body scroll
        document.body.style.overflow = '';
        
        // Wait for transition then hide
        setTimeout(() => {
            if (!lightbox.classList.contains('active')) {
                lightbox.style.display = 'none';
                lightboxImg.src = ''; // Clear memory
            }
        }, 300);
    }
    
    // 4. Attach Event Listeners to Images
    // We delegate to document to handle any dynamic content and ensure robustness
    document.addEventListener('click', function(e) {
        // Check if clicked element is an example image OR inside the container
        // This handles clicks on the label, overlay, or the image itself
        const container = e.target.closest('.example-image-container');
        
        if (container) {
            // Find the image within this container
            const img = container.querySelector('img');
            if (img) {
                e.preventDefault();
                e.stopPropagation();
                openLightbox(img);
                return; // Stop processing
            }
        }
        
        // Check if clicked element is close button
        if (e.target.matches('.lightbox-close') || e.target.closest('.lightbox-close')) {
            e.preventDefault();
            closeLightbox();
        }
        
        // Check if clicked element is lightbox background
        if (e.target === lightbox) {
            closeLightbox();
        }
    });
    
    // Keyboard support
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && lightbox.classList.contains('active')) {
            closeLightbox();
        }
    });
});
