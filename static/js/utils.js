/**
 * Utility functions for mGFD CloudGenerator
 * Contains common helper functions used across different modules.
 * 
 * @namespace Utils
 */

const Utils = {
    /**
     * Formats a file size in bytes to a human-readable string with appropriate units.
     * Converts bytes to the most appropriate unit (Bytes, KB, MB, GB) and formats
     * the result with proper decimal precision for optimal readability.
     * 
     * @param {number} bytes - The file size in bytes to be formatted
     * @returns {string} The formatted file size string with appropriate unit
     */
    formatFileSize: function(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    /**
     * Shows an error message in the upload zone.
     * Standardizes the error display across different modules.
     * 
     * @param {HTMLElement} zone - The upload zone element
     * @param {HTMLElement} content - The content element to update
     * @param {string} title - The error title
     * @param {string} message - The error message
     * @param {string} [resetFn='resetUpload'] - The name of the global function to call to reset
     */
    showUploadError: function(zone, content, title, message, resetFn = 'resetUpload') {
        if (!zone || !content) return;

        zone.classList.remove('uploading', 'success');
        zone.classList.add('error');
        
        content.innerHTML = `
            <div class="upload-error">
                <div class="error-icon">
                    <i class="fas fa-exclamation-triangle"></i>
                </div>
                <h4>${title}</h4>
                <p>${message}</p>
                <button class="control-btn secondary small" onclick="${resetFn}()">
                    <i class="fas fa-redo"></i>
                    Try Again
                </button>
            </div>
        `;
    }
};
