/**
 * Main JavaScript file for Survey Data Analysis Platform
 * Handles frontend interactions, AJAX calls, and UI enhancements
 */

// Global variables
let processingInterval = null;
let chartInstances = {};

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApplication();
});

/**
 * Initialize all application components
 */
function initializeApplication() {
    // Initialize Bootstrap components
    initializeBootstrapComponents();
    
    // Initialize form handlers
    initializeFormHandlers();
    
    // Initialize drag and drop
    initializeDragAndDrop();
    
    // Initialize processing status checks
    initializeProcessingStatusCheck();
    
    // Initialize charts
    initializeCharts();
    
    // Initialize tooltips and popovers
    initializeTooltips();
    
    // Initialize theme handling
    initializeThemeHandling();

    // Initialize scroll-reveal animations
    initializeScrollReveal();
    
    console.log('Survey Data Analysis Platform initialized successfully');
}

/**
 * Initialize scroll-reveal animations
 */
function initializeScrollReveal() {
    const revealElements = document.querySelectorAll('.reveal');

    function revealOnScroll() {
        const windowHeight = window.innerHeight;
        revealElements.forEach(el => {
            const elementTop = el.getBoundingClientRect().top;
            if (elementTop < windowHeight - 100) {
                el.classList.add('active');
            }
        });
    }

    window.addEventListener('scroll', revealOnScroll);
    revealOnScroll(); // Initial check
}

/**
 * Initialize Bootstrap components
 */
function initializeBootstrapComponents() {
    // Initialize all tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl, {
            trigger: 'hover focus'
        });
    });

    // Initialize all popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Initialize all toasts
    const toastElList = [].slice.call(document.querySelectorAll('.toast'));
    toastElList.map(function (toastEl) {
        return new bootstrap.Toast(toastEl);
    });
}

/**
 * Initialize form handlers
 */
function initializeFormHandlers() {
    // File upload form handling
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        initializeFileUpload();
    }

    // Analysis configuration form
    const configForm = document.querySelector('form[action*="process_data"]');
    if (configForm) {
        initializeConfigurationForm();
    }

    // Weight options toggle
    const applyWeightsCheckbox = document.getElementById('applyWeights');
    const weightOptions = document.getElementById('weightOptions');
    if (applyWeightsCheckbox && weightOptions) {
        applyWeightsCheckbox.addEventListener('change', function() {
            weightOptions.style.display = this.checked ? 'block' : 'none';
            if (this.checked) {
                weightOptions.classList.add('fade-in');
            }
        });
    }
}

/**
 * Initialize file upload functionality
 */
function initializeFileUpload() {
    const fileInput = document.getElementById('file');
    const uploadForm = document.getElementById('uploadForm');
    const uploadBtn = document.getElementById('uploadBtn');
    const filePreview = document.getElementById('filePreview');
    const fileInfo = document.getElementById('fileInfo');
    const uploadProgress = document.getElementById('uploadProgress');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');

    if (!fileInput || !uploadForm) return;

    // File selection handler
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            displayFilePreview(file);
            validateFile(file);
        } else if (filePreview) {
            filePreview.style.display = 'none';
        }
    });

    // Form submission handler
    uploadForm.addEventListener('submit', function(e) {
        const file = fileInput.files[0];
        if (!file) {
            e.preventDefault();
            showAlert('Please select a file to upload.', 'warning');
            return;
        }

        if (!validateFile(file)) {
            e.preventDefault();
            return;
        }

        // Show progress
        showUploadProgress();
    });

    /**
     * Display file preview information
     */
    function displayFilePreview(file) {
        if (!fileInfo || !filePreview) return;

        const fileSize = formatFileSize(file.size);
        const fileType = file.type || 'Unknown';
        const fileName = file.name;
        const fileExtension = fileName.split('.').pop().toLowerCase();

        fileInfo.innerHTML = `
            <div class="row g-2">
                <div class="col-sm-3"><strong>Name:</strong></div>
                <div class="col-sm-9">${escapeHtml(fileName)}</div>
                <div class="col-sm-3"><strong>Size:</strong></div>
                <div class="col-sm-9">${fileSize}</div>
                <div class="col-sm-3"><strong>Type:</strong></div>
                <div class="col-sm-9">
                    <span class="badge bg-${getFileTypeBadgeColor(fileExtension)}">
                        ${fileExtension.toUpperCase()}
                    </span>
                </div>
            </div>
        `;

        filePreview.style.display = 'block';
        filePreview.classList.add('fade-in');
    }

    /**
     * Validate uploaded file
     */
    function validateFile(file) {
        const maxSize = 16 * 1024 * 1024; // 16MB
        const allowedTypes = ['csv', 'xlsx', 'xls'];
        const fileExtension = file.name.split('.').pop().toLowerCase();

        if (file.size > maxSize) {
            showAlert('File size exceeds 16MB limit. Please choose a smaller file.', 'error');
            return false;
        }

        if (!allowedTypes.includes(fileExtension)) {
            showAlert('Invalid file type. Please upload CSV or Excel files only.', 'error');
            return false;
        }

        return true;
    }

    /**
     * Show upload progress
     */
    function showUploadProgress() {
        if (!uploadBtn || !uploadProgress || !progressBar || !progressText) return;

        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Uploading...';
        uploadProgress.style.display = 'block';

        // Simulate progress animation
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress > 95) {
                clearInterval(progressInterval);
                progress = 95;
                progressText.textContent = 'Processing file...';
            }

            progressBar.style.width = progress + '%';
            progressBar.setAttribute('aria-valuenow', progress);

            if (progress < 30) {
                progressText.textContent = 'Uploading file...';
            } else if (progress < 60) {
                progressText.textContent = 'Validating data...';
            } else if (progress < 90) {
                progressText.textContent = 'Analyzing structure...';
            } else {
                progressText.textContent = 'Preparing analysis...';
            }
        }, 200);
    }
}

/**
 * Initialize configuration form
 */
function initializeConfigurationForm() {
    // Add form validation
    const forms = document.querySelectorAll('.needs-validation');
    Array.prototype.slice.call(forms).forEach(function (form) {
        form.addEventListener('submit', function (event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });

    // Imputation method help text
    const imputationSelect = document.querySelector('select[name="imputation_method"]');
    if (imputationSelect) {
        imputationSelect.addEventListener('change', function() {
            updateImputationHelp(this.value);
        });
    }

    // Outlier method help text
    const outlierSelect = document.querySelector('select[name="outlier_method"]');
    if (outlierSelect) {
        outlierSelect.addEventListener('change', function() {
            updateOutlierHelp(this.value);
        });
    }
}

/**
 * Initialize drag and drop functionality
 */
function initializeDragAndDrop() {
    const dropZones = document.querySelectorAll('.file-upload-area, .card-body');
    
    dropZones.forEach(zone => {
        zone.addEventListener('dragover', handleDragOver);
        zone.addEventListener('dragleave', handleDragLeave);
        zone.addEventListener('drop', handleDrop);
    });

    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.add('drag-over');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('drag-over');
    }

    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        const fileInput = document.getElementById('file');
        
        if (files.length > 0 && fileInput) {
            fileInput.files = files;
            const event = new Event('change', { bubbles: true });
            fileInput.dispatchEvent(event);
        }
    }
}

/**
 * Initialize processing status checking
 */
function initializeProcessingStatusCheck() {
    const processingElements = document.querySelectorAll('[data-processing-status]');
    
    if (processingElements.length > 0) {
        startProcessingStatusCheck();
    }
}

/**
 * Start checking processing status
 */
function startProcessingStatusCheck() {
    if (processingInterval) {
        clearInterval(processingInterval);
    }

    processingInterval = setInterval(() => {
        checkProcessingStatus();
    }, 5000); // Check every 5 seconds
}

/**
 * Check processing status via API
 */
function checkProcessingStatus() {
    const sessionId = getSessionIdFromUrl();
    if (!sessionId) return;

    fetch(`/api/processing_status/${sessionId}`)
        .then(response => response.json())
        .then(data => {
            updateProcessingStatus(data);
        })
        .catch(error => {
            console.error('Error checking processing status:', error);
        });
}

/**
 * Update processing status in UI
 */
function updateProcessingStatus(statusData) {
    const statusElements = document.querySelectorAll('[data-processing-status]');
    
    statusElements.forEach(element => {
        updateStatusElement(element, statusData);
    });

    // If processing is complete, stop checking and redirect if needed
    if (statusData.status === 'completed') {
        clearInterval(processingInterval);
        processingInterval = null;
        
        // Show success message
        showAlert('Processing completed successfully!', 'success');
        
        // Auto-redirect to results page after 2 seconds
        setTimeout(() => {
            const sessionId = getSessionIdFromUrl();
            if (sessionId) {
                window.location.href = `/results/${sessionId}`;
            }
        }, 2000);
    } else if (statusData.status === 'error') {
        clearInterval(processingInterval);
        processingInterval = null;
        
        showAlert('Processing failed. Please try again.', 'error');
    }
}

/**
 * Initialize charts
 */
function initializeCharts() {
    // Initialize Chart.js defaults
    Chart.defaults.responsive = true;
    Chart.defaults.maintainAspectRatio = false;
    Chart.defaults.plugins.legend.position = 'bottom';

    // Find and initialize all chart canvases
    const chartCanvases = document.querySelectorAll('canvas[data-chart-type]');
    chartCanvases.forEach(canvas => {
        initializeChart(canvas);
    });
}

/**
 * Initialize individual chart
 */
function initializeChart(canvas) {
    const chartType = canvas.getAttribute('data-chart-type');
    const chartData = JSON.parse(canvas.getAttribute('data-chart-data') || '{}');
    
    if (chartInstances[canvas.id]) {
        chartInstances[canvas.id].destroy();
    }

    const ctx = canvas.getContext('2d');
    chartInstances[canvas.id] = new Chart(ctx, {
        type: chartType,
        data: chartData,
        options: getChartOptions(chartType)
    });
}

/**
 * Get chart configuration options
 */
function getChartOptions(chartType) {
    const baseOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom'
            }
        }
    };

    switch (chartType) {
        case 'bar':
            return {
                ...baseOptions,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            };
        case 'line':
            return {
                ...baseOptions,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                elements: {
                    line: {
                        tension: 0.4
                    }
                }
            };
        case 'pie':
        case 'doughnut':
            return {
                ...baseOptions,
                cutout: chartType === 'doughnut' ? '50%' : 0
            };
        default:
            return baseOptions;
    }
}

/**
 * Initialize tooltips
 */
function initializeTooltips() {
    // Re-initialize tooltips for dynamically added content
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]:not([data-tooltip-initialized])'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        tooltipTriggerEl.setAttribute('data-tooltip-initialized', 'true');
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize theme handling
 */
function initializeThemeHandling() {
    // Handle theme changes if needed
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
}

/**
 * Utility Functions
 */

/**
 * Format file size in human readable format
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Get file type badge color
 */
function getFileTypeBadgeColor(extension) {
    const colorMap = {
        'csv': 'info',
        'xlsx': 'success',
        'xls': 'success',
        'pdf': 'danger',
        'html': 'warning'
    };
    return colorMap[extension] || 'secondary';
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, function(m) { return map[m]; });
}

/**
 * Show alert message
 */
function showAlert(message, type = 'info', duration = 5000) {
    const alertContainer = document.createElement('div');
    alertContainer.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
    alertContainer.innerHTML = `
        <i class="fas fa-${getAlertIcon(type)} me-2"></i>
        ${escapeHtml(message)}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    // Insert at the top of the main container
    const mainContainer = document.querySelector('main .container');
    if (mainContainer) {
        mainContainer.insertBefore(alertContainer, mainContainer.firstChild);
    }

    // Auto-remove after duration
    setTimeout(() => {
        if (alertContainer.parentNode) {
            alertContainer.remove();
        }
    }, duration);
}

/**
 * Get alert icon based on type
 */
function getAlertIcon(type) {
    const iconMap = {
        'success': 'check-circle',
        'error': 'exclamation-triangle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };
    return iconMap[type] || 'info-circle';
}

/**
 * Get session ID from current URL
 */
function getSessionIdFromUrl() {
    const pathParts = window.location.pathname.split('/');
    const sessionIndex = pathParts.findIndex(part => part === 'configure' || part === 'results' || part === 'process');
    return sessionIndex !== -1 && pathParts[sessionIndex + 1] ? pathParts[sessionIndex + 1] : null;
}

/**
 * Update status element with new data
 */
function updateStatusElement(element, statusData) {
    const statusText = element.querySelector('.status-text');
    const statusIcon = element.querySelector('.status-icon');
    const statusBadge = element.querySelector('.status-badge');

    if (statusText) {
        statusText.textContent = statusData.message || statusData.status;
    }

    if (statusIcon) {
        statusIcon.className = `fas fa-${getStatusIcon(statusData.status)} status-icon`;
    }

    if (statusBadge) {
        statusBadge.className = `badge bg-${getStatusColor(statusData.status)}`;
        statusBadge.textContent = statusData.status.charAt(0).toUpperCase() + statusData.status.slice(1);
    }
}

/**
 * Get status icon
 */
function getStatusIcon(status) {
    const iconMap = {
        'uploaded': 'clock',
        'processing': 'spinner fa-spin',
        'completed': 'check-circle',
        'error': 'exclamation-triangle'
    };
    return iconMap[status] || 'clock';
}

/**
 * Get status color
 */
function getStatusColor(status) {
    const colorMap = {
        'uploaded': 'secondary',
        'processing': 'warning',
        'completed': 'success',
        'error': 'danger'
    };
    return colorMap[status] || 'secondary';
}

/**
 * Update imputation method help text
 */
function updateImputationHelp(method) {
    const helpTexts = {
        'mean': 'Replaces missing values with the arithmetic mean of the column.',
        'median': 'Replaces missing values with the median value of the column.',
        'mode': 'Replaces missing values with the most frequent value.',
        'knn': 'Uses K-Nearest Neighbors to estimate missing values based on similar records.',
        'none': 'Skips missing value imputation - missing values will remain as-is.'
    };

    const helpElement = document.getElementById('imputationHelp');
    if (helpElement) {
        helpElement.textContent = helpTexts[method] || '';
    }
}

/**
 * Update outlier method help text
 */
function updateOutlierHelp(method) {
    const helpTexts = {
        'iqr': 'Identifies outliers using the Interquartile Range method (Q1 - 1.5*IQR, Q3 + 1.5*IQR).',
        'zscore': 'Identifies outliers using Z-Score method (values beyond ±3 standard deviations).',
        'winsorization': 'Caps extreme values at the 5th and 95th percentiles.',
        'none': 'Skips outlier detection and handling.'
    };

    const helpElement = document.getElementById('outlierHelp');
    if (helpElement) {
        helpElement.textContent = helpTexts[method] || '';
    }
}

/**
 * Toggle application theme
 */
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-bs-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-bs-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Update charts if theme affects them
    Object.values(chartInstances).forEach(chart => {
        chart.update();
    });
}

/**
 * Cleanup function for page unload
 */
window.addEventListener('beforeunload', function() {
    // Clear intervals
    if (processingInterval) {
        clearInterval(processingInterval);
    }

    // Destroy chart instances
    Object.values(chartInstances).forEach(chart => {
        if (chart && typeof chart.destroy === 'function') {
            chart.destroy();
        }
    });
});

// Export functions for use in other scripts
window.SurveyAnalysisPlatform = {
    showAlert,
    formatFileSize,
    initializeChart,
    updateProcessingStatus,
    getSessionIdFromUrl
};
