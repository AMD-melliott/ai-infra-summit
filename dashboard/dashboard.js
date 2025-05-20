// Dashboard JavaScript for real-time updates and visualization
// Completes the demo dashboard HTML

// Dashboard Configuration
const DASHBOARD_CONFIG = {
    SHOW_VRAM_PROGRESS_BARS: true, // Set to true to show VRAM progress bars, false to hide them
    UPDATE_INTERVAL: 15000          // Update interval in milliseconds
};

// Helper function to format bytes to GB
function formatGB(bytes) {
    // Ensure bytes is a number and not NaN
    if (typeof bytes !== 'number' || isNaN(bytes)) {
        return 'N/A';
    }
    if (bytes === 0) return '0.0';
    // Assuming input is already in GB from the backend
    return bytes.toFixed(1);
}

// Function to initially render all partition cards from static config
function initializePartitionDisplay(config) {
    const container = document.getElementById('partition-usage-container');
    container.innerHTML = ''; // Clear any previous placeholders

    // Assuming config is an object like {"0": {"name": ..., "bg_color": ...}, ...}
    // Get sorted keys to ensure consistent order
    const partitionIds = Object.keys(config).sort((a, b) => parseInt(a) - parseInt(b));

    if (partitionIds.length === 0) {
        container.innerHTML = '<div class="col"><p class="text-muted">No partition configuration loaded. Waiting for live data...</p></div>';
        return;
    }

    partitionIds.forEach(id => {
        const partitionConfig = config[id] || {}; // Handle potential missing entries
        const partitionName = typeof partitionConfig === 'object' ? partitionConfig.name : partitionConfig; 
        const bgColor = typeof partitionConfig === 'object' ? partitionConfig.bg_color : null;
        const displayName = partitionName || `GPU ${id}`; // Fallback display name

        const partitionCol = document.createElement('div');
        partitionCol.className = 'col';
        // Use a unique ID for the card itself and its elements
        partitionCol.innerHTML = `
            <div class="card h-100" id="partition-card-${id}" style="${bgColor ? `background-color: ${bgColor}; border-color: #495057; color: #fff;` : 'border-color: #ced4da;'}">
                <div class="card-header small fw-bold" style="${bgColor ? 'background-color: rgba(0,0,0,0.15); border-bottom: 1px solid rgba(255,255,255,0.2);' : 'background-color: #f8f9fa;'}">
                    ${displayName}<br><span class="text-muted small" style="${bgColor ? 'color: rgba(255,255,255,0.7);' : ''}">(GPU ${id})</span>
                </div>
                <div class="card-body p-2">
                    <p class="card-text mb-1 small">
                        VRAM: <span id="partition-used-${id}" class="fw-bold">--</span> / <span id="partition-total-${id}" class="fw-bold">--</span> GB
                    </p>
                    <div class="progress mb-2" style="height: 20px; background-color: ${bgColor ? 'rgba(0,0,0,0.25)' : '#e9ecef'}; position: relative;">
                        <div id="partition-progress-${id}" class="progress-bar bg-info d-flex align-items-center justify-content-center" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
                            <span id="partition-percent-${id}" class="fw-bold small" style="color: #fff; text-shadow: 1px 1px 1px rgba(0,0,0,0.4);">--%</span>
                        </div>
                    </div>
                    <!-- GPU Util Section Commented Out START -->
                    <!--
                    <p class="card-text mb-1 small">
                        GPU Util:
                    </p>
                    <div class="progress" style="height: 20px; background-color: ${bgColor ? 'rgba(0,0,0,0.25)' : '#e9ecef'}; position: relative;">
                        <div id="gpu-util-progress-${id}" class="progress-bar bg-success d-flex align-items-center justify-content-center" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
                            <span id="gpu-util-percent-${id}" class="fw-bold small" style="color: #fff; text-shadow: 1px 1px 1px rgba(0,0,0,0.4);">--%</span>
                        </div>
                    </div>
                    -->
                    <!-- GPU Util Section Commented Out END -->
                </div>
            </div>
        `;
        container.appendChild(partitionCol);
    });
}

// Function to update partition usage display based on live metrics
function updatePartitionDisplay(partitions) {
    if (!partitions) { // Allow empty array to clear/update to empty state
        console.warn("Received null/undefined partition data from /api/metrics");
        // Potentially clear all partition data to N/A if that's desired for a full loss of signal
        // For now, do nothing if partitions is null/undefined
        return;
    }
    
    // Get all existing partition card IDs from the config (or DOM if config not stored)
    const displayedPartitionIds = new Set();
    document.querySelectorAll('[id^="partition-card-"]').forEach(card => {
        displayedPartitionIds.add(card.id.replace('partition-card-', ''));
    });

    partitions.forEach(partition => {
        const cardElement = document.getElementById(`partition-card-${partition.id}`);
        if (!cardElement) {
            // This case should ideally be handled by ensuring initializePartitionDisplay creates all necessary cards
            // or by dynamically adding cards here if a new, unexpected partition appears.
            // For now, we log if a card from metrics doesn't exist in the initial config.
            // console.warn(`Card element not found for partition ID: ${partition.id}. Metrics received for unconfigured partition.`);
            // To dynamically add, you'd need more info or make assumptions about its display properties.
            return; 
        }
        
        displayedPartitionIds.delete(String(partition.id)); // Mark as updated

        // Update background color if provided by metrics and different (though usually from config)
        if (partition.bg_color && cardElement.style.backgroundColor !== partition.bg_color) {
            cardElement.style.backgroundColor = partition.bg_color;
            // Adjust text/border colors for contrast if bg_color is set
            cardElement.style.color = '#fff';
            cardElement.style.borderColor = '#495057';
            const header = cardElement.querySelector('.card-header');
            if (header) header.style.backgroundColor = 'rgba(0,0,0,0.15)';
            const progressContainer = cardElement.querySelector('.progress');
            if (progressContainer) progressContainer.style.backgroundColor = 'rgba(0,0,0,0.25)';
        }


        const usedGB = parseFloat(partition.vram_used);
        const totalGB = parseFloat(partition.vram_total);

        const usedFormatted = formatGB(usedGB);
        const totalFormatted = formatGB(totalGB);

        const usedSpan = document.getElementById(`partition-used-${partition.id}`);
        const totalSpan = document.getElementById(`partition-total-${partition.id}`);
        if (usedSpan) usedSpan.textContent = usedFormatted;
        if (totalSpan) totalSpan.textContent = totalFormatted;
        
        const progressBar = document.getElementById(`partition-progress-${partition.id}`);
        const progressContainer = progressBar ? progressBar.parentElement : null;
        const percentSpan = document.getElementById(`partition-percent-${partition.id}`);

        if (progressContainer) {
            // Set visibility based on configuration
            progressContainer.style.display = DASHBOARD_CONFIG.SHOW_VRAM_PROGRESS_BARS ? 'block' : 'none';
        }

        if (progressBar && DASHBOARD_CONFIG.SHOW_VRAM_PROGRESS_BARS) {
            // Calculate and display percentage when progress bars are enabled
            let usagePercent = 0;
            let percentFixed = "N/A";

            if (typeof usedGB === 'number' && !isNaN(usedGB) && typeof totalGB === 'number' && !isNaN(totalGB) && totalGB > 0) {
                usagePercent = (usedGB / totalGB) * 100;
                // Cap at 100% for display purposes, even if actual usage reports higher
                usagePercent = Math.min(usagePercent, 100);
                percentFixed = usagePercent.toFixed(1);
            } else if (typeof usedGB === 'number' && !isNaN(usedGB) && typeof totalGB === 'number' && !isNaN(totalGB) && totalGB === 0 && usedGB === 0) {
                usagePercent = 0; // Explicitly set to 0 if 0/0
                percentFixed = "0.0";
            }

            const currentProgress = typeof usagePercent === 'number' && !isNaN(usagePercent) ? Math.max(0, Math.min(100, usagePercent)) : 0;
            progressBar.style.width = `${currentProgress.toFixed(1)}%`;
            progressBar.setAttribute('aria-valuenow', currentProgress.toFixed(1));
            
            // Update color based on usage
            if (currentProgress > 90) {
                progressBar.classList.remove('bg-info', 'bg-warning', 'bg-secondary');
                progressBar.classList.add('bg-danger');
            } else if (currentProgress > 70) {
                progressBar.classList.remove('bg-info', 'bg-danger', 'bg-secondary');
                progressBar.classList.add('bg-warning');
            } else {
                progressBar.classList.remove('bg-warning', 'bg-danger', 'bg-secondary');
                progressBar.classList.add('bg-info');
            }

            if (percentSpan) {
                percentSpan.style.display = 'inline';
                percentSpan.textContent = percentFixed !== "N/A" ? `${percentFixed}%` : "N/A";
            }
        } else if (progressBar) {
            // Set progress bar to a neutral, static state when disabled
            progressBar.style.width = '0%';
            progressBar.setAttribute('aria-valuenow', '0');
            progressBar.classList.remove('bg-info', 'bg-warning', 'bg-danger');
            progressBar.classList.add('bg-secondary');
            
            if (percentSpan) {
                percentSpan.style.display = 'none';
            }
        }

    });
    
    // For any partitions that were in config but not in the latest metrics, set their data to N/A
    displayedPartitionIds.forEach(staleId => {
        const usedSpan = document.getElementById(`partition-used-${staleId}`);
        const totalSpan = document.getElementById(`partition-total-${staleId}`);
        const percentSpan = document.getElementById(`partition-percent-${staleId}`);
        const progressBar = document.getElementById(`partition-progress-${staleId}`);

        if (usedSpan) usedSpan.textContent = 'N/A';
        if (totalSpan) totalSpan.textContent = 'N/A';
        if (percentSpan) percentSpan.textContent = 'N/A';
        if (progressBar) {
            progressBar.style.width = '0%';
            progressBar.setAttribute('aria-valuenow', '0');
            progressBar.className = 'progress-bar bg-secondary d-flex align-items-center justify-content-center';
        }
        console.warn(`Partition ID ${staleId} was in config/DOM but not in latest metrics. Marked as N/A.`);
    });
}

// Function to update summary display
function updateSummaryDisplay(summary) {
    const totalVramAvailableEl = document.getElementById('total-vram-available');
    if (totalVramAvailableEl) {
        totalVramAvailableEl.textContent = formatGB(summary.total_vram_available);
    } else {
        console.warn("Element with ID 'total-vram-available' not found in the DOM.");
    }

    const totalVramUsedEl = document.getElementById('total-vram-used');
    if (totalVramUsedEl) {
        totalVramUsedEl.textContent = formatGB(summary.total_vram_used);
    } else {
        console.warn("Element with ID 'total-vram-used' not found in the DOM.");
    }

    const totalVramFreeEl = document.getElementById('total-vram-free');
    if (totalVramFreeEl) {
        totalVramFreeEl.textContent = formatGB(summary.total_vram_free);
    } else {
        console.warn("Element with ID 'total-vram-free' not found in the DOM.");
    }
}

// Function to fetch initial config and render layout
async function initializeDashboard() {
    try {
        const response = await fetch('/api/config');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const configData = await response.json();
        initializePartitionDisplay(configData); // This renders the layout based on config
        
        // Perform an initial metrics fetch immediately after setting up the layout
        // to populate data quickly, then start the interval.
        updateMetrics(); 

    } catch (error) {
        console.error('Error fetching or processing initial config:', error);
        const container = document.getElementById('partition-usage-container');
        if (container) {
            container.innerHTML = '<div class="col"><p class="text-danger">Failed to load initial partition config.</p></div>';
        }
    }
    // Start fetching metrics after initial layout is done
    // updateMetrics(); // Initial metrics fetch
    // setInterval(updateMetrics, updateInterval);
}

// Function to update metrics periodically
function updateMetrics() {
    fetch('/api/metrics')
        .then(response => {
            if (!response.ok) {
                // Store the error status to potentially use in UI updates
                const errorStatus = response.status; 
                throw new Error(`HTTP error! status: ${errorStatus}`);
            }
            return response.json();
        })
        .then(data => {
            // Clear any previous global error messages if successful
            const errorDisplay = document.getElementById('metrics-error-message');
            if (errorDisplay) errorDisplay.style.display = 'none';

            // Update Partition Display and Summary
            updatePartitionDisplay(data.partitions || []); // Pass empty array if data.partitions is undefined/null
            updateSummaryDisplay(data.summary || { total_vram_available: 0, total_vram_used: 0, total_vram_free: 0 });

            // Update Service Metrics and Charts (assuming data.services exists)
            if (data.services) {
                const services = data.services;
            

                // --- Update vLLM Card with Aggregated Data ---
                const vllmAggregateData = services['vllm_aggregate'];
                if (vllmAggregateData) {
                    document.getElementById('vllm-requests-running').textContent = vllmAggregateData.requests_running ?? '--';
                    document.getElementById('vllm-requests-waiting').textContent = vllmAggregateData.requests_waiting ?? '--';
                    document.getElementById('vllm-kv-cache-usage').textContent = vllmAggregateData.kv_cache_usage ?? '--';
                    // Optionally, add instance count/errors somewhere if needed
                    // console.log(`vLLM Instances: ${vllmAggregateData.instance_count}, Errors: ${vllmAggregateData.fetch_errors}`);
                } else {
                    // Handle case where aggregated data is missing (shouldn't happen if backend logic is correct)
                     document.getElementById('vllm-requests-running').textContent = 'N/A';
                     document.getElementById('vllm-requests-waiting').textContent = 'N/A';
                     document.getElementById('vllm-kv-cache-usage').textContent = 'N/A';
                }
            }
        })
        .catch(error => {
            console.error('Error fetching or processing metrics:', error);
            
            // Display a global error message
            const errorDisplay = document.getElementById('metrics-error-message'); 
            if (errorDisplay) {
                errorDisplay.textContent = `Failed to load live metrics: ${error.message}. Displaying last known data or N/A.`;
                errorDisplay.style.display = 'block'; 
            } else {
                 console.warn("No dedicated error message element found. Error logged to console.")
            }

            // On error, we might not want to clear all data immediately.
            // updatePartitionDisplay is now designed to mark stale partitions as N/A if they disappear from metrics.
            // If the entire /api/metrics call fails, updatePartitionDisplay won't be called with new data,
            // so existing data remains, which is the desired behavior ("Displaying last known data").
            // However, summary and service metrics might need explicit handling to show 'Error' or 'N/A'.

            // Clear summary on error, or set to N/A
            updateSummaryDisplay({ total_vram_available: 'N/A', total_vram_used: 'N/A', total_vram_free: 'N/A' });
            
            // Clear service metrics (vLLM aggregate) on error, or set to N/A
             if (document.getElementById('vllm-requests-running')) { // Check if elements exist
                 document.getElementById('vllm-requests-running').textContent = 'N/A';
                 document.getElementById('vllm-requests-waiting').textContent = 'N/A';
                 document.getElementById('vllm-kv-cache-usage').textContent = 'N/A';
             }
             // Potentially set other service metrics (SGLang, Flux) to N/A as well if they are critical.
        });
}


function animatePartitions() {
    const partitions = document.querySelectorAll('.gpu-partition');
    
    // Subtle pulse animation for active partitions
    partitions.forEach(partition => {
        // Add a subtle pulse effect
        setInterval(() => {
            partition.style.opacity = '0.8';
            setTimeout(() => {
                partition.style.opacity = '1';
            }, 500);
        }, 5000 + Math.random() * 2000); // Random offset for each partition
    });
}

animatePartitions();

// Event listeners for UI interaction (keep as is for now)
document.querySelectorAll('.service-links a').forEach(link => {
    link.addEventListener('click', function(event) {
        // Simple check if it's not the monitoring link
        if (!this.href.includes('/monitoring/grafana/')) {
            // Prevent default if it's a placeholder '#' link
            if(this.getAttribute('href') === '#') {
              event.preventDefault();
            }
            // Add loading indicator when clicking service links
            const icon = this.querySelector('i');
            if (icon) { // Check if icon exists
                const originalClass = icon.className;
                icon.className = 'fas fa-spinner fa-spin';

                setTimeout(() => {
                    // Check if icon still exists before resetting class
                    if (this.querySelector('i')) {
                         this.querySelector('i').className = originalClass;
                    }
                }, 1000);
            }
        }
        // Allow default behavior for actual links like monitoring
    });
});

// Initialize the dashboard on load
initializeDashboard();

// Set interval for updating metrics after initial load and first fetch
setInterval(updateMetrics, DASHBOARD_CONFIG.UPDATE_INTERVAL);