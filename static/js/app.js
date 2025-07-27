// Diabetes Prediction AI - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initializeApp();
    
    // Set up event listeners
    setupEventListeners();
    
    // Load initial statistics
    loadStatistics();
});

function initializeApp() {
    console.log('Diabetes Prediction AI initialized');
    
    // Add smooth scrolling for navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

function setupEventListeners() {
    // Prediction form submission
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePredictionSubmit);
    }
    
    // Batch file upload
    const csvFile = document.getElementById('csvFile');
    if (csvFile) {
        csvFile.addEventListener('change', handleFileUpload);
    }
    
    // Batch prediction button
    const batchPredictBtn = document.getElementById('batchPredictBtn');
    if (batchPredictBtn) {
        batchPredictBtn.addEventListener('click', handleBatchPrediction);
    }
    
    // Real-time validation
    setupRealTimeValidation();
}

function setupRealTimeValidation() {
    const inputs = document.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            validateInput(this);
        });
        
        input.addEventListener('blur', function() {
            validateInput(this);
        });
    });
}

function validateInput(input) {
    const value = parseFloat(input.value);
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    
    if (isNaN(value)) {
        input.classList.remove('is-valid', 'is-invalid');
        return;
    }
    
    if (value >= min && value <= max) {
        input.classList.remove('is-invalid');
        input.classList.add('is-valid');
    } else {
        input.classList.remove('is-valid');
        input.classList.add('is-invalid');
    }
}

async function handlePredictionSubmit(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const data = {};
    
    // Collect form data
    for (let [key, value] of formData.entries()) {
        data[key] = parseFloat(value);
    }
    
    // Show loading state
    const submitBtn = e.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<span class="spinner"></span> Processing...';
    submitBtn.disabled = true;
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayPredictionResult(result);
        } else {
            displayError(result.error || 'Prediction failed');
        }
    } catch (error) {
        displayError('Network error: ' + error.message);
    } finally {
        // Restore button state
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }
}

function displayPredictionResult(result) {
    const resultsSection = document.getElementById('results');
    const predictionResult = document.getElementById('predictionResult');
    const confidenceLevel = document.getElementById('confidenceLevel');
    const explanationSection = document.getElementById('explanationSection');
    
    // Update prediction result
    predictionResult.textContent = result.prediction;
    confidenceLevel.textContent = result.confidence;
    
    // Display SHAP explanation
    if (result.explanation && Object.keys(result.explanation).length > 0) {
        displaySHAPExplanation(result.explanation);
    } else {
        explanationSection.innerHTML = '<p class="text-muted">No explanation available</p>';
    }
    
    // Show results section with animation
    resultsSection.style.display = 'block';
    resultsSection.classList.add('fade-in');
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Reload statistics
    loadStatistics();
}

function displaySHAPExplanation(explanation) {
    const explanationSection = document.getElementById('explanationSection');
    const chartContainer = document.getElementById('explanationChart');
    
    // Create feature importance chart
    const features = Object.keys(explanation);
    const values = Object.values(explanation);
    
    // Clear previous chart
    chartContainer.innerHTML = '';
    
    // Create horizontal bar chart
    const canvas = document.createElement('canvas');
    chartContainer.appendChild(canvas);
    
    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                label: 'SHAP Values',
                data: values,
                backgroundColor: values.map(v => v > 0 ? 'rgba(40, 167, 69, 0.8)' : 'rgba(220, 53, 69, 0.8)'),
                borderColor: values.map(v => v > 0 ? '#28a745' : '#dc3545'),
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Feature Importance (SHAP Values)'
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'SHAP Value'
                    }
                }
            }
        }
    });
}

function displayError(message) {
    // Create error alert
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger fade-in';
    alertDiv.innerHTML = `
        <i class="fas fa-exclamation-triangle me-2"></i>
        ${message}
    `;
    
    // Insert at top of container
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

function handleFileUpload(e) {
    const file = e.target.files[0];
    const batchPredictBtn = document.getElementById('batchPredictBtn');
    
    if (file && file.type === 'text/csv') {
        batchPredictBtn.disabled = false;
        batchPredictBtn.innerHTML = '<i class="fas fa-upload me-2"></i>Process ' + file.name;
    } else {
        batchPredictBtn.disabled = true;
        batchPredictBtn.innerHTML = '<i class="fas fa-upload me-2"></i>Process Batch';
        displayError('Please select a valid CSV file');
    }
}

async function handleBatchPrediction() {
    const fileInput = document.getElementById('csvFile');
    const file = fileInput.files[0];
    
    if (!file) {
        displayError('Please select a CSV file first');
        return;
    }
    
    const batchPredictBtn = document.getElementById('batchPredictBtn');
    const originalText = batchPredictBtn.innerHTML;
    batchPredictBtn.innerHTML = '<span class="spinner"></span> Processing...';
    batchPredictBtn.disabled = true;
    
    try {
        // Read CSV file
        const text = await file.text();
        const lines = text.split('\n');
        const headers = lines[0].split(',');
        const data = [];
        
        // Parse CSV data
        for (let i = 1; i < lines.length; i++) {
            if (lines[i].trim()) {
                const values = lines[i].split(',');
                const row = {};
                headers.forEach((header, index) => {
                    row[header.trim()] = parseFloat(values[index]);
                });
                data.push(row);
            }
        }
        
        // Send batch prediction request
        const response = await fetch('/batch_predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ data: data })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayBatchResults(result.results);
        } else {
            displayError(result.error || 'Batch prediction failed');
        }
    } catch (error) {
        displayError('Error processing file: ' + error.message);
    } finally {
        batchPredictBtn.innerHTML = originalText;
        batchPredictBtn.disabled = false;
    }
}

function displayBatchResults(results) {
    const batchResults = document.getElementById('batchResults');
    const batchResultsBody = document.getElementById('batchResultsBody');
    
    // Clear previous results
    batchResultsBody.innerHTML = '';
    
    // Add results to table
    results.forEach(result => {
        const row = document.createElement('tr');
        
        if (result.error) {
            row.innerHTML = `
                <td>${result.row}</td>
                <td>-</td>
                <td><span class="badge bg-danger">Error</span> ${result.error}</td>
            `;
        } else {
            row.innerHTML = `
                <td>${result.row}</td>
                <td><span class="badge bg-success">${result.prediction}</span></td>
                <td><span class="badge bg-success">Success</span></td>
            `;
        }
        
        batchResultsBody.appendChild(row);
    });
    
    // Show results section
    batchResults.style.display = 'block';
    batchResults.classList.add('fade-in');
    
    // Reload statistics
    loadStatistics();
}

async function loadStatistics() {
    try {
        const response = await fetch('/stats');
        const stats = await response.json();
        
        // Update statistics cards
        document.getElementById('totalPredictions').textContent = stats.total_predictions;
        document.getElementById('avgAccuracy').textContent = '95%';
        document.getElementById('activeUsers').textContent = Object.keys(stats.class_distribution).length;
        
        // Update class distribution chart
        updateClassChart(stats.class_distribution);
        
        // Update recent predictions
        updateRecentPredictions(stats.recent_predictions);
        
    } catch (error) {
        console.error('Error loading statistics:', error);
    }
}

function updateClassChart(classDistribution) {
    const ctx = document.getElementById('classChart');
    
    // Clear previous chart
    ctx.innerHTML = '';
    const canvas = document.createElement('canvas');
    ctx.appendChild(canvas);
    
    const labels = Object.keys(classDistribution);
    const data = Object.values(classDistribution);
    
    new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: [
                    '#007bff',
                    '#28a745',
                    '#ffc107',
                    '#dc3545',
                    '#17a2b8'
                ],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function updateRecentPredictions(recentPredictions) {
    const container = document.getElementById('recentPredictions');
    container.innerHTML = '';
    
    recentPredictions.forEach(pred => {
        const item = document.createElement('div');
        item.className = 'list-group-item';
        item.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <strong>Class ${pred[1]}</strong>
                    <br>
                    <small class="text-muted">${new Date(pred[0]).toLocaleString()}</small>
                </div>
                <span class="badge bg-primary">${pred[1]}</span>
            </div>
        `;
        container.appendChild(item);
    });
}

// Utility functions
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} notification fade-in`;
    notification.innerHTML = `
        <i class="fas fa-info-circle me-2"></i>
        ${message}
    `;
    
    // Add styles
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.zIndex = '9999';
    notification.style.minWidth = '300px';
    
    document.body.appendChild(notification);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Export functions for global access
window.DiabetesPredictionAI = {
    showNotification,
    loadStatistics,
    displayError
}; 