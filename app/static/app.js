// App state
let state = {
    currentStep: 1,
    selectedFile: null,
    problemType: 'classification',
    targetColumn: '',
    columns: [],
    analysis: null,
    recommendations: [],
    selectedAlgorithm: '',
    features: [],
    historyChart: null,
    metrics: {}
};

// DOM Elements
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const fileUploadSuccess = document.getElementById('file-upload-success');
const uploadedFilename = document.getElementById('uploaded-filename');
const btnGotoStep2 = document.getElementById('btn-goto-step2');
const btnGotoStep3 = document.getElementById('btn-goto-step3');
const targetSelect = document.getElementById('target-select');
const targetSelectionGroup = document.getElementById('target-selection-group');
const btnRunAnalysis = document.getElementById('btn-run-analysis');
const analysisDashboardSection = document.getElementById('analysis-dashboard-section');
const previewTableWrapper = document.getElementById('preview-table-wrapper');
const recommendationsContainer = document.getElementById('recommendations-container');
const featureTypeBreakdown = document.getElementById('feature-type-breakdown');
const classBalanceSection = document.getElementById('class-balance-section');
const classBalanceTable = document.getElementById('class-balance-table');

// Step 3
const featureCheckboxesContainer = document.getElementById('feature-checkboxes-container');
const btnSelectAllFeatures = document.getElementById('btn-select-all-features');
const btnClearFeatures = document.getElementById('btn-clear-features');
const algoSelect = document.getElementById('algo-select');
const hyperparamsContainer = document.getElementById('hyperparams-container');
const splitRatioSlider = document.getElementById('split-ratio-slider');
const splitRatioVal = document.getElementById('split-ratio-val');
const btnTrainModel = document.getElementById('btn-train-model');

// Step 4
const trainingLoadingCard = document.getElementById('training-loading-card');
const trainingProgressFill = document.getElementById('training-progress-fill');
const trainingResultsSection = document.getElementById('training-results-section');
const metricsCardsContainer = document.getElementById('metrics-cards-container');
const confusionMatrixContainer = document.getElementById('confusion-matrix-container');
const cmRenderingTarget = document.getElementById('cm-rendering-target');
const clusterDistContainer = document.getElementById('cluster-dist-container');
const clusterDistTable = document.getElementById('cluster-dist-table');
const predictionFormInputs = document.getElementById('prediction-form-inputs');
const predictionForm = document.getElementById('prediction-form');
const predictionResultPanel = document.getElementById('prediction-result-panel');
const predictedValue = document.getElementById('predicted-value');
const btnRestartApp = document.getElementById('btn-restart-app');

// Status update helper
function updateStatus(text, active = false) {
    document.getElementById('status-text').innerText = text;
    const dot = document.getElementById('status-dot');
    if (active) {
        dot.className = 'status-dot active';
    } else {
        dot.className = 'status-dot';
    }
}

// Navigation
function gotoStep(stepNum) {
    // Deactivate current steps
    document.querySelectorAll('.step-section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.step-indicator').forEach(ind => {
        const step = parseInt(ind.dataset.step);
        if (step === stepNum) {
            ind.className = 'step-indicator active';
        } else if (step < stepNum) {
            ind.className = 'step-indicator completed';
        } else {
            ind.className = 'step-indicator';
        }
    });

    document.getElementById(`step-${stepNum}`).classList.add('active');
    state.currentStep = stepNum;
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Initialize event listeners
document.addEventListener('DOMContentLoaded', () => {
    setupUploadZone();
    setupProblemTypeSelector();
    setupStepNavigators();
    setupAnalysisTriggers();
    setupPreprocessingListeners();
    setupTrainingTrigger();
    setupPredictionTrigger();
    setupBatchPredictionHandlers();
    setupThemeHandler();

    btnRestartApp.addEventListener('click', () => {
        window.location.reload();
    });
});

// Setup Upload Zone drag and drop
function setupUploadZone() {
    uploadZone.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
        }, false);
    });

    uploadZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
}

// Upload CSV API call
async function handleFileUpload(file) {
    state.selectedFile = file;
    updateStatus('Uploading CSV...', true);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Upload failed');
        }
        
        const data = await response.json();
        state.columns = data.columns;
        
        // Show success UI
        uploadedFilename.innerText = data.filename;
        fileUploadSuccess.style.display = 'block';
        
        // Enable Next Button
        btnGotoStep2.className = 'btn';
        btnGotoStep2.disabled = false;
        
        // Render preview table
        renderPreviewTable(data.preview, data.columns);
        
        // Fill target select
        populateTargetSelect(data.columns);
        
        updateStatus('Dataset loaded successfully.');
    } catch (err) {
        updateStatus('Error: ' + err.message);
        alert('File upload failed: ' + err.message);
    }
}

// Preview table generator
function renderPreviewTable(rows, columns) {
    if (!rows || rows.length === 0) return;
    
    let html = '<table><thead><tr>';
    columns.forEach(col => {
        html += `<th>${col.name}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    rows.forEach(row => {
        html += '<tr>';
        columns.forEach(col => {
            html += `<td>${row[col.name] !== undefined ? row[col.name] : ''}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    previewTableWrapper.innerHTML = html;
}

// Target select dropdown populator
function populateTargetSelect(columns) {
    // Clear
    targetSelect.innerHTML = '<option value="">-- Choose Target --</option>';
    columns.forEach(col => {
        targetSelect.innerHTML += `<option value="${col.name}">${col.name} (${col.type})</option>`;
    });
}

// Problem type selector styling updates
function setupProblemTypeSelector() {
    document.querySelectorAll('.radio-card').forEach(card => {
        card.addEventListener('click', () => {
            document.querySelectorAll('.radio-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            const radio = card.querySelector('input[type="radio"]');
            radio.checked = true;
            
            state.problemType = radio.value;
            
            // If clustering is selected, hide the target column selector (as it is unsupervised)
            if (state.problemType === 'clustering') {
                targetSelectionGroup.style.display = 'none';
                state.targetColumn = '';
            } else {
                targetSelectionGroup.style.display = 'block';
            }
        });
    });
}

// Step navigation buttons setup
function setupStepNavigators() {
    btnGotoStep2.addEventListener('click', () => {
        if (state.selectedFile) {
            gotoStep(2);
            updateStatus('Analyzing dataset properties...');
        }
    });

    btnGotoStep3.addEventListener('click', () => {
        gotoStep(3);
        setupStep3Config();
        updateStatus('Configure preprocessing and training.');
    });

    document.querySelectorAll('.btn-back-step').forEach(btn => {
        btn.addEventListener('click', () => {
            const stepNum = parseInt(btn.dataset.target);
            gotoStep(stepNum);
        });
    });
}

// Analysis and recommendation triggers
function setupAnalysisTriggers() {
    btnRunAnalysis.addEventListener('click', async () => {
        if (state.problemType !== 'clustering' && !targetSelect.value) {
            alert('Please select a target column (Y) for supervised learning.');
            return;
        }
        
        state.targetColumn = targetSelect.value;
        updateStatus('Computing recommendation heuristics...', true);
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    problem_type: state.problemType,
                    target_column: state.targetColumn || null
                })
            });
            
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Analysis failed');
            }
            
            const data = await response.json();
            state.analysis = data.analysis;
            state.recommendations = data.recommendations;
            
            // Render report & recommendations
            renderAnalysisReport(data.analysis);
            renderRecommendations(data.recommendations);
            
            // Enable next button
            btnGotoStep3.className = 'btn';
            btnGotoStep3.disabled = false;
            
            // Show dashboard
            analysisDashboardSection.style.display = 'block';
            updateStatus('AI Recommendations loaded. Select your algorithm.');
            
            // Auto scroll down to see recommendations
            setTimeout(() => {
                analysisDashboardSection.scrollIntoView({ behavior: 'smooth' });
            }, 100);
            
        } catch (err) {
            updateStatus('Analysis Error: ' + err.message);
            alert('Analysis failed: ' + err.message);
        }
    });
}

// Renders dataset statistics
function renderAnalysisReport(analysis) {
    document.getElementById('stat-rows').innerText = analysis.n_rows.toLocaleString();
    document.getElementById('stat-cols').innerText = analysis.n_cols.toLocaleString();
    document.getElementById('stat-missing').innerText = analysis.missing_values.total_missing.toLocaleString();
    document.getElementById('stat-duplicates').innerText = analysis.duplicates.duplicate_count.toLocaleString();
    
    // Feature types
    featureTypeBreakdown.innerHTML = '';
    const feats = analysis.features;
    
    const types = [
        { label: 'Numeric', count: feats.numeric.length, class: 'type-numeric' },
        { label: 'Categorical', count: feats.categorical.length, class: 'type-categorical' },
        { label: 'Boolean', count: feats.boolean.length, class: 'type-boolean' },
        { label: 'Text', count: feats.text.length, class: 'type-text' }
    ];
    
    types.forEach(t => {
        if (t.count > 0) {
            featureTypeBreakdown.innerHTML += `
                <div class="feature-tag-card">
                    <span class="feature-tag-name">${t.label} Features</span>
                    <span class="feature-tag-type ${t.class}">${t.count}</span>
                </div>
            `;
        }
    });
    
    // Class balance (classification)
    if (state.problemType === 'classification' && analysis.class_balance.distribution) {
        classBalanceSection.style.display = 'block';
        const tbody = classBalanceTable.querySelector('tbody');
        tbody.innerHTML = '';
        
        const dist = analysis.class_balance.distribution;
        const total = Object.values(dist).reduce((a, b) => a + b, 0);
        
        for (const [cls, count] of Object.entries(dist)) {
            const pct = ((count / total) * 100).toFixed(1);
            tbody.innerHTML += `
                <tr>
                    <td><strong>${cls}</strong></td>
                    <td>${count.toLocaleString()}</td>
                    <td>${pct}%</td>
                </tr>
            `;
        }
    } else {
        classBalanceSection.style.display = 'none';
    }
}

// Renders the top-3 algorithms list
function renderRecommendations(recs) {
    recommendationsContainer.innerHTML = '';
    
    // Auto select the top rank
    if (recs.length > 0) {
        state.selectedAlgorithm = recs[0].algorithm;
    }
    
    recs.forEach((rec, index) => {
        const rankClass = `rank-${rec.rank}`;
        const isSelected = state.selectedAlgorithm === rec.algorithm ? 'selected' : '';
        const reasonsHtml = rec.reasons.map(r => `<li>${r}</li>`).join('');
        
        recommendationsContainer.innerHTML += `
            <div class="rec-card ${rankClass} ${isSelected}" data-algo="${rec.algorithm}">
                <div class="rec-header">
                    <div class="rec-title-wrap">
                        <div class="rank-badge">${rec.rank}</div>
                        <div class="rec-algorithm">${rec.algorithm}</div>
                    </div>
                    <div class="rec-confidence">${rec.confidence}% confidence</div>
                </div>
                <div class="confidence-progress-bar">
                    <div class="confidence-fill" style="width: ${rec.confidence}%"></div>
                </div>
                <ul class="rec-reasons">
                    ${reasonsHtml}
                </ul>
            </div>
        `;
    });
    
    // Setup selection behavior on cards
    document.querySelectorAll('.rec-card').forEach(card => {
        card.addEventListener('click', () => {
            document.querySelectorAll('.rec-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            state.selectedAlgorithm = card.dataset.algo;
            updateStatus(`Selected: ${state.selectedAlgorithm}`);
        });
    });
    
    // Trigger icons reload
    lucide.createIcons();
}

// Set up preprocessing UI options and feature check list
function setupStep3Config() {
    // Populate target feature checkboxes
    featureCheckboxesContainer.innerHTML = '';
    
    // Sort columns so numeric are first or original order
    state.columns.forEach(col => {
        // Skip target column in feature list
        if (col.name === state.targetColumn) return;
        
        // Find matching type class
        let typeClass = 'type-numeric';
        let friendlyType = 'numeric';
        if (state.analysis.features.categorical.includes(col.name)) {
            typeClass = 'type-categorical';
            friendlyType = 'categorical';
        } else if (state.analysis.features.boolean.includes(col.name)) {
            typeClass = 'type-boolean';
            friendlyType = 'boolean';
        } else if (state.analysis.features.text.includes(col.name)) {
            typeClass = 'type-text';
            friendlyType = 'text';
        }
        
        // Default checked unless it is a text column or target
        const isChecked = friendlyType !== 'text' ? 'checked' : '';
        
        featureCheckboxesContainer.innerHTML += `
            <label class="checkbox-item">
                <input type="checkbox" name="selected_features" value="${col.name}" ${isChecked}>
                <span>${col.name}</span>
                <span class="feature-tag-type ${typeClass}" style="margin-left:auto; font-size:0.7rem; padding: 1px 6px;">${friendlyType}</span>
            </label>
        `;
    });

    // Populate algorithm dropdown
    algoSelect.innerHTML = '';
    let algorithms = [];
    if (state.problemType === 'regression') {
        algorithms = ['Multiple Linear Regression', 'Polynomial Regression', 'Linear Regression (Simple)'];
    } else if (state.problemType === 'classification') {
        algorithms = ['Logistic Regression', 'K-Nearest Neighbors (KNN)', 'Naive Bayes'];
    } else { // clustering
        algorithms = ['K-Means (K-means++)', 'Hierarchical Clustering', 'K-Means (Random Init)'];
    }
    
    algorithms.forEach(algo => {
        const isSelected = state.selectedAlgorithm === algo ? 'selected' : '';
        algoSelect.innerHTML += `<option value="${algo}" ${isSelected}>${algo}</option>`;
    });
    
    // Load hyperparameters
    renderHyperparameters(algoSelect.value);
}

// Preprocessing listener setup
function setupPreprocessingListeners() {
    splitRatioSlider.addEventListener('input', (e) => {
        const testPct = e.target.value;
        const trainPct = 100 - testPct;
        splitRatioVal.innerText = `${trainPct}% / ${testPct}%`;
    });

    btnSelectAllFeatures.addEventListener('click', () => {
        document.querySelectorAll('input[name="selected_features"]').forEach(cb => {
            cb.checked = true;
        });
    });

    btnClearFeatures.addEventListener('click', () => {
        document.querySelectorAll('input[name="selected_features"]').forEach(cb => {
            cb.checked = false;
        });
    });

    algoSelect.addEventListener('change', (e) => {
        state.selectedAlgorithm = e.target.value;
        renderHyperparameters(state.selectedAlgorithm);
    });
}

// Hyperparameters dynamic form rendering
function renderHyperparameters(algorithm) {
    hyperparamsContainer.innerHTML = '';
    
    if (algorithm === 'K-Nearest Neighbors (KNN)') {
        hyperparamsContainer.innerHTML = `
            <div class="form-group">
                <label for="param-k">Number of Neighbors (K)</label>
                <input type="number" id="param-k" value="5" min="1" max="50">
            </div>
            <div class="form-group">
                <label for="param-metric">Distance Metric</label>
                <select id="param-metric">
                    <option value="euclidean">Euclidean</option>
                    <option value="manhattan">Manhattan</option>
                </select>
            </div>
        `;
    } else if (algorithm === 'Logistic Regression') {
        hyperparamsContainer.innerHTML = `
            <div class="form-group">
                <label for="param-lr">Learning Rate</label>
                <input type="number" id="param-lr" value="0.01" step="0.001" min="0.0001" max="1">
            </div>
            <div class="form-group">
                <label for="param-epochs">Training Epochs</label>
                <input type="number" id="param-epochs" value="1000" step="50" min="10">
            </div>
        `;
    } else if (algorithm === 'Naive Bayes') {
        hyperparamsContainer.innerHTML = `
            <p style="grid-column: span 2; color: var(--text-muted); font-size: 0.88rem; text-align: center; padding: 20px 0;">
                No hyperparameters needed for Naive Bayes (calculates probabilities from class means & variances).
            </p>
        `;
    } else if (algorithm === 'Linear Regression (Simple)' || algorithm === 'Multiple Linear Regression') {
        hyperparamsContainer.innerHTML = `
            <div class="form-group">
                <label for="param-solver">Solver Method</label>
                <select id="param-solver">
                    <option value="gradient_descent">Gradient Descent</option>
                    <option value="ols">Closed Form OLS</option>
                </select>
            </div>
            <div class="form-group" id="gd-lr-group">
                <label for="param-lr">Learning Rate</label>
                <input type="number" id="param-lr" value="0.01" step="0.001" min="0.0001" max="1">
            </div>
            <div class="form-group" id="gd-epochs-group">
                <label for="param-epochs">Epochs</label>
                <input type="number" id="param-epochs" value="1000" step="50" min="10">
            </div>
        `;
        
        // Hide/show Learning rate and Epochs if solver is OLS
        const solver = document.getElementById('param-solver');
        const updateSolverUI = () => {
            const isOLS = solver.value === 'ols';
            document.getElementById('gd-lr-group').style.display = isOLS ? 'none' : 'block';
            document.getElementById('gd-epochs-group').style.display = isOLS ? 'none' : 'block';
        };
        solver.addEventListener('change', updateSolverUI);
        updateSolverUI();
        
    } else if (algorithm === 'Polynomial Regression') {
        hyperparamsContainer.innerHTML = `
            <div class="form-group">
                <label for="param-degree">Polynomial Degree</label>
                <input type="number" id="param-degree" value="2" min="1" max="5">
            </div>
            <div class="form-group">
                <label for="param-solver">Solver Method</label>
                <select id="param-solver">
                    <option value="ols">Closed Form OLS</option>
                    <option value="gradient_descent">Gradient Descent</option>
                </select>
            </div>
            <div class="form-group" id="gd-lr-group" style="display:none;">
                <label for="param-lr">Learning Rate</label>
                <input type="number" id="param-lr" value="0.01" step="0.001" min="0.0001" max="1">
            </div>
            <div class="form-group" id="gd-epochs-group" style="display:none;">
                <label for="param-epochs">Epochs</label>
                <input type="number" id="param-epochs" value="1000" step="50" min="10">
            </div>
        `;
        const solver = document.getElementById('param-solver');
        const updateSolverUI = () => {
            const isOLS = solver.value === 'ols';
            document.getElementById('gd-lr-group').style.display = isOLS ? 'none' : 'block';
            document.getElementById('gd-epochs-group').style.display = isOLS ? 'none' : 'block';
        };
        solver.addEventListener('change', updateSolverUI);
        updateSolverUI();
        
    } else if (algorithm === 'K-Means (K-means++)' || algorithm === 'K-Means (Random Init)') {
        hyperparamsContainer.innerHTML = `
            <div class="form-group">
                <label for="param-k">Number of Clusters (K)</label>
                <input type="number" id="param-k" value="3" min="2" max="15">
            </div>
            <div class="form-group">
                <label for="param-iter">Max Iterations</label>
                <input type="number" id="param-iter" value="300" step="10">
            </div>
        `;
    } else if (algorithm === 'Hierarchical Clustering') {
        hyperparamsContainer.innerHTML = `
            <div class="form-group">
                <label for="param-k">Number of Clusters (K)</label>
                <input type="number" id="param-k" value="3" min="2" max="15">
            </div>
            <div class="form-group">
                <label for="param-linkage">Linkage Type</label>
                <select id="param-linkage">
                    <option value="average">Average Linkage</option>
                    <option value="single">Single Linkage</option>
                    <option value="complete">Complete Linkage</option>
                </select>
            </div>
        `;
    }
}

// Training execution logic
function setupTrainingTrigger() {
    btnTrainModel.addEventListener('click', async () => {
        // Collect checked features
        const features = [];
        document.querySelectorAll('input[name="selected_features"]:checked').forEach(cb => {
            features.push(cb.value);
        });
        
        if (features.length === 0) {
            alert('Please select at least one feature column (X) to train the model.');
            return;
        }
        
        // Save features in state
        state.features = features;
        
        // Collect hyperparameters
        const hyperparams = {};
        const kEl = document.getElementById('param-k');
        const metricEl = document.getElementById('param-metric');
        const lrEl = document.getElementById('param-lr');
        const epochsEl = document.getElementById('param-epochs');
        const solverEl = document.getElementById('param-solver');
        const degreeEl = document.getElementById('param-degree');
        const iterEl = document.getElementById('param-iter');
        const linkageEl = document.getElementById('param-linkage');
        
        if (kEl) hyperparams['k'] = parseInt(kEl.value);
        if (metricEl) hyperparams['metric'] = metricEl.value;
        if (lrEl) hyperparams['learning_rate'] = parseFloat(lrEl.value);
        if (epochsEl) hyperparams['epochs'] = parseInt(epochsEl.value);
        if (solverEl) hyperparams['solver'] = solverEl.value;
        if (degreeEl) hyperparams['degree'] = parseInt(degreeEl.value);
        if (iterEl) hyperparams['max_iter'] = parseInt(iterEl.value);
        if (linkageEl) hyperparams['linkage'] = linkageEl.value;

        const requestBody = {
            algorithm: algoSelect.value,
            problem_type: state.problemType,
            features: features,
            target_column: state.targetColumn || null,
            preprocessing: {
                impute_numeric: document.getElementById('impute-numeric-select').value,
                impute_categorical: document.getElementById('impute-categorical-select').value,
                scale_type: document.getElementById('scaler-select').value,
                test_size: parseFloat(splitRatioSlider.value) / 100.0,
                random_state: 42
            },
            hyperparams: hyperparams
        };

        // Go to step 4, show progress bar loading
        gotoStep(4);
        trainingLoadingCard.style.display = 'block';
        trainingResultsSection.style.display = 'none';
        updateStatus('Training algorithm from scratch...', true);
        
        // Start progress bar animation (0 to 90% in 1.5s)
        let prog = 0;
        trainingProgressFill.style.width = '0%';
        const progressInterval = setInterval(() => {
            if (prog < 90) {
                prog += Math.random() * 8;
                if (prog > 90) prog = 90;
                trainingProgressFill.style.width = `${prog}%`;
            }
        }, 100);

        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });
            
            clearInterval(progressInterval);
            
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Training failed');
            }
            
            const data = await response.json();
            
            // Finish progress bar
            trainingProgressFill.style.width = '100%';
            
            setTimeout(() => {
                trainingLoadingCard.style.display = 'none';
                trainingResultsSection.style.display = 'block';
                
                // Show evaluation metrics dashboard
                renderEvaluationDashboard(data.results);
                
                // Show loss curves Chart
                renderTrainingHistoryChart(data.algorithm, data.history);
                
                // Build Prediction Playground Inputs
                generatePredictionPlayground();
                
                updateStatus('Model training complete.');
            }, 300);
            
        } catch (err) {
            clearInterval(progressInterval);
            trainingLoadingCard.style.display = 'none';
            updateStatus('Training Failed.');
            alert('Training error: ' + err.message);
            gotoStep(3); // go back
        }
    });
}

// Render Model Metrics
function renderEvaluationDashboard(results) {
    metricsCardsContainer.innerHTML = '';
    confusionMatrixContainer.style.display = 'none';
    clusterDistContainer.style.display = 'none';
    
    if (state.problemType === 'regression') {
        const train = results.train;
        const test = results.test;
        
        metricsCardsContainer.innerHTML = `
            <div class="metric-card highlight">
                <div class="metric-title">Test R² Score</div>
                <div class="metric-val">${test.r2.toFixed(4)}</div>
                <div class="metric-sub">Train R²: ${train.r2.toFixed(4)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Test RMSE</div>
                <div class="metric-val">${test.rmse.toFixed(4)}</div>
                <div class="metric-sub">Train RMSE: ${train.rmse.toFixed(4)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Test MAE</div>
                <div class="metric-val">${test.mae.toFixed(4)}</div>
                <div class="metric-sub">Train MAE: ${train.mae.toFixed(4)}</div>
            </div>
        `;
    } else if (state.problemType === 'classification') {
        const train = results.train;
        const test = results.test;
        
        metricsCardsContainer.innerHTML = `
            <div class="metric-card highlight">
                <div class="metric-title">Test Accuracy</div>
                <div class="metric-val">${(test.accuracy * 100).toFixed(1)}%</div>
                <div class="metric-sub">Train Accuracy: ${(train.accuracy * 100).toFixed(1)}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Test F1-Score</div>
                <div class="metric-val">${test.f1_score.toFixed(4)}</div>
                <div class="metric-sub">Macro-averaged F1</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Precision / Recall</div>
                <div class="metric-val">${test.precision.toFixed(2)}</div>
                <div class="metric-sub">Recall: ${test.recall.toFixed(2)}</div>
            </div>
        `;
        
        // Render confusion matrix
        if (results.confusion_matrix) {
            confusionMatrixContainer.style.display = 'block';
            renderConfusionMatrix(results.confusion_matrix);
        }
    } else { // clustering
        const train = results.train;
        
        metricsCardsContainer.innerHTML = `
            <div class="metric-card highlight">
                <div class="metric-title">Final Inertia (WCSS)</div>
                <div class="metric-val">${train.inertia.toLocaleString(undefined, {maximumFractionDigits:2})}</div>
                <div class="metric-sub">Lower values indicate tighter clusters</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Number of Clusters</div>
                <div class="metric-val">${Object.keys(train.distribution).length}</div>
                <div class="metric-sub">Partition groups created</div>
            </div>
        `;
        
        // Render cluster distribution table
        clusterDistContainer.style.display = 'block';
        const tbody = clusterDistTable.querySelector('tbody');
        tbody.innerHTML = '';
        
        const dist = train.distribution;
        const total = Object.values(dist).reduce((a, b) => a + b, 0);
        
        for (const [clusterId, count] of Object.entries(dist)) {
            const ratio = ((count / total) * 100).toFixed(1);
            tbody.innerHTML += `
                <tr>
                    <td><strong>Cluster ${clusterId}</strong></td>
                    <td>${count.toLocaleString()}</td>
                    <td>${ratio}%</td>
                </tr>
            `;
        }
    }
}

// Confusion Matrix generator
function renderConfusionMatrix(cm) {
    const matrix = cm.matrix;
    const labels = cm.labels;
    const n = labels.length;
    
    // Find sum of each row (true class count) to calculate cell densities for color scale
    const rowSums = matrix.map(row => row.reduce((a, b) => a + b, 0));
    
    let html = `<table class="cm-table">`;
    
    // Table Header (Predicted class title)
    html += `<tr><td colspan="${n + 1}" class="cm-header-title">Predicted Class</td></tr>`;
    
    // Columns headers
    html += `<tr><td></td>`; // empty corner cell
    labels.forEach(l => {
        html += `<td class="cm-label-x"><strong>${l}</strong></td>`;
    });
    html += `</tr>`;
    
    // Matrix rows
    for (let r = 0; r < n; r++) {
        // True class column header
        html += `<tr><td class="cm-label-y"><strong>${labels[r]}</strong></td>`;
        for (let c = 0; c < n; c++) {
            const val = matrix[r][c];
            const sum = rowSums[r];
            const density = sum > 0 ? (val / sum) : 0;
            
            // Generate glowing background style based on cell density (using purple primary color)
            const alpha = 0.05 + density * 0.75;
            const bgStyle = `background: rgba(139, 92, 246, ${alpha});`;
            const textColorStyle = density > 0.5 ? 'color: #ffffff;' : 'color: var(--text-secondary);';
            
            html += `<td class="cm-cell" style="${bgStyle} ${textColorStyle}" title="True: ${labels[r]}, Predicted: ${labels[c]}">${val}</td>`;
        }
        html += `</tr>`;
    }
    html += `</table>`;
    
    cmRenderingTarget.innerHTML = html;
}

// Training History Chart using Chart.js
function renderTrainingHistoryChart(algoName, history) {
    const canvas = document.getElementById('history-chart');
    const ctx = canvas.getContext('2d');
    
    // Cache parameters for theme changes
    state.lastChartAlgo = algoName;
    state.lastChartData = history;
    
    // Destroy previous instance
    if (state.historyChart) {
        state.historyChart.destroy();
    }
    
    // Label depending on metric
    let yLabel = 'Loss';
    if (state.problemType === 'clustering') {
        yLabel = algoName === 'Hierarchical Clustering' ? 'Merge Linkage Distance' : 'Inertia';
    }
    
    document.getElementById('chart-description').innerText = `Tracks model ${yLabel.toLowerCase()} optimizations over iterations.`;
    
    const labels = history.map((_, idx) => idx + 1);
    
    // Visual settings
    let borderColor = '#8b5cf6'; // purple
    let fillGradientStart = 'rgba(139, 92, 246, 0.15)';
    
    if (state.problemType === 'regression') {
        borderColor = '#3b82f6'; // blue
        fillGradientStart = 'rgba(59, 130, 246, 0.15)';
    } else if (state.problemType === 'clustering') {
        borderColor = '#10b981'; // emerald
        fillGradientStart = 'rgba(16, 185, 129, 0.15)';
    }

    // Theme-dependent colors
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const gridColor = isLight ? 'rgba(15, 23, 42, 0.05)' : 'rgba(255, 255, 255, 0.03)';
    const tickColor = isLight ? '#475569' : '#6b7280';
    const titleColor = isLight ? '#0f172a' : '#9ca3af';

    state.historyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: yLabel,
                data: history,
                borderColor: borderColor,
                backgroundColor: fillGradientStart,
                borderWidth: 2,
                pointRadius: history.length > 50 ? 0 : 3,
                pointHoverRadius: 5,
                fill: true,
                tension: 0.15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: gridColor },
                    ticks: { color: tickColor, maxTicksLimit: 10 },
                    title: { display: true, text: 'Iterations / Epochs', color: titleColor }
                },
                y: {
                    grid: { color: gridColor },
                    ticks: { color: tickColor },
                    title: { display: true, text: yLabel, color: titleColor }
                }
            }
        }
    });
}

// Generate Playground input fields
function generatePredictionPlayground() {
    predictionFormInputs.innerHTML = '';
    predictionResultPanel.style.display = 'none';
    
    state.features.forEach(featName => {
        // Find feature metadata
        const colMeta = state.columns.find(c => c.name === featName);
        if (!colMeta) return;
        
        const isNumeric = state.analysis.features.numeric.includes(featName) || state.analysis.features.boolean.includes(featName);
        
        let inputHtml = '';
        if (isNumeric) {
            // Read mean to set as placeholder / default
            const details = state.analysis.numeric_details[featName];
            const defaultValue = details ? details.mean.toFixed(2) : '0';
            
            inputHtml = `
                <div class="form-group">
                    <label for="pred-${featName}">${featName}</label>
                    <input type="number" id="pred-${featName}" step="any" value="${defaultValue}" required>
                </div>
            `;
        } else {
            // Categorical features: show select option list
            const uniqueVals = colMeta.unique_values || [];
            let options = uniqueVals.map(v => `<option value="${v}">${v}</option>`).join('');
            if (options === '') {
                options = '<option value="Missing">Missing</option>';
            }
            
            inputHtml = `
                <div class="form-group">
                    <label for="pred-${featName}">${featName}</label>
                    <select id="pred-${featName}" required>
                        ${options}
                    </select>
                </div>
            `;
        }
        
        predictionFormInputs.innerHTML += inputHtml;
    });
}

// Predict button listener
function setupPredictionTrigger() {
    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const inputs = {};
        state.features.forEach(featName => {
            const inputEl = document.getElementById(`pred-${featName}`);
            if (inputEl) {
                if (inputEl.type === 'number') {
                    inputs[featName] = parseFloat(inputEl.value);
                } else {
                    inputs[featName] = inputEl.value;
                }
            }
        });
        
        updateStatus('Running prediction inference...', true);
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ inputs: inputs })
            });
            
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Prediction failed');
            }
            
            const data = await response.json();
            
            // Format output label
            let resultText = data.prediction;
            if (typeof resultText === 'number' && !Number.isInteger(resultText)) {
                resultText = resultText.toFixed(6);
            }
            
            predictedValue.innerText = resultText;
            predictionResultPanel.style.display = 'block';
            
            // Scroll to results
            setTimeout(() => {
                predictionResultPanel.scrollIntoView({ behavior: 'smooth' });
            }, 100);
            
            updateStatus('Prediction calculated.');
            
        } catch (err) {
            updateStatus('Prediction Failed.');
            alert('Prediction error: ' + err.message);
        }
    });
}

function setupBatchPredictionHandlers() {
    const batchUploadZone = document.getElementById('batch-upload-zone');
    const batchFileInput = document.getElementById('batch-file-input');
    const batchFileSuccess = document.getElementById('batch-file-success');
    const batchFilename = document.getElementById('batch-filename');
    const btnRunBatch = document.getElementById('btn-run-batch');
    const batchPredictionResults = document.getElementById('batch-prediction-results');
    const batchPreviewTableWrapper = document.getElementById('batch-preview-table-wrapper');
    
    let batchFile = null;
    
    batchUploadZone.addEventListener('click', () => batchFileInput.click());
    
    batchFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            loadBatchFile(e.target.files[0]);
        }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        batchUploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            batchUploadZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        batchUploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            batchUploadZone.classList.remove('dragover');
        }, false);
    });

    batchUploadZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            loadBatchFile(files[0]);
        }
    });
    
    function loadBatchFile(file) {
        batchFile = file;
        batchFilename.innerText = file.name;
        batchFileSuccess.style.display = 'block';
        
        btnRunBatch.className = 'btn btn-success';
        btnRunBatch.disabled = false;
        updateStatus(`Test dataset "${file.name}" loaded.`);
    }
    
    btnRunBatch.addEventListener('click', async () => {
        if (!batchFile) return;
        
        updateStatus('Running batch predictions...', true);
        btnRunBatch.disabled = true;
        btnRunBatch.innerText = 'Calculating Inferences...';
        
        const formData = new FormData();
        formData.append('file', batchFile);
        
        try {
            const response = await fetch('/api/predict/batch', {
                method: 'POST',
                body: formData
            });
            
            btnRunBatch.disabled = false;
            btnRunBatch.innerText = 'Run Batch Prediction';
            btnRunBatch.className = 'btn btn-success';
            
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Batch prediction failed');
            }
            
            const data = await response.json();
            
            // Show preview table
            renderBatchPreviewTable(data.preview, data.columns);
            
            // Render batch evaluation metrics if available
            const metricsContainer = document.getElementById('batch-metrics-container');
            const metricsGrid = document.getElementById('batch-metrics-grid');
            
            if (data.metrics) {
                metricsContainer.style.display = 'block';
                metricsGrid.innerHTML = '';
                
                if (state.problemType === 'classification') {
                    metricsGrid.innerHTML = `
                        <div class="stat-card" style="padding: 10px; border-color: rgba(16, 185, 129, 0.2); background: rgba(0,0,0,0.2);">
                            <div class="stat-val" style="font-size: 1.2rem; color: var(--secondary);">${(data.metrics.accuracy * 100).toFixed(1)}%</div>
                            <div class="stat-lbl" style="font-size: 0.65rem; margin-top: 2px;">Accuracy</div>
                        </div>
                        <div class="stat-card" style="padding: 10px; background: rgba(0,0,0,0.2);">
                            <div class="stat-val" style="font-size: 1.2rem; color: white;">${data.metrics.f1_score.toFixed(4)}</div>
                            <div class="stat-lbl" style="font-size: 0.65rem; margin-top: 2px;">F1-Score</div>
                        </div>
                        <div class="stat-card" style="padding: 10px; background: rgba(0,0,0,0.2);">
                            <div class="stat-val" style="font-size: 1.2rem; color: white;">${data.metrics.precision.toFixed(2)}</div>
                            <div class="stat-lbl" style="font-size: 0.65rem; margin-top: 2px;">Precision</div>
                        </div>
                        <div class="stat-card" style="padding: 10px; background: rgba(0,0,0,0.2);">
                            <div class="stat-val" style="font-size: 1.2rem; color: white;">${data.metrics.recall.toFixed(2)}</div>
                            <div class="stat-lbl" style="font-size: 0.65rem; margin-top: 2px;">Recall</div>
                        </div>
                    `;
                } else if (state.problemType === 'regression') {
                    metricsGrid.innerHTML = `
                        <div class="stat-card" style="padding: 10px; border-color: rgba(16, 185, 129, 0.2); background: rgba(0,0,0,0.2);">
                            <div class="stat-val" style="font-size: 1.2rem; color: var(--secondary);">${data.metrics.r2.toFixed(4)}</div>
                            <div class="stat-lbl" style="font-size: 0.65rem; margin-top: 2px;">R² Score</div>
                        </div>
                        <div class="stat-card" style="padding: 10px; background: rgba(0,0,0,0.2);">
                            <div class="stat-val" style="font-size: 1.2rem; color: white;">${data.metrics.rmse.toFixed(4)}</div>
                            <div class="stat-lbl" style="font-size: 0.65rem; margin-top: 2px;">RMSE</div>
                        </div>
                        <div class="stat-card" style="padding: 10px; background: rgba(0,0,0,0.2);">
                            <div class="stat-val" style="font-size: 1.2rem; color: white;">${data.metrics.mae.toFixed(4)}</div>
                            <div class="stat-lbl" style="font-size: 0.65rem; margin-top: 2px;">MAE</div>
                        </div>
                    `;
                }
            } else {
                metricsContainer.style.display = 'none';
            }
            
            batchPredictionResults.style.display = 'block';
            updateStatus('Batch predictions computed successfully.');
            
            // Scroll to batch results
            setTimeout(() => {
                batchPredictionResults.scrollIntoView({ behavior: 'smooth' });
            }, 100);
            
            // Refresh icons inside batch panel
            lucide.createIcons();
            
        } catch (err) {
            btnRunBatch.disabled = false;
            btnRunBatch.innerText = 'Run Batch Prediction';
            btnRunBatch.className = 'btn btn-success';
            updateStatus('Batch Prediction Failed.');
            alert('Batch prediction error: ' + err.message);
        }
    });
    
    function renderBatchPreviewTable(rows, columns) {
        if (!rows || rows.length === 0) return;
        
        // Find the prediction column (which is the last column)
        const predCol = columns[columns.length - 1];
        
        let html = '<table><thead><tr>';
        columns.forEach(col => {
            if (col === predCol) {
                html += `<th style="background: rgba(16, 185, 129, 0.2); border-bottom: 2px solid var(--secondary);">${col}</th>`;
            } else {
                html += `<th>${col}</th>`;
            }
        });
        html += '</tr></thead><tbody>';
        
        rows.forEach(row => {
            html += '<tr>';
            columns.forEach(col => {
                if (col === predCol) {
                    html += `<td style="color: var(--secondary); font-weight: 700; background: rgba(16, 185, 129, 0.03);">${row[col] !== undefined ? row[col] : ''}</td>`;
                } else {
                    html += `<td>${row[col] !== undefined ? row[col] : ''}</td>`;
                }
            });
            html += '</tr>';
        });
        html += '</tbody></table>';
        batchPreviewTableWrapper.innerHTML = html;
    }
}

function setupThemeHandler() {
    const toggleBtn = document.getElementById('btn-theme-toggle');
    const icon = document.getElementById('theme-toggle-icon');
    const text = document.getElementById('theme-toggle-text');
    
    // Check local storage for theme preference
    const currentTheme = localStorage.getItem('theme') || 'dark';
    if (currentTheme === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
        icon.setAttribute('data-lucide', 'moon');
        text.innerText = 'Dark Mode';
    } else {
        document.documentElement.setAttribute('data-theme', 'dark');
        icon.setAttribute('data-lucide', 'sun');
        text.innerText = 'Light Mode';
    }
    lucide.createIcons();
    
    toggleBtn.addEventListener('click', () => {
        const isLight = document.documentElement.getAttribute('data-theme') === 'light';
        if (isLight) {
            document.documentElement.setAttribute('data-theme', 'dark');
            localStorage.setItem('theme', 'dark');
            icon.setAttribute('data-lucide', 'sun');
            text.innerText = 'Light Mode';
        } else {
            document.documentElement.setAttribute('data-theme', 'light');
            localStorage.setItem('theme', 'light');
            icon.setAttribute('data-lucide', 'moon');
            text.innerText = 'Dark Mode';
        }
        lucide.createIcons();
        
        // Re-render the loss chart if it exists
        if (state.historyChart && state.lastChartData) {
            renderTrainingHistoryChart(state.lastChartAlgo, state.lastChartData);
        }
    });
}
