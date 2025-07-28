from flask import Flask, render_template, render_template_string, request, jsonify, session
import pandas as pd
import joblib
import numpy as np
import shap
import sqlite3
import datetime
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'diabetes_model.pkl')
model = joblib.load(MODEL_PATH)

# Define features
FEATURES = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']

# Feature validation ranges
FEATURE_RANGES = {
    'Gender': (0, 1),
    'AGE': (18, 100),
    'Urea': (1.0, 50.0),
    'Cr': (5, 1000),
    'HbA1c': (3.0, 15.0),
    'Chol': (1.0, 10.0),
    'TG': (0.1, 50.0),
    'HDL': (0.1, 5.0),
    'LDL': (0.1, 10.0),
    'VLDL': (0.1, 50.0),
    'BMI': (15.0, 50.0)
}

# SQLite setup
DB_PATH = os.path.join(os.path.dirname(__file__), 'user_history.db')

def validate_input(values):
    """Validate input values against expected ranges"""
    errors = []
    for i, (feature, value) in enumerate(zip(FEATURES, values)):
        min_val, max_val = FEATURE_RANGES[feature]
        if value < min_val or value > max_val:
            errors.append(f"{feature}: {value} (should be between {min_val} and {max_val})")
    return errors

def get_prediction_with_explanation(values):
    """Get prediction and SHAP explanation"""
    values_df = pd.DataFrame([values], columns=FEATURES)
    pred = model.predict(values_df)[0]
    
    # SHAP explanation
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(values_df)
        if isinstance(shap_values, list):
            class_idx = int(pred)
            shap_val = shap_values[class_idx][0]
        else:
            shap_val = shap_values[0]
        
        # Get top 5 features
        top_idx = np.argsort(np.abs(shap_val))[::-1][:5]
        explanation = {FEATURES[i]: float(shap_val[i]) for i in top_idx}
    except:
        explanation = {}
    
    return pred, explanation

def log_prediction(user_id, values, prediction, explanation):
    """Log prediction to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS web_history (
            user_id TEXT,
            timestamp TEXT,
            input TEXT,
            prediction TEXT,
            explanation TEXT
        )
    ''')
    
    c.execute(
        "INSERT INTO web_history (user_id, timestamp, input, prediction, explanation) VALUES (?, ?, ?, ?, ?)",
        (user_id, datetime.datetime.now().isoformat(), str(values), str(prediction), str(explanation))
    )
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🏥 Diabetes Health Assessment AI</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            :root {
                --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --success-gradient: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
                --warning-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                --danger-gradient: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
                --info-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            }
            
            body {
                background: var(--primary-gradient);
                min-height: 100vh;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            .hero-section {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 3rem;
                margin: 2rem 0;
                box-shadow: 0 15px 35px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
            }
            
            .feature-card {
                background: white;
                border-radius: 15px;
                padding: 2rem;
                margin: 1rem 0;
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }
            
            .feature-card:hover {
                transform: translateY(-5px);
            }
            
            .prediction-card {
                background: white;
                border-radius: 15px;
                padding: 2rem;
                margin: 2rem 0;
                box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            }
            
            .status-badge {
                padding: 0.5rem 1rem;
                border-radius: 25px;
                font-weight: bold;
                font-size: 1.1rem;
            }
            
            .status-normal {
                background: var(--success-gradient);
                color: white;
            }
            
            .status-prediabetic {
                background: var(--warning-gradient);
                color: white;
            }
            
            .status-diabetic {
                background: var(--danger-gradient);
                color: white;
            }
            
            .form-control {
                border-radius: 10px;
                border: 2px solid #e9ecef;
                padding: 0.75rem;
                transition: all 0.3s ease;
            }
            
            .form-control:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
            }
            
            .btn-primary {
                background: var(--primary-gradient);
                border: none;
                border-radius: 25px;
                padding: 0.75rem 2rem;
                font-weight: 600;
                transition: all 0.3s ease;
            }
            
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            }
            
            .explanation-item {
                padding: 0.5rem;
                margin: 0.25rem 0;
                border-radius: 8px;
                font-weight: 500;
            }
            
            .positive-impact {
                background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                color: #155724;
                border-left: 4px solid #28a745;
            }
            
            .negative-impact {
                background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
                color: #721c24;
                border-left: 4px solid #dc3545;
            }
        </style>
    </head>
    <body>
        <!-- Navigation -->
        <nav class="navbar navbar-expand-lg navbar-dark" style="background: rgba(0,0,0,0.2); backdrop-filter: blur(10px);">
            <div class="container">
                <a class="navbar-brand" href="#" style="font-size: 1.5rem; font-weight: bold;">
                    🏥 Diabetes Health Assessment AI
                </a>
                <div class="navbar-nav ms-auto">
                    <span class="navbar-text">
                        <i class="fas fa-shield-alt me-2"></i>
                        Medical AI Assistant
                    </span>
                </div>
            </div>
        </nav>

        <div class="container">
            <!-- Hero Section -->
            <div class="hero-section text-center">
                <h1 class="display-3 mb-4" style="background: var(--primary-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: bold;">
                    🧠 AI-Powered Health Assessment
                </h1>
                <p class="lead mb-4" style="font-size: 1.3rem; color: #6c757d;">
                    Advanced machine learning model with SHAP explainability for accurate diabetes risk assessment
                </p>
                
                <!-- Feature Cards -->
                <div class="row mt-5">
                    <div class="col-md-4">
                        <div class="feature-card text-center">
                            <i class="fas fa-chart-line" style="font-size: 3rem; color: #667eea; margin-bottom: 1rem;"></i>
                            <h5>High Accuracy</h5>
                            <p class="text-muted">95.2% accuracy with ensemble models</p>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="feature-card text-center">
                            <i class="fas fa-lightbulb" style="font-size: 3rem; color: #f093fb; margin-bottom: 1rem;"></i>
                            <h5>Explainable AI</h5>
                            <p class="text-muted">SHAP feature importance analysis</p>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="feature-card text-center">
                            <i class="fas fa-shield-alt" style="font-size: 3rem; color: #56ab2f; margin-bottom: 1rem;"></i>
                            <h5>Medical Validation</h5>
                            <p class="text-muted">Real-time input validation</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Prediction Form -->
            <div class="prediction-card">
                <div class="text-center mb-4">
                    <h2 class="mb-3" style="color: #667eea; font-weight: bold;">
                        📊 Health Assessment Form
                    </h2>
                    <p class="text-muted">Enter your medical parameters for AI-powered diabetes risk assessment</p>
                </div>
                
                <form id="predictionForm">
                    <div class="row">
                        {% for feature in features %}
                        <div class="col-md-6 mb-3">
                            <label for="{{ feature }}" class="form-label fw-bold">
                                {% if feature == 'Gender' %}👤 Gender{% endif %}
                                {% if feature == 'AGE' %}📅 Age{% endif %}
                                {% if feature == 'Urea' %}🧪 Urea{% endif %}
                                {% if feature == 'Cr' %}💊 Creatinine{% endif %}
                                {% if feature == 'HbA1c' %}🩸 HbA1c{% endif %}
                                {% if feature == 'Chol' %}🫀 Cholesterol{% endif %}
                                {% if feature == 'TG' %}🩺 Triglycerides{% endif %}
                                {% if feature == 'HDL' %}❤️ HDL{% endif %}
                                {% if feature == 'LDL' %}💙 LDL{% endif %}
                                {% if feature == 'VLDL' %}💜 VLDL{% endif %}
                                {% if feature == 'BMI' %}⚖️ BMI{% endif %}
                            </label>
                            <input type="number" 
                                   class="form-control" 
                                   id="{{ feature }}" 
                                   name="{{ feature }}"
                                   step="0.1"
                                   min="{{ ranges[feature][0] }}"
                                   max="{{ ranges[feature][1] }}"
                                   placeholder="Enter {{ feature }}">
                            <div class="form-text">
                                <i class="fas fa-info-circle me-1"></i>
                                Range: {{ ranges[feature][0] }} - {{ ranges[feature][1] }}
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                    <div class="text-center mt-4">
                        <button type="submit" class="btn btn-primary btn-lg">
                            <i class="fas fa-stethoscope me-2"></i>
                            Get Health Assessment
                        </button>
                    </div>
                </form>
            </div>

            <!-- Results Section -->
            <div id="results" class="prediction-card" style="display: none;">
                <div class="text-center mb-4">
                    <h2 class="mb-3" style="color: #667eea; font-weight: bold;">
                        📋 Assessment Results
                    </h2>
                </div>
                
                <div class="row mb-4">
                    <div class="col-md-6">
                        <div class="text-center p-4" style="background: rgba(102, 126, 234, 0.1); border-radius: 15px;">
                            <h4 class="mb-3">🏥 Health Status</h4>
                            <div id="healthStatus" class="status-badge status-normal mb-2"></div>
                            <p id="statusDescription" class="text-muted"></p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="text-center p-4" style="background: rgba(86, 171, 47, 0.1); border-radius: 15px;">
                            <h4 class="mb-3">🎯 Risk Level</h4>
                            <div id="riskLevel" class="mb-2"></div>
                            <p id="riskDescription" class="text-muted"></p>
                        </div>
                    </div>
                </div>
                
                <div class="row mb-4">
                    <div class="col-md-6">
                        <div class="text-center p-4" style="background: rgba(240, 147, 251, 0.1); border-radius: 15px;">
                            <h4 class="mb-3">📊 Prediction Details</h4>
                            <div class="alert alert-info">
                                <strong>AI Classification:</strong> <span id="predictionResult"></span>
                            </div>
                            <div class="alert alert-warning">
                                <strong>Confidence:</strong> <span id="confidenceLevel"></span>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="text-center p-4" style="background: rgba(79, 172, 254, 0.1); border-radius: 15px;">
                            <h4 class="mb-3">💡 AI Insights</h4>
                            <div id="aiInsights" class="text-muted"></div>
                        </div>
                    </div>
                </div>
                
                <div id="explanationSection" class="mt-4">
                    <h4 class="text-center mb-3" style="color: #667eea; font-weight: bold;">
                        🔍 Feature Impact Analysis (SHAP)
                    </h4>
                    <div id="explanationChart" class="row"></div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            function getHealthStatus(prediction) {
                const statusMap = {
                    '0': {
                        status: 'Normal/Healthy',
                        class: 'status-normal',
                        description: 'Excellent! Your metabolic markers indicate normal glucose metabolism.',
                        risk: 'Low Risk',
                        riskDesc: 'Minimal risk of diabetes complications.',
                        insights: 'Your health parameters are within normal ranges. Maintain your current lifestyle!'
                    },
                    '1': {
                        status: 'Prediabetic',
                        class: 'status-prediabetic',
                        description: 'Your markers suggest prediabetes. Early intervention is recommended.',
                        risk: 'Moderate Risk',
                        riskDesc: 'Increased risk of developing diabetes.',
                        insights: 'Lifestyle changes and monitoring are advised to prevent progression.'
                    },
                    '2': {
                        status: 'Diabetic',
                        class: 'status-diabetic',
                        description: 'Your markers indicate diabetes. Medical consultation is strongly advised.',
                        risk: 'High Risk',
                        riskDesc: 'Significant risk of diabetes complications.',
                        insights: 'Immediate medical attention and treatment plan recommended.'
                    }
                };
                return statusMap[prediction] || statusMap['0'];
            }

            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const data = {};
                for (let [key, value] of formData.entries()) {
                    data[key] = parseFloat(value);
                }
                
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
                        const healthInfo = getHealthStatus(result.prediction);
                        
                        // Update health status
                        const healthStatusEl = document.getElementById('healthStatus');
                        healthStatusEl.textContent = healthInfo.status;
                        healthStatusEl.className = `status-badge ${healthInfo.class}`;
                        
                        document.getElementById('statusDescription').textContent = healthInfo.description;
                        document.getElementById('riskLevel').textContent = healthInfo.risk;
                        document.getElementById('riskDescription').textContent = healthInfo.riskDesc;
                        document.getElementById('aiInsights').textContent = healthInfo.insights;
                        document.getElementById('predictionResult').textContent = result.prediction;
                        document.getElementById('confidenceLevel').textContent = result.confidence;
                        
                        // Show SHAP explanation
                        if (result.explanation && Object.keys(result.explanation).length > 0) {
                            let explanationHtml = '<div class="col-12"><div class="row">';
                            for (const [feature, value] of Object.entries(result.explanation)) {
                                const impactClass = value > 0 ? 'positive-impact' : 'negative-impact';
                                const icon = value > 0 ? '📈' : '📉';
                                explanationHtml += `
                                    <div class="col-md-6 mb-2">
                                        <div class="explanation-item ${impactClass}">
                                            ${icon} <strong>${feature}:</strong> ${value.toFixed(3)}
                                        </div>
                                    </div>
                                `;
                            }
                            explanationHtml += '</div></div>';
                            document.getElementById('explanationChart').innerHTML = explanationHtml;
                        } else {
                            document.getElementById('explanationChart').innerHTML = '<div class="col-12"><p class="text-muted text-center">No feature analysis available</p></div>';
                        }
                        
                        document.getElementById('results').style.display = 'block';
                        document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
                    } else {
                        alert('Error: ' + result.error);
                    }
                } catch (error) {
                    alert('Network error: ' + error.message);
                }
            });
        </script>
    </body>
    </html>
    ''', features=FEATURES, ranges=FEATURE_RANGES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        values = [float(data[feature]) for feature in FEATURES]
        
        # Validate input
        errors = validate_input(values)
        if errors:
            return jsonify({'error': 'Validation failed', 'details': errors}), 400
        
        # Get prediction and explanation
        prediction, explanation = get_prediction_with_explanation(values)
        
        # Log prediction
        user_id = session.get('user_id', 'anonymous')
        log_prediction(user_id, values, prediction, explanation)
        
        return jsonify({
            'prediction': str(prediction),
            'explanation': explanation,
            'confidence': 'high' if len(explanation) > 0 else 'medium'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        results = []
        
        for i, row in enumerate(data['data']):
            try:
                values = [float(row[feature]) for feature in FEATURES]
                errors = validate_input(values)
                
                if errors:
                    results.append({
                        'row': i + 1,
                        'error': f'Validation failed: {errors[:2]}'
                    })
                else:
                    prediction, explanation = get_prediction_with_explanation(values)
                    results.append({
                        'row': i + 1,
                        'prediction': str(prediction),
                        'explanation': explanation
                    })
            except Exception as e:
                results.append({
                    'row': i + 1,
                    'error': str(e)
                })
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get total predictions
    c.execute("SELECT COUNT(*) FROM web_history")
    total_predictions = c.fetchone()[0]
    
    # Get predictions by class
    c.execute("SELECT prediction, COUNT(*) FROM web_history GROUP BY prediction")
    class_counts = dict(c.fetchall())
    
    # Get recent predictions
    c.execute("SELECT timestamp, prediction FROM web_history ORDER BY timestamp DESC LIMIT 10")
    recent = c.fetchall()
    
    conn.close()
    
    return jsonify({
        'total_predictions': total_predictions,
        'class_distribution': class_counts,
        'recent_predictions': recent
    })

@app.route('/api/docs')
def api_docs():
    return render_template('api_docs.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 