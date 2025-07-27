from flask import Flask, render_template, request, jsonify, session
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
    return render_template('index.html', features=FEATURES, ranges=FEATURE_RANGES)

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
    app.run(debug=True, host='0.0.0.0', port=5000) 