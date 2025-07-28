from flask import Flask, render_template_string
import joblib
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'diabetes_model.pkl')
model = joblib.load(MODEL_PATH)

@app.route('/')
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Diabetes Prediction AI - Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .form-group { margin: 10px 0; }
            label { display: inline-block; width: 100px; }
            input { padding: 5px; width: 200px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Diabetes Prediction AI - Test</h1>
            <p>This is a test version to verify the web interface is working.</p>
            
            <form id="predictionForm">
                <div class="form-group">
                    <label>Gender:</label>
                    <input type="number" name="Gender" step="0.1" min="0" max="1" value="0">
                </div>
                <div class="form-group">
                    <label>AGE:</label>
                    <input type="number" name="AGE" step="0.1" min="18" max="100" value="50">
                </div>
                <div class="form-group">
                    <label>Urea:</label>
                    <input type="number" name="Urea" step="0.1" min="1" max="50" value="4.7">
                </div>
                <div class="form-group">
                    <label>Cr:</label>
                    <input type="number" name="Cr" step="0.1" min="5" max="1000" value="46">
                </div>
                <div class="form-group">
                    <label>HbA1c:</label>
                    <input type="number" name="HbA1c" step="0.1" min="3" max="15" value="4.9">
                </div>
                <div class="form-group">
                    <label>Chol:</label>
                    <input type="number" name="Chol" step="0.1" min="1" max="10" value="4.2">
                </div>
                <div class="form-group">
                    <label>TG:</label>
                    <input type="number" name="TG" step="0.1" min="0.1" max="50" value="0.9">
                </div>
                <div class="form-group">
                    <label>HDL:</label>
                    <input type="number" name="HDL" step="0.1" min="0.1" max="5" value="2.4">
                </div>
                <div class="form-group">
                    <label>LDL:</label>
                    <input type="number" name="LDL" step="0.1" min="0.1" max="10" value="1.4">
                </div>
                <div class="form-group">
                    <label>VLDL:</label>
                    <input type="number" name="VLDL" step="0.1" min="0.1" max="50" value="0.5">
                </div>
                <div class="form-group">
                    <label>BMI:</label>
                    <input type="number" name="BMI" step="0.1" min="15" max="50" value="24.0">
                </div>
                <button type="submit">Get Prediction</button>
            </form>
            
            <div id="result" style="margin-top: 20px; padding: 10px; background: #f0f0f0; display: none;">
                <h3>Prediction Result:</h3>
                <p id="predictionText"></p>
            </div>
        </div>
        
        <script>
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
                        document.getElementById('predictionText').textContent = 'Predicted Class: ' + result.prediction;
                        document.getElementById('result').style.display = 'block';
                    } else {
                        document.getElementById('predictionText').textContent = 'Error: ' + result.error;
                        document.getElementById('result').style.display = 'block';
                    }
                } catch (error) {
                    document.getElementById('predictionText').textContent = 'Network error: ' + error.message;
                    document.getElementById('result').style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """
    return html

@app.route('/predict', methods=['POST'])
def predict():
    from flask import request, jsonify
    import pandas as pd
    
    try:
        data = request.get_json()
        features = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
        values = [float(data[feature]) for feature in features]
        
        # Create DataFrame with feature names
        values_df = pd.DataFrame([values], columns=features)
        prediction = model.predict(values_df)[0]
        
        return jsonify({
            'prediction': str(prediction),
            'confidence': 'high'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 