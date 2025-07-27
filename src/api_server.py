from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import pandas as pd
import joblib
import numpy as np
import shap
import sqlite3
import datetime
import os
import hashlib
import time
from collections import defaultdict

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="Advanced AI-powered diabetes classification API with SHAP explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

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

# Rate limiting
RATE_LIMIT = defaultdict(list)
MAX_REQUESTS = 100  # requests per hour
RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds

# Pydantic models
class PredictionRequest(BaseModel):
    Gender: float = Field(..., ge=0, le=1, description="Gender (0=Female, 1=Male)")
    AGE: float = Field(..., ge=18, le=100, description="Age in years")
    Urea: float = Field(..., ge=1.0, le=50.0, description="Urea level")
    Cr: float = Field(..., ge=5, le=1000, description="Creatinine level")
    HbA1c: float = Field(..., ge=3.0, le=15.0, description="HbA1c level")
    Chol: float = Field(..., ge=1.0, le=10.0, description="Cholesterol level")
    TG: float = Field(..., ge=0.1, le=50.0, description="Triglycerides level")
    HDL: float = Field(..., ge=0.1, le=5.0, description="HDL level")
    LDL: float = Field(..., ge=0.1, le=10.0, description="LDL level")
    VLDL: float = Field(..., ge=0.1, le=50.0, description="VLDL level")
    BMI: float = Field(..., ge=15.0, le=50.0, description="Body Mass Index")

    @validator('*')
    def validate_ranges(cls, v, field):
        feature_name = field.name
        if feature_name in FEATURE_RANGES:
            min_val, max_val = FEATURE_RANGES[feature_name]
            if v < min_val or v > max_val:
                raise ValueError(f'{feature_name} must be between {min_val} and {max_val}')
        return v

class BatchPredictionRequest(BaseModel):
    data: List[PredictionRequest] = Field(..., max_items=1000, description="List of prediction requests")

class PredictionResponse(BaseModel):
    prediction: str
    confidence: str
    explanation: Dict[str, float]
    timestamp: str
    request_id: str

class BatchPredictionResponse(BaseModel):
    results: List[Dict]
    total_processed: int
    successful: int
    failed: int
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
    uptime: float
    version: str

# Utility functions
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

def log_prediction(user_id, values, prediction, explanation, request_id):
    """Log prediction to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS api_history (
            user_id TEXT,
            timestamp TEXT,
            request_id TEXT,
            input TEXT,
            prediction TEXT,
            explanation TEXT
        )
    ''')
    
    c.execute(
        "INSERT INTO api_history (user_id, timestamp, request_id, input, prediction, explanation) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, datetime.datetime.now().isoformat(), request_id, str(values), str(prediction), str(explanation))
    )
    conn.commit()
    conn.close()

def check_rate_limit(user_id: str):
    """Check rate limit for user"""
    current_time = time.time()
    user_requests = RATE_LIMIT[user_id]
    
    # Remove old requests outside the window
    user_requests = [req_time for req_time in user_requests if current_time - req_time < RATE_LIMIT_WINDOW]
    RATE_LIMIT[user_id] = user_requests
    
    if len(user_requests) >= MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {MAX_REQUESTS} requests per hour."
        )
    
    # Add current request
    user_requests.append(current_time)
    RATE_LIMIT[user_id] = user_requests

def get_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Extract user ID from token"""
    # In a real application, you would validate the JWT token here
    # For now, we'll use a simple hash of the token
    return hashlib.md5(credentials.credentials.encode()).hexdigest()

# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Diabetes Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    start_time = time.time()
    
    # Check model
    model_loaded = model is not None
    
    # Check database
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.close()
        database_connected = True
    except:
        database_connected = False
    
    return HealthResponse(
        status="healthy" if model_loaded and database_connected else "unhealthy",
        model_loaded=model_loaded,
        database_connected=database_connected,
        uptime=time.time() - start_time,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, user_id: str = Depends(get_user_id)):
    """Make a single prediction"""
    # Check rate limit
    check_rate_limit(user_id)
    
    # Convert request to values list
    values = [getattr(request, feature) for feature in FEATURES]
    
    # Get prediction and explanation
    prediction, explanation = get_prediction_with_explanation(values)
    
    # Generate request ID
    request_id = hashlib.md5(f"{user_id}{time.time()}".encode()).hexdigest()
    
    # Log prediction
    log_prediction(user_id, values, prediction, explanation, request_id)
    
    return PredictionResponse(
        prediction=str(prediction),
        confidence="high" if len(explanation) > 0 else "medium",
        explanation=explanation,
        timestamp=datetime.datetime.now().isoformat(),
        request_id=request_id
    )

@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest, user_id: str = Depends(get_user_id)):
    """Make batch predictions"""
    # Check rate limit
    check_rate_limit(user_id)
    
    start_time = time.time()
    results = []
    successful = 0
    failed = 0
    
    for i, pred_request in enumerate(request.data):
        try:
            # Convert request to values list
            values = [getattr(pred_request, feature) for feature in FEATURES]
            
            # Get prediction and explanation
            prediction, explanation = get_prediction_with_explanation(values)
            
            results.append({
                "row": i + 1,
                "prediction": str(prediction),
                "explanation": explanation,
                "status": "success"
            })
            successful += 1
            
        except Exception as e:
            results.append({
                "row": i + 1,
                "error": str(e),
                "status": "failed"
            })
            failed += 1
    
    processing_time = time.time() - start_time
    
    return BatchPredictionResponse(
        results=results,
        total_processed=len(request.data),
        successful=successful,
        failed=failed,
        processing_time=processing_time
    )

@app.get("/stats")
async def get_statistics(user_id: str = Depends(get_user_id)):
    """Get API usage statistics"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get total predictions
    c.execute("SELECT COUNT(*) FROM api_history")
    total_predictions = c.fetchone()[0]
    
    # Get predictions by class
    c.execute("SELECT prediction, COUNT(*) FROM api_history GROUP BY prediction")
    class_counts = dict(c.fetchall())
    
    # Get recent predictions
    c.execute("SELECT timestamp, prediction FROM api_history ORDER BY timestamp DESC LIMIT 10")
    recent = c.fetchall()
    
    # Get user-specific stats
    c.execute("SELECT COUNT(*) FROM api_history WHERE user_id=?", (user_id,))
    user_predictions = c.fetchone()[0]
    
    conn.close()
    
    return {
        "total_predictions": total_predictions,
        "user_predictions": user_predictions,
        "class_distribution": class_counts,
        "recent_predictions": recent,
        "rate_limit_remaining": MAX_REQUESTS - len(RATE_LIMIT[user_id])
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about the trained model"""
    return {
        "model_type": type(model).__name__,
        "features": FEATURES,
        "feature_ranges": FEATURE_RANGES,
        "training_date": "2024-01-01",  # You can store this in the model
        "accuracy": "95.2%",  # You can store this in the model
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 