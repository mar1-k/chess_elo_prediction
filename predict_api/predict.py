from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
import logging
import uvicorn
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chess ELO Rating Prediction API",
    description="API for predicting chess players' ELO ratings based on game features",
    version="1.0.0"
)

# Global variables for models and features
model_black = None
model_white = None
feature_columns = None

def load_models():
    """Load pickled models and features"""
    global model_black, model_white, feature_columns
    
    try:
        # Load black player model
        logger.info("Loading black player model...")
        with open("xgb_model_black.bin", 'rb') as f:
            model_black = pickle.load(f)
        logger.info("Black player model loaded successfully")
        
        # Load white player model
        logger.info("Loading white player model...")
        with open("xgb_model_white.bin", 'rb') as f:
            model_white = pickle.load(f)
        logger.info("White player model loaded successfully")
        
        # Load feature columns
        logger.info("Loading feature columns...")
        with open("feature_columns.bin", "rb") as f:
            feature_columns = pickle.load(f)
        logger.info(f"Feature columns loaded successfully. Found {len(feature_columns)} features")
        
    except Exception as e:
        logger.error(f"Error loading models or features: {str(e)}")
        raise RuntimeError(f"Failed to load models or features: {str(e)}")

# Load models at startup
load_models()

class Features(BaseModel):
    White_playTime_hours: float
    White_total_matches: float
    White_title_value: float
    Black_playTime_hours: float
    Black_total_matches: float
    Black_title_value: float
    TotalMoves: float
    White_avgEval: float
    Black_avgEval: float
    White_avgMoveTime: float
    Black_avgMoveTime: float
    White_blunders: float
    Black_blunders: float
    White_mistakes: float
    Black_mistakes: float

class PredictionRequest(BaseModel):
    features: Features

class PredictionResponse(BaseModel):
    black_elo: float
    white_elo: float

@app.get("/")
def read_root():
    return {
        "message": "Chess ELO Rating Prediction API", 
        "endpoints": [
            "/predict - POST endpoint for ELO predictions",
            "/predict-info - GET endpoint for prediction format information",
            "/status - GET endpoint for API status",
            "/ - GET endpoint for API information"
        ]
    }

@app.get("/status")
async def get_status():
    """Returns the current status of the models and features"""
    return {
        "xgboost_version": xgb.__version__,
        "models_loaded": {
            "black_player": model_black is not None,
            "white_player": model_white is not None
        },
        "features_loaded": feature_columns is not None,
        "num_features": len(feature_columns) if feature_columns is not None else 0
    }

@app.get("/predict-info")
def get_prediction_info():
    """Returns information about how to format the prediction request"""
    example_request = {
        "features": {
            "White_playTime_hours": 0,
            "White_total_matches": 0,
            "White_title_value": 0,
            "Black_playTime_hours": 0,
            "Black_total_matches": 0,
            "Black_title_value": 0,
            "TotalMoves": 0,
            "White_avgEval": 0,
            "Black_avgEval": 0,
            "White_avgMoveTime": 0,
            "Black_avgMoveTime": 0,
            "White_blunders": 0,
            "Black_blunders": 0,
            "White_mistakes": 0,
            "Black_mistakes": 0
        }
    }
    
    return {
        "endpoint": "/predict",
        "method": "POST",
        "request_format": {
            "description": "Dictionary of feature names and their float values",
            "required_features": list(Features.__annotations__.keys()),
            "example_request": example_request
        },
        "response_format": {
            "black_elo": "float: predicted ELO rating for black player",
            "white_elo": "float: predicted ELO rating for white player"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model_black is None or model_white is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    try:
        # Convert features to DataFrame
        df = pd.DataFrame([request.features.dict()])
        
        # Create DMatrix from DataFrame
        dmatrix = xgb.DMatrix(df)
        
        # Make predictions
        black_elo = float(model_black.predict(dmatrix)[0])
        white_elo = float(model_white.predict(dmatrix)[0])
        
        return PredictionResponse(
            black_elo=black_elo,
            white_elo=white_elo
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=8000)