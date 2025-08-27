"""
Production-ready FastAPI service for entrepreneur readiness scoring and summarization.
"""
import os
import time
import uuid
from typing import Optional, Dict, Any
import logging

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import requests
import skops.io as sio
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.skops")
MODEL_ID = os.getenv("MODEL_ID", "ethnmcl/entrepreneur-readiness-xgb")
HF_SUMMARY_MODEL = os.getenv("HF_SUMMARY_MODEL", "ethnmcl/gpt2-entrepreneur-agent")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://*.hf.space,https://huggingface.co").split(",")

# Validate required environment variables
if not HF_API_KEY:
    logger.error("HF_API_KEY environment variable is required")
    raise RuntimeError("HF_API_KEY environment variable is required")

# Initialize FastAPI app
app = FastAPI(
    title="Entrepreneur Readiness API",
    description="Production API for entrepreneur readiness scoring and summarization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global model variable
model = None

def load_model():
    """Load the XGBoost model from skops artifact."""
    global model
    if model is None:
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            model = sio.load(MODEL_PATH, trusted=True)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    return model

class ScoreInput(BaseModel):
    """Input schema for scoring endpoint."""
    # Required numeric fields
    savings: float = Field(..., description="Savings amount")
    monthly_income: float = Field(..., description="Monthly income")
    monthly_expenses: float = Field(..., description="Monthly expenses")
    monthly_entertainment: float = Field(..., description="Monthly entertainment expenses")
    sales_skills: float = Field(..., description="Sales skills rating")
    dependents: float = Field(..., description="Number of dependents")
    age: float = Field(..., description="Age")
    assets: float = Field(..., description="Assets value")
    risk_level: float = Field(..., description="Risk tolerance level")
    confidence: float = Field(..., description="Confidence level")
    idea_difficulty: float = Field(..., description="Idea difficulty rating")
    
    # Optional fields
    entrepreneurial_readiness_score: Optional[float] = None
    savings_ratio: Optional[float] = None
    expense_ratio: Optional[float] = None
    entertainment_ratio: Optional[float] = None
    entrepreneur_type: Optional[str] = None
    education_level: Optional[str] = None
    funding_source: Optional[str] = None

    @validator('savings_ratio', pre=True, always=True)
    def compute_savings_ratio(cls, v, values):
        if v is not None:
            return v
        monthly_income = values.get('monthly_income', 0)
        savings = values.get('savings', 0)
        return savings / monthly_income if monthly_income != 0 else 0

    @validator('expense_ratio', pre=True, always=True)
    def compute_expense_ratio(cls, v, values):
        if v is not None:
            return v
        monthly_income = values.get('monthly_income', 0)
        monthly_expenses = values.get('monthly_expenses', 0)
        return monthly_expenses / monthly_income if monthly_income != 0 else 0

    @validator('entertainment_ratio', pre=True, always=True)
    def compute_entertainment_ratio(cls, v, values):
        if v is not None:
            return v
        monthly_income = values.get('monthly_income', 0)
        monthly_entertainment = values.get('monthly_entertainment', 0)
        return monthly_entertainment / monthly_income if monthly_income != 0 else 0

class SummarizeInput(BaseModel):
    """Input schema for summarization endpoint."""
    input: ScoreInput = Field(..., description="Original input data")
    score: float = Field(..., description="Computed score")
    band: str = Field(..., description="Score band (Low/Med/High)")

class ScoreResponse(BaseModel):
    """Response schema for scoring endpoint."""
    score: float
    band: str
    meta: Dict[str, Any]

class SummarizeResponse(BaseModel):
    """Response schema for summarization endpoint."""
    summary: str
    meta: Dict[str, Any]

def get_score_band(score: float) -> str:
    """Determine score band based on score value."""
    if score < 50:
        return "Low"
    elif score < 75:
        return "Med"
    else:
        return "High"

def prepare_model_input(data: ScoreInput) -> np.ndarray:
    """Prepare input data for model prediction."""
    # Convert input to feature array - adjust this based on your model's expected features
    features = [
        data.savings,
        data.monthly_income,
        data.monthly_expenses,
        data.monthly_entertainment,
        data.sales_skills,
        data.dependents,
        data.age,
        data.assets,
        data.risk_level,
        data.confidence,
        data.idea_difficulty,
        data.savings_ratio or 0,
        data.expense_ratio or 0,
        data.entertainment_ratio or 0
    ]
    return np.array(features).reshape(1, -1)

def call_huggingface_api(prompt: str, max_retries: int = 3) -> str:
    """Call Hugging Face Inference API with retry logic."""
    url = f"https://api-inference.huggingface.co/models/{HF_SUMMARY_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                headers=headers,
                json={"inputs": prompt, "parameters": {"max_length": 200, "temperature": 0.7}},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
                return str(result).strip()
            
            elif response.status_code in [429, 503]:
                wait_time = 2 ** attempt
                logger.warning(f"API rate limited, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            
            else:
                logger.error(f"HF API error: {response.status_code} - {response.text}")
                raise HTTPException(status_code=500, detail="Summary generation failed")
                
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail="Summary generation failed")
            time.sleep(2 ** attempt)
    
    raise HTTPException(status_code=500, detail="Summary generation failed after retries")

def generate_summary_prompt(data: ScoreInput, score: float, band: str) -> str:
    """Generate prompt for summarization model."""
    return f"""
Entrepreneur Profile Analysis:
- Score: {score}/100 ({band} readiness)
- Age: {data.age}, Dependents: {data.dependents}
- Monthly Income: ${data.monthly_income:,.0f}
- Savings: ${data.savings:,.0f} (ratio: {data.savings_ratio:.2f})
- Risk Level: {data.risk_level}/10
- Confidence: {data.confidence}/10
- Sales Skills: {data.sales_skills}/10

Provide a 2-4 sentence assessment of this entrepreneur's readiness, ending with one specific actionable step prefixed with 'Next:'.
""".strip()

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()

@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/v1/score", response_model=ScoreResponse)
async def score_endpoint(data: ScoreInput):
    """Score entrepreneur readiness based on input data."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Load model if not already loaded
        model_instance = load_model()
        
        # Prepare input for model
        model_input = prepare_model_input(data)
        
        # Make prediction
        prediction = model_instance.predict(model_input)[0]
        
        # Normalize score to 0-100 range
        if 0 <= prediction <= 1:
            score = prediction * 100
        else:
            score = prediction
        
        # Ensure score is within bounds
        score = max(0, min(100, float(score)))
        
        # Determine band
        band = get_score_band(score)
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        return ScoreResponse(
            score=round(score, 1),
            band=band,
            meta={
                "duration_ms": duration_ms,
                "model_id": MODEL_ID,
                "request_id": request_id
            }
        )
        
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

@app.post("/v1/summarize", response_model=SummarizeResponse)
async def summarize_endpoint(data: SummarizeInput):
    """Generate summary based on scoring results."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Generate prompt
        prompt = generate_summary_prompt(data.input, data.score, data.band)
        
        # Call Hugging Face API
        summary = call_huggingface_api(prompt)
        
        # Ensure summary ends with actionable step
        if "Next:" not in summary:
            summary += f" Next: Focus on improving your {data.band.lower()} readiness score through targeted skill development."
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        return SummarizeResponse(
            summary=summary,
            meta={
                "duration_ms": duration_ms,
                "model_id": HF_SUMMARY_MODEL,
                "request_id": request_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=1)
