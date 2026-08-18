import time
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum

# --- ALIGNED SCHEMAS ---
class PaymentType(str, Enum):
    CARD = "card"
    WIRE = "wire"
    CRYPTO = "crypto"

class DeviceCategory(str, Enum):
    MOBILE = "mobile"
    DESKTOP = "desktop"

class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    payment_type: PaymentType = Field(..., description="Method of payment")
    device_category: DeviceCategory = Field(..., description="Device used for transaction")

class TransactionResponse(BaseModel):
    predicted_category: str
    confidence: float
    is_retryable: bool
    recommended_action: str
    latency_ms: float
# -----------------------

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ML model into memory...")
    try:
        ml_models["fraud_classifier"] = joblib.load("models/v20260808_0716ac8.pkl")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    yield 
    
    print("Shutting down and cleaning up...")
    ml_models.clear()

app = FastAPI(
    title="Payment Fraud Classification API",
    description="Live inference API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {
        "status": "200 OK", 
        "model_loaded": "fraud_classifier" in ml_models
    }

@app.post("/classify", response_model=TransactionResponse)
async def classify_transaction(data: TransactionRequest):
    start_time = time.time()
    
    if "fraud_classifier" not in ml_models:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # Convert Pydantic object into a DataFrame for scikit-learn
    input_df = pd.DataFrame([data.model_dump()])
    
    try:
        model = ml_models["fraud_classifier"]
        prediction = model.predict(input_df)
        
        is_fraud = int(prediction[0])
        category = "Fraudulent" if is_fraud == 1 else "Legitimate"
        
        latency = (time.time() - start_time) * 1000
        
        return TransactionResponse(
            predicted_category=category,
            confidence=0.95,
            is_retryable=False if is_fraud == 1 else True,
            recommended_action="Block" if is_fraud == 1 else "Process",
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))