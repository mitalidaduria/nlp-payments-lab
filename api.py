import time
from enum import Enum
from typing import List
from fastapi import FastAPI, status
from pydantic import BaseModel, Field

app = FastAPI(
    title="Payment Fraud Classification API",
    description="Live inference API running on Google Colab",
    version="1.0.0"
)

class PaymentGateway(str, Enum):
    RAZORPAY = "Razorpay"
    STRIPE = "Stripe"
    PAYTM = "Paytm"
    PAYPAL = "PayPal"

class TransactionRequest(BaseModel):
    gateway: PaymentGateway
    amount: float = Field(..., gt=0, le=500000)
    hour_of_day: int = Field(..., ge=0, le=23)
    txn_per_hour: int = Field(..., ge=0)
    user_age_days: int = Field(..., ge=0)

class TransactionResponse(BaseModel):
    predicted_category: str
    confidence: float
    is_retryable: bool
    recommended_action: str
    latency_ms: float

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "service": "fraud-classifier-colab"}

@app.post("/classify", response_model=TransactionResponse)
def classify_transaction(txn: TransactionRequest):
    start_time = time.perf_counter()

    is_high_risk = (txn.amount > 4000 and txn.hour_of_day in [1, 2, 3, 4]) or (txn.user_age_days < 10 and txn.txn_per_hour > 10)

    confidence = 0.92 if is_high_risk else 0.15
    category = "High Risk Fraud" if is_high_risk else "Legitimate"
    action = "Block & Review" if is_high_risk else "Approve"
    retryable = not is_high_risk
    latency = round((time.perf_counter() - start_time) * 1000, 2)

    return TransactionResponse(
        predicted_category=category,
        confidence=confidence,
        is_retryable=retryable,
        recommended_action=action,
        latency_ms=latency
    )