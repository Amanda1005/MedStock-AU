
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from anthropic import Anthropic

app = FastAPI(
    title       = "MedStock-AU API",
    description = "AI-Powered Pharmaceutical Demand Forecasting and Inventory Optimisation",
    version     = "1.0.0"
)

client = Anthropic()

# Load data on startup
df        = pd.read_csv('../data/processed/pharmacy_demand_with_anomalies.csv', parse_dates=["date"])
forecasts = pd.read_csv('../data/processed/forecasts_rpa_paracetamol.csv', parse_dates=["date"])

# ── Request/Response Models ──────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    reply: str
    conversation_history: List[dict]

class StockRequest(BaseModel):
    location: str
    medication: str

class ForecastRequest(BaseModel):
    location: str
    medication: str
    days: Optional[int] = 7

# ── Helper Functions ─────────────────────────────────────────────
def get_current_stock(location, medication):
    subset = df[
        (df["location"].str.lower().str.contains(location.lower())) &
        (df["medication"].str.lower().str.contains(medication.lower()))
    ].sort_values("date")
    if subset.empty:
        return None
    latest = subset.iloc[-1]
    avg_demand = subset.tail(7)["demand_units"].mean()
    reorder_pt = latest["reorder_point"]
    simulated_stock = max(0, int(reorder_pt - avg_demand * np.random.uniform(0.3, 0.9)))
    return {
        "location"     : latest["location"],
        "medication"   : latest["medication"],
        "stock_units"  : simulated_stock,
        "reorder_point": int(reorder_pt),
        "avg_demand_7d": round(avg_demand, 1),
        "status"       : "LOW" if simulated_stock < reorder_pt * 0.5 else
                         "CRITICAL" if simulated_stock < reorder_pt * 0.2 else "OK",
    }

def get_demand_forecast(location, medication, days=7):
    subset = df[
        (df["location"].str.lower().str.contains(location.lower())) &
        (df["medication"].str.lower().str.contains(medication.lower()))
    ].sort_values("date").tail(30)
    if subset.empty:
        return None
    avg_demand = subset["demand_units"].mean()
    trend = subset["demand_units"].iloc[-7:].mean() - subset["demand_units"].iloc[:7].mean()
    forecast = [max(0, int(avg_demand + trend * 0.1 + np.random.normal(0, avg_demand * 0.05)))
                for _ in range(days)]
    return {
        "location"      : subset.iloc[-1]["location"],
        "medication"    : subset.iloc[-1]["medication"],
        "forecast_days" : days,
        "daily_forecast": forecast,
        "avg_forecast"  : round(np.mean(forecast), 1),
        "trend"         : "INCREASING" if trend > 5 else "DECREASING" if trend < -5 else "STABLE",
    }

# ── API Endpoints ────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "MedStock-AU API is running",
        "version": "1.0.0",
        "endpoints": ["/stock", "/forecast", "/anomalies", "/report", "/chat", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": str(datetime.now())}

@app.post("/stock")
def stock_endpoint(request: StockRequest):
    result = get_current_stock(request.location, request.medication)
    if not result:
        raise HTTPException(status_code=404, detail="Location or medication not found")
    return result

@app.post("/forecast")
def forecast_endpoint(request: ForecastRequest):
    result = get_demand_forecast(request.location, request.medication, request.days)
    if not result:
        raise HTTPException(status_code=404, detail="Location or medication not found")
    return result

@app.get("/anomalies")
def anomalies_endpoint(location: Optional[str] = None, days: int = 7):
    latest_date = df["date"].max()
    cutoff = latest_date - timedelta(days=days)
    subset = df[df["date"] >= cutoff]
    if location:
        subset = subset[subset["location"].str.lower().str.contains(location.lower())]
    anomalies = subset[subset["if_anomaly"] == 1][
        ["date", "location", "medication", "demand_units"]
    ].sort_values("date", ascending=False)
    return {
        "total_anomalies": len(anomalies),
        "period_days"    : days,
        "records"        : anomalies.head(10).to_dict("records")
    }

@app.get("/report")
def report_endpoint(location: Optional[str] = None):
    latest = df["date"].max()
    recent = df[df["date"] >= latest - timedelta(days=7)]
    if location:
        recent = recent[recent["location"].str.lower().str.contains(location.lower())]
    summary = recent.groupby("location").agg(
        avg_demand   = ("demand_units", "mean"),
        anomaly_count= ("if_anomaly", "sum"),
        total_cost   = ("unit_cost_aud", lambda x: (x * recent.loc[x.index, "demand_units"]).sum())
    ).round(2).reset_index()
    return summary.to_dict("records")

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        system_prompt = """You are MedStock Assistant, an AI-powered pharmaceutical inventory chatbot for Sydney pharmacies.
You have data for: RPA Hospital, Westmead Hospital, St Vincent's Hospital, Prince of Wales Hospital,
Chemist Warehouse Epping, Chemist Warehouse Sydney CBD, Priceline Pitt Street, TerryWhite Chemmart Parramatta.
Be concise, professional, and specific with numbers."""

        history = request.conversation_history or []
        history.append({"role": "user", "content": request.message})

        response = client.messages.create(
            model     = "claude-sonnet-4-5",
            max_tokens= 400,
            system    = system_prompt,
            messages  = history
        )

        reply = response.content[0].text
        history.append({"role": "assistant", "content": reply})

        return ChatResponse(reply=reply, conversation_history=history)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
