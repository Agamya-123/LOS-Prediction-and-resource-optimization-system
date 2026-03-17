from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager

from ml_utils import (
    load_real_dataset,
    train_and_compare_models,
    save_model_artifacts,
    load_model_artifacts,
    predict_patient_stay
)

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AgamyaEngine")

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'hospital_db')]

# Global variable to store model artifacts
MODEL_ARTIFACTS = None

# Pydantic Models
class PatientInput(BaseModel):
    Age: int
    Gender: str
    Admission_Type: str
    Department: str
    Insurance_Type: str
    Num_Comorbidities: int
    Visitors_Count: int
    Blood_Sugar_Level: int
    Admission_Deposit: int
    Diagnosis: str
    Severity_Score: int
    Ward_Type: str

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    probabilities: Dict[str, float]
    contributing_factors: List[str] = []
    is_anomaly: Optional[bool] = False
    recommended_actions: List[str] = []
    shap_explanation: Optional[Dict[str, float]] = None

class Patient(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_name: str
    patient_data: Dict[str, Any]
    prediction: int
    prediction_label: str
    confidence: float
    bed_number: Optional[int] = None
    admission_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    predicted_discharge: Optional[str] = None

class Bed(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    bed_number: int
    status: str
    patient_id: Optional[str] = None
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_ARTIFACTS
    # Startup Sequence
    try:
        logger.info("Initializing Hospital AI Infrastructure...")
        # Verify DB connection
        await client.admin.command('ping')
        
        # Initialize beds if collection is empty
        if await db.beds.count_documents({}) == 0:
            logger.info("Universal Bed Repository is empty. Initializing 50 static beds...")
            beds = [Bed(bed_number=i, status="available").model_dump() for i in range(1, 51)]
            for b in beds: b['last_updated'] = b['last_updated'].isoformat()
            await db.beds.insert_many(beds)
            logger.info("Static Bed Initialization Successful.")
    except Exception as e:
        logger.warning(f"Infrastructure initialization advisory: {e}")
    
    # Load ML Model
    logger.info("Retrieving Predictive Model Artifacts...")
    MODEL_ARTIFACTS = load_model_artifacts()
    if MODEL_ARTIFACTS:
        logger.info("Model benchmarks retrieved. System ready.")
    else:
        logger.info("No predictive model found. System idling until optimization.")

    yield
    
    # Shutdown logic
    client.close()
    logger.info("AI Engine Shutdown Complete.")

app = FastAPI(title="Agamya Healthcare AI Engine", lifespan=lifespan)
api_router = APIRouter(prefix="/api")

# Routes
@api_router.get("/")
async def root():
    return {"status": "Agamya AI Engine Online", "mission": "High Precision Healthcare Logistics"}

@api_router.post("/train")
async def train_model():
    """Train the ML model and save artifacts"""
    global MODEL_ARTIFACTS
    try:
        logger.info("Initiating Comprehensive Model Optimization...")
        df = load_real_dataset()
        if df is None:
             raise HTTPException(status_code=404, detail="Dataset not found or failed to load")

        training_results = train_and_compare_models(df)
        save_model_artifacts(training_results)
        MODEL_ARTIFACTS = load_model_artifacts()
        
        return {
            "message": "Model optimization completed successfully",
            "model_comparison": training_results['model_comparison'],
            "best_model": training_results['best_model_name'],
            "best_auc": training_results['best_auc'],
            "feature_importance": training_results['feature_importance'],
            "dataset_size": len(df)
        }
    except Exception as e:
        logger.error(f"Training cycle failure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/model/info")
async def get_model_info():
    """Get metadata about the current active model"""
    if MODEL_ARTIFACTS is None:
        raise HTTPException(status_code=404, detail="No active model found. Optimization required.")
    return MODEL_ARTIFACTS['metadata']

@api_router.post("/predict", response_model=PredictionResponse)
async def predict(data: PatientInput):
    """Real-time stay duration prediction"""
    if MODEL_ARTIFACTS is None:
        raise HTTPException(status_code=404, detail="AI engine is offline (no model).")
    try:
        return predict_patient_stay(data.model_dump(), MODEL_ARTIFACTS)
    except Exception as e:
        logger.error(f"Runtime prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/patients")
async def create_patient(patient_input: PatientInput, patient_name: str):
    """Admit patient with automated bed assignment and ML prognosis"""
    if MODEL_ARTIFACTS is None:
        raise HTTPException(status_code=404, detail="AI engine offline.")
    
    try:
        prediction = predict_patient_stay(patient_input.model_dump(), MODEL_ARTIFACTS)
        available_bed = await db.beds.find_one({"status": "available"})
        bed_number = available_bed['bed_number'] if available_bed else None
        
        patient = Patient(
            patient_name=patient_name,
            patient_data=patient_input.model_dump(),
            prediction=prediction['prediction'],
            prediction_label=prediction['prediction_label'],
            confidence=prediction['confidence'],
            bed_number=bed_number,
            predicted_discharge="Long Stay" if prediction['prediction'] == 1 else "Short Stay"
        )
        
        doc = patient.model_dump()
        doc['admission_date'] = doc['admission_date'].isoformat()
        await db.patients.insert_one(doc)
        
        if bed_number:
            await db.beds.update_one(
                {"bed_number": bed_number},
                {"$set": {
                    "status": "occupied",
                    "patient_id": patient.id,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }}
            )
        
        return {"patient_id": patient.id, "bed_number": bed_number, **prediction}
    except Exception as e:
        logger.error(f"Patient admission protocol failure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/patients")
async def get_patients(limit: int = 100):
    """Retrieve all active patient records"""
    return await db.patients.find({}, {"_id": 0}).sort("admission_date", -1).to_list(limit)

@api_router.delete("/patients/{patient_id}")
async def discharge_patient(patient_id: str):
    """Process patient discharge and release resources"""
    patient = await db.patients.find_one({"id": patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if patient.get('bed_number'):
        await db.beds.update_one(
            {"bed_number": patient['bed_number']},
            {"$set": {
                "status": "cleaning",
                "patient_id": None,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }}
        )
    
    await db.patients.delete_one({"id": patient_id})
    return {"message": "Patient discharge finalized"}

@api_router.get("/beds")
async def get_beds():
    """Real-time bed occupancy overview"""
    return await db.beds.find({}, {"_id": 0}).sort("bed_number", 1).to_list(100)

@api_router.patch("/beds/{bed_number}")
async def update_bed_status(bed_number: int, status: str):
    """Manual bed status override"""
    if status not in ["available", "occupied", "cleaning"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    result = await db.beds.update_one(
        {"bed_number": bed_number},
        {"$set": {
            "status": status,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Bed identifier error")
    
    return {"message": "Resource status updated"}

@api_router.get("/analytics/stats")
async def get_stats():
    """Aggregate hospital-wide statistics"""
    total_beds = await db.beds.count_documents({})
    occupied = await db.beds.count_documents({"status": "occupied"})
    available = await db.beds.count_documents({"status": "available"})
    cleaning = await db.beds.count_documents({"status": "cleaning"})
    
    total_patients = await db.patients.count_documents({})
    short_stay = await db.patients.count_documents({"prediction": 0})
    long_stay = await db.patients.count_documents({"prediction": 1})
    
    return {
        "total_beds": total_beds,
        "occupied_beds": occupied,
        "available_beds": available,
        "cleaning_beds": cleaning,
        "occupancy_rate": round((occupied / total_beds * 100), 1) if total_beds > 0 else 0,
        "total_patients": total_patients,
        "short_stay_count": short_stay,
        "long_stay_count": long_stay
    }

@api_router.get("/analytics/predictions")
async def get_predictions_analytics():
    """Prognostic history for visualization"""
    return await db.patients.find({}, {"_id": 0, "admission_date": 1, "prediction": 1}).to_list(1000)

app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
