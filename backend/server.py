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
from ml_utils import (
    load_real_dataset,
    train_and_compare_models,
    save_model_artifacts,
    load_model_artifacts,
    predict_patient_stay
)

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db_name = os.environ.get('DB_NAME', 'hospital_db')
db = client[db_name]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Global variable to store model artifacts
MODEL_ARTIFACTS = None

# Pydantic Models
class PatientInput(BaseModel):
    Age: int
    Gender: str          # Male, Female
    Admission_Type: str  # Emergency, Urgent, Elective
    Department: str      # Cardiology, etc.
    Comorbidity: str     # Diabetes, etc.
    Procedures: int      # Number of procedures

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    probabilities: Dict[str, float]

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
    status: str  # "available", "occupied", "cleaning"
    patient_id: Optional[str] = None
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Initialize beds on startup
@app.on_event("startup")
async def startup_event():
    global MODEL_ARTIFACTS
    
    # Check if beds exist, if not create them
    try:
        bed_count = await db.beds.count_documents({})
        if bed_count == 0:
            beds = []
            for i in range(1, 51):  # Create 50 beds
                bed = Bed(bed_number=i, status="available")
                bed_dict = bed.model_dump()
                bed_dict['last_updated'] = bed_dict['last_updated'].isoformat()
                beds.append(bed_dict)
            await db.beds.insert_many(beds)
            logger.info("Initialized 50 beds")
    except Exception as e:
        logger.warning(f"Could not connect to DB on startup or init beds: {e}")
    
    # Load model if exists
    MODEL_ARTIFACTS = load_model_artifacts()
    if MODEL_ARTIFACTS:
        logger.info("Model artifacts loaded successfully")
    else:
        logger.info("No trained model found. Train model first using /api/train endpoint")

# Routes
@api_router.get("/")
async def root():
    return {"message": "Hospital Bed Management System API"}

@api_router.post("/train")
async def train_model():
    """Train the ML model and save artifacts"""
    global MODEL_ARTIFACTS
    
    try:
        # Load REAL dataset
        logger.info("Loading Indian Hospital Dataset...")
        df = load_real_dataset()
        
        if df is None:
             raise HTTPException(status_code=404, detail="Dataset not found or failed to load")

        # Train models
        logger.info("Training models...")
        training_results = train_and_compare_models(df)
        
        # Save model artifacts
        output_dir = save_model_artifacts(training_results)
        logger.info(f"Model saved at {output_dir}")
        
        # Load artifacts to global variable
        MODEL_ARTIFACTS = load_model_artifacts()
        
        return {
            "message": "Model training completed successfully",
            "model_comparison": training_results['model_comparison'],
            "best_model": training_results['best_model_name'],
            "best_auc": training_results['best_auc'],
            "feature_importance": training_results['feature_importance'],
            "dataset_size": len(df)
        }
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/model/info")
async def get_model_info():
    """Get information about the trained model"""
    if MODEL_ARTIFACTS is None:
        raise HTTPException(status_code=404, detail="No trained model found. Train model first.")
    
    return MODEL_ARTIFACTS['metadata']

@api_router.post("/predict", response_model=PredictionResponse)
async def predict(patient_data: PatientInput):
    """Predict length of stay for a patient"""
    if MODEL_ARTIFACTS is None:
        raise HTTPException(status_code=404, detail="No trained model found. Train model first using /api/train")
    
    try:
        patient_dict = patient_data.model_dump()
        prediction = predict_patient_stay(patient_dict, MODEL_ARTIFACTS)
        return prediction
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/patients")
async def create_patient(patient_input: PatientInput, patient_name: str):
    """Create a new patient with prediction"""
    if MODEL_ARTIFACTS is None:
        raise HTTPException(status_code=404, detail="No trained model found. Train model first.")
    
    try:
        # Get prediction
        patient_dict = patient_input.model_dump()
        prediction = predict_patient_stay(patient_dict, MODEL_ARTIFACTS)
        
        # Find available bed
        available_bed = await db.beds.find_one({"status": "available"})
        bed_number = available_bed['bed_number'] if available_bed else None
        
        # Calculate predicted discharge
        predicted_days = "3 days" if prediction['prediction'] == 0 else "10 days"
        
        # Create patient
        patient = Patient(
            patient_name=patient_name,
            patient_data=patient_dict,
            prediction=prediction['prediction'],
            prediction_label=prediction['prediction_label'],
            confidence=prediction['confidence'],
            bed_number=bed_number,
            predicted_discharge=predicted_days
        )
        
        patient_doc = patient.model_dump()
        patient_doc['admission_date'] = patient_doc['admission_date'].isoformat()
        
        result = await db.patients.insert_one(patient_doc)
        
        # Update bed status if assigned
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
        logger.error(f"Patient creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/patients")
async def get_patients(limit: int = 100):
    """Get all patients"""
    patients = await db.patients.find({}, {"_id": 0}).sort("admission_date", -1).to_list(limit)
    return patients

@api_router.get("/patients/{patient_id}")
async def get_patient(patient_id: str):
    """Get a specific patient"""
    patient = await db.patients.find_one({"id": patient_id}, {"_id": 0})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

@api_router.delete("/patients/{patient_id}")
async def discharge_patient(patient_id: str):
    """Discharge a patient and free up bed"""
    patient = await db.patients.find_one({"id": patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Free up bed
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
    return {"message": "Patient discharged successfully"}

@api_router.get("/beds")
async def get_beds():
    """Get all beds with their status"""
    beds = await db.beds.find({}, {"_id": 0}).sort("bed_number", 1).to_list(100)
    return beds

@api_router.patch("/beds/{bed_number}")
async def update_bed_status(bed_number: int, status: str):
    """Update bed status"""
    if status not in ["available", "occupied", "cleaning"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    result = await db.beds.update_one(
        {"bed_number": bed_number},
        {"$set": {
            "status": status,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Bed not found")
    
    return {"message": "Bed status updated"}

@api_router.get("/analytics/stats")
async def get_stats():
    """Get dashboard statistics"""
    total_beds = await db.beds.count_documents({})
    occupied_beds = await db.beds.count_documents({"status": "occupied"})
    available_beds = await db.beds.count_documents({"status": "available"})
    cleaning_beds = await db.beds.count_documents({"status": "cleaning"})
    
    total_patients = await db.patients.count_documents({})
    short_stay = await db.patients.count_documents({"prediction": 0})
    long_stay = await db.patients.count_documents({"prediction": 1})
    
    occupancy_rate = (occupied_beds / total_beds * 100) if total_beds > 0 else 0
    
    return {
        "total_beds": total_beds,
        "occupied_beds": occupied_beds,
        "available_beds": available_beds,
        "cleaning_beds": cleaning_beds,
        "occupancy_rate": round(occupancy_rate, 1),
        "total_patients": total_patients,
        "short_stay_count": short_stay,
        "long_stay_count": long_stay
    }

@api_router.get("/analytics/predictions")
async def get_predictions_analytics():
    """Get predictions over time for charts"""
    patients = await db.patients.find({}, {"_id": 0, "admission_date": 1, "prediction": 1}).to_list(1000)
    return patients

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
