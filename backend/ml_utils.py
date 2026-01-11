import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import json
from datetime import datetime, timezone
import os

# Define expected columns for the API
FEATURE_COLS = ['Age', 'Gender', 'Admission_Type', 'Department', 'Comorbidity', 'Procedures']
TARGET_COL = 'Stay_Category' # Derived from Stay_Duration

def load_real_dataset():
    """Load the Indian Hospital Dataset"""
    try:
        df = pd.read_csv('indian_hospital_data_large.csv')
        df['Stay_Category'] = (df['Stay_Duration'] > 7).astype(int)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def train_and_compare_models(df):
    """Train multiple models and compare their performance"""
    print("DEBUG: Starting training...")
    
    # 1. Select Features
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL]
    
    # 2. Encode Categoricals
    encoders = {}
    cat_cols = ['Gender', 'Admission_Type', 'Department', 'Comorbidity']
    
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna("Unknown").astype(str)
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
        
    # 3. Handle Numerical
    num_cols = ['Age', 'Procedures']
    imputer = SimpleImputer(strategy='median')
    X[num_cols] = imputer.fit_transform(X[num_cols])
    
    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"DEBUG: X_train type: {type(X_train)}")
    print(f"DEBUG: X_train columns: {X_train.columns.tolist()}")

    # 5. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_auc = 0
    best_model = None
    
    print("Training models...")
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)
        else:
            print(f"DEBUG: Fitting {name} with DataFrame...")
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        
        results[name] = {'auc': float(auc), 'accuracy': float(acc)}
        print(f"{name} - AUC: {auc:.4f}, Acc: {acc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model = model
            
    # Feature Importance
    if hasattr(best_model, 'feature_importances_'):
        fi = dict(zip(FEATURE_COLS, best_model.feature_importances_.tolist()))
    elif hasattr(best_model, 'coef_'):
        fi = dict(zip(FEATURE_COLS, np.abs(best_model.coef_[0]).tolist()))
    else:
        fi = {}
        
    return {
        'model_comparison': results,
        'best_model_name': best_model_name,
        'best_auc': float(best_auc),
        'feature_importance': fi,
        'scaler': scaler,
        'trained_model': best_model,
        'encoders': encoders,
        'imputer': imputer
    }

def save_model_artifacts(train_results, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Model
    joblib.dump(train_results['trained_model'], os.path.join(output_dir, 'best_model.pkl'))
    joblib.dump(train_results['scaler'], os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(train_results['imputer'], os.path.join(output_dir, 'imputer.pkl'))
    joblib.dump(train_results['encoders'], os.path.join(output_dir, 'encoders.pkl'))
    
    # Save Metadata
    metadata = {
        'best_model': train_results['best_model_name'],
        'best_auc': train_results['best_auc'],
        'feature_importance': train_results['feature_importance'],
        'feature_columns': FEATURE_COLS,
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'mappings': {col: dict(zip(enc.classes_, range(len(enc.classes_)))) 
                     for col, enc in train_results['encoders'].items()}
    }
    
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
        
    return output_dir

def load_model_artifacts(model_dir='models'):
    try:
        return {
            'model': joblib.load(os.path.join(model_dir, 'best_model.pkl')),
            'scaler': joblib.load(os.path.join(model_dir, 'scaler.pkl')),
            'imputer': joblib.load(os.path.join(model_dir, 'imputer.pkl')),
            'encoders': joblib.load(os.path.join(model_dir, 'encoders.pkl')),
            'metadata': json.load(open(os.path.join(model_dir, 'model_metadata.json')))
        }
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None

def predict_patient_stay(patient_data, artifacts):
    """
    patient_data: dict with keys matching FEATURE_COLS
    """
    model = artifacts['model']
    scaler = artifacts['scaler']
    encoders = artifacts['encoders']
    imputer = artifacts['imputer']
    
    print("DEBUG: Prediction Request Data:", patient_data)

    # Create DF
    df = pd.DataFrame([patient_data])
    
    # Ensure columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
            
    # Encode Categoricals
    for col, enc in encoders.items():
        known_classes = set(enc.classes_)
        df[col] = df[col].astype(str).map(lambda x: x if x in known_classes else "Unknown")
        try:
             df[col] = enc.transform(df[col])
        except ValueError:
             # Fallback
             df[col] = 0
             
    # Impute
    # FIX: ONLY IMPUTE NUMERICAL COLUMNS
    num_cols = ['Age', 'Procedures']
    # Ensure num_cols exist
    for col in num_cols:
        if col not in df.columns:
            df[col] = 0.0 # Default fallback
            
    try:
        df[num_cols] = imputer.transform(df[num_cols])
    except Exception as e:
        print(f"DEBUG: Imputer transform failed: {e}")
        # In case of failure, prevent crash if possible, but imputer really should work on correct cols
        pass
    
    is_logreg = artifacts['metadata']['best_model'] == 'Logistic Regression'
    print(f"DEBUG: Best model is LogReg? {is_logreg}")

    if is_logreg:
        X_final = scaler.transform(df[FEATURE_COLS])
        prediction = model.predict(X_final)[0]
        probability = model.predict_proba(X_final)[0]
    else:
        # Trees used raw encoded features.
        # CRITICAL: Convert to numpy array to bypass strict feature name checking causing issues.
        X_final = df[FEATURE_COLS].values
        print(f"DEBUG: X_final type: {type(X_final)}")
        print(f"DEBUG: X_final shape: {X_final.shape}")
        
        prediction = model.predict(X_final)[0]
        probability = model.predict_proba(X_final)[0]
    
    return {
        'prediction': int(prediction),
        'prediction_label': 'Long Stay (>7 days)' if prediction == 1 else 'Short Stay (≤7 days)',
        'confidence': float(probability[prediction]),
        'probabilities': {'short_stay': float(probability[0]), 'long_stay': float(probability[1])}
    }
