import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timezone

# Define comprehensive features expected by the new model
FEATURE_COLS = [
    'Age', 'Gender', 'Admission_Type', 'Insurance_Type', 
    'Num_Comorbidities', 'Visitors_Count', 'Blood_Sugar_Level', 'Admission_Deposit',
    'Department', 'Diagnosis', 'Severity_Score', 'Ward_Type'
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/best_hospital_stay_model_comprehensive.pkl')

def load_model_artifacts():
    """Load the pre-trained comprehensive model pipeline"""
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model not found at {MODEL_PATH}")
            return None
            
        data = joblib.load(MODEL_PATH)
        return {
            "model": data["model"],
            "metadata": data["metadata"]
        }
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return None

def predict_patient_stay(patient_data, artifacts):
    """
    Predict length of stay using the comprehensive pipeline.
    patient_data: dict containing keys for all FEATURE_COLS
    """
    model = artifacts['model']
    
    # Create DataFrame from input
    # Ensure all required columns are present, fill missing with defaults if necessary
    input_data = {}
    
    for col in FEATURE_COLS:
        val = patient_data.get(col)
        # Handle renaming or mapping if the API input names differ lightly
        # For now, assuming API sends matching keys or we map them in server.py
        if val is None:
             # Safe defaults for missing values to prevent crash
             if col == 'Visitors_Count': val = 0
             elif col == 'Num_Comorbidities': val = 0
             elif col == 'Severity_Score': val = 1
             elif col == 'Blood_Sugar_Level': val = 120
             elif col == 'Admission_Deposit': val = 5000
             else: val = "Unknown"
        input_data[col] = [val]

    df = pd.DataFrame(input_data)
    
    # The pipeline handles all encoding/scaling internaly
    try:
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        
        # Generate Contributing Factors (Heuristic from previous project)
        contributing_factors = []
        
        # Extract values (they are lists in input_data)
        severity = input_data.get('Severity_Score', [0])[0]
        ward = input_data.get('Ward_Type', [''])[0]
        diagnosis = input_data.get('Diagnosis', [''])[0]
        age = input_data.get('Age', [0])[0]
        comorb = input_data.get('Num_Comorbidities', [0])[0]
        adm_type = input_data.get('Admission_Type', [''])[0]
        
        if severity >= 4:
            contributing_factors.append(f"Critical Severity (Score: {severity})")
        if ward == "ICU":
            contributing_factors.append("ICU Admission")
        if diagnosis in ['Stroke', 'Heart Failure', 'Hip Fracture']:
            contributing_factors.append(f"High Risk Condition: {diagnosis}")
        if age > 70:
            contributing_factors.append(f"Elderly Patient ({age})")
        if comorb > 2:
            contributing_factors.append(f"Multiple Comorbidities ({comorb})")
        if adm_type == "Trauma":
            contributing_factors.append("Trauma Case")
            
        return {
            'prediction': int(prediction),
            'prediction_label': 'Long Stay (>7 days)' if prediction == 1 else 'Short Stay (≤7 days)',
            'confidence': float(probability[prediction]),
            'probabilities': {'short_stay': float(probability[0]), 'long_stay': float(probability[1])},
            'contributing_factors': contributing_factors
        }
    except Exception as e:
        print(f"Prediction logic error: {e}")
        raise e

# Legacy function placeholder if called by existing server code, though we should update server.py
def load_real_dataset():
    """Load the comprehensive healthcare dataset"""
    try:
        data_path = os.path.join(os.path.dirname(__file__), 'healthcare_dataset_comprehensive.csv')
        if not os.path.exists(data_path):
            print(f"Dataset not found at {data_path}")
            return None
        return pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def train_and_compare_models(df):
    """
    Train models using the dataset and return comparison results.
    Generates a binary target: 0 for <=7 days, 1 for >7 days.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    # 1. Prepare Target
    # comprehensive dataset has 'Stay_Days'
    if 'Stay_Days' not in df.columns:
        raise ValueError("Dataset missing 'Stay_Days' column")
        
    y = (df['Stay_Days'] > 7).astype(int)
    X = df[FEATURE_COLS]
    
    # 2. Preprocessing Pipeline
    numeric_features = ['Age', 'Num_Comorbidities', 'Visitors_Count', 'Blood_Sugar_Level', 'Admission_Deposit', 'Severity_Score']
    categorical_features = ['Gender', 'Admission_Type', 'Insurance_Type', 'Department', 'Diagnosis', 'Ward_Type']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    # 3. Model Pipelines
    models = {
        'RandomForest': Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))]),
        'XGBoost': Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
    }
    
    # 4. Train and Evaluate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {
        'model_comparison': {},
        'best_model_name': None,
        'best_auc': -1,
        'feature_importance': {}
    }
    
    best_model_pipeline = None
    
    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_probs = pipeline.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_probs)
        acc = accuracy_score(y_test, y_pred)
        
        results['model_comparison'][name] = {'auc': float(auc), 'accuracy': float(acc)}
        
        if auc > results['best_auc']:
            results['best_auc'] = float(auc)
            results['best_model_name'] = name
            best_model_pipeline = pipeline
            
    # Extract feature importance from the best model
    if results['best_model_name']:
        try:
            pipeline = best_model_pipeline
            classifier = pipeline.named_steps['classifier']
            preprocessor = pipeline.named_steps['preprocessor']
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                # Get feature names from preprocessor using get_feature_names_out()
                # This returns names like 'num__Age', 'cat__Gender_Male'
                feature_names = preprocessor.get_feature_names_out()
                
                # Clean prefix "num__" or "cat__" for better readability
                clean_names = [name.split('__')[-1] for name in feature_names]
                
                # Create dictionary and sort & CAST TO FLOAT
                feat_imp = {k: float(v) for k, v in zip(clean_names, importances)}
                
                # --- AGGREGATE FEATURE IMPORTANCE ---
                # Group one-hot encoded features back to their original names
                aggregated_imp = {}
                for feature, score in feat_imp.items():
                    # Check if it's a one-hot feature (e.g., "Department_Cardiology" -> "Department")
                    # We check against our known categorical list to be safe, or just use the prefix logic
                    base_feature = feature
                    
                    # Logic: If feature starts with a known categorical column name followed by _, group it
                    # Known categorical features: 'Gender', 'Admission_Type', 'Insurance_Type', 'Department', 'Diagnosis', 'Ward_Type'
                    for cat_col in categorical_features:
                        if feature.startswith(f"{cat_col}_"):
                            base_feature = cat_col
                            break
                    
                    if base_feature in aggregated_imp:
                        aggregated_imp[base_feature] += score
                    else:
                        aggregated_imp[base_feature] = score

                # Round for cleaner display
                aggregated_imp = {k: round(v, 4) for k, v in aggregated_imp.items()}
                
                # Sort descending
                sorted_imp = dict(sorted(aggregated_imp.items(), key=lambda item: item[1], reverse=True))
                
                results['feature_importance'] = sorted_imp
            else:
                results['feature_importance'] = {}
                
        except Exception as e:
            print(f"Could not extract feature importance: {e}")
            results['feature_importance'] = {}
            
    # Return the fit pipeline and results
    results['best_pipeline'] = best_model_pipeline
    return results

def save_model_artifacts(results):
    """Save the best model pipeline and metadata"""
    try:
        model = results['best_pipeline']
        metadata = {
            'best_model': results['best_model_name'],
            'best_auc': results['best_auc'],
            'feature_importance': results['feature_importance'],
            'model_comparison': results['model_comparison'],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        joblib.dump({
            'model': model,
            'metadata': metadata
        }, MODEL_PATH)
        
        return os.path.dirname(MODEL_PATH)
    except Exception as e:
        print(f"Error saving artifacts: {e}")
        return None
