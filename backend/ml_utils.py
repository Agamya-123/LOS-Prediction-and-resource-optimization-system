import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timezone
import shap

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
            "anomaly_detector": data.get("anomaly_detector"),
            "shap_explainer": data.get("shap_explainer"),
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
        
        # --- NEW FEATURE: Actionable AI Recommendations ---
        recommendations = []
        if input_data.get('Blood_Sugar_Level', [0])[0] > 140:
            recommendations.append("Endocrinology Consult for Hyperglycemia")
        if severity >= 4:
            recommendations.append("Priority ICU Escalation Protocol & Continuous Vitals")
        if age > 65 and comorb >= 2:
            recommendations.append("Geriatric Palliative & High-Risk Fall Precautions")
        if diagnosis in ['Stroke', 'Heart Failure']:
            recommendations.append(f"Immediate {diagnosis} Rapid Response Pathway")
        if len(recommendations) == 0:
            recommendations.append("Standard Care Protocol")

        # --- NEW FEATURE: Clinical Anomaly Detection ---
        is_anomaly = False
        anomaly_score = 1
        if 'anomaly_detector' in artifacts and artifacts['anomaly_detector'] is not None:
            anomaly_detector = artifacts['anomaly_detector']
            try:
                preprocessor = model.named_steps['preprocessor']
                df_transformed = preprocessor.transform(df)
                anomaly_score = int(anomaly_detector.predict(df_transformed)[0])
                raw_score = anomaly_detector.score_samples(df_transformed)[0]
                print(f"[ANOMALY CHECK] Score: {anomaly_score}, Raw Score: {raw_score:.4f}, Is Anomaly: {anomaly_score == -1}")
                if anomaly_score == -1:
                    is_anomaly = True
                    # Push anomaly warning to top of factors
                    contributing_factors.insert(0, "⚠️ ANOMALY DETECTED: Patient's clinical presentation is highly unusual (Top 5% outlier). Review data for entry errors or rare conditions.")
            except Exception as e:
                print(f"Error during anomaly detection: {e}")

        # --- NEW FEATURE: Personalized XAI (SHAP) ---
        patient_shap_explanation = {}
        if 'shap_explainer' in artifacts and artifacts['shap_explainer'] is not None:
            try:
                explainer = artifacts['shap_explainer']
                preprocessor = model.named_steps['preprocessor']
                df_transformed = preprocessor.transform(df)
                
                # Calculate SHAP values for this specific patient
                shap_values = explainer.shap_values(df_transformed)
                # Some explainers return list (one for each class), take index 1 for positive class (Long Stay)
                if isinstance(shap_values, list):
                     shap_vals_patient = shap_values[1][0]
                else:
                     shap_vals_patient = shap_values[0]

                # Map values back to feature names
                feature_names = preprocessor.get_feature_names_out()
                clean_names = [name.split('__')[-1] for name in feature_names]
                
                shap_dict = {}
                for feature, score in zip(clean_names, shap_vals_patient):
                     base_feature = feature
                     for col in df.columns:
                         if feature.startswith(f"{col}_"):
                             base_feature = col
                             break
                     shap_dict[base_feature] = shap_dict.get(base_feature, 0) + score
                
                patient_shap_explanation = dict(sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True))

                # Dynamically append top 3 SHAP factors into contributing factors!
                # Always show top 3 by absolute value for full XAI transparency
                sorted_features = sorted(patient_shap_explanation.items(), key=lambda item: abs(item[1]), reverse=True)
                top_3_driving = sorted_features[:3]
                for p_factor, shap_val in top_3_driving:
                     val = df[p_factor].iloc[0] if p_factor in df.columns else p_factor
                     direction = "↑ Higher Stay" if shap_val > 0 else "↓ Lower Stay"
                     contributing_factors.append(f"🔬 AI Risk Driver: {p_factor} = {val} (SHAP: {shap_val:+.3f} {direction})")

            except Exception as e:
                print(f"Error during SHAP calculation: {e}")

        return {
            'prediction': int(prediction),
            'prediction_label': 'Long Stay (>7 days)' if prediction == 1 else 'Short Stay (≤7 days)',
            'confidence': float(probability[prediction]),
            'probabilities': {'short_stay': float(probability[0]), 'long_stay': float(probability[1])},
            'contributing_factors': contributing_factors,
            'is_anomaly': is_anomaly,
            'recommended_actions': recommendations,
            'shap_explanation': patient_shap_explanation  # Return true explainability
        }
    except Exception as e:
        print(f"Prediction logic error: {e}")
        raise e

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
    Train 5 models (Logistic, RF, GB, HistGB, Voting).
    Returns comparison results and best pipeline.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    
    # Models
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (
        RandomForestClassifier, 
        GradientBoostingClassifier, 
        HistGradientBoostingClassifier,
        VotingClassifier,
        IsolationForest
    )
    from sklearn.metrics import roc_auc_score, accuracy_score
    import heapq

    # 1. Prepare Target
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
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    # 3. Define Base Models
    base_models = {
        'LogisticRegression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'RandomForest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'GradientBoosting': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]),
        'HistGradientBoosting': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', HistGradientBoostingClassifier(random_state=42))
        ])
    }
    
    # 4. Train Base Models
    # Use stratify to ensure balanced split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    results = {
        'model_comparison': {},
        'best_model_name': None,
        'best_auc': -1,
        'feature_importance': {}
    }
    
    trained_estimators = []
    
    print("Starting training of base models...")
    for name, pipeline in base_models.items():
        try:
            print(f"Training {name}...")
            # HistGradientBoosting handles dense/sparse automatically in modern sklearn
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            if hasattr(pipeline, "predict_proba"):
                y_probs = pipeline.predict_proba(X_test)[:, 1]
            else:
                y_probs = y_pred 
                
            auc = roc_auc_score(y_test, y_probs)
            acc = accuracy_score(y_test, y_pred)
            
            results['model_comparison'][name] = {'auc': float(auc), 'accuracy': float(acc)}
            trained_estimators.append((name, pipeline, auc))
            
        except Exception as e:
            print(f"Failed to train {name}: {e}")

    # 5. Select Top 3 Models
    trained_estimators.sort(key=lambda x: x[2], reverse=True)
    top_3_estimators = trained_estimators[:3]
    print(f"Top 3 Models selected for Ensemble: {[x[0] for x in top_3_estimators]}")
    
    # 6. Train Voting Ensemble
    ensemble_estimators = []
    for name, _, _ in top_3_estimators:
        # Re-use the pipeline structure from base_models to be safe
        ensemble_estimators.append((name, base_models[name]))
        
    voting_clf = VotingClassifier(estimators=ensemble_estimators, voting='soft')
    
    print("Training Voting Ensemble...")
    try:
        voting_clf.fit(X_train, y_train)
        
        v_pred = voting_clf.predict(X_test)
        v_probs = voting_clf.predict_proba(X_test)[:, 1]
        
        v_auc = roc_auc_score(y_test, v_probs)
        v_acc = accuracy_score(y_test, v_pred)
        
        results['model_comparison']['VotingEnsemble'] = {'auc': float(v_auc), 'accuracy': float(v_acc)}
        
        # Add to comparison list
        trained_estimators.append(('VotingEnsemble', voting_clf, v_auc))
        
    except Exception as e:
        print(f"Failed to train Voting Ensemble: {e}")
        
    # 7. Identify Best Model
    # Sort again, ensemble might be top
    trained_estimators.sort(key=lambda x: x[2], reverse=True)
    best_name, best_pipeline_obj, best_auc_val = trained_estimators[0]
    
    results['best_model_name'] = best_name
    results['best_auc'] = float(best_auc_val)
    results['best_pipeline'] = best_pipeline_obj
    
    print(f"Best Model: {best_name} (AUC: {best_auc_val:.4f})")

    # --- NEW FEATURE: Train Clinical Anomaly Detector ---
    print("Training Clinical Anomaly Detector (Isolation Forest)...")
    try:
        # Preprocess features using the best pipeline's preprocessor to train IsolationForest
        best_preprocessor = best_pipeline_obj.named_steps['preprocessor']
        X_train_transformed = best_preprocessor.transform(X_train)
        
        # Detect top 5% clinical outliers
        anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        anomaly_detector.fit(X_train_transformed)
        
        results['anomaly_detector'] = anomaly_detector
        print("Anomaly Detector trained successfully.")
    except Exception as e:
        print(f"Failed to train Anomaly Detector: {e}")
        results['anomaly_detector'] = None

    # --- NEW FEATURE: Build Global SHAP Explainer ---
    print("Building Global SHAP Explainer...")
    try:
        best_preprocessor = best_pipeline_obj.named_steps['preprocessor']
        X_train_transformed = best_preprocessor.transform(X_train)
        
        # We need the inner classifier to build SHAP
        clf = best_pipeline_obj.named_steps['classifier']
        
        # Sample background dataset for SHAP to avoid massive memory usage
        background_sample = shap.sample(X_train_transformed, 100)
        
        # TreeExplainer works for RF, GB, HistGB. LinearExplainer for Logistic
        if best_name in ['RandomForest', 'GradientBoosting', 'HistGradientBoosting']:
            explainer = shap.TreeExplainer(clf, feature_perturbation='interventional', data=background_sample)
        elif best_name == 'LogisticRegression':
            explainer = shap.LinearExplainer(clf, background_sample)
        else:
            # Fallback Kernel explainer for Ensembles (Very slow, but works)
            explainer = shap.KernelExplainer(clf.predict_proba, background_sample)
            
        results['shap_explainer'] = explainer
        print("SHAP Explainer built successfully.")
    except Exception as e:
        print(f"Failed to build SHAP explainer: {e}")
        results['shap_explainer'] = None

    # 8. Feature Importance
    try:
        # Helper to get feature importance from a pipeline
        def get_importance(pip):
            clf = pip.named_steps['classifier']
            pre = pip.named_steps['preprocessor']
            if hasattr(clf, 'feature_importances_'):
                return clf.feature_importances_, pre
            if hasattr(clf, 'coef_'):
                return clf.coef_[0], pre # Logistic Regression
            return None, None

        importances = None
        preproc_ref = None
        
        if best_name == 'VotingEnsemble' or best_name == 'HistGradientBoosting':
            # Use a model that natively supports feature importance (RF or GB)
            print("Using fallback model for global feature importance visualization.")
            for name, pipeline, _ in trained_estimators:
                if name not in ['VotingEnsemble', 'HistGradientBoosting']:
                    importances, preproc_ref = get_importance(pipeline)
                    if importances is not None:
                        break
        else:
            importances, preproc_ref = get_importance(best_pipeline_obj)

        if importances is not None:
             # Get feature names
            feature_names = preproc_ref.get_feature_names_out()
            clean_names = [name.split('__')[-1] for name in feature_names]
            feat_imp = {k: float(v) for k, v in zip(clean_names, importances)}
            
            # Aggregate
            aggregated_imp = {}
            for feature, score in feat_imp.items():
                base_feature = feature
                for cat_col in categorical_features:
                    if feature.startswith(f"{cat_col}_"):
                        base_feature = cat_col
                        break
                aggregated_imp[base_feature] = aggregated_imp.get(base_feature, 0) + abs(score)

            results['feature_importance'] = dict(sorted(aggregated_imp.items(), key=lambda item: item[1], reverse=True))
        else:
             results['feature_importance'] = {}

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Feature importance extraction warning: {e}")
        results['feature_importance'] = {}

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
            'anomaly_detector': results.get('anomaly_detector'),
            'shap_explainer': results.get('shap_explainer'),
            'metadata': metadata
        }, MODEL_PATH)
        
        return os.path.dirname(MODEL_PATH)
    except Exception as e:
        print(f"Error saving artifacts: {e}")
        return None
