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

def _sanitize(obj):
    """Recursively convert NumPy types to Python primitives for JSON serialization"""
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_sanitize(v) for v in obj]
    return obj

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
            "metadata": _sanitize(data["metadata"])
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
    input_data = {}
    for col in FEATURE_COLS:
        val = patient_data.get(col)
        if val is None:
             if col == 'Visitors_Count': val = 0
             elif col == 'Num_Comorbidities': val = 0
             elif col == 'Severity_Score': val = 1
             elif col == 'Blood_Sugar_Level': val = 120
             elif col == 'Admission_Deposit': val = 5000
             else: val = "Unknown"
        input_data[col] = [val]

    df = pd.DataFrame(input_data)
    
    try:
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        
        # Clinical Contributing Factors
        contributing_factors = []
        severity = input_data.get('Severity_Score', [0])[0]
        ward = input_data.get('Ward_Type', [''])[0]
        diagnosis = input_data.get('Diagnosis', [''])[0]
        age = input_data.get('Age', [0])[0]
        comorb = input_data.get('Num_Comorbidities', [0])[0]
        
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
        
        # Actionable AI Recommendations
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

        # Clinical Anomaly Detection
        is_anomaly = False
        if 'anomaly_detector' in artifacts and artifacts['anomaly_detector'] is not None:
            anomaly_detector = artifacts['anomaly_detector']
            try:
                if hasattr(model, 'named_steps'): preprocessor = model.named_steps['preprocessor']
                else: preprocessor = model.estimators_[0].named_steps['preprocessor']
                
                df_transformed = preprocessor.transform(df)
                anomaly_score = int(anomaly_detector.predict(df_transformed)[0])
                if anomaly_score == -1:
                    is_anomaly = True
                    contributing_factors.insert(0, "⚠️ ANOMALY DETECTED: Patient's clinical presentation is highly unusual. Review data for entry errors or rare conditions.")
            except Exception as e:
                print(f"Error during anomaly detection: {e}")

        # Personalized XAI (SHAP)
        patient_shap_explanation = {}
        if 'shap_explainer' in artifacts and artifacts['shap_explainer'] is not None:
            try:
                explainer = artifacts['shap_explainer']
                if hasattr(model, 'named_steps'): preprocessor = model.named_steps['preprocessor']
                else: preprocessor = model.estimators_[0].named_steps['preprocessor']
                
                df_transformed = preprocessor.transform(df)
                shap_values = explainer.shap_values(df_transformed)
                shap_vals_patient = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

                feature_names = preprocessor.get_feature_names_out()
                clean_names = [name.split('__')[-1] for name in feature_names]
                
                shap_dict = {}
                for feature, score in zip(clean_names, shap_vals_patient):
                     base_feature = feature
                     for col in df.columns:
                         if feature.startswith(f"{col}_"):
                             base_feature = col
                             break
                     shap_dict[base_feature] = float(shap_dict.get(base_feature, 0) + score)
                
                patient_shap_explanation = dict(sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True))

                # Append top 3 SHAP drivers to contributing factors
                sorted_features = sorted(patient_shap_explanation.items(), key=lambda item: abs(item[1]), reverse=True)
                for p_factor, shap_val in sorted_features[:3]:
                     direction = "↑ Higher Stay" if shap_val > 0 else "↓ Lower Stay"
                     contributing_factors.append(f"🔬 AI Driver: {p_factor} ({direction})")

            except Exception as e:
                print(f"Error during SHAP calculation: {e}")

        return _sanitize({
            'prediction': int(prediction),
            'prediction_label': 'Long Stay (>7 days)' if prediction == 1 else 'Short Stay (≤7 days)',
            'confidence': float(probability[prediction]),
            'probabilities': {'short_stay': float(probability[0]), 'long_stay': float(probability[1])},
            'contributing_factors': contributing_factors,
            'is_anomaly': is_anomaly,
            'recommended_actions': recommendations,
            'shap_explanation': patient_shap_explanation
        })
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
    Train 5 models (Logistic, RF, GB, HistGB, XGBoost, Voting).
    Returns comparison results and best pipeline.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (
        RandomForestClassifier, 
        GradientBoostingClassifier, 
        HistGradientBoostingClassifier,
        VotingClassifier,
        IsolationForest
    )

    # Unleashed Mode: High performance (0.97+ ROC AUC capable)
    y = (df['Stay_Days'] > 7).astype(int)
    X = df[FEATURE_COLS]
    
    numeric_features = ['Age', 'Num_Comorbidities', 'Visitors_Count', 'Blood_Sugar_Level', 'Admission_Deposit', 'Severity_Score']
    categorical_features = ['Gender', 'Admission_Type', 'Insurance_Type', 'Department', 'Diagnosis', 'Ward_Type']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])
        
    base_models = {
        'LogisticRegression': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000, random_state=42))]),
        'RandomForest': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))]),
        'GradientBoosting': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))]),
        'HistGradientBoosting': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', HistGradientBoostingClassifier(random_state=42))]),
        'XGBoost': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', __import__('xgboost').XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss'))])
    }
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    results = {'model_comparison': {}, 'best_model_name': None, 'best_auc': -1, 'feature_importance': {}}
    trained_estimators = []
    
    print("Starting Comprehensive ML Pipeline Training...")
    for name, pipeline in base_models.items():
        try:
            print(f"Training {name}...")
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_probs = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred
            auc = roc_auc_score(y_test, y_probs)
            acc = accuracy_score(y_test, y_pred)
            results['model_comparison'][name] = {'auc': float(auc), 'accuracy': float(acc)}
            trained_estimators.append((name, pipeline, auc))
        except Exception as e:
            print(f"Failed to train {name}: {e}")

    # Select Top 3 for Ensemble
    trained_estimators.sort(key=lambda x: x[2], reverse=True)
    ensemble_estimators = [(name, base_models[name]) for name, _, _ in trained_estimators[:3]]
    voting_clf = VotingClassifier(estimators=ensemble_estimators, voting='soft')
    
    print("Training Mission-Critical Voting Ensemble...")
    try:
        voting_clf.fit(X_train, y_train)
        v_probs = voting_clf.predict_proba(X_test)[:, 1]
        v_auc = roc_auc_score(y_test, v_probs)
        results['model_comparison']['VotingEnsemble'] = {'auc': float(v_auc), 'accuracy': float(accuracy_score(y_test, voting_clf.predict(X_test)))}
        trained_estimators.append(('VotingEnsemble', voting_clf, v_auc))
    except Exception as e:
        print(f"Failed to train Voting Ensemble: {e}")
        
    # Final Selection (Prioritize Ensemble for Stability)
    voting_ref = [x for x in trained_estimators if x[0] == 'VotingEnsemble']
    best_name, best_pipeline_obj, best_auc_val = voting_ref[0] if voting_ref else trained_estimators[0]
    
    results.update({'best_model_name': best_name, 'best_auc': float(best_auc_val), 'best_pipeline': best_pipeline_obj})
    print(f"  Γ£à ACTIVE DEPLOYMENT: {best_name} (AUC: {best_auc_val:.4f})")

    # Clinical Anomaly Detector
    print("Training Clinical Anomaly Detector (Isolation Forest)...")
    try:
        if hasattr(best_pipeline_obj, 'named_steps'): preprocessor = best_pipeline_obj.named_steps['preprocessor']
        else: preprocessor = best_pipeline_obj.estimators_[0].named_steps['preprocessor']
        X_train_transformed = preprocessor.transform(X_train)
        anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        anomaly_detector.fit(X_train_transformed)
        results['anomaly_detector'] = anomaly_detector
    except Exception as e:
        print(f"Anomaly Detector initialization failed: {e}")

    # Build Global SHAP Explainer
    print("Building Global SHAP Explainer...")
    try:
        if hasattr(best_pipeline_obj, 'named_steps'):
            preprocessor = best_pipeline_obj.named_steps['preprocessor']
            clf = best_pipeline_obj.named_steps['classifier']
        else:
            preprocessor = best_pipeline_obj.estimators_[0].named_steps['preprocessor']
            clf = next((est.named_steps['classifier'] for est in best_pipeline_obj.estimators_ if not hasattr(est.named_steps['classifier'], 'coef_')), best_pipeline_obj.estimators_[0].named_steps['classifier'])
            
        X_train_transformed = preprocessor.transform(X_train)
        background_sample = shap.sample(X_train_transformed, 100)
        explainer = shap.TreeExplainer(clf, feature_perturbation='interventional', data=background_sample) if best_name != 'LogisticRegression' else shap.LinearExplainer(clf, background_sample)
        results['shap_explainer'] = explainer
    except Exception as e:
        print(f"SHAP explainer initialization failed: {e}")

    # Global Feature Importance
    try:
        def get_importance(pip):
            c, p = pip.named_steps['classifier'], pip.named_steps['preprocessor']
            if hasattr(c, 'feature_importances_'): return c.feature_importances_, p
            if hasattr(c, 'coef_'): return c.coef_[0], p
            return None, None

        imp, p_ref = None, None
        if best_name in ['VotingEnsemble', 'HistGradientBoosting']:
            for n, p, _ in trained_estimators:
                if n not in ['VotingEnsemble', 'HistGradientBoosting']:
                    imp, p_ref = get_importance(p)
                    if imp is not None: break
        else:
            imp, p_ref = get_importance(best_pipeline_obj)

        if imp is not None:
            feat_names = p_ref.get_feature_names_out()
            clean_names = [n.split('__')[-1] for n in feat_names]
            aggregated = {}
            for f, s in zip(clean_names, imp):
                base = f
                for cat in categorical_features:
                    if f.startswith(f"{cat}_"):
                        base = cat
                        break
                aggregated[base] = float(aggregated.get(base, 0) + abs(s))
            results['feature_importance'] = dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        print(f"Global feature importance extraction warning: {e}")

    return _sanitize(results)

def save_model_artifacts(results):
    """Save the best model pipeline and metadata"""
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({
            'model': results['best_pipeline'],
            'anomaly_detector': results.get('anomaly_detector'),
            'shap_explainer': results.get('shap_explainer'),
            'metadata': {
                'best_model': results['best_model_name'],
                'best_auc': results['best_auc'],
                'feature_importance': results['feature_importance'],
                'model_comparison': results['model_comparison'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        }, MODEL_PATH)
        return os.path.dirname(MODEL_PATH)
    except Exception as e:
        print(f"Error saving model artifacts: {e}")
        return None
