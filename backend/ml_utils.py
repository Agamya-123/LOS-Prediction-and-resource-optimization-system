"""
==============================================================================
  UNIVERSAL ML UTILITIES — Auto-Adaptive Healthcare LOS Pipeline
  
  This module provides:
  - Auto-detection of target columns from ANY healthcare CSV
  - Auto-discovery of numeric/categorical features  
  - Auto-dropping of IDs, names, dates, and junk columns
  - Regularized model training (anti-overfitting)
  - Anomaly detection (Isolation Forest)
  - SHAP explainability
  - Feature importance extraction
  - All with ZERO manual code changes per dataset
==============================================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
import glob
from datetime import datetime, timezone
import shap

# Known feature columns for the current default dataset (used for prediction input)
FEATURE_COLS = [
    'Age', 'Gender', 'Admission_Type', 'Insurance_Type',
    'Num_Comorbidities', 'Visitors_Count', 'Blood_Sugar_Level', 'Admission_Deposit',
    'Department', 'Diagnosis', 'Severity_Score', 'Ward_Type'
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/best_hospital_stay_model_comprehensive.pkl')

# ============================================================================
#  UTILITIES
# ============================================================================

def _sanitize(obj):
    """Recursively convert NumPy types to Python primitives for JSON serialization"""
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_sanitize(v) for v in obj]
    return obj


# ============================================================================
#  UNIVERSAL AUTO-DETECTION FUNCTIONS
# ============================================================================

def auto_detect_target(df):
    """
    Automatically find or compute the LOS target column from any dataset.
    Strategies:
      1. Look for columns with 'stay' or 'los' or 'length' in the name
      2. Look for admission + discharge date columns and compute the difference
    Returns (modified_df, target_column_name)
    """
    cols_lower = {c.lower().replace(' ', '').replace('_', ''): c for c in df.columns}

    # Strategy 1: Direct LOS column
    los_keywords = ['lengthofstay', 'staydays', 'stay_days', 'los', 'lengthstay']
    for keyword in los_keywords:
        for cl, original in cols_lower.items():
            if keyword in cl:
                print(f"  [AUTO-DETECT] Found direct target column: '{original}'")
                return df, original

    # Strategy 2: Compute from date columns
    date_pairs = [
        ('date of admission', 'discharge date'),
        ('dateofadmission', 'dischargedate'),
        ('admissiondate', 'dischargedate'),
        ('admit_date', 'discharge_date'),
        ('vdate', 'discharged'),
    ]
    for adm_key, dis_key in date_pairs:
        adm_col = None
        dis_col = None
        for cl, original in cols_lower.items():
            if adm_key.replace(' ', '').replace('_', '') in cl:
                adm_col = original
            if dis_key.replace(' ', '').replace('_', '') in cl:
                dis_col = original
        if adm_col and dis_col:
            print(f"  [AUTO-DETECT] Computing LOS from '{adm_col}' and '{dis_col}'")
            try:
                df[adm_col] = pd.to_datetime(df[adm_col], errors='coerce')
                df[dis_col] = pd.to_datetime(df[dis_col], errors='coerce')
                df['_computed_los'] = (df[dis_col] - df[adm_col]).dt.days
                df = df.dropna(subset=['_computed_los'])
                df['_computed_los'] = df['_computed_los'].astype(int)
                print(f"  [AUTO-DETECT] Computed LOS range: {df['_computed_los'].min()} to {df['_computed_los'].max()} days")
                return df, '_computed_los'
            except Exception as e:
                print(f"  [AUTO-DETECT] Date computation failed: {e}")

    raise ValueError(f"Cannot detect target column. Available: {df.columns.tolist()}")


def auto_drop_columns(df, target_col):
    """
    Automatically drop columns that should NOT be features:
    IDs, names, dates, high-cardinality free text, and the target itself.
    """
    drop_keywords = ['id', 'eid', 'facid', 'name', 'doctor', 'hospital',
                     'date', 'vdate', 'discharged', '_computed_los']

    cols_to_drop = []
    for col in df.columns:
        if col == target_col:
            cols_to_drop.append(col)
            continue
        cl = col.lower().replace(' ', '').replace('_', '')
        for kw in drop_keywords:
            if kw in cl:
                cols_to_drop.append(col)
                break
        else:
            # Drop if nearly-unique text (likely an ID or name)
            if df[col].nunique() > 0.9 * len(df) and df[col].dtype == 'object':
                cols_to_drop.append(col)
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                cols_to_drop.append(col)

    cols_to_drop = list(set(cols_to_drop))
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    print(f"  [AUTO-DROP] Dropped {len(cols_to_drop)} columns: {cols_to_drop}")
    print(f"  [AUTO-DROP] Remaining features: {len(X.columns)}")
    return X


def auto_discover_features(X):
    """
    Classify columns as numeric or categorical automatically.
    Object dtype or <15 unique values → categorical. Everything else → numeric.
    """
    categorical_features = []
    numeric_features = []
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].nunique() < 15:
            categorical_features.append(col)
        else:
            numeric_features.append(col)
    print(f"  [AUTO-DISCOVER] Numeric:     {len(numeric_features)} features")
    print(f"  [AUTO-DISCOVER] Categorical: {len(categorical_features)} features")
    return numeric_features, categorical_features


# ============================================================================
#  LOAD / SAVE MODEL
# ============================================================================

def load_model_artifacts():
    """Load the pre-trained model pipeline and all artifacts"""
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model not found at {MODEL_PATH}")
            return None

        data = joblib.load(MODEL_PATH)
        return {
            "model": data["model"],
            "anomaly_detector": data.get("anomaly_detector"),
            "shap_explainer": data.get("shap_explainer"),
            "metadata": _sanitize(data["metadata"]),
            "feature_cols": data.get("feature_cols", FEATURE_COLS),
        }
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return None


def save_model_artifacts(results):
    """Save the best model pipeline and all metadata"""
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({
            'model': results['best_pipeline'],
            'anomaly_detector': results.get('anomaly_detector'),
            'shap_explainer': results.get('shap_explainer'),
            'feature_cols': results.get('feature_cols', FEATURE_COLS),
            'metadata': {
                'best_model': results['best_model_name'],
                'best_auc': results['best_auc'],
                'feature_importance': results.get('feature_importance', {}),
                'model_comparison': results['model_comparison'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        }, MODEL_PATH)
        print(f"  [SAVED] Model artifacts to {MODEL_PATH}")
        return os.path.dirname(MODEL_PATH)
    except Exception as e:
        print(f"Error saving model artifacts: {e}")
        return None


# ============================================================================
#  PREDICTION
# ============================================================================

def predict_patient_stay(patient_data, artifacts):
    """
    Predict length of stay using the active model pipeline.
    Works with both the legacy fixed-column model AND the universal model.
    """
    model = artifacts['model']

    # Use model's stored feature list or fallback to FEATURE_COLS
    model_features = artifacts.get('feature_cols', FEATURE_COLS)

    # Build input DataFrame from patient data, filling defaults for missing columns
    input_data = {}
    for col in model_features:
        val = patient_data.get(col)
        if val is None:
            # Smart defaults based on column name patterns
            cl = col.lower()
            if 'visit' in cl or 'count' in cl or 'comorb' in cl:
                val = 0
            elif 'severity' in cl or 'score' in cl:
                val = 1
            elif 'sugar' in cl or 'glucose' in cl:
                val = 120
            elif 'deposit' in cl or 'amount' in cl or 'billing' in cl:
                val = 5000
            elif 'age' in cl:
                val = 50
            else:
                val = "Unknown"
        input_data[col] = [val]

    df = pd.DataFrame(input_data)

    try:
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]

        # Clinical Contributing Factors (smart — works with any column names)
        contributing_factors = []
        raw = {k: v[0] for k, v in input_data.items()}

        # Auto-detect relevant clinical values from whatever columns exist
        severity = _get_val(raw, ['Severity_Score', 'severity', 'Severity'], 0)
        ward = _get_val(raw, ['Ward_Type', 'ward', 'Ward'], '')
        diagnosis = _get_val(raw, ['Diagnosis', 'diagnosis', 'Medical_Condition', 'Medical Condition'], '')
        age = _get_val(raw, ['Age', 'age'], 0)
        comorb = _get_val(raw, ['Num_Comorbidities', 'comorbidities', 'comorb'], 0)
        blood_sugar = _get_val(raw, ['Blood_Sugar_Level', 'glucose', 'blood_sugar', 'bloodsugar'], 0)

        if isinstance(severity, (int, float)) and severity >= 4:
            contributing_factors.append(f"Critical Severity (Score: {severity})")
        if isinstance(ward, str) and ward.upper() == "ICU":
            contributing_factors.append("ICU Admission")
        if isinstance(diagnosis, str) and diagnosis in ['Stroke', 'Heart Failure', 'Hip Fracture']:
            contributing_factors.append(f"High Risk Condition: {diagnosis}")
        if isinstance(age, (int, float)) and age > 70:
            contributing_factors.append(f"Elderly Patient ({age})")
        if isinstance(comorb, (int, float)) and comorb > 2:
            contributing_factors.append(f"Multiple Comorbidities ({comorb})")

        # Actionable AI Recommendations
        recommendations = []
        if isinstance(blood_sugar, (int, float)) and blood_sugar > 140:
            recommendations.append("Endocrinology Consult for Hyperglycemia")
        if isinstance(severity, (int, float)) and severity >= 4:
            recommendations.append("Priority ICU Escalation Protocol & Continuous Vitals")
        if isinstance(age, (int, float)) and age > 65 and isinstance(comorb, (int, float)) and comorb >= 2:
            recommendations.append("Geriatric Palliative & High-Risk Fall Precautions")
        if isinstance(diagnosis, str) and diagnosis in ['Stroke', 'Heart Failure']:
            recommendations.append(f"Immediate {diagnosis} Rapid Response Pathway")
        if not recommendations:
            recommendations.append("Standard Care Protocol")

        # Anomaly Detection
        is_anomaly = False
        if artifacts.get('anomaly_detector') is not None:
            try:
                preprocessor = _get_preprocessor(model)
                df_transformed = preprocessor.transform(df)
                anomaly_score = int(artifacts['anomaly_detector'].predict(df_transformed)[0])
                if anomaly_score == -1:
                    is_anomaly = True
                    contributing_factors.insert(0,
                        "⚠️ ANOMALY DETECTED: Patient's clinical presentation is highly unusual. "
                        "Review data for entry errors or rare conditions.")
            except Exception as e:
                print(f"Error during anomaly detection: {e}")

        # SHAP Explainability
        patient_shap_explanation = {}
        if artifacts.get('shap_explainer') is not None:
            try:
                explainer = artifacts['shap_explainer']
                preprocessor = _get_preprocessor(model)
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

                patient_shap_explanation = dict(
                    sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True))

                sorted_features = sorted(patient_shap_explanation.items(),
                                         key=lambda item: abs(item[1]), reverse=True)
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


def _get_val(data_dict, keys, default):
    """Get a value from a dict trying multiple possible key names."""
    for k in keys:
        if k in data_dict:
            return data_dict[k]
    return default


def _get_preprocessor(model):
    """Extract the preprocessor from a Pipeline or VotingClassifier."""
    if hasattr(model, 'named_steps'):
        return model.named_steps['preprocessor']
    elif hasattr(model, 'estimators_'):
        return model.estimators_[0].named_steps['preprocessor']
    raise ValueError("Cannot extract preprocessor from model")


# ============================================================================
#  DATASET LOADING (Universal)
# ============================================================================

def load_real_dataset(csv_path=None):
    """
    Load a healthcare dataset. If csv_path is specified, use that.
    Otherwise, auto-detect the default dataset in the backend directory.
    """
    try:
        if csv_path and os.path.exists(csv_path):
            print(f"  [LOAD] Loading specified dataset: {csv_path}")
            return pd.read_csv(csv_path)

        # Default: look for the comprehensive dataset
        default_path = os.path.join(os.path.dirname(__file__), 'healthcare_dataset_comprehensive.csv')
        if os.path.exists(default_path):
            print(f"  [LOAD] Loading default dataset: {default_path}")
            return pd.read_csv(default_path)

        # Fallback: find any CSV in the backend folder
        csvs = glob.glob(os.path.join(os.path.dirname(__file__), '*.csv'))
        if csvs:
            print(f"  [LOAD] Auto-detected dataset: {csvs[0]}")
            return pd.read_csv(csvs[0])

        print("  [ERROR] No dataset found")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


# ============================================================================
#  UNIVERSAL TRAINING PIPELINE (Regularized, Anti-Overfitting)
# ============================================================================

def train_and_compare_models(df):
    """
    Universal training pipeline that auto-adapts to any healthcare CSV.
    Uses regularized models to prevent overfitting.
    Includes: Anomaly Detection, SHAP, Feature Importance.
    """
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        VotingClassifier,
        IsolationForest
    )
    import xgboost as xgb

    print("\n  === UNIVERSAL ML PIPELINE (Regularized) ===")

    # Step 1: Auto-detect target
    print("\n  [STEP 1] Auto-detecting target column...")
    df, target_col = auto_detect_target(df)
    y = (df[target_col] > 7).astype(int)
    print(f"  Class Distribution: Short={int((y==0).sum())} ({(y==0).mean()*100:.1f}%), "
          f"Long={int((y==1).sum())} ({(y==1).mean()*100:.1f}%)")

    # Step 2: Auto-drop junk columns
    print("\n  [STEP 2] Auto-dropping non-feature columns...")
    X = auto_drop_columns(df, target_col)

    # Step 3: Auto-discover feature types
    print("\n  [STEP 3] Auto-discovering feature types...")
    numeric_features, categorical_features = auto_discover_features(X)
    feature_cols_used = list(X.columns)

    if not numeric_features and not categorical_features:
        raise ValueError("No usable features found in the dataset.")

    # Step 4: Build preprocessor
    transformers = []
    if numeric_features:
        transformers.append(('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features))
    if categorical_features:
        transformers.append(('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20))
        ]), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    # Step 5: Build regularized models (anti-overfitting)
    base_models = {
        'LogisticRegression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, C=0.1, penalty='l2', random_state=42))
        ]),
        'RandomForest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=20,
                max_features='sqrt', random_state=42
            ))
        ]),
        'GradientBoosting': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=30, random_state=42
            ))
        ]),
        'HistGradientBoosting': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', HistGradientBoostingClassifier(
                max_depth=6, learning_rate=0.05, min_samples_leaf=50,
                max_iter=300, random_state=42
            ))
        ]),
        'XGBoost': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=5.0,
                eval_metric='logloss', random_state=42
            ))
        ]),
    }

    # Step 6: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  [STEP 4] Data split — Train: {len(X_train)} | Test: {len(X_test)}")

    # Step 7: Train all models
    print("\n  [STEP 5] Training regularized models...")
    results = {'model_comparison': {}, 'best_model_name': None, 'best_auc': -1, 'feature_importance': {}}
    trained_estimators = []

    for name, pipeline in base_models.items():
        try:
            print(f"    Training {name}...", end=" ")
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_probs = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_probs)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)

            results['model_comparison'][name] = {
                'auc': float(auc), 'accuracy': float(acc),
                'f1': float(f1), 'precision': float(prec), 'recall': float(rec)
            }
            trained_estimators.append((name, pipeline, auc))
            print(f"AUC={auc:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")

    # Step 8: Voting Ensemble
    trained_estimators.sort(key=lambda x: x[2], reverse=True)
    top3 = [(name, base_models[name]) for name, _, _ in trained_estimators[:3]]
    voting_clf = VotingClassifier(estimators=top3, voting='soft')

    print("    Training Voting Ensemble...", end=" ")
    try:
        voting_clf.fit(X_train, y_train)
        v_probs = voting_clf.predict_proba(X_test)[:, 1]
        v_pred = voting_clf.predict(X_test)
        v_auc = roc_auc_score(y_test, v_probs)
        v_acc = accuracy_score(y_test, v_pred)
        v_f1 = f1_score(y_test, v_pred, zero_division=0)
        v_prec = precision_score(y_test, v_pred, zero_division=0)
        v_rec = recall_score(y_test, v_pred, zero_division=0)

        results['model_comparison']['VotingEnsemble'] = {
            'auc': float(v_auc), 'accuracy': float(v_acc),
            'f1': float(v_f1), 'precision': float(v_prec), 'recall': float(v_rec)
        }
        trained_estimators.append(('VotingEnsemble', voting_clf, v_auc))
        print(f"AUC={v_auc:.4f}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Step 9: Select best (prioritize ensemble for stability)
    voting_ref = [x for x in trained_estimators if x[0] == 'VotingEnsemble']
    best_name, best_pipeline_obj, best_auc_val = voting_ref[0] if voting_ref else trained_estimators[0]
    results.update({
        'best_model_name': best_name,
        'best_auc': float(best_auc_val),
        'best_pipeline': best_pipeline_obj,
        'feature_cols': feature_cols_used,
    })
    print(f"\n  ✅ ACTIVE DEPLOYMENT: {best_name} (AUC: {best_auc_val:.4f})")

    # Step 10: Anomaly Detector
    print("  Training Anomaly Detector (Isolation Forest)...")
    try:
        preprocessor_ref = _get_preprocessor(best_pipeline_obj)
        X_train_transformed = preprocessor_ref.transform(X_train)
        anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        anomaly_detector.fit(X_train_transformed)
        results['anomaly_detector'] = anomaly_detector
        print("  ✅ Anomaly detector trained")
    except Exception as e:
        print(f"  ⚠️ Anomaly detector failed: {e}")

    # Step 11: SHAP Explainer
    print("  Building SHAP Explainer...")
    try:
        if hasattr(best_pipeline_obj, 'named_steps'):
            preprocessor_ref = best_pipeline_obj.named_steps['preprocessor']
            clf = best_pipeline_obj.named_steps['classifier']
        else:
            preprocessor_ref = best_pipeline_obj.estimators_[0].named_steps['preprocessor']
            clf = next(
                (est.named_steps['classifier'] for est in best_pipeline_obj.estimators_
                 if not hasattr(est.named_steps['classifier'], 'coef_')),
                best_pipeline_obj.estimators_[0].named_steps['classifier']
            )

        X_train_transformed = preprocessor_ref.transform(X_train)
        background_sample = shap.sample(X_train_transformed, 100)
        if best_name == 'LogisticRegression':
            explainer = shap.LinearExplainer(clf, background_sample)
        else:
            explainer = shap.TreeExplainer(clf, feature_perturbation='interventional',
                                           data=background_sample)
        results['shap_explainer'] = explainer
        print("  ✅ SHAP explainer built")
    except Exception as e:
        print(f"  ⚠️ SHAP explainer failed: {e}")

    # Step 12: Feature Importance
    print("  Extracting feature importance...")
    try:
        def get_importance(pip):
            c, p = pip.named_steps['classifier'], pip.named_steps['preprocessor']
            if hasattr(c, 'feature_importances_'):
                return c.feature_importances_, p
            if hasattr(c, 'coef_'):
                return c.coef_[0], p
            return None, None

        imp, p_ref = None, None
        if best_name in ['VotingEnsemble', 'HistGradientBoosting']:
            for n, p, _ in trained_estimators:
                if n not in ['VotingEnsemble', 'HistGradientBoosting']:
                    imp, p_ref = get_importance(p)
                    if imp is not None:
                        break
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
            results['feature_importance'] = dict(
                sorted(aggregated.items(), key=lambda x: x[1], reverse=True))
            print("  ✅ Feature importance extracted")
    except Exception as e:
        print(f"  ⚠️ Feature importance warning: {e}")

    return _sanitize(results)
