import sys
import os
import json
sys.path.append(os.getcwd())
import pandas as pd
from ml_utils import load_real_dataset, train_and_compare_models

try:
    print("Loading dataset...")
    df = load_real_dataset()
    if df is None:
        print("Dataset load failed")
        sys.exit(1)

    print(f"Dataset Columns: {df.columns.tolist()}")

    print("Training models...")
    results = train_and_compare_models(df)

    model = results['trained_model']
    print(f"Best Model: {results['best_model_name']}")

    if hasattr(model, 'feature_names_in_'):
        features = model.feature_names_in_.tolist()
        print(f"Model Feature Names In: {features}")
        with open('model_features.json', 'w') as f:
            json.dump(features, f)
    else:
        print("Model does not have feature_names_in_")
        
except Exception as e:
    print(f"Error: {e}")
    with open('model_error.txt', 'w') as f:
        f.write(str(e))

print("Training setup complete.")
