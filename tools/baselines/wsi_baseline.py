"""
WSI Survival Analysis Baseline
Trains classical Cox Proportional Hazards and Machine Learning regression models 
on histopathology features alone to establish a prognostic performance baseline.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.enhanced_data_client import get_data_client
from tools.enhanced_pathology import extract_features_for_dataset

def run_wsi_baseline(cohort_id: str, random_seed: int = 42) -> None:
    print(f"=== Running WSI Survival Baseline for {cohort_id} ===")
    
    np.random.seed(random_seed)
    data_client = get_data_client()
    
    # 1. Fetch clinical data
    print("Loading clinical data...")
    df_clinical = data_client.get_survival_analysis_ready_data(
        project_id=cohort_id,
        missing_data_strategy="impute_median"
    )
    
    if df_clinical.empty:
        print(f"Error: Clinical data is empty for cohort {cohort_id}.")
        return
        
    print(f"Loaded {len(df_clinical)} clinical records.")
    
    # 2. Get pathology features
    pathology_csv_path = f"data/pathology_{cohort_id.lower()}_features.csv"
    print(f"Checking pathology features at '{pathology_csv_path}'...")
    
    # Generate features if missing
    if not os.path.exists(pathology_csv_path):
        print("Pathology features not found. Extracting/generating features...")
        case_ids = df_clinical['submitter_id'].tolist()
        extract_features_for_dataset(
            wsi_dir="data/wsi_slides", 
            output_csv_path=pathology_csv_path,
            case_ids=case_ids,
            mock_mode=True
        )
        
    # Merge pathology features
    df_merged = data_client.merge_pathology_features(df_clinical, pathology_csv_path)
    
    # Filter down to samples that have pathology features (prefixed with pathology_)
    pathology_cols = [c for c in df_merged.columns if c.startswith('pathology_') and c != 'pathology_schema_version']
    if not pathology_cols:
        print("Error: No pathology features found in merged dataset.")
        return
        
    print(f"Pathology feature columns: {pathology_cols}")
    
    # Drop samples missing pathology data
    df_analysis = df_merged.dropna(subset=pathology_cols)
    print(f"Cohort size with complete pathology features: {len(df_analysis)}")
    
    if len(df_analysis) < 10:
        print("Error: Too few samples for analysis.")
        return
        
    # 3. Train-test split
    train_df, test_df = train_test_split(df_analysis, test_size=0.3, random_state=random_seed)
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    results = {
        "cohort_id": cohort_id,
        "sample_size": len(df_analysis),
        "random_seed": random_seed,
        "features_used": pathology_cols,
        "models": {}
    }
    
    # 4. Model A: Cox Proportional Hazards (on pathology features alone)
    print("\n--- Training Cox Proportional Hazards Model ---")
    cox_cols = pathology_cols + ['time', 'event']
    train_cox = train_df[cox_cols].copy()
    test_cox = test_df[cox_cols].copy()
    
    # Fill remaining NaNs with 0
    train_cox = train_cox.fillna(0)
    test_cox = test_cox.fillna(0)
    
    # Remove zero variance columns
    var_cols = train_cox[pathology_cols].var()
    keep_cols = [c for c in pathology_cols if var_cols[c] > 0]
    
    if keep_cols:
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(train_cox[keep_cols + ['time', 'event']], duration_col='time', event_col='event')
            
            # Predict and evaluate
            train_preds = cph.predict_partial_hazard(train_cox[keep_cols])
            test_preds = cph.predict_partial_hazard(test_cox[keep_cols])
            
            # Higher hazard means shorter survival -> minus sign for standard concordance
            train_c_index = concordance_index(train_cox['time'], -train_preds, train_cox['event'])
            test_c_index = concordance_index(test_cox['time'], -test_preds, test_cox['event'])
            
            print(f"Cox PH Train C-index: {train_c_index:.4f}")
            print(f"Cox PH Test C-index: {test_c_index:.4f}")
            
            results["models"]["CoxPH"] = {
                "train_c_index": float(train_c_index),
                "test_c_index": float(test_c_index),
                "coefficients": cph.summary['coef'].to_dict()
            }
        except Exception as e:
            print(f"Cox PH fitting failed: {e}")
            results["models"]["CoxPH"] = {"error": str(e)}
    else:
        print("No pathology features with non-zero variance found.")
        
    # 5. Model B: Random Forest Regressor (predict survival time)
    print("\n--- Training Random Forest Regressor ---")
    X_train = train_df[pathology_cols].fillna(0)
    y_train = train_df['time']
    X_test = test_df[pathology_cols].fillna(0)
    y_test = test_df['time']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=random_seed)
    rf.fit(X_train, y_train)
    
    rf_train_preds = rf.predict(X_train)
    rf_test_preds = rf.predict(X_test)
    
    # Higher predicted survival time means longer survival
    rf_train_c_index = concordance_index(y_train, rf_train_preds, train_df['event'])
    rf_test_c_index = concordance_index(y_test, rf_test_preds, test_df['event'])
    
    print(f"RF Regressor Train C-index: {rf_train_c_index:.4f}")
    print(f"RF Regressor Test C-index: {rf_test_c_index:.4f}")
    
    results["models"]["RandomForest"] = {
        "train_c_index": float(rf_train_c_index),
        "test_c_index": float(rf_test_c_index),
        "feature_importances": dict(zip(pathology_cols, [float(x) for x in rf.feature_importances_]))
    }
    
    # Save results
    os.makedirs("data", exist_ok=True)
    out_path = f"data/wsi_baseline_{cohort_id.lower()}_metrics.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nBaseline metrics saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WSI survival baseline models")
    parser.add_argument("--cohort", type=str, default="TCGA-SKCM", help="TCGA cohort project ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    run_wsi_baseline(args.cohort, args.seed)
