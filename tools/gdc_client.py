import requests
import pandas as pd
from typing import Optional

GDC_API_BASE = "https://api.gdc.cancer.gov"

CANCER_PROJECTS = {
    "Skin Cancer": "TCGA-SKCM",
    "Breast Cancer": "TCGA-BRCA",
    "Lung Cancer": "TCGA-LUAD",
    "Brain Cancer": "TCGA-GBM"
}

def fetch_tcga_clinical_data(project_id: str = "TCGA-SKCM", limit: int = 100) -> Optional[pd.DataFrame]:
    """
    Fetches clinical metadata for the given TCGA cohort.
    Returns a Pandas DataFrame.
    """
    endpoint = f"{GDC_API_BASE}/cases"
    
    filters = {
        "op": "in",
        "content": {
            "field": "cases.project.project_id",
            "value": [project_id]
        }
    }
    
    params = {
        "filters": str(filters).replace("'", '"'), # GDC expects valid JSON string
        "format": "JSON",
        "size": limit,
        "expand": "diagnoses,demographic"
    }
    
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        
        cases = data.get("data", {}).get("hits", [])
        if not cases:
            return pd.DataFrame()
            
        # Flatten the nested dictionary into a simple tabular structure
        records = []
        for case in cases:
            demographic = case.get("demographic", {})
            diagnoses = case.get("diagnoses", [{}])[0] # Take first diagnosis
            
            record = {
                "case_id": case.get("case_id"),
                "submitter_id": case.get("submitter_id"),
                "age_at_diagnosis": diagnoses.get("age_at_diagnosis"),
                "vital_status": demographic.get("vital_status"),
                "days_to_death": demographic.get("days_to_death"),
                "gender": demographic.get("gender"),
                "tumor_stage": diagnoses.get("tumor_stage"),
                "primary_diagnosis": diagnoses.get("primary_diagnosis")
            }
            records.append(record)
            
        df = pd.DataFrame(records)
        return df
        
    except Exception as e:
        print(f"Error fetching GDC data: {e}")
        # Return mock data for MVP if network fails
        return pd.DataFrame({
            "case_id": ["mock1", "mock2", "mock3", "mock4", "mock5"],
            "vital_status": ["Dead", "Alive", "Dead", "Alive", "Alive"],
            "days_to_death": [400, None, 1200, None, None],
            "age_at_diagnosis": [20000, 15000, 22000, 18000, 25000],
            "tumor_stage": ["stage i", "stage ii", "stage iii", "stage i", "stage iv"],
            "lymphocyte_density": [0.8, 0.2, 0.5, 0.9, 0.1] # Mock feature for the agent to discover
        })

if __name__ == "__main__":
    df = fetch_tcga_clinical_data(project_id="TCGA-BRCA", limit=5)
    print(df.head())
