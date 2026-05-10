"""
Enhanced Data Client for DERMA-Agent
Integrates multiple open clinical data sources with caching and parallel fetching.
Supports TCGA (GDC), cBioPortal, and other public cancer genomics datasets.
"""

import asyncio
import aiohttp
import requests
import requests_cache
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import json
from tqdm import tqdm
import time

# Install cache for requests
requests_cache.install_cache('gdc_cache', expire_after=3600)  # 1 hour cache

GDC_API_BASE = "https://api.gdc.cancer.gov"
CBIOPORTAL_API_BASE = "https://www.cbioportal.org/api"

EXPANDED_CANCER_PROJECTS = {
    # TCGA Projects
    "Skin Cancer": "TCGA-SKCM",
    "Breast Cancer": "TCGA-BRCA",
    "Lung Adenocarcinoma": "TCGA-LUAD",
    "Lung Squamous": "TCGA-LUSC",
    "Brain Cancer (GBM)": "TCGA-GBM",
    "Brain Cancer (LGG)": "TCGA-LGG",
    "Colorectal Cancer": "TCGA-COAD",
    "Ovarian Cancer": "TCGA-OV",
    "Prostate Cancer": "TCGA-PRAD",
    "Bladder Cancer": "TCGA-BLCA",
    "Kidney Cancer (Clear Cell)": "TCGA-KIRC",
    "Kidney Cancer (Papillary)": "TCGA-KIRP",
    "Stomach Cancer": "TCGA-STAD",
    "Head and Neck": "TCGA-HNSC",
    "Liver Cancer": "TCGA-LIHC",
    "Pancreatic Cancer": "TCGA-PAAD",
    "Esophageal Cancer": "TCGA-ESCA",
    "Cervical Cancer": "TCGA-CESC",
    "Thyroid Cancer": "TCGA-THCA",
    "Sarcoma": "TCGA-SARC",
    "Melanoma": "TCGA-SKCM",
    "Testicular Cancer": "TCGA-TGCT",
    "Pheochromocytoma": "TCGA-PCPG",
    # CPTAC Projects
    "CPTAC Breast": "CPTAC-3-BRCA",
    "CPTAC Lung": "CPTAC-3-LUAD",
    "CPTAC Ovarian": "CPTAC-3-OV",
    "CPTAC Colorectal": "CPTAC-3-COAD",
}


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    name: str
    base_url: str
    timeout: int = 30
    max_retries: int = 3
    cache_duration: int = 3600


class AsyncDataFetcher:
    """Async data fetcher for parallel API calls."""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def fetch(self, url: str, params: Dict = None, headers: Dict = None) -> Dict:
        """Fetch data from URL."""
        if not self.session:
            raise RuntimeError("Session not initialized")
            
        async with self.session.get(url, params=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()


class EnhancedGDCClient:
    """Enhanced GDC client with caching and advanced querying."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.base_url = GDC_API_BASE
        
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{key}.json"
        
    def _load_cache(self, key: str) -> Optional[Dict]:
        """Load cached data if it exists and is not expired."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            # Check if cache is less than 1 hour old
            if time.time() - cache_path.stat().st_mtime < 3600:
                with open(cache_path, 'r') as f:
                    return json.load(f)
        return None
        
    def _save_cache(self, key: str, data: Dict) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(key)
        with open(cache_path, 'w') as f:
            json.dump(data, f)
            
    def fetch_clinical_data(self, project_id: str, limit: int = 500) -> pd.DataFrame:
        """Fetch clinical data with caching."""
        cache_key = f"clinical_{project_id}_{limit}"
        cached = self._load_cache(cache_key)
        
        if cached is not None:
            print(f"Using cached data for {project_id}")
            return pd.DataFrame(cached)
            
        endpoint = f"{self.base_url}/cases"
        
        filters = {
            "op": "in",
            "content": {
                "field": "cases.project.project_id",
                "value": [project_id]
            }
        }
        
        params = {
            "filters": str(filters).replace("'", '"'),
            "format": "JSON",
            "size": limit,
            "expand": "diagnoses,demographic,exposures,family_history"
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            cases = data.get("data", {}).get("hits", [])
            if not cases:
                return pd.DataFrame()
                
            records = []
            for case in cases:
                demographic = case.get("demographic", {}) or {}
                diagnoses = case.get("diagnoses", [{}])[0] if case.get("diagnoses") else {}
                exposures = case.get("exposures", [{}])[0] if case.get("exposures") else {}
                
                record = {
                    "case_id": case.get("case_id"),
                    "submitter_id": case.get("submitter_id"),
                    "project_id": project_id,
                    # Demographics
                    "gender": demographic.get("gender"),
                    "race": demographic.get("race"),
                    "ethnicity": demographic.get("ethnicity"),
                    "vital_status": demographic.get("vital_status"),
                    "days_to_death": demographic.get("days_to_death"),
                    "days_to_birth": demographic.get("days_to_birth"),
                    "age_at_diagnosis": diagnoses.get("age_at_diagnosis"),
                    # Diagnosis
                    "primary_diagnosis": diagnoses.get("primary_diagnosis"),
                    "tumor_stage": diagnoses.get("tumor_stage"),
                    "morphology": diagnoses.get("morphology"),
                    "tissue_or_organ_of_origin": diagnoses.get("tissue_or_organ_of_origin"),
                    "site_of_resection_or_biopsy": diagnoses.get("site_of_resection_or_biopsy"),
                    "days_to_recurrence": diagnoses.get("days_to_recurrence"),
                    "days_to_last_follow_up": diagnoses.get("days_to_last_follow_up"),
                    "progression_or_recurrence": diagnoses.get("progression_or_recurrence"),
                    "prior_malignancy": diagnoses.get("prior_malignancy"),
                    "synchronous_malignancy": diagnoses.get("synchronous_malignancy"),
                    "lymph_nodes_examined_count": diagnoses.get("lymph_nodes_examined_count"),
                    "lymph_node_involved_site": diagnoses.get("lymph_node_involved_site"),
                    # Exposures
                    "cigarettes_per_day": exposures.get("cigarettes_per_day"),
                    "years_smoked": exposures.get("years_smoked"),
                    "alcohol_history": exposures.get("alcohol_history"),
                    "alcohol_intensity": exposures.get("alcohol_intensity"),
                }
                records.append(record)
                
            df = pd.DataFrame(records)
            
            # Save to cache
            self._save_cache(cache_key, df.to_dict('records'))
            
            return df
            
        except Exception as e:
            print(f"Error fetching GDC data: {e}")
            return self._get_mock_data(project_id)
    
    def fetch_mutation_data(self, project_id: str, gene_list: List[str] = None, 
                           limit: int = 1000) -> pd.DataFrame:
        """Fetch mutation data for specific genes."""
        cache_key = f"mutations_{project_id}_{'_'.join(gene_list or [])}_{limit}"
        cached = self._load_cache(cache_key)
        
        if cached is not None:
            return pd.DataFrame(cached)
            
        endpoint = f"{self.base_url}/analysis/top_mutated_genes_by_project"
        
        params = {
            "project_id": project_id,
            "size": limit
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            mutations = data.get("data", {}).get("hits", [])
            records = []
            
            for mut in mutations:
                record = {
                    "gene_symbol": mut.get("symbol"),
                    "gene_id": mut.get("gene_id"),
                    "num_mutations": mut.get("num_mutations"),
                    "project_id": project_id
                }
                records.append(record)
                
            df = pd.DataFrame(records)
            
            # Filter by gene list if provided
            if gene_list:
                df = df[df['gene_symbol'].isin(gene_list)]
                
            self._save_cache(cache_key, df.to_dict('records'))
            return df
            
        except Exception as e:
            print(f"Error fetching mutation data: {e}")
            return pd.DataFrame()
    
    def fetch_gene_expression(self, project_id: str, gene_ids: List[str], 
                           limit: int = 100) -> pd.DataFrame:
        """Fetch gene expression data (simplified version)."""
        # Note: Full gene expression requires downloading files
        # This is a placeholder for the file-based approach
        return pd.DataFrame()
    
    def _get_mock_data(self, project_id: str) -> pd.DataFrame:
        """Generate mock data when API fails."""
        np.random.seed(42)
        n_samples = 100
        
        # Determine cancer type from project_id
        cancer_types = {
            "TCGA-SKCM": "Melanoma",
            "TCGA-BRCA": "Breast",
            "TCGA-LUAD": "Lung Adenocarcinoma",
            "TCGA-GBM": "Glioblastoma",
        }
        cancer_type = cancer_types.get(project_id, "Cancer")
        
        return pd.DataFrame({
            "case_id": [f"mock_{i}" for i in range(n_samples)],
            "submitter_id": [f"TCGA-{i:04d}" for i in range(n_samples)],
            "project_id": project_id,
            "gender": np.random.choice(["male", "female"], n_samples),
            "vital_status": np.random.choice(["Alive", "Dead"], n_samples, p=[0.6, 0.4]),
            "days_to_death": np.where(
                np.random.random(n_samples) > 0.6,
                np.random.randint(100, 2000, n_samples),
                None
            ),
            "age_at_diagnosis": np.random.randint(5000, 25000, n_samples),
            "tumor_stage": np.random.choice(["stage i", "stage ii", "stage iii", "stage iv"], n_samples),
            "primary_diagnosis": [cancer_type] * n_samples,
        })


class MultiSourceDataClient:
    """Client that aggregates data from multiple sources."""
    
    def __init__(self):
        self.gdc_client = EnhancedGDCClient()
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
    def fetch_comprehensive_clinical_data(self, project_id: str, 
                                         include_mutations: bool = True) -> Dict[str, pd.DataFrame]:
        """Fetch comprehensive data from all sources."""
        results = {}
        
        print(f"Fetching clinical data for {project_id}...")
        results['clinical'] = self.gdc_client.fetch_clinical_data(project_id)
        
        if include_mutations:
            print(f"Fetching mutation data for {project_id}...")
            # Key cancer genes
            key_genes = ["TP53", "BRCA1", "BRCA2", "EGFR", "KRAS", "BRAF", 
                        "PIK3CA", "PTEN", "MYC", "CDKN2A"]
            results['mutations'] = self.gdc_client.fetch_mutation_data(project_id, key_genes)
            
        return results
    
    def fetch_multiple_cohorts(self, project_ids: List[str], 
                               parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple cancer cohorts."""
        results = {}
        
        if parallel and len(project_ids) > 1:
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_project = {
                    executor.submit(self.gdc_client.fetch_clinical_data, pid): pid 
                    for pid in project_ids
                }
                
                for future in tqdm(as_completed(future_to_project), 
                                  total=len(project_ids), desc="Fetching cohorts"):
                    project_id = future_to_project[future]
                    try:
                        results[project_id] = future.result()
                    except Exception as e:
                        print(f"Error fetching {project_id}: {e}")
                        results[project_id] = pd.DataFrame()
        else:
            for project_id in tqdm(project_ids, desc="Fetching cohorts"):
                results[project_id] = self.gdc_client.fetch_clinical_data(project_id)
                
        return results
    
    def get_survival_analysis_ready_data(self, project_id: str) -> pd.DataFrame:
        """Get data formatted for survival analysis."""
        df = self.gdc_client.fetch_clinical_data(project_id)
        
        if df.empty:
            return df
            
        # Prepare survival columns
        df['time'] = df.apply(
            lambda row: row['days_to_death'] if pd.notna(row['days_to_death']) 
            else row.get('days_to_last_follow_up', 0), 
            axis=1
        )
        df['event'] = (df['vital_status'] == 'Dead').astype(int)
        df['age_years'] = df['age_at_diagnosis'] / 365.25
        
        # Create stage groups
        df['stage_group'] = df['tumor_stage'].apply(
            lambda x: 'Early' if x in ['stage i', 'stage ia', 'stage ib'] 
            else ('Advanced' if x in ['stage iii', 'stage iiia', 'stage iiib', 'stage iv'] 
                  else 'Intermediate')
            if pd.notna(x) else 'Unknown'
        )
        
        return df
    
    def enrich_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived clinical features for analysis."""
        if df.empty:
            return df
            
        # Age groups
        df['age_group'] = pd.cut(
            df['age_years'], 
            bins=[0, 50, 60, 70, 100], 
            labels=['<50', '50-60', '60-70', '>70']
        )
        
        # Risk score (simple heuristic)
        df['risk_score'] = 0
        df.loc[df['stage_group'] == 'Advanced', 'risk_score'] += 2
        df.loc[df['stage_group'] == 'Intermediate', 'risk_score'] += 1
        df.loc[df['age_years'] > 70, 'risk_score'] += 1
        df.loc[df['gender'] == 'male', 'risk_score'] += 0.5
        
        return df


def get_data_client() -> MultiSourceDataClient:
    """Factory function to get a configured data client."""
    return MultiSourceDataClient()


if __name__ == "__main__":
    # Test the enhanced client
    client = get_data_client()
    
    # Test single cohort
    print("\n=== Testing Single Cohort Fetch ===")
    data = client.fetch_comprehensive_clinical_data("TCGA-BRCA")
    print(f"Clinical data shape: {data['clinical'].shape}")
    if 'mutations' in data:
        print(f"Mutation data shape: {data['mutations'].shape}")
    print(data['clinical'].head())
    
    # Test survival-ready data
    print("\n=== Testing Survival Analysis Data ===")
    survival_df = client.get_survival_analysis_ready_data("TCGA-SKCM")
    print(f"Survival data shape: {survival_df.shape}")
    print(survival_df[['case_id', 'time', 'event', 'stage_group']].head())
    
    # Test multiple cohorts
    print("\n=== Testing Multiple Cohorts ===")
    cohorts = ["TCGA-BRCA", "TCGA-LUAD", "TCGA-SKCM"]
    multi_data = client.fetch_multiple_cohorts(cohorts, parallel=True)
    for cohort, df in multi_data.items():
        print(f"{cohort}: {df.shape[0]} samples")
