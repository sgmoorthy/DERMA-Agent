"""
Fast Parallel Discovery Engine for DERMA-Agent
Implements high-performance hypothesis generation and testing with parallel execution.
"""

import os
import json
import asyncio
import time
from typing import TypedDict, List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import traceback
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from tools.enhanced_data_client import get_data_client, EXPANDED_CANCER_PROJECTS
from tools.knowledge_fabric import KnowledgeFabric, create_default_knowledge_fabric
from tools.enhanced_clinical_stats import EnhancedStatsEngine


@dataclass
class DiscoveryResult:
    """Result of a discovery iteration."""
    hypothesis: str
    test_code: str
    execution_result: str
    conclusion: str
    p_value: Optional[float] = None
    hazard_ratio: Optional[float] = None
    confidence_interval: Optional[tuple] = None
    significant: bool = False
    iteration: int = 0
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "hypothesis": self.hypothesis,
            "test_code": self.test_code,
            "execution_result": self.execution_result,
            "conclusion": self.conclusion,
            "p_value": self.p_value,
            "hazard_ratio": self.hazard_ratio,
            "confidence_interval": self.confidence_interval,
            "significant": self.significant,
            "iteration": self.iteration,
            "execution_time": self.execution_time
        }


@dataclass
class DiscoveryConfig:
    """Configuration for the discovery engine."""
    max_iterations: int = 3
    parallel_workers: int = 4
    hypothesis_per_cohort: int = 3
    significance_threshold: float = 0.05
    use_knowledge_fabric: bool = True
    auto_correct_errors: bool = True
    save_intermediate: bool = True
    timeout_seconds: int = 120


class FastDiscoveryEngine:
    """
    High-performance discovery engine with parallel hypothesis testing.
    """
    
    def __init__(self, config: DiscoveryConfig = None, knowledge_fabric: KnowledgeFabric = None):
        self.config = config or DiscoveryConfig()
        self.knowledge_fabric = knowledge_fabric
        self.data_client = get_data_client()
        self.stats_engine = EnhancedStatsEngine()
        self.ledger: List[Dict] = []
        self.results: List[DiscoveryResult] = []
        
        # Initialize LLM
        self.llm = self._init_llm()
        
    def _init_llm(self) -> Optional[ChatOpenAI]:
        """Initialize the LLM with API key."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not set. Using mock mode.")
            return None
        try:
            return ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.2)
        except Exception as e:
            print(f"Failed to initialize LLM: {e}")
            return None
    
    def _generate_hypothesis_batch(self, df: pd.DataFrame, cancer_type: str, 
                                   n_hypotheses: int) -> List[str]:
        """Generate multiple hypotheses in a single LLM call."""
        columns = df.columns.tolist()
        
        # Use knowledge fabric if available
        knowledge_context = ""
        if self.knowledge_fabric and self.config.use_knowledge_fabric:
            # Query relevant knowledge
            cancer_node_id = cancer_type.replace(" ", "_")
            related = self.knowledge_fabric.query_pattern({
                "label": "Gene"
            })
            
            if related:
                genes = [r["node"].id for r in related[:5]]
                knowledge_context = f"\nRelevant genes for this cancer type: {', '.join(genes)}"
        
        prompt = f"""Act as a Cancer Bioinformatics AI Agent.

Available clinical data columns: {columns}
Cancer Type: {cancer_type}
Sample size: {len(df)}
{knowledge_context}

Generate {n_hypotheses} distinct, testable hypotheses about survival or clinical outcomes 
using these columns. Each hypothesis should be statistically testable with the available data.

Format: Return a numbered list with one hypothesis per line.
Be specific about which columns to use.

Examples of good hypotheses:
- Higher tumor stage correlates with worse survival outcomes
- Age at diagnosis is a significant predictor of survival time
- Gender differences exist in survival patterns
"""
        
        if self.llm:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            
            # Parse hypotheses from response
            hypotheses = []
            for line in content.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                    # Remove numbering/bullets
                    hyp = line.lstrip('0123456789.-*) ').strip()
                    if hyp and len(hyp) > 20:
                        hypotheses.append(hyp)
            
            return hypotheses[:n_hypotheses]
        else:
            # Mock hypotheses
            return [
                f"Tumor stage is associated with survival time in {cancer_type}",
                f"Age at diagnosis predicts survival outcomes in {cancer_type}",
                f"Gender differences exist in survival patterns for {cancer_type}"
            ][:n_hypotheses]
    
    def _test_single_hypothesis(self, hypothesis: str, df: pd.DataFrame, 
                                iteration: int) -> DiscoveryResult:
        """Test a single hypothesis."""
        start_time = time.time()
        
        # Generate test code
        code = self._generate_test_code(hypothesis, df, iteration)
        
        # Execute test
        try:
            result = self.stats_engine.execute_survival_analysis(code, df)
            execution_time = time.time() - start_time
            
            # Parse results
            p_value = self._extract_p_value(result)
            hr = self._extract_hazard_ratio(result)
            
            # Generate conclusion
            conclusion = self._generate_conclusion(hypothesis, result, p_value)
            
            return DiscoveryResult(
                hypothesis=hypothesis,
                test_code=code,
                execution_result=result,
                conclusion=conclusion,
                p_value=p_value,
                hazard_ratio=hr,
                significant=(p_value is not None and p_value < self.config.significance_threshold),
                iteration=iteration,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            
            return DiscoveryResult(
                hypothesis=hypothesis,
                test_code=code,
                execution_result=error_msg,
                conclusion="Test failed due to execution error",
                iteration=iteration,
                execution_time=execution_time
            )
    
    def _generate_test_code(self, hypothesis: str, df: pd.DataFrame, 
                           iteration: int) -> str:
        """Generate Python code to test the hypothesis."""
        columns = df.columns.tolist()
        
        # Check for previous errors
        previous_error = ""
        if iteration > 0:
            previous_error = "Previous attempt failed. Ensure all column names exist and handle missing values."
        
        prompt = f"""Write Python code to test this hypothesis:
"{hypothesis}"

Available dataframe columns: {columns}
The dataframe is named 'df' and is already loaded.

Requirements:
1. Use lifelines (KaplanMeierFitter, CoxPHFitter) for survival analysis
2. Print the p-value, hazard ratio, and confidence interval
3. Handle missing values gracefully
4. Include data validation checks
5. Print a clear summary of findings

{previous_error}

Return ONLY Python code without markdown formatting.
"""
        
        if self.llm:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            code = response.content.replace("```python", "").replace("```", "").strip()
            return code
        else:
            # Mock code
            return self._get_mock_test_code(hypothesis, columns)
    
    def _get_mock_test_code(self, hypothesis: str, columns: List[str]) -> str:
        """Generate mock test code."""
        has_stage = 'tumor_stage' in columns
        has_age = 'age_years' in columns
        
        code = """from lifelines import KaplanMeierFitter, CoxPHFitter
import pandas as pd
import numpy as np

print("Testing hypothesis...")

# Prepare survival data
df_clean = df.dropna(subset=['time', 'event'])
print(f"Samples after cleaning: {len(df_clean)}")

if len(df_clean) < 10:
    print("Insufficient data for analysis")
else:
"""
        if has_stage:
            code += """
    # Test by tumor stage
    kmf = KaplanMeierFitter()
    
    stages = df_clean['stage_group'].dropna().unique()
    if len(stages) >= 2:
        for stage in stages[:2]:
            mask = df_clean['stage_group'] == stage
            if mask.sum() >= 5:
                kmf.fit(df_clean.loc[mask, 'time'], 
                       event_observed=df_clean.loc[mask, 'event'], 
                       label=stage)
                print(f"Stage {stage}: Median survival = {kmf.median_survival_time_:.1f} days")
        
        # Cox model
        df_model = df_clean.dropna(subset=['stage_group', 'age_years'])
        if len(df_model) > 10:
            df_model = pd.get_dummies(df_model, columns=['stage_group'], drop_first=True)
            stage_cols = [c for c in df_model.columns if c.startswith('stage_group_')]
            if stage_cols:
                cph = CoxPHFitter()
                cph.fit(df_model[['time', 'event'] + stage_cols], 
                       duration_col='time', event_col='event')
                print(f"\\nCox Model Summary:")
                print(cph.summary)
"""
        else:
            code += """
    # Basic survival analysis
    kmf = KaplanMeierFitter()
    kmf.fit(df_clean['time'], event_observed=df_clean['event'])
    print(f"Median survival: {kmf.median_survival_time_:.1f} days")
"""
        
        return code
    
    def _extract_p_value(self, result: str) -> Optional[float]:
        """Extract p-value from execution result."""
        import re
        # Look for p-value patterns
        patterns = [
            r'p[\s]*[=<>]+[\s]*([0-9.]+(?:e-?\d+)?)',
            r'p-value[=:\s]+([0-9.]+(?:e-?\d+)?)',
            r'p\s*value[=:\s]+([0-9.]+(?:e-?\d+)?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, result.lower())
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
        return None
    
    def _extract_hazard_ratio(self, result: str) -> Optional[float]:
        """Extract hazard ratio from execution result."""
        import re
        patterns = [
            r'hr[=:\s]+([0-9.]+)',
            r'hazard ratio[=:\s]+([0-9.]+)',
            r'exp\(coef\)[=:\s]+([0-9.]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, result.lower())
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
        return None
    
    def _generate_conclusion(self, hypothesis: str, result: str, 
                            p_value: Optional[float]) -> str:
        """Generate a conclusion from test results."""
        if p_value is not None:
            if p_value < 0.01:
                return f"Strong evidence supports the hypothesis (p={p_value:.4f})"
            elif p_value < 0.05:
                return f"Moderate evidence supports the hypothesis (p={p_value:.4f})"
            else:
                return f"No significant evidence for the hypothesis (p={p_value:.4f})"
        
        if "Error" in result or "Traceback" in result:
            return "Test could not be completed due to technical issues"
        
        return "Test completed but statistical significance unclear"
    
    def discover_single_cohort(self, project_id: str, cancer_type: str,
                                 progress_callback: Optional[Callable] = None) -> List[DiscoveryResult]:
        """Run discovery on a single cancer cohort."""
        print(f"\n🔬 Discovering {cancer_type} (Project: {project_id})")
        
        # Fetch and prepare data
        print("  📊 Fetching clinical data...")
        df = self.data_client.get_survival_analysis_ready_data(project_id)
        df = self.data_client.enrich_with_derived_features(df)
        
        if df.empty:
            print(f"  ⚠️ No data available for {project_id}")
            return []
        
        print(f"  ✓ Loaded {len(df)} samples")
        
        # Generate hypotheses
        print("  💡 Generating hypotheses...")
        hypotheses = self._generate_hypothesis_batch(
            df, cancer_type, self.config.hypothesis_per_cohort
        )
        print(f"  ✓ Generated {len(hypotheses)} hypotheses")
        
        # Test hypotheses in parallel
        results = []
        
        if self.config.parallel_workers > 1 and len(hypotheses) > 1:
            with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                futures = {
                    executor.submit(self._test_single_hypothesis, hyp, df, 0): hyp 
                    for hyp in hypotheses
                }
                
                for future in tqdm(as_completed(futures), total=len(hypotheses), 
                                 desc="  🧪 Testing", leave=False):
                    result = future.result()
                    results.append(result)
                    
                    if progress_callback:
                        progress_callback(result)
        else:
            for hypothesis in tqdm(hypotheses, desc="  🧪 Testing", leave=False):
                result = self._test_single_hypothesis(hypothesis, df, 0)
                results.append(result)
                
                if progress_callback:
                    progress_callback(result)
        
        # Log results
        for result in results:
            self._log_result(result, project_id)
        
        # Print summary
        significant = sum(1 for r in results if r.significant)
        print(f"  ✓ Discovery complete: {significant}/{len(results)} significant findings")
        
        return results
    
    def discover_multiple_cohorts(self, cohorts: Dict[str, str],
                                  parallel_cohorts: bool = False) -> Dict[str, List[DiscoveryResult]]:
        """Run discovery on multiple cancer cohorts."""
        all_results = {}
        
        cohort_items = list(cohorts.items())
        
        if parallel_cohorts and len(cohorts) > 1:
            # Parallel across cohorts (use ProcessPoolExecutor for true parallelism)
            with ProcessPoolExecutor(max_workers=min(4, len(cohorts))) as executor:
                futures = {
                    executor.submit(self._discover_cohort_wrapper, cancer_type, project_id): 
                    (cancer_type, project_id)
                    for cancer_type, project_id in cohort_items
                }
                
                for future in tqdm(as_completed(futures), total=len(cohorts), 
                                 desc="🔬 Cohort Discovery"):
                    cancer_type, project_id = futures[future]
                    try:
                        results = future.result()
                        all_results[project_id] = results
                    except Exception as e:
                        print(f"Error in {project_id}: {e}")
                        all_results[project_id] = []
        else:
            # Sequential processing
            for cancer_type, project_id in tqdm(cohort_items, desc="🔬 Cohort Discovery"):
                results = self.discover_single_cohort(project_id, cancer_type)
                all_results[project_id] = results
        
        return all_results
    
    def _discover_cohort_wrapper(self, cancer_type: str, project_id: str) -> List[DiscoveryResult]:
        """Wrapper for parallel cohort discovery."""
        # Need to reinitialize in subprocess
        engine = FastDiscoveryEngine(self.config, self.knowledge_fabric)
        return engine.discover_single_cohort(project_id, cancer_type)
    
    def _log_result(self, result: DiscoveryResult, project_id: str) -> None:
        """Log a discovery result."""
        entry = {
            "project_id": project_id,
            **result.to_dict()
        }
        self.ledger.append(entry)
        self.results.append(result)
    
    def get_significant_findings(self) -> List[DiscoveryResult]:
        """Get all statistically significant findings."""
        return [r for r in self.results if r.significant]
    
    def save_ledger(self, filepath: str) -> None:
        """Save discovery ledger to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.ledger, f, indent=2, default=str)
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of all discoveries."""
        total_tests = len(self.results)
        significant = self.get_significant_findings()
        
        report = {
            "total_hypotheses_tested": total_tests,
            "significant_findings": len(significant),
            "significance_rate": len(significant) / total_tests if total_tests > 0 else 0,
            "by_cancer_type": {},
            "top_findings": []
        }
        
        # Group by project
        project_results = {}
        for entry in self.ledger:
            pid = entry.get("project_id", "unknown")
            if pid not in project_results:
                project_results[pid] = []
            project_results[pid].append(entry)
        
        for pid, entries in project_results.items():
            sig_count = sum(1 for e in entries if e.get("significant"))
            report["by_cancer_type"][pid] = {
                "total": len(entries),
                "significant": sig_count,
                "rate": sig_count / len(entries) if entries else 0
            }
        
        # Top findings by p-value
        sorted_results = sorted(
            [r for r in self.results if r.p_value is not None],
            key=lambda x: x.p_value or 1.0
        )
        
        for r in sorted_results[:5]:
            report["top_findings"].append({
                "hypothesis": r.hypothesis[:100] + "..." if len(r.hypothesis) > 100 else r.hypothesis,
                "p_value": r.p_value,
                "hazard_ratio": r.hazard_ratio,
                "significant": r.significant
            })
        
        return report


def run_fast_discovery(
    cancer_types: List[str] = None,
    config: DiscoveryConfig = None,
    output_dir: str = "discoveries"
) -> Dict[str, Any]:
    """
    Main entry point for fast discovery.
    
    Args:
        cancer_types: List of cancer types to analyze (or None for all)
        config: Discovery configuration
        output_dir: Directory to save results
    
    Returns:
        Summary report of discoveries
    """
    # Setup
    config = config or DiscoveryConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load or create knowledge fabric
    kg_path = output_path / "knowledge_fabric.json"
    if kg_path.exists():
        print("📚 Loading existing knowledge fabric...")
        knowledge_fabric = KnowledgeFabric.load(str(kg_path))
    else:
        print("📚 Building knowledge fabric...")
        knowledge_fabric = create_default_knowledge_fabric(str(kg_path))
    
    # Initialize engine
    engine = FastDiscoveryEngine(config, knowledge_fabric)
    
    # Select cohorts
    if cancer_types:
        cohorts = {ct: EXPANDED_CANCER_PROJECTS[ct] 
                  for ct in cancer_types if ct in EXPANDED_CANCER_PROJECTS}
    else:
        # Default subset for faster testing
        cohorts = {
            "Skin Cancer": "TCGA-SKCM",
            "Breast Cancer": "TCGA-BRCA",
            "Lung Adenocarcinoma": "TCGA-LUAD",
        }
    
    print(f"\n🚀 Starting Fast Discovery on {len(cohorts)} cohorts")
    print(f"   Parallel workers: {config.parallel_workers}")
    print(f"   Hypotheses per cohort: {config.hypothesis_per_cohort}")
    
    # Run discovery
    start_time = time.time()
    results = engine.discover_multiple_cohorts(cohorts, parallel_cohorts=False)
    elapsed = time.time() - start_time
    
    # Save results
    ledger_file = output_path / f"discovery_ledger_{int(time.time())}.json"
    engine.save_ledger(str(ledger_file))
    
    # Generate report
    report = engine.generate_summary_report()
    report["execution_time_seconds"] = elapsed
    report["cohorts_analyzed"] = list(cohorts.keys())
    
    report_file = output_path / f"discovery_report_{int(time.time())}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 DISCOVERY COMPLETE")
    print(f"{'='*60}")
    print(f"Execution time: {elapsed:.1f} seconds")
    print(f"Total hypotheses tested: {report['total_hypotheses_tested']}")
    print(f"Significant findings: {report['significant_findings']}")
    print(f"Significance rate: {report['significance_rate']*100:.1f}%")
    
    if report['top_findings']:
        print(f"\n🏆 Top Findings:")
        for i, finding in enumerate(report['top_findings'][:3], 1):
            print(f"  {i}. p={finding['p_value']:.4f}: {finding['hypothesis'][:60]}...")
    
    print(f"\n💾 Results saved to:")
    print(f"   - {ledger_file}")
    print(f"   - {report_file}")
    
    return report


if __name__ == "__main__":
    # Test the discovery engine
    print("Testing Fast Discovery Engine...")
    
    config = DiscoveryConfig(
        max_iterations=2,
        parallel_workers=2,
        hypothesis_per_cohort=2,
        use_knowledge_fabric=True
    )
    
    # Run on a single cohort for testing
    report = run_fast_discovery(
        cancer_types=["Breast Cancer"],
        config=config,
        output_dir="test_discoveries"
    )
    
    print("\nTest complete!")
