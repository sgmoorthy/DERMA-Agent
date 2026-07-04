"""
Fast Parallel Discovery Engine for DermaMind.ai
Implements high-performance hypothesis generation and testing with parallel execution.
"""

import asyncio
import json
import os
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from derma_agent.secrets import get_secret
from tools.enhanced_clinical_stats import EnhancedStatsEngine
from tools.enhanced_data_client import EXPANDED_CANCER_PROJECTS, get_data_client
from tools.knowledge_fabric import KnowledgeFabric, create_default_knowledge_fabric


@dataclass
class DiscoveryResult:
    """Result of a discovery iteration."""

    hypothesis: str
    test_code: str
    execution_result: str
    conclusion: str
    p_value: Optional[float] = None
    adjusted_p_value: Optional[float] = None
    hazard_ratio: Optional[float] = None
    confidence_interval: Optional[tuple] = None
    significant: bool = False
    raw_significant: bool = False
    iteration: int = 0
    execution_time: float = 0.0
    kg_confidence_score: Optional[float] = None
    kg_evidence_path: Optional[str] = None
    fdr_method: str = "benjamini-hochberg"
    fdr_alpha: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "hypothesis": self.hypothesis,
            "test_code": self.test_code,
            "execution_result": self.execution_result,
            "conclusion": self.conclusion,
            "p_value": self.p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "hazard_ratio": self.hazard_ratio,
            "confidence_interval": self.confidence_interval,
            "significant": self.significant,
            "raw_significant": self.raw_significant,
            "iteration": self.iteration,
            "execution_time": self.execution_time,
            "kg_confidence_score": self.kg_confidence_score,
            "kg_evidence_path": self.kg_evidence_path,
            "fdr_method": self.fdr_method,
            "fdr_alpha": self.fdr_alpha,
        }


@dataclass
class DiscoveryConfig:
    """Configuration for the discovery engine."""

    experiment_name: str = "default_experiment"
    schema_version: str = "1.0"
    random_seed: int = 42
    max_iterations: int = 3
    parallel_workers: int = 4
    hypothesis_per_cohort: int = 3
    significance_threshold: float = 0.05
    use_knowledge_fabric: bool = True
    auto_correct_errors: bool = True
    save_intermediate: bool = True
    timeout_seconds: int = 120

    # Data Layer
    missing_data_strategy: str = (
        "impute_median"  # 'drop', 'impute_median', 'impute_mean'
    )
    censoring_treatment: str = "right_censored"
    local_mirror_dir: Optional[str] = None

    # LLM
    llm_model: str = "gpt-4-turbo-preview"
    temperature: float = 0.2

    @classmethod
    def load_from_yaml(cls, filepath: str) -> "DiscoveryConfig":
        """Load configuration from a YAML file."""
        try:
            import yaml

            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
            # Filter keys to match dataclass fields
            valid_keys = {
                k: v for k, v in data.items() if k in cls.__dataclass_fields__
            }
            return cls(**valid_keys)
        except ImportError:
            print("PyYAML not installed, attempting JSON fallback")
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                valid_keys = {
                    k: v for k, v in data.items() if k in cls.__dataclass_fields__
                }
                return cls(**valid_keys)
            except Exception as e:
                print(f"Error loading JSON fallback: {e}")
                return cls()
        except Exception as e:
            print(f"Error loading config from {filepath}: {e}. Using defaults.")
            return cls()

    def save_to_yaml(self, filepath: str) -> None:
        """Save configuration to a YAML file."""
        data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        try:
            import yaml

            with open(filepath, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False)
        except ImportError:
            print("PyYAML not installed, saving as JSON")
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)


class FastDiscoveryEngine:
    """
    High-performance discovery engine with parallel hypothesis testing.
    """

    def __init__(
        self, config: DiscoveryConfig = None, knowledge_fabric: KnowledgeFabric = None
    ):
        self.config = config or DiscoveryConfig()

        # Enforce deterministic random seed
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        self.knowledge_fabric = knowledge_fabric
        self.data_client = get_data_client()
        self.stats_engine = EnhancedStatsEngine(random_seed=self.config.random_seed)
        self.ledger: List[Dict] = []
        self.results: List[DiscoveryResult] = []

        # Initialize LLM
        self.llm = self._init_llm()

    def _init_llm(self) -> Optional[ChatOpenAI]:
        """Initialize the LLM with API key."""
        api_key = get_secret("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not set. Using mock mode.")
            return None
        try:
            return ChatOpenAI(
                model="gpt-4-turbo-preview", temperature=0.2, api_key=api_key
            )
        except Exception as e:
            print(f"Failed to initialize LLM: {e}")
            return None

    def _generate_hypothesis_batch(
        self, df: pd.DataFrame, cancer_type: str, n_hypotheses: int
    ) -> List[str]:
        """Generate multiple hypotheses in a single LLM call."""
        columns = df.columns.tolist()

        # Use knowledge fabric if available
        knowledge_context = ""
        if self.knowledge_fabric and self.config.use_knowledge_fabric:
            # Query relevant knowledge
            cancer_node_id = cancer_type.replace(" ", "_")
            related = self.knowledge_fabric.query_pattern({"label": "Gene"})

            if related:
                genes = [r["node"].id for r in related[:5]]
                knowledge_context = (
                    f"\nRelevant genes for this cancer type: {', '.join(genes)}"
                )

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
            for line in content.split("\n"):
                line = line.strip()
                if line and (
                    line[0].isdigit() or line.startswith("-") or line.startswith("*")
                ):
                    # Remove numbering/bullets
                    hyp = line.lstrip("0123456789.-*) ").strip()
                    if hyp and len(hyp) > 20:
                        hypotheses.append(hyp)

            return hypotheses[:n_hypotheses]
        else:
            # Mock hypotheses
            return [
                f"Tumor stage is associated with survival time in {cancer_type}",
                f"Age at diagnosis predicts survival outcomes in {cancer_type}",
                f"Gender differences exist in survival patterns for {cancer_type}",
            ][:n_hypotheses]

    def _test_single_hypothesis(
        self, hypothesis: str, df: pd.DataFrame, iteration: int
    ) -> DiscoveryResult:
        """Test a single hypothesis."""
        start_time = time.time()

        # Generate test code
        code = self._generate_test_code(hypothesis, df, iteration)

        # Compute Knowledge Fabric confidence if available
        kg_score = None
        kg_path = None
        if self.knowledge_fabric and self.config.use_knowledge_fabric:
            try:
                kg_info = self.knowledge_fabric.calculate_hypothesis_prior_score(
                    hypothesis
                )
                kg_score = kg_info.get("score")
                kg_path = kg_info.get("evidence_path")
            except Exception as e:
                print(f"Error querying knowledge fabric for hypothesis score: {e}")

        # Execute test
        try:
            result = self.stats_engine.execute_survival_analysis(code, df)
            execution_time = time.time() - start_time

            # Parse results
            p_value = self._extract_p_value(result)
            hr = self._extract_hazard_ratio(result)

            # Generate conclusion
            conclusion = self._generate_conclusion(hypothesis, result, p_value)

            raw_significant = (
                p_value is not None and p_value < self.config.significance_threshold
            )
            return DiscoveryResult(
                hypothesis=hypothesis,
                test_code=code,
                execution_result=result,
                conclusion=conclusion,
                p_value=p_value,
                hazard_ratio=hr,
                significant=raw_significant,
                raw_significant=raw_significant,
                iteration=iteration,
                execution_time=execution_time,
                kg_confidence_score=kg_score,
                kg_evidence_path=kg_path,
                fdr_alpha=self.config.significance_threshold,
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
                execution_time=execution_time,
                kg_confidence_score=kg_score,
                kg_evidence_path=kg_path,
                fdr_alpha=self.config.significance_threshold,
            )

    def _generate_test_code(
        self, hypothesis: str, df: pd.DataFrame, iteration: int
    ) -> str:
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
        has_stage = "tumor_stage" in columns
        has_age = "age_years" in columns

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
            r"p[\s]*[=<>]+[\s]*([0-9.]+(?:e-?\d+)?)",
            r"p-value[=:\s]+([0-9.]+(?:e-?\d+)?)",
            r"p\s*value[=:\s]+([0-9.]+(?:e-?\d+)?)",
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
            r"hr[=:\s]+([0-9.]+)",
            r"hazard ratio[=:\s]+([0-9.]+)",
            r"exp\(coef\)[=:\s]+([0-9.]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, result.lower())
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
        return None

    def _generate_conclusion(
        self, hypothesis: str, result: str, p_value: Optional[float]
    ) -> str:
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

    @staticmethod
    def _benjamini_hochberg(
        p_values: List[float], alpha: float
    ) -> tuple[List[float], List[bool]]:
        """Apply Benjamini-Hochberg FDR correction to a list of raw p-values."""
        if not p_values:
            return [], []

        pvals = np.asarray(p_values, dtype=float)
        n_tests = len(pvals)
        order = np.argsort(pvals)
        ranked = pvals[order]

        adjusted_ranked = np.empty(n_tests, dtype=float)
        running_min = 1.0
        for idx in range(n_tests - 1, -1, -1):
            rank = idx + 1
            candidate = ranked[idx] * n_tests / rank
            running_min = min(running_min, candidate)
            adjusted_ranked[idx] = min(running_min, 1.0)

        adjusted = np.empty(n_tests, dtype=float)
        adjusted[order] = adjusted_ranked

        thresholds = alpha * (np.arange(1, n_tests + 1) / n_tests)
        passing = np.where(ranked <= thresholds)[0]
        significant_ranked = np.zeros(n_tests, dtype=bool)
        if len(passing) > 0:
            significant_ranked[: passing[-1] + 1] = True

        significant = np.empty(n_tests, dtype=bool)
        significant[order] = significant_ranked
        return adjusted.tolist(), significant.tolist()

    def _apply_global_fdr_correction(self) -> None:
        """Recompute BH/FDR significance across the full agent session."""
        tested_results = [
            (idx, result)
            for idx, result in enumerate(self.results)
            if result.p_value is not None
        ]
        tested_count = len(tested_results)

        for result in self.results:
            result.fdr_method = "benjamini-hochberg"
            result.fdr_alpha = self.config.significance_threshold
            if result.p_value is None:
                result.adjusted_p_value = None
                result.significant = False

        if tested_count:
            adjusted_p_values, significant_flags = self._benjamini_hochberg(
                [
                    float(result.p_value)
                    for _, result in tested_results
                    if result.p_value is not None
                ],
                self.config.significance_threshold,
            )
            for (_, result), adjusted_p, significant in zip(
                tested_results, adjusted_p_values, significant_flags
            ):
                result.adjusted_p_value = float(adjusted_p)
                result.significant = bool(significant)

        for idx, entry in enumerate(self.ledger):
            result = self.results[idx]
            entry.update(
                {
                    "p_value": result.p_value,
                    "adjusted_p_value": result.adjusted_p_value,
                    "significant": result.significant,
                    "raw_significant": result.raw_significant,
                    "fdr_method": result.fdr_method,
                    "fdr_alpha": result.fdr_alpha,
                    "global_test_count": tested_count,
                }
            )

    def discover_single_cohort(
        self,
        project_id: str,
        cancer_type: str,
        progress_callback: Optional[Callable] = None,
    ) -> List[DiscoveryResult]:
        """Run discovery on a single cancer cohort."""
        print(f"\n🔬 Discovering {cancer_type} (Project: {project_id})")

        # Fetch and prepare data
        print("  📊 Fetching clinical data...")
        df = self.data_client.get_survival_analysis_ready_data(
            project_id=project_id,
            missing_data_strategy=self.config.missing_data_strategy,
            censoring_treatment=self.config.censoring_treatment,
            local_mirror_dir=self.config.local_mirror_dir,
        )
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
            with ThreadPoolExecutor(
                max_workers=self.config.parallel_workers
            ) as executor:
                futures = {
                    executor.submit(self._test_single_hypothesis, hyp, df, 0): hyp
                    for hyp in hypotheses
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(hypotheses),
                    desc="  🧪 Testing",
                    leave=False,
                ):
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
        print(
            f"  ✓ Discovery complete: {significant}/{len(results)} significant findings"
        )

        return results

    def discover_multiple_cohorts(
        self, cohorts: Dict[str, str], parallel_cohorts: bool = False
    ) -> Dict[str, List[DiscoveryResult]]:
        """Run discovery on multiple cancer cohorts."""
        all_results = {}

        cohort_items = list(cohorts.items())

        if parallel_cohorts and len(cohorts) > 1:
            # Parallel across cohorts (use ProcessPoolExecutor for true parallelism)
            with ProcessPoolExecutor(max_workers=min(4, len(cohorts))) as executor:
                futures = {
                    executor.submit(
                        self._discover_cohort_wrapper, cancer_type, project_id
                    ): (cancer_type, project_id)
                    for cancer_type, project_id in cohort_items
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(cohorts),
                    desc="🔬 Cohort Discovery",
                ):
                    cancer_type, project_id = futures[future]
                    try:
                        results = future.result()
                        all_results[project_id] = results
                    except Exception as e:
                        print(f"Error in {project_id}: {e}")
                        all_results[project_id] = []
        else:
            # Sequential processing
            for cancer_type, project_id in tqdm(
                cohort_items, desc="🔬 Cohort Discovery"
            ):
                results = self.discover_single_cohort(project_id, cancer_type)
                all_results[project_id] = results

        return all_results

    def _discover_cohort_wrapper(
        self, cancer_type: str, project_id: str
    ) -> List[DiscoveryResult]:
        """Wrapper for parallel cohort discovery."""
        # Need to reinitialize in subprocess
        engine = FastDiscoveryEngine(self.config, self.knowledge_fabric)
        return engine.discover_single_cohort(project_id, cancer_type)

    def _log_result(self, result: DiscoveryResult, project_id: str) -> None:
        """Log a discovery result with experiment metadata."""
        self.results.append(result)
        entry = {
            "project_id": project_id,
            "experiment_name": self.config.experiment_name,
            "schema_version": self.config.schema_version,
            "random_seed": self.config.random_seed,
            "llm_model": self.config.llm_model,
            "knowledge_fabric_version": self.knowledge_fabric.version
            if (self.knowledge_fabric and hasattr(self.knowledge_fabric, "version"))
            else "1.0",
            **result.to_dict(),
        }
        self.ledger.append(entry)
        self._apply_global_fdr_correction()

    def get_significant_findings(self) -> List[DiscoveryResult]:
        """Get all statistically significant findings."""
        return [r for r in self.results if r.significant]

    def save_ledger(self, filepath: str) -> None:
        """Save discovery ledger to file wrapped with metadata and provenance."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        provenance = {}
        if hasattr(self.data_client, "get_provenance_metadata"):
            provenance = self.data_client.get_provenance_metadata()

        output_data = {
            "schema_version": self.config.schema_version,
            "experiment_name": self.config.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                k: v for k, v in self.config.__dict__.items() if not k.startswith("_")
            },
            "provenance": provenance,
            "ledger": self.ledger,
        }
        with open(filepath, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of all discoveries."""
        total_tests = len(self.results)
        significant = self.get_significant_findings()
        raw_significant = sum(1 for r in self.results if r.raw_significant)
        tested_count = sum(1 for r in self.results if r.p_value is not None)

        report = {
            "schema_version": self.config.schema_version,
            "experiment_name": self.config.experiment_name,
            "config": {
                k: v for k, v in self.config.__dict__.items() if not k.startswith("_")
            },
            "fdr_method": "benjamini-hochberg",
            "fdr_alpha": self.config.significance_threshold,
            "global_test_count": tested_count,
            "total_hypotheses_tested": total_tests,
            "significant_findings": len(significant),
            "raw_significant_findings": raw_significant,
            "significance_rate": len(significant) / total_tests
            if total_tests > 0
            else 0,
            "by_cancer_type": {},
            "top_findings": [],
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
            raw_sig_count = sum(1 for e in entries if e.get("raw_significant"))
            report["by_cancer_type"][pid] = {
                "total": len(entries),
                "significant": sig_count,
                "raw_significant": raw_sig_count,
                "rate": sig_count / len(entries) if entries else 0,
            }

        # Top findings by adjusted p-value first, raw p-value as fallback
        sorted_results = sorted(
            [r for r in self.results if r.p_value is not None],
            key=lambda x: (
                x.adjusted_p_value
                if x.adjusted_p_value is not None
                else x.p_value or 1.0,
                x.p_value or 1.0,
            ),
        )

        for r in sorted_results[:5]:
            report["top_findings"].append(
                {
                    "hypothesis": r.hypothesis[:100] + "..."
                    if len(r.hypothesis) > 100
                    else r.hypothesis,
                    "p_value": r.p_value,
                    "adjusted_p_value": r.adjusted_p_value,
                    "hazard_ratio": r.hazard_ratio,
                    "significant": r.significant,
                    "raw_significant": r.raw_significant,
                }
            )

        return report


def run_fast_discovery(
    cancer_types: List[str] = None,
    config: DiscoveryConfig = None,
    output_dir: str = "discoveries",
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
        cohorts = {
            ct: EXPANDED_CANCER_PROJECTS[ct]
            for ct in cancer_types
            if ct in EXPANDED_CANCER_PROJECTS
        }
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
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"📊 DISCOVERY COMPLETE")
    print(f"{'=' * 60}")
    print(f"Execution time: {elapsed:.1f} seconds")
    print(f"Total hypotheses tested: {report['total_hypotheses_tested']}")
    print(f"Significant findings: {report['significant_findings']}")
    print(f"Significance rate: {report['significance_rate'] * 100:.1f}%")

    if report["top_findings"]:
        print(f"\n🏆 Top Findings:")
        for i, finding in enumerate(report["top_findings"][:3], 1):
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
        use_knowledge_fabric=True,
    )

    # Run on a single cohort for testing
    report = run_fast_discovery(
        cancer_types=["Breast Cancer"], config=config, output_dir="test_discoveries"
    )

    print("\nTest complete!")
