"""
Enhanced Clinical Statistics Engine for DermaMind.ai
Provides advanced survival analysis, ML models, and statistical testing capabilities.
"""

import io
import sys
import traceback
import warnings
import ast
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from lifelines.plotting import add_at_risk_counts
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import scipy.stats as stats

warnings.filterwarnings('ignore')


class SafetyValidationError(Exception):
    pass


class ASTCodeValidator(ast.NodeVisitor):
    """Statically inspects Python code to ensure only safe libraries are imported and dangerous functions are not called."""
    def __init__(self, allowed_modules=None):
        self.allowed_modules = allowed_modules or {
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'lifelines', 
            'sklearn', 'scipy', 'statsmodels', 'math'
        }
        
    def visit_Import(self, node):
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            if module_name not in self.allowed_modules:
                raise SafetyValidationError(f"Import of module '{module_name}' is not allowed for security reasons.")
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module.split('.')[0]
            if module_name not in self.allowed_modules:
                raise SafetyValidationError(f"Import from module '{module_name}' is not allowed for security reasons.")
        self.generic_visit(node)
        
    def visit_Call(self, node):
        # Block dangerous builtins
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in {'open', 'eval', 'exec', '__import__', 'getattr', 'setattr', 'delattr', 'locals', 'globals', 'compile', 'input'}:
                raise SafetyValidationError(f"Call to builtin function '{func_name}' is disallowed for security reasons.")
        elif isinstance(node.func, ast.Attribute):
            # Check for dunder methods
            attr_name = node.func.attr
            if attr_name.startswith('__'):
                raise SafetyValidationError(f"Access to private/dunder attribute '{attr_name}' is disallowed for security reasons.")
        self.generic_visit(node)
        
    def visit_Attribute(self, node):
        if node.attr.startswith('__'):
            raise SafetyValidationError(f"Access to private/dunder attribute '{node.attr}' is disallowed for security reasons.")
        self.generic_visit(node)


def _validate_code_safety_detail(code: str, allowed_modules=None) -> Tuple[bool, Optional[str]]:
    """Internal helper: validates code safety, returning (is_safe, error_message)."""
    try:
        tree = ast.parse(code)
        validator = ASTCodeValidator(allowed_modules)
        validator.visit(tree)
        return True, None
    except SafetyValidationError as e:
        return False, str(e)
    except SyntaxError as e:
        return False, f"Syntax Error: {str(e)}"
    except Exception as e:
        return False, f"Validation Error: {str(e)}"


def validate_code_safety(code: str, allowed_modules=None) -> bool:
    """Validate Python code for imports and calls to ensure it is sandboxed and secure.
    Returns True if code is safe, False otherwise.
    Use _validate_code_safety_detail() to also retrieve the error message.
    """
    is_safe, _ = _validate_code_safety_detail(code, allowed_modules)
    return is_safe


@dataclass
class SurvivalAnalysisResult:
    """Container for survival analysis results."""
    median_survival: Optional[float]
    p_value: Optional[float]
    hazard_ratio: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    c_index: Optional[float]
    log_rank_p: Optional[float]
    model_summary: str
    plot_data: Optional[Dict]


class EnhancedStatsEngine:
    """
    Enhanced statistical engine for clinical data analysis.
    Supports survival analysis, machine learning models, and visualization.
    """
    
    def __init__(self, random_seed: int = 42):
        self.execution_history = []
        self.random_seed = random_seed
        
    def execute_survival_analysis(self, code: str, df: pd.DataFrame) -> str:
        """
        Execute Python code for survival analysis in a controlled environment.
        Statically validates code safety before execution.
        
        Args:
            code: Python code string to execute
            df: DataFrame to provide as context
            
        Returns:
            Execution output as string
        """
        # Statically validate code safety first
        is_safe, error_msg = _validate_code_safety_detail(code)
        if not is_safe:
            output = f"Security/Validation Error: {error_msg}"
            self.execution_history.append({
                'code': code,
                'output': output,
                'success': False,
                'error': 'Safety Validation Failed'
            })
            return output

        stdout_buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = stdout_buffer
        
        # Extended execution context with all necessary imports
        exec_globals = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'KaplanMeierFitter': KaplanMeierFitter,
            'CoxPHFitter': CoxPHFitter,
            'logrank_test': logrank_test,
            'concordance_index': concordance_index,
            'add_at_risk_counts': add_at_risk_counts,
            'stats': stats,
            'df': df,
            'train_test_split': train_test_split,
            'RandomForestClassifier': RandomForestClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'LogisticRegression': LogisticRegression,
            'StandardScaler': StandardScaler,
            'LabelEncoder': LabelEncoder,
            'cross_val_score': cross_val_score,
            'classification_report': classification_report,
            'roc_auc_score': roc_auc_score,
            'roc_curve': roc_curve,
        }
        
        output = ""
        try:
            exec(code, exec_globals)
            output = stdout_buffer.getvalue()
            
            # Store successful execution
            self.execution_history.append({
                'code': code,
                'output': output,
                'success': True
            })
            
        except Exception as e:
            output = stdout_buffer.getvalue() + "\n" + traceback.format_exc()
            
            self.execution_history.append({
                'code': code,
                'output': output,
                'success': False,
                'error': str(e)
            })
            
        finally:
            sys.stdout = original_stdout
            
        return output

    def execute_analysis_dsl(self, df: pd.DataFrame, dsl_input: Any) -> Any:
        """
        Execute an analysis request specified in a structured DSL format, avoiding exec() entirely.
        Accepts df as first argument, then dsl_input (dict or JSON string).
        Supports both 'template' and 'analysis_type' keys for the operation name.
        Returns a structured dict with keys appropriate to the analysis type.
        """
        import json
        if isinstance(dsl_input, str):
            try:
                dsl = json.loads(dsl_input)
            except Exception as e:
                return {"error": f"DSL Parsing Error: Invalid JSON - {str(e)}"}
        elif isinstance(dsl_input, dict):
            dsl = dsl_input
        else:
            return {"error": "DSL Error: Input must be a dictionary or a JSON string."}

        # Support both 'template' (test-facing API) and 'analysis_type' (legacy key)
        analysis_type = dsl.get("template") or dsl.get("analysis_type")
        params = dsl.get("parameters", {})

        try:
            if analysis_type == "kaplan_meier":
                time_col = params.get("time_col", "time")
                event_col = params.get("event_col", "event")
                group_col = params.get("group_col")

                res = self.run_kaplan_meier(
                    df, time_col=time_col, event_col=event_col, group_col=group_col
                )
                result = {
                    "p_value": res.log_rank_p,
                    "median_survival": float(res.median_survival) if res.median_survival else None,
                    "summary_text": res.model_summary,
                    "plot_data": res.plot_data,
                }

            elif analysis_type == "cox_regression":
                time_col = params.get("time_col", "time")
                event_col = params.get("event_col", "event")
                predictor_cols = params.get("predictor_cols")

                res = self.run_cox_regression(
                    df, time_col=time_col, event_col=event_col, predictor_cols=predictor_cols
                )
                result = {
                    "p_value": res.p_value,
                    "hazard_ratio": res.hazard_ratio,
                    "c_index": res.c_index,
                    "summary_text": res.model_summary,
                    "plot_data": res.plot_data,
                }

            elif analysis_type == "ml_survival":
                feature_cols = params.get("feature_cols", [])
                target_col = params.get("target_col", "event")
                time_col = params.get("time_col", "time")
                model_type = params.get("model_type", "random_forest")

                ml_res = self.run_ml_survival_prediction(
                    df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    time_col=time_col,
                    model_type=model_type,
                )
                if "error" in ml_res:
                    result = {"error": ml_res["error"]}
                else:
                    result = {
                        # AUC used as discrimination proxy for c_index
                        "c_index": ml_res.get("auc_score"),
                        # CV mean AUC used as accuracy proxy
                        "accuracy": ml_res.get("cv_mean"),
                        "feature_importances": ml_res.get("feature_importance", {}),
                        "summary_text": (
                            f"ML Survival ({ml_res['model_type']}): "
                            f"AUC={ml_res['auc_score']:.4f}, "
                            f"CV Mean AUC={ml_res['cv_mean']:.4f} "
                            f"(+/- {ml_res['cv_std']:.4f})"
                        ),
                    }

            elif analysis_type == "univariate_test":
                feature_col = params.get("feature_col")
                target_col = params.get("target_col", "event")
                test_type = params.get("test_type", "t_test")

                if feature_col not in df.columns or target_col not in df.columns:
                    result = {"error": f"Columns {feature_col} or {target_col} not found in dataframe."}
                else:
                    df_clean = df[[feature_col, target_col]].dropna()
                    if len(df_clean) < 10:
                        result = {"error": "Insufficient data for univariate test."}
                    elif test_type == "t_test":
                        groups = df_clean[target_col].unique()
                        if len(groups) == 2:
                            g1 = df_clean[df_clean[target_col] == groups[0]][feature_col]
                            g2 = df_clean[df_clean[target_col] == groups[1]][feature_col]
                            stat, p = stats.ttest_ind(g1, g2, equal_var=False)
                            result = {
                                "statistic": float(stat), "p_value": float(p), "test_type": "t_test",
                                "summary_text": f"T-test {feature_col} by {target_col}: t={stat:.4f}, p={p:.6f}",
                            }
                        else:
                            result = {"error": "Target variable must have exactly 2 categories for t-test."}
                    elif test_type == "chi_square":
                        contingency = pd.crosstab(df_clean[feature_col], df_clean[target_col])
                        chi2, p, dof, _ = stats.chi2_contingency(contingency)
                        result = {
                            "statistic": float(chi2), "p_value": float(p), "test_type": "chi_square",
                            "summary_text": f"Chi-square {feature_col} vs {target_col}: chi2={chi2:.4f}, p={p:.6f}",
                        }
                    elif test_type == "correlation":
                        r, p = stats.pearsonr(df_clean[feature_col], df_clean[target_col])
                        result = {
                            "statistic": float(r), "p_value": float(p), "test_type": "correlation",
                            "summary_text": f"Pearson {feature_col} vs {target_col}: r={r:.4f}, p={p:.6f}",
                        }
                    else:
                        result = {"error": f"Unknown test_type '{test_type}'."}
            else:
                result = {"error": f"Unknown analysis type '{analysis_type}'."}

        except Exception as e:
            result = {"error": f"Execution Error: {str(e)}\n{traceback.format_exc()}"}

        self.execution_history.append({"dsl": dsl, "result": result})
        return result

    
    def run_kaplan_meier(self, df: pd.DataFrame, 
                        time_col: str = 'time',
                        event_col: str = 'event',
                        group_col: Optional[str] = None,
                        group_values: Optional[List] = None) -> SurvivalAnalysisResult:
        """
        Run Kaplan-Meier survival analysis.
        
        Args:
            df: DataFrame with survival data
            time_col: Column with time to event
            event_col: Column with event indicator (1=event, 0=censored)
            group_col: Optional column to group by
            group_values: Optional specific values to use from group_col
            
        Returns:
            SurvivalAnalysisResult with statistics
        """
        # Clean data
        df_clean = df.dropna(subset=[time_col, event_col])
        
        if len(df_clean) < 10:
            return SurvivalAnalysisResult(
                median_survival=None,
                p_value=None,
                hazard_ratio=None,
                confidence_interval=None,
                c_index=None,
                log_rank_p=None,
                model_summary="Insufficient data (<10 samples)",
                plot_data=None
            )
        
        kmf = KaplanMeierFitter()
        
        # Get times and events
        times = df_clean[time_col].values
        events = df_clean[event_col].values
        
        # Fit overall model
        kmf.fit(times, event_observed=events)
        median_survival = kmf.median_survival_time_
        
        results = {
            'median_survival': median_survival,
            'survival_times': kmf.survival_function_.index.tolist(),
            # Use .iloc[:, 0] — lifelines >= 0.27 names the column after the label, not 'KM_estimate'
            'survival_probs': kmf.survival_function_.iloc[:, 0].tolist()
        }
        
        p_value = None
        log_rank_p = None
        
        # Group comparison if specified
        if group_col and group_col in df_clean.columns:
            groups = df_clean[group_col].dropna().unique()
            
            if group_values:
                groups = [g for g in groups if g in group_values]
            
            if len(groups) >= 2:
                group_curves = {}
                
                for group in groups[:2]:  # Compare first two groups
                    mask = df_clean[group_col] == group
                    if mask.sum() >= 5:
                        kmf_group = KaplanMeierFitter()
                        kmf_group.fit(
                            df_clean.loc[mask, time_col],
                            event_observed=df_clean.loc[mask, event_col],
                            label=str(group)
                        )
                        group_curves[group] = {
                            'times': kmf_group.survival_function_.index.tolist(),
                            # Use .iloc[:, 0] for lifelines >= 0.27 compatibility
                            'probs': kmf_group.survival_function_.iloc[:, 0].tolist()
                        }
                
                # Log-rank test if we have two groups
                if len(group_curves) == 2:
                    g1, g2 = list(group_curves.keys())
                    mask1 = df_clean[group_col] == g1
                    mask2 = df_clean[group_col] == g2
                    
                    if mask1.sum() > 0 and mask2.sum() > 0:
                        try:
                            lr_result = logrank_test(
                                df_clean.loc[mask1, time_col],
                                df_clean.loc[mask2, time_col],
                                event_observed_A=df_clean.loc[mask1, event_col],
                                event_observed_B=df_clean.loc[mask2, event_col]
                            )
                            log_rank_p = lr_result.p_value
                        except:
                            pass
                
                results['group_curves'] = group_curves
        
        summary = f"""Kaplan-Meier Analysis:
- Total samples: {len(df_clean)}
- Events: {events.sum()}
- Censored: {len(df_clean) - events.sum()}
- Median survival: {median_survival:.1f} days (if reached)
"""
        if log_rank_p is not None:
            summary += f"- Log-rank test p-value: {log_rank_p:.4f}"
        
        return SurvivalAnalysisResult(
            median_survival=median_survival,
            p_value=p_value,
            hazard_ratio=None,
            confidence_interval=None,
            c_index=None,
            log_rank_p=log_rank_p,
            model_summary=summary,
            plot_data=results
        )
    
    def run_cox_regression(self, df: pd.DataFrame,
                          time_col: str = 'time',
                          event_col: str = 'event',
                          predictor_cols: List[str] = None) -> SurvivalAnalysisResult:
        """
        Run Cox Proportional Hazards regression.
        
        Args:
            df: DataFrame with survival data
            time_col: Time column
            event_col: Event indicator column
            predictor_cols: List of predictor columns
            
        Returns:
            SurvivalAnalysisResult with model statistics
        """
        if predictor_cols is None:
            # Auto-select numeric columns
            predictor_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            predictor_cols = [c for c in predictor_cols 
                            if c not in [time_col, event_col]]
        
        # Prepare data
        model_cols = [time_col, event_col] + predictor_cols
        df_model = df[model_cols].dropna()
        
        if len(df_model) < 20:
            return SurvivalAnalysisResult(
                median_survival=None,
                p_value=None,
                hazard_ratio=None,
                confidence_interval=None,
                c_index=None,
                log_rank_p=None,
                model_summary="Insufficient data (<20 samples)",
                plot_data=None
            )
        
        try:
            cph = CoxPHFitter()
            cph.fit(df_model, duration_col=time_col, event_col=event_col)
            
            # Extract key statistics
            summary = cph.summary
            
            # Get hazard ratios
            hrs = np.exp(summary['coef']).to_dict()
            p_values = summary['p'].to_dict()
            
            # Concordance index
            c_index = cph.concordance_index_
            
            # Build summary string — use public log_likelihood_ratio_test() API
            try:
                lrt_p = cph.log_likelihood_ratio_test().p_value
            except Exception:
                lrt_p = float('nan')
            
            summary_str = f"""Cox Proportional Hazards Model:
- Concordance index: {c_index:.3f}
- Log-likelihood ratio test p-value: {lrt_p:.4f}

Hazard Ratios:
"""
            for var, hr in hrs.items():
                p = p_values.get(var, 1.0)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                summary_str += f"  {var}: HR={hr:.2f}, p={p:.4f} {sig}\n"
            
            # Return first predictor's stats (highest significance or first listed)
            first_var = list(hrs.keys())[0] if hrs else None
            return SurvivalAnalysisResult(
                median_survival=None,
                p_value=p_values.get(first_var) if first_var else None,
                hazard_ratio=hrs.get(first_var) if first_var else None,
                confidence_interval=None,
                c_index=c_index,
                log_rank_p=None,
                model_summary=summary_str,
                plot_data={'hazard_ratios': hrs, 'p_values': p_values}
            )
            
        except Exception as e:
            return SurvivalAnalysisResult(
                median_survival=None,
                p_value=None,
                hazard_ratio=None,
                confidence_interval=None,
                c_index=None,
                log_rank_p=None,
                model_summary=f"Model fitting failed: {str(e)}",
                plot_data=None
            )
    
    def run_ml_survival_prediction(self, df: pd.DataFrame,
                                    feature_cols: List[str],
                                    target_col: str = 'event',
                                    time_col: str = 'time',
                                    model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Run machine learning prediction for survival outcomes.
        
        Args:
            df: DataFrame
            feature_cols: Features to use
            target_col: Target variable (e.g., event within X days)
            time_col: Time column
            model_type: 'random_forest', 'gradient_boosting', or 'logistic'
            
        Returns:
            Dictionary with model results
        """
        # Prepare data
        model_cols = feature_cols + [target_col]
        df_model = df[model_cols].dropna()
        
        if len(df_model) < 30:
            return {'error': 'Insufficient data for ML model'}
        
        X = df_model[feature_cols]
        y = df_model[target_col]
        
        # Handle categorical features
        X = pd.get_dummies(X, drop_first=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Select model
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_seed)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=self.random_seed)
        else:
            model = LogisticRegression(random_state=self.random_seed, max_iter=1000)
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_prob)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(X.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            importance = dict(zip(X.columns, np.abs(model.coef_[0])))
        else:
            importance = {}
        
        return {
            'model_type': model_type,
            'auc_score': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(sorted(importance.items(), 
                                             key=lambda x: x[1], reverse=True)[:10]),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def compare_cohorts(self, cohorts_data: Dict[str, pd.DataFrame],
                       time_col: str = 'time',
                       event_col: str = 'event') -> Dict[str, Any]:
        """
        Compare survival across multiple cohorts.
        
        Args:
            cohorts_data: Dictionary of cohort name -> DataFrame
            time_col: Time column name
            event_col: Event column name
            
        Returns:
            Comparison statistics
        """
        results = {}
        
        for cohort_name, df in cohorts_data.items():
            km_result = self.run_kaplan_meier(df, time_col, event_col)
            results[cohort_name] = {
                'median_survival': km_result.median_survival,
                'n_samples': len(df),
                'n_events': df[event_col].sum() if event_col in df.columns else 0
            }
        
        return results


def quick_survival_summary(df: pd.DataFrame, 
                          time_col: str = 'time',
                          event_col: str = 'event') -> str:
    """
    Generate a quick text summary of survival data.
    
    Args:
        df: DataFrame with survival data
        time_col: Time column
        event_col: Event column
        
    Returns:
        Summary string
    """
    if time_col not in df.columns or event_col not in df.columns:
        return f"Missing required columns: {time_col}, {event_col}"
    
    df_clean = df.dropna(subset=[time_col, event_col])
    
    if len(df_clean) == 0:
        return "No valid survival data"
    
    n_total = len(df_clean)
    n_events = int(df_clean[event_col].sum())
    n_censored = n_total - n_events
    
    # Try Kaplan-Meier
    try:
        kmf = KaplanMeierFitter()
        kmf.fit(df_clean[time_col], event_observed=df_clean[event_col])
        median_survival = kmf.median_survival_time_
        median_str = f"{median_survival:.1f} days" if median_survival else "not reached"
    except:
        median_str = "could not calculate"
    
    summary = f"""Survival Data Summary:
- Total samples: {n_total}
- Events (deaths): {n_events} ({100*n_events/n_total:.1f}%)
- Censored: {n_censored} ({100*n_censored/n_total:.1f}%)
- Median survival: {median_str}
- Follow-up range: {df_clean[time_col].min():.0f} - {df_clean[time_col].max():.0f} days
"""
    
    return summary


if __name__ == "__main__":
    # Test the enhanced stats engine
    print("Testing Enhanced Clinical Stats Engine...")
    
    engine = EnhancedStatsEngine()
    
    # Create mock data
    np.random.seed(42)
    n = 100
    mock_df = pd.DataFrame({
        'time': np.random.exponential(500, n) + 100,
        'event': np.random.binomial(1, 0.4, n),
        'age': np.random.normal(60, 10, n),
        'stage': np.random.choice(['I', 'II', 'III', 'IV'], n),
        'treatment': np.random.choice(['A', 'B'], n)
    })
    
    print("\n1. Kaplan-Meier Test:")
    km_result = engine.run_kaplan_meier(mock_df, group_col='stage')
    print(km_result.model_summary)
    
    print("\n2. Cox Regression Test:")
    # Create numeric stage
    mock_df['stage_num'] = mock_df['stage'].map({'I': 1, 'II': 2, 'III': 3, 'IV': 4})
    cox_result = engine.run_cox_regression(mock_df, predictor_cols=['age', 'stage_num'])
    print(cox_result.model_summary)
    
    print("\n3. ML Prediction Test:")
    ml_result = engine.run_ml_survival_prediction(
        mock_df, 
        feature_cols=['age', 'stage_num'],
        target_col='event'
    )
    print(f"AUC: {ml_result.get('auc_score', 'N/A')}")
    print(f"Feature importance: {ml_result.get('feature_importance', {})}")
    
    print("\n4. Code Execution Test:")
    test_code = """
print("Testing code execution...")
kmf = KaplanMeierFitter()
kmf.fit(df['time'], event_observed=df['event'])
print(f"Median survival: {kmf.median_survival_time_:.1f} days")
print(f"C-index: {kmf.concordance_index_:.3f}")
"""
    output = engine.execute_survival_analysis(test_code, mock_df)
    print(output)
    
    print("\nAll tests complete!")
