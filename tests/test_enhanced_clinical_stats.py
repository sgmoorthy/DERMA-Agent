"""
Unit tests for the Enhanced Clinical Statistics Engine.
Verifies code execution safety, AST validation, and correct Analysis DSL template behavior.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.enhanced_clinical_stats import ASTCodeValidator, validate_code_safety, EnhancedStatsEngine

class TestASTCodeValidator(unittest.TestCase):
    """Tests the security boundary checks of the AST validator."""
    
    def test_safe_imports_allowed(self):
        """Whitelisted imports should pass validation."""
        safe_code = """
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import math

df = pd.DataFrame({'time': [1, 2, 3], 'event': [1, 0, 1]})
print(np.mean(df['time']))
"""
        self.assertTrue(validate_code_safety(safe_code))
        
    def test_unsafe_imports_blocked(self):
        """Unwhitelisted imports should be blocked."""
        unsafe_code_1 = "import os; os.system('echo hack')"
        unsafe_code_2 = "import sys; sys.exit(0)"
        unsafe_code_3 = "from subprocess import Popen"
        
        self.assertFalse(validate_code_safety(unsafe_code_1))
        self.assertFalse(validate_code_safety(unsafe_code_2))
        self.assertFalse(validate_code_safety(unsafe_code_3))
        
    def test_dangerous_builtins_blocked(self):
        """Dangerous built-in functions must be blocked."""
        dangerous_codes = [
            "eval('1+1')",
            "exec('import os')",
            "open('file.txt', 'r')",
            "__import__('os')",
            "getattr(object, 'attr')",
            "globals()",
            "locals()"
        ]
        for code in dangerous_codes:
            with self.subTest(code=code):
                self.assertFalse(validate_code_safety(code))
                
    def test_dunder_attributes_blocked(self):
        """Access to dunder attributes to prevent sandbox escape must be blocked."""
        dunder_codes = [
            "x = df.__class__",
            "x = [].__class__.__base__.__subclasses__()",
            "x = func.__globals__",
            "x = object.__subclasses__()"
        ]
        for code in dunder_codes:
            with self.subTest(code=code):
                self.assertFalse(validate_code_safety(code))


class TestAnalysisDSL(unittest.TestCase):
    """Tests the Analysis DSL templates on toy datasets."""
    
    def setUp(self):
        # Create a toy survival dataset
        # 2 groups (A and B), where Group B has significantly better survival
        np.random.seed(42)
        n = 100
        group = np.random.choice(['A', 'B'], size=n)
        
        # Hazard ratio: B has lower hazard (longer survival)
        hazard_a = 0.05
        hazard_b = 0.01
        
        time = []
        event = []
        for g in group:
            h = hazard_a if g == 'A' else hazard_b
            t = np.random.exponential(1.0 / h)
            # Right censor at 150 days
            if t > 150:
                time.append(150.0)
                event.append(0)
            else:
                time.append(t)
                event.append(1)
                
        self.df = pd.DataFrame({
            'time': time,
            'event': event,
            'group': group,
            'age': np.random.normal(60, 10, size=n),
            'gene_mutated': np.random.choice([0, 1], p=[0.8, 0.2], size=n)
        })
        
        self.engine = EnhancedStatsEngine(random_seed=42)
        
    def test_dsl_kaplan_meier(self):
        """Verifies Kaplan-Meier analysis through the JSON DSL."""
        dsl_query = {
            "template": "kaplan_meier",
            "parameters": {
                "time_col": "time",
                "event_col": "event",
                "group_col": "group"
            }
        }
        res = self.engine.execute_analysis_dsl(self.df, dsl_query)
        self.assertIn("p_value", res)
        self.assertIn("summary_text", res)
        # B survives longer, so p-value should be small (log-rank test)
        self.assertTrue(res["p_value"] < 0.05)
        
    def test_dsl_cox_regression(self):
        """Verifies Cox Proportional Hazards through the JSON DSL."""
        # Convert group to binary indicator
        df_copy = self.df.copy()
        df_copy['is_group_b'] = (df_copy['group'] == 'B').astype(int)
        
        dsl_query = {
            "template": "cox_regression",
            "parameters": {
                "time_col": "time",
                "event_col": "event",
                "predictor_cols": ["is_group_b", "age"]
            }
        }
        res = self.engine.execute_analysis_dsl(df_copy, dsl_query)
        self.assertIn("p_value", res)
        self.assertIn("hazard_ratio", res)
        self.assertIn("summary_text", res)
        # is_group_b has lower hazard (hazard ratio < 1.0)
        self.assertTrue(res["hazard_ratio"] < 1.0)
        
    def test_dsl_ml_survival(self):
        """Verifies ML classification-based survival predictions through the JSON DSL."""
        # Classify survival at 50 days threshold
        dsl_query = {
            "template": "ml_survival",
            "parameters": {
                "feature_cols": ["age", "gene_mutated"],
                "target_col": "group",
                "time_col": "time",
                "model_type": "random_forest"
            }
        }
        res = self.engine.execute_analysis_dsl(self.df, dsl_query)
        self.assertIn("c_index", res)
        self.assertIn("accuracy", res)
        self.assertIn("feature_importances", res)

if __name__ == "__main__":
    unittest.main()
