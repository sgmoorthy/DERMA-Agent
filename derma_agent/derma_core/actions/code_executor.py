import sys
import io
from contextlib import redirect_stdout
from derma_core.actions.safety_policy import SafetyPolicy

import builtins

# Prepare whitelisted builtins dictionary
def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root_module = name.split('.')[0]
    if root_module not in SafetyPolicy.ALLOWED_MODULES:
        raise ImportError(f"Import of module '{root_module}' is not allowed by Safety Policy.")
    return builtins.__import__(name, globals, locals, fromlist, level)

SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "range": range,
    "list": list,
    "dict": dict,
    "set": set,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "sum": sum,
    "max": max,
    "min": min,
    "abs": abs,
    "round": round,
    "enumerate": enumerate,
    "zip": zip,
    "any": any,
    "all": all,
    "isinstance": isinstance,
    "__import__": safe_import,
}

class CodeExecutor:
    @staticmethod
    def execute(code: str, context: dict) -> str:
        """
        Executes dynamically generated statistical python code within a strictly restricted sandbox.
        Captures standard output and handles exceptions cleanly.
        """
        # Inject whitelisted analytical modules into execution scope
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from lifelines import KaplanMeierFitter, CoxPHFitter
        
        restricted_globals = {
            "__builtins__": SAFE_BUILTINS,
            "pd": pd,
            "np": np,
            "plt": plt,
            "KaplanMeierFitter": KaplanMeierFitter,
            "CoxPHFitter": CoxPHFitter,
        }
        
        # Inject caller context (e.g. dataframes)
        restricted_locals = context.copy()
        
        output = io.StringIO()
        with redirect_stdout(output):
            try:
                exec(code, restricted_globals, restricted_locals)
            except Exception as e:
                # Capture standard traceback info in a clean way
                return f"Execution Error: {type(e).__name__} - {str(e)}"
                
        return output.getvalue()
