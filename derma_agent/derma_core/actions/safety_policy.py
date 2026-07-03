"""
Safety Policy Module
Defines execution sandboxing constraints and import whitelists.
"""

import os
from typing import Set

class SafetyPolicy:
    # Whitelisted modules for statistical and clinical analysis
    ALLOWED_MODULES: Set[str] = {
        "pandas", "numpy", "matplotlib", "seaborn", 
        "lifelines", "sklearn", "scipy", "statsmodels", "math"
    }

    # Blocked built-in functions
    BLOCKED_BUILTINS: Set[str] = {
        "open", "eval", "exec", "__import__", "getattr", 
        "setattr", "delattr", "locals", "globals", "compile", "input"
    }

    # Restrict file output modifications. Allow writing only to discoveries/ or data/
    ALLOWED_WRITE_PATHS: Set[str] = {
        "discoveries", "data"
    }

    @classmethod
    def is_path_write_allowed(cls, filepath: str) -> bool:
        """
        Validate whether a filepath resides inside allowed folders and prevents path traversal (e.g., ../).
        """
        resolved = os.path.abspath(filepath)
        current_dir = os.path.abspath(os.getcwd())
        
        # Prevent accessing files outside current workspace
        if not resolved.startswith(current_dir):
            return False
            
        # Extract relative path components
        rel_path = os.path.relpath(resolved, current_dir)
        path_parts = rel_path.split(os.sep)
        
        # Check if the destination folder is in whitelisted output pathways
        if path_parts and path_parts[0] in cls.ALLOWED_WRITE_PATHS:
            return True
            
        return False
