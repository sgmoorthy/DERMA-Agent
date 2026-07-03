import ast
from typing import Tuple, Optional
from derma_core.actions.safety_policy import SafetyPolicy

class ASTSecurityInspector(ast.NodeVisitor):
    """
    Statically inspects Python code to ensure only safe libraries are imported, 
    no blocked builtins are accessed, and private/dunder features are not touched.
    """
    def visit_Import(self, node):
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            if module_name not in SafetyPolicy.ALLOWED_MODULES:
                raise ValueError(f"Unauthorized import of module '{module_name}'")
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module.split('.')[0]
            if module_name not in SafetyPolicy.ALLOWED_MODULES:
                raise ValueError(f"Unauthorized import from module '{module_name}'")
        self.generic_visit(node)
        
    def visit_Call(self, node):
        # Block dangerous builtins
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in SafetyPolicy.BLOCKED_BUILTINS:
                raise ValueError(f"Unauthorized function call to '{func_name}'")
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name.startswith('__'):
                raise ValueError(f"Unauthorized access to private/dunder attribute '{attr_name}'")
        self.generic_visit(node)
        
    def visit_Attribute(self, node):
        if node.attr.startswith('__'):
            raise ValueError(f"Unauthorized access to private/dunder attribute '{node.attr}'")
        self.generic_visit(node)


class CriticAgent:
    """
    The Critic evaluates generated code for security vulnerabilities 
    and checks scientific/logical validity BEFORE execution.
    """
    def __init__(self):
        pass
        
    def evaluate_code(self, code: str, expected_columns: list = None) -> Tuple[bool, Optional[str]]:
        """
        Verify safety and basic logical structure.
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, validation_report_or_error)
        """
        # 1. Security Check via AST static inspection
        try:
            tree = ast.parse(code)
            inspector = ASTSecurityInspector()
            inspector.visit(tree)
        except SyntaxError as e:
            return False, f"Syntax Error: {str(e)}"
        except ValueError as e:
            return False, f"Security Violation: {str(e)}"
        except Exception as e:
            return False, f"Validation Failure: {str(e)}"
            
        # 2. Basic Scientific/Logical Validity Checks
        # Verify it tries to run a fitter or do pandas analysis
        code_lower = code.lower()
        if "fit(" not in code_lower and "fitter" not in code_lower and "groupby(" not in code_lower:
            return False, "Logical Warning: Code does not appear to perform any fitting or group analysis."
            
        # If expected columns are given, check if code references a column that is not present
        if expected_columns:
            # Simple check: search if any column names are in the code
            col_found = False
            for col in expected_columns:
                if col in code:
                    col_found = True
                    break
            if not col_found:
                return False, f"Logical Warning: Code does not use any of the available clinical columns: {expected_columns}"
                
        return True, "Code passed security and logic checks successfully."
