import io
import sys
import traceback
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt

def execute_codeact(code: str, df_context: pd.DataFrame = None) -> str:
    """
    Executes dynamically generated python code from the agent.
    Captures stdout and tracebacks for self-correction.
    
    Args:
        code (str): The python code to execute.
        df_context (pd.DataFrame): The clinical data dataframe injected into the execution context.
        
    Returns:
        str: The captured standard output or the error traceback.
    """
    # Create a string buffer to capture stdout
    stdout_buffer = io.StringIO()
    # Save original stdout
    original_stdout = sys.stdout
    sys.stdout = stdout_buffer

    # Define the execution context (globals available to the agent's code)
    exec_globals = {
        'pd': pd,
        'KaplanMeierFitter': KaplanMeierFitter,
        'CoxPHFitter': CoxPHFitter,
        'plt': plt,
        'df': df_context
    }
    
    output = ""
    try:
        # Execute the code in the restricted environment
        exec(code, exec_globals)
        output = stdout_buffer.getvalue()
    except Exception:
        # Capture traceback for self-correction
        output = stdout_buffer.getvalue() + "\n" + traceback.format_exc()
    finally:
        # Restore stdout
        sys.stdout = original_stdout
        
    return output

if __name__ == "__main__":
    # Test the execution
    mock_code = \"\"\"
print("Checking dataframe columns:")
print(df.columns)
kmf = KaplanMeierFitter()
# Create mock time and event
df['time'] = [10, 20, 30, 40]
df['event'] = [1, 0, 1, 1]
kmf.fit(df['time'], event_observed=df['event'])
print("Median survival:", kmf.median_survival_time_)
\"\"\"
    
    mock_df = pd.DataFrame({"id": [1, 2, 3, 4]})
    result = execute_codeact(mock_code, df_context=mock_df)
    print("Code Execution Result:")
    print(result)
