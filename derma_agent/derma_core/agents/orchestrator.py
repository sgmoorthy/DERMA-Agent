import os
import json
from typing import TypedDict, List, Dict, Any, Annotated
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from tools.clinical_stats import execute_codeact
from tools.gdc_client import fetch_tcga_clinical_data

# Define the Knowledge State Data Structure (KSDS)
class AgentState(TypedDict, total=False):
    messages: Annotated[List[Any], operator.add]
    current_hypothesis: str
    python_code: str
    execution_result: str
    ledger: List[Dict[str, str]]
    iteration: int
    data_context: Any # pd.DataFrame
    project_id: str
    cancer_type: str

# To use Anthropic or Gemini, uncomment the respective lines and import:
# from langchain_anthropic import ChatAnthropic
# llm = ChatAnthropic(model="claude-3-opus-20240229")
# OR
# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Defaulting to OpenAI (requires OPENAI_API_KEY environment variable)
def get_llm():
    try:
        return ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return None

def fetch_data_node(state: AgentState):
    """Initial node to fetch clinical data."""
    project_id = state.get("project_id", "TCGA-SKCM")
    cancer_type = state.get("cancer_type", "Skin Cancer")
    print(f"Agent: Fetching {cancer_type} clinical data...")
    df = fetch_tcga_clinical_data(project_id=project_id)
    return {"data_context": df}

def formulate_hypothesis_node(state: AgentState):
    """The Brain reasons and formulates a hypothesis based on data."""
    print("Agent: Formulating hypothesis...")
    llm = get_llm()
    if not llm:
        return {"current_hypothesis": "Mock Hypothesis: High lymphocyte density correlates with survival."}
        
    df = state.get("data_context")
    columns = df.columns.tolist() if df is not None else []
    
    cancer_type = state.get("cancer_type", "Skin Cancer")
    prompt = f"""
    Act as a Bioinformatics AI Agent.
    Available clinical data columns: {columns}
    Formulate a single, testable hypothesis about {cancer_type} pathology using these columns.
    Return ONLY the hypothesis text.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    hypothesis = response.content.strip()
    
    # Log to ledger
    ledger_entry = {"step": "Hypothesis Formulation", "content": hypothesis}
    return {"current_hypothesis": hypothesis, "ledger": [ledger_entry]}

def write_code_node(state: AgentState):
    """Agent writes python code to validate the hypothesis."""
    print("Agent: Writing analysis code...")
    llm = get_llm()
    hypothesis = state.get("current_hypothesis")
    previous_error = state.get("execution_result", "")
    
    if not llm:
        # Mock code for MVP
        mock_code = "print('Validating mock hypothesis...')\n" \
                    "kmf = KaplanMeierFitter()\n" \
                    "if 'vital_status' in df.columns:\n" \
                    "    df['event'] = df['vital_status'] == 'Dead'\n" \
                    "    # Using a dummy numeric column for time if days_to_death is missing\n" \
                    "    df['time'] = df['days_to_death'].fillna(100)\n" \
                    "    kmf.fit(df['time'], event_observed=df['event'])\n" \
                    "    print('Median survival:', kmf.median_survival_time_)\n"
        return {"python_code": mock_code}
        
    prompt = f"""
    Act as a Bioinformatics AI Agent.
    Hypothesis: {hypothesis}
    Write Python code to test this hypothesis using `lifelines` (KaplanMeierFitter or CoxPHFitter) and `pandas`.
    The data is available in the global variable `df`.
    Print the key statistical findings (e.g., p-value, hazard ratio) using `print()`.
    DO NOT include markdown block markers (```python) in your response, return pure python code.
    """
    if previous_error and "Traceback" in previous_error:
        prompt += f"\n\nYour previous code failed with this error:\n{previous_error}\nPlease fix the error."
        
    response = llm.invoke([HumanMessage(content=prompt)])
    code = response.content.replace("```python", "").replace("```", "").strip()
    return {"python_code": code}

def execute_code_node(state: AgentState):
    """The Hands execute the generated python code."""
    print("Agent: Executing code...")
    code = state.get("python_code")
    df = state.get("data_context")
    
    result = execute_codeact(code, df_context=df)
    
    ledger_entry = {
        "step": "Execution", 
        "code_snippet": code[:100] + "...", 
        "result": result
    }
    
    return {"execution_result": result, "ledger": [ledger_entry], "iteration": state.get("iteration", 0) + 1}

def evaluate_result_node(state: AgentState):
    """Evaluates if the execution was successful or needs self-correction."""
    result = state.get("execution_result", "")
    iteration = state.get("iteration", 0)
    
    print("Agent: Evaluating results...")
    
    if "Traceback" in result and iteration < 3:
        print(f"Agent: Error detected, self-correcting (Iteration {iteration})...")
        return "write_code" # Go back and fix code
        
    # If successful or max iterations reached, conclude
    llm = get_llm()
    if llm:
        prompt = f"Analyze these statistical results and conclude if the hypothesis is supported: {result}"
        conclusion = llm.invoke([HumanMessage(content=prompt)]).content
    else:
        conclusion = "Mock Conclusion: The hypothesis is partially supported by the mock execution."
        
    return {"ledger": [{"step": "Conclusion", "content": conclusion}]}

def route_evaluation(state: AgentState):
    """Router function after execution."""
    result = state.get("execution_result", "")
    iteration = state.get("iteration", 0)
    
    if "Traceback" in result and iteration < 3:
        return "write_code"
    return "conclude"

def build_graph():
    """Builds the LangGraph state machine."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("fetch_data", fetch_data_node)
    workflow.add_node("formulate", formulate_hypothesis_node)
    workflow.add_node("write_code", write_code_node)
    workflow.add_node("execute_code", execute_code_node)
    workflow.add_node("conclude", evaluate_result_node)
    
    workflow.set_entry_point("fetch_data")
    workflow.add_edge("fetch_data", "formulate")
    workflow.add_edge("formulate", "write_code")
    workflow.add_edge("write_code", "execute_code")
    
    # Conditional edge for self-correction
    workflow.add_conditional_edges(
        "execute_code",
        route_evaluation,
        {
            "write_code": "write_code",
            "conclude": "conclude"
        }
    )
    
    workflow.add_edge("conclude", END)
    
    return workflow.compile()

def run_agent_workflow(project_id: str = "TCGA-SKCM", cancer_type: str = "Skin Cancer"):
    """Runs the workflow and saves the ledger."""
    app = build_graph()
    initial_state = {
        "messages": [], 
        "ledger": [], 
        "iteration": 0,
        "project_id": project_id,
        "cancer_type": cancer_type
    }
    
    print(f"Starting SPARK Agent Workflow for {cancer_type}...")
    final_state = app.invoke(initial_state)
    
    # Save scientific ledger to JSON for the Streamlit UI to read
    ledger_file = f"ledger_{project_id}.json"
    with open(ledger_file, "w") as f:
        # We need to make sure the dataframe isn't serialized
        clean_ledger = final_state.get("ledger", [])
        json.dump(clean_ledger, f, indent=4)
        
    print(f"Workflow complete. Ledger saved to {ledger_file}.")
    return final_state

if __name__ == "__main__":
    run_agent_workflow()
