import os
import argparse
from dotenv import load_dotenv
from agents.orchestrator import run_agent_workflow
from tools.gdc_client import CANCER_PROJECTS

def main():
    parser = argparse.ArgumentParser(description="Run the SPARK Pathology Agent for a specific cancer type.")
    parser.add_argument(
        "--cancer", 
        type=str, 
        choices=["skin", "breast", "lung", "brain"], 
        default="skin",
        help="The cancer type to run the agent for (skin, breast, lung, brain)."
    )
    args = parser.parse_args()

    cancer_mapping = {
        "skin": "Skin Cancer",
        "breast": "Breast Cancer",
        "lung": "Lung Cancer",
        "brain": "Brain Cancer"
    }
    
    cancer_name = cancer_mapping[args.cancer]
    project_id = CANCER_PROJECTS[cancer_name]

    print("="*50)
    print(f" SPARK: Autonomous {cancer_name} Pathology Agent ")
    print("="*50)
    
    # Load environment variables (e.g., OPENAI_API_KEY)
    load_dotenv()
    
    # Run the ReAct loop
    final_state = run_agent_workflow(project_id=project_id, cancer_type=cancer_name)
    
    print("\nDiscovery Loop Concluded.")
    print("Run `streamlit run app.py` to view the MVP Dashboard.")

if __name__ == "__main__":
    main()
