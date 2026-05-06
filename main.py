import os
from dotenv import load_dotenv
from agents.orchestrator import run_agent_workflow

def main():
    print("="*50)
    print(" SPARK: Autonomous Skin Cancer Pathology Agent ")
    print("="*50)
    
    # Load environment variables (e.g., OPENAI_API_KEY)
    load_dotenv()
    
    # Run the ReAct loop
    final_state = run_agent_workflow()
    
    print("\nDiscovery Loop Concluded.")
    print("Run `streamlit run app.py` to view the MVP Dashboard.")

if __name__ == "__main__":
    main()
