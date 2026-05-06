# SPARK: Autonomous Skin Cancer Pathology Agent (MVP)

An agentic framework for autonomous scientific discovery in cancer pathology, inspired by the SPARK architecture (Nature Medicine 2026).

## Overview

This MVP implements a "Perception-Action" loop where an AI agent (The Brain) autonomously:
1. Formulates hypotheses based on available clinical data (TCGA-SKCM).
2. Perceives histopathology images (The Eyes) using zero-shot segmentation (SAM placeholder).
3. Executes dynamic code (The Hands) to statistically validate the hypothesis using `lifelines` (Kaplan-Meier survival analysis).
4. Maintains a **Scientific Ledger** of its discoveries, self-correcting if code execution fails.

## Architecture

- **Orchestrator (`agents/orchestrator.py`)**: Built with `LangGraph` for stateful multi-agent workflows. Uses `langchain-openai` by default, but can be easily swapped for Anthropic or Google Gemini models.
- **Pathology Toolkit (`tools/pathology_utils.py`)**: Uses `tiffslide` and `tiatoolbox` (for WSI), and implements a lightweight mock for `Segment Anything Model (SAM)` to generate saliency maps.
- **Data Access (`tools/gdc_client.py`)**: Automated fetcher for open-access clinical metadata using the GDC API.
- **Validation Engine (`tools/clinical_stats.py`)**: A `CodeAct` execution environment where the agent can run dynamic Python code to calculate p-values and hazard ratios, capturing standard output and tracebacks.
- **UI Dashboard (`app.py`)**: A Streamlit interface to visualize the spatial heatmaps, the agent's scientific ledger, and the generated survival curves.

## Installation

```bash
# Clone the repository
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`

# Install dependencies
pip install -r requirements.txt
```

## Configuration

The agent uses OpenAI by default. You need to set your API key:
```bash
export OPENAI_API_KEY="your_api_key_here"
```
*(Note: To use Anthropic or Gemini, see the commented code in `agents/orchestrator.py`)*

## Usage

### 1. Run the Discovery Loop
Let the agent autonomously formulate and test hypotheses:
```bash
python main.py
```
This will run the agent and generate a `ledger.json` file.

### 2. View the Dashboard
Launch the MVP Visual Elements to see the results:
```bash
streamlit run app.py
```

### 3. Interactive Notebook
Explore the agent's inner workings interactively:
```bash
jupyter notebook notebooks/demo_discovery.ipynb
```

## Disclaimer
The **CodeAct** environment (`tools/clinical_stats.py`) uses `exec()` to run dynamically generated code. For this MVP, it runs locally on your machine. Be aware of the security implications of executing AI-generated code.
