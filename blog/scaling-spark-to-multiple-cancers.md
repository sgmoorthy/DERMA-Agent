# Scaling the SPARK Pathology Agent: From Skin Cancer to a Multi-Oncology Framework

*Published: May 2026*

When we first introduced the SPARK Autonomous Pathology Agent (inspired by the groundbreaking 2026 *Nature Medicine* paper), our primary objective was to build a functional MVP focused on Skin Cutaneous Melanoma (TCGA-SKCM). The agent successfully demonstrated a closed "Perception-Action" loop: pulling clinical metadata, extracting zero-shot features from Whole Slide Images (WSIs), formulating hypotheses, and writing dynamic Python code to statistically validate them using survival analysis.

Today, we are thrilled to announce a major architectural shift: **SPARK is no longer just a Skin Cancer agent.** We have upgraded the framework to support multiple oncology domains out-of-the-box, effectively transforming it into a generalized AI discovery platform for cancer pathology.

## The Motivation for Generalization

Cancer is not a monolithic disease; the molecular and morphological features that govern survival in melanoma differ vastly from those in glioblastoma or lung adenocarcinoma. To make SPARK a truly useful tool for researchers, the underlying data engines and agent personas needed to be decoupled from a single disease context. 

We needed a system where the "Brain" (the LLM Orchestrator) could dynamically switch specializations, and the "Hands" (the Data and Execution engines) could context-switch to the appropriate cohort data without hardcoding.

## What's New Under the Hood?

### 1. Dynamic Data Engine (`gdc_client.py`)
Previously, our Genomic Data Commons (GDC) client was hardcoded to fetch only the `TCGA-SKCM` cohort. We've refactored the client to be entirely dynamic. By maintaining a mapping of `CANCER_PROJECTS`, the agent can now dynamically target the GDC API for:
- **Skin Cancer** (`TCGA-SKCM`)
- **Breast Cancer** (`TCGA-BRCA`)
- **Lung Cancer** (`TCGA-LUAD`)
- **Brain Cancer** (`TCGA-GBM`)

This means the agent now formulates its hypotheses based on the specific demographic and diagnostic nuances of the requested cancer type.

### 2. Context-Aware Orchestration (`orchestrator.py`)
The `AgentState` in our LangGraph orchestrator has been expanded. It now carries explicit variables for `project_id` and `cancer_type`. 

When the prompt is sent to the LLM (OpenAI, Anthropic, or Gemini), it injects this persona. Instead of a generic prompt, the agent is instructed: *"Act as a Bioinformatics AI Agent specializing in Breast Cancer pathology."* This contextual grounding drastically improves the relevance of the formulated hypotheses.

Furthermore, the agent's **Scientific Ledger** is now saved dynamically (e.g., `ledger_TCGA-BRCA.json`). This prevents context pollution—your lung cancer discoveries won't accidentally overwrite your brain cancer research!

### 3. A Multi-Specialty UI Dashboard
Our Streamlit interface has been upgraded to reflect this new multi-agent reality. Users will now see an **Agent Configuration** sidebar. By simply selecting an "Agent Specialty" from a dropdown, the dashboard instantly re-routes, loading the specific scientific ledger and context for that cancer type. 

### 4. CLI Argument Parsing
For the power users, the main execution loop now supports `argparse`. Spinning up a specialized agent is as simple as running:
```bash
python main.py --cancer breast
```

## What's Next?
By generalizing the data ingestion and orchestrator state, we've laid the groundwork for SPARK to scale to all 33 TCGA cancer types. Our immediate next steps involve enhancing the "Eyes" of the agent—upgrading the mocked Segment Anything Model (SAM) integration to process massive, multi-gigabyte WSIs for these new cancer types natively.

We invite researchers and developers to clone the repository, test the different agent personas, and contribute to the open-source effort to accelerate pathology discoveries!
