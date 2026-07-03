# 🔬 DERMA-Agent AILAB Scaffold

This is the modular, professional-grade scaffolding for the `DERMA-Agent` framework designed for computational pathology generative discovery.

## Directory Structure

```text
derma_agent/
├── app.py                  # Entry point (Streamlit AILAB Dashboard)
├── requirements.txt        # Scaffolding dependencies
├── README.md               # Setup and configuration guide
├── derma_core/             # The computational core
│   ├── __init__.py
│   ├── perception/         # WSI patch ingestion & Foundation Model hooks (TITAN/APOLLO)
│   │   └── wsi_engine.py
│   ├── knowledge_fabric/   # Local graph memory (NetworkX Gene-Pathway-Disease mapping)
│   │   └── graph_memory.py
│   ├── actions/            # CodeAct Sandbox execution
│   │   ├── code_executor.py # Restricted sandbox environment
│   │   ├── critic_agent.py  # AST Static inspector & logic validation
│   │   └── safety_policy.py # Whitelisted modules and security boundaries
│   └── memory/             # Episodic memory hooks
│       └── research_log.py  # crewai-soul persistent narrations with local fallback
└── web_interface/          # Visual interface rendering components
    ├── components.py
    └── dashboard.py
```

---

## Setup & Run

### 1. Install Dependencies
Initialize your virtual environment and install the scaffold requirements:
```bash
pip install -r derma_agent/requirements.txt
```

### 2. Configure API Keys
Configure OpenAI or Anthropic API keys for language modeling and reasoning capabilities:
```bash
# On Linux/macOS
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# On Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Initialize crewai-soul Environment
The Episodic Memory module utilizes `crewai-soul` for persistent research thoughts.
Ensure `crewai` and `crewai-soul` are installed. If `crewai-soul` is not active, the system automatically falls back to saving thoughts in a structured local JSON file at `data/research_narrative_log.json`.

To configure crewai-soul:
```python
from crewai_soul import SoulMemory

# Initialized inside research_log.py:
memory = SoulMemory(agent_name="Digital_Pathologist")
```

### 4. Deploy Dashboard
Start the Streamlit research interface:
```bash
streamlit run derma_agent/app.py
```

---

## Safety Features

- **Static AST Inspection**: The `CriticAgent` inspects generated code's Abstract Syntax Tree BEFORE execution to check that all imports match whitelisted modules (`pandas`, `numpy`, `lifelines`, `scipy`, `matplotlib`, `seaborn`) and blocks access to unsafe builtins (`open`, `eval`, `exec`, `__import__`).
- **Restricted execution namespace**: `CodeExecutor` bounds the environment to clean whitelisted builtins and variables to prevent unauthorized filesystem changes.
