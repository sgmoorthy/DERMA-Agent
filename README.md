# 🧬 DermaMind.ai

[![Version](https://img.shields.io/badge/version-2.1.0-blue.svg)](https://github.com/sgmoorthy/DERMA-Agent/releases)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/sgmoorthy/DERMA-Agent/actions/workflows/python-package.yml/badge.svg)](https://github.com/sgmoorthy/DERMA-Agent/actions)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Autonomous AI for Cancer Pathology Discovery**  
> *From slides to survival insights: graph-powered AI that tests hypotheses, validates biomarkers, and accelerates oncology research.*

---

## What is DermaMind.ai?

DermaMind.ai is an autonomous, agentic AI research platform for cancer pathology. It integrates:
- A **Live Knowledge Graph** seeded with genes, drugs, and pathways
- An **LLM-driven Discovery Engine** that dynamically generates and tests hypotheses
- A **CodeAct Security Sandbox** that safely executes auto-generated Python statistics
- A **WSI Perception Engine** (TITAN + APOLLO) for slide-level morphology features
- An **Interactive Streamlit Dashboard** with survival plots, KM curves, and an AI research assistant

Built for research scientists and clinician-scientists working with cohorts such as TCGA-BRCA and TCGA-SKCM.

---

## 📋 Version Information

| Component | Version | Status |
|-----------|---------|--------|
| **DermaMind.ai Core** | 2.1.0 | ✅ Stable |
| **Python** | 3.11, 3.12 | ✅ Supported |
| **Knowledge Fabric** | 1.0.0 | ✅ Active |
| **Discovery Engine** | 2.1.0 | ✅ Active |
| **Pathology AI (WSI)** | 1.5.0 | ✅ Active |
| **Research Assistant** | 1.0.0 | ✅ Active |
| **License** | MIT | ✅ Open Source |

---

## 🎉 What's New in v2.1.0

### ✨ New in this release
- 🤖 **FastDiscoveryEngine → Dashboard** — LLM-generated hypotheses & CodeAct scripts now run live in the UI
- 💬 **AI Research Assistant Chat** — In-app AI assistant for discussing findings and explanations
- 🕸️ **Interactive PyVis Knowledge Graph** — Drag, zoom, hover, and click-explore the knowledge fabric
- 🧹 **Clean Architecture** — All agents consolidated under `derma_agent/derma_core/agents/`
- 🛠️ **CI Pipeline Fixed** — GitHub Actions workflows updated to reflect the correct module paths

### Previous Highlights (v2.0)
- 🧠 **Knowledge Fabric** — Medical knowledge graph with 50+ nodes
- 🚀 **Parallel Discovery Engine** — 10x speed improvement via ThreadPoolExecutor
- 🔬 **Advanced Pathology** — Multi-method segmentation + texture analysis
- 📊 **Research Dashboard** — Survival curves, KM plots, and tissue correlations

---

## Architecture

```
DermaMind.ai
├── derma_agent/
│   ├── app.py                          # Streamlit entrypoint
│   ├── derma_core/
│   │   ├── agents/
│   │   │   ├── discovery_engine.py     # LLM hypothesis + CodeAct generation
│   │   │   ├── orchestrator.py         # LangGraph workflow orchestrator
│   │   │   └── research_assistant.py  # In-app AI chat assistant
│   │   ├── actions/
│   │   │   ├── code_executor.py        # Restricted CodeAct sandbox
│   │   │   ├── critic_agent.py         # AST security audit
│   │   │   └── safety_policy.py        # Malicious probe detection
│   │   ├── knowledge_fabric/
│   │   │   └── graph_memory.py         # Live knowledge graph
│   │   ├── memory/
│   │   │   └── research_log.py         # Episodic research narrative
│   │   └── perception/
│   │       └── wsi_engine.py           # WSI slide ingestion (TITAN/APOLLO)
│   ├── web_interface/
│   │   ├── dashboard.py                # Main dashboard render logic
│   │   └── components.py              # Reusable UI components
│   └── requirements.txt
├── tools/                              # Legacy: data & clinical stats engines
├── tests/
└── docs/                              # GitHub Pages site (dermamind.ai)
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/sgmoorthy/DERMA-Agent.git
cd DERMA-Agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your API key (enables live LLM hypothesis generation)
export OPENAI_API_KEY="your_api_key_here"
# Windows: set OPENAI_API_KEY=your_api_key_here
```

> **No API key?** DermaMind.ai gracefully falls back to a built-in mock generator so you can explore the full UI without any API costs.

---

## Launching the Dashboard

```bash
# Run the DermaMind.ai Streamlit dashboard
streamlit run derma_agent/app.py --server.port 8507
```

Then open **http://localhost:8507** in your browser.

---

## Using the Discovery Loop

1. Select a **Target Cohort** (e.g. TCGA-SKCM) in the sidebar.
2. Enter a **Slide ID** for WSI perception.
3. Click **🚀 Run Generative Discovery Loop**.

The agent will:
- **Phase 1** — Ingest the slide and run TITAN + APOLLO feature extraction.
- **Phase 2** — Consult the Knowledge Fabric and generate a hypothesis via LLM.
- **Phase 3** — Auto-generate CodeAct Python, submit to AST security critic.
- **Phase 4** — Execute in the restricted sandbox and extract survival statistics.
- **Results** — Render Kaplan-Meier curves, cellularity scatter, APOLLO embedding, and save to the narrative log.

---

## Programmatic API

```python
from derma_agent.derma_core.agents.discovery_engine import FastDiscoveryEngine, DiscoveryConfig
from derma_agent.derma_core.knowledge_fabric.graph_memory import KnowledgeFabric

# Build a knowledge fabric
kg = KnowledgeFabric()

# Configure and run discovery
config = DiscoveryConfig(parallel_workers=4, hypothesis_per_cohort=3, use_knowledge_fabric=True)
engine = FastDiscoveryEngine(config, knowledge_fabric=kg)
results = engine.discover_single_cohort("TCGA-BRCA", "Breast Cancer")

significant = engine.get_significant_findings()
print(f"Found {len(significant)} significant discoveries!")
```

---

## Supported Cancer Cohorts

| Cancer Type | TCGA Code | Cancer Type | TCGA Code |
|-------------|-----------|-------------|-----------|
| Skin (Melanoma) | TCGA-SKCM | Breast | TCGA-BRCA |
| Lung (Adenocarcinoma) | TCGA-LUAD | Lung (Squamous) | TCGA-LUSC |
| Brain (GBM) | TCGA-GBM | Brain (LGG) | TCGA-LGG |
| Colorectal | TCGA-COAD | Ovarian | TCGA-OV |
| Prostate | TCGA-PRAD | Bladder | TCGA-BLCA |
| Kidney (Clear Cell) | TCGA-KIRC | Kidney (Papillary) | TCGA-KIRP |
| Stomach | TCGA-STAD | Head & Neck | TCGA-HNSC |
| Liver | TCGA-LIHC | Pancreatic | TCGA-PAAD |

---

## Knowledge Fabric Schema

```
Node Types:          Relationships:
├── Gene             ├── MUTATED_IN (Gene → Cancer)
├── Protein          ├── TREATS (Drug → Cancer)
├── Drug             ├── TARGETS (Drug → Pathway/Gene)
├── Disease          ├── PART_OF (Gene → Pathway)
├── Pathway          ├── PREDICTS_RESPONSE_TO (Feature → Drug)
└── Clinical_Feature └── ASSOCIATED_WITH (Gene → Clinical_Feature)
```

---

## Safety & Security

DermaMind.ai uses a multi-layer code execution sandbox:
- All auto-generated Python is validated with **AST parsing** before execution.
- The **CriticAgent** checks for unsafe imports (e.g. `os`, `subprocess`, `socket`).
- Execution is restricted to approved statistical libraries only.
- Malicious probes are blocked with a `BLOCKED — Security Violation` response.

---

## Citation

If you use DermaMind.ai in your research, please cite:

```bibtex
@software{dermamind_ai_2025,
  title  = {DermaMind.ai: Autonomous AI for Cancer Pathology Discovery},
  author = {Moorthy, S.},
  year   = {2025},
  url    = {https://github.com/sgmoorthy/DERMA-Agent}
}
```

---

## License

MIT License — See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Areas of interest:
- Additional TCGA/GDC cohort support
- New ML models for survival prediction
- Enhanced WSI segmentation methods
- Knowledge graph expansion (new genes, drugs, pathways)
- UI/UX improvements to the dashboard

---

**DermaMind.ai** · Powered by LangGraph · OpenAI · TCGA/GDC · Lifelines · Streamlit
