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

DermaMind.ai is an agentic oncology research framework that combines:
- a **WSI perception layer** for slide-level morphology features
- a **knowledge fabric** for biomedical grounding
- a **CodeAct-style execution loop** for dynamic statistical validation
- **survival analysis** and ML summaries for hypothesis testing
- **interactive dashboards** for exploration, provenance, and review

The project is designed for research scientists and clinician-scientists working with cohorts such as `TCGA-SKCM` and `TCGA-BRCA`, while remaining extensible to broader TCGA/GDC exploration.

---

## What changed recently?

The current workspace includes a research-grade documentation and math-alignment pass.

### Research-grade upgrades
- **Session-wide Benjamini–Hochberg FDR correction** in `derma_agent/derma_core/agents/discovery_engine.py`
- **Attention-style pooled slide representations** in `derma_agent/derma_core/perception/wsi_engine.py`
- **Dashboard support for raw p-values and adjusted q-values**
- **New tests** for FDR correction and WSI pooling in `tests/test_research_math_alignment.py`
- **Expanded documentation** including a walkthrough and multiple architecture/framework blog posts

### Why this matters
These changes make DermaMind.ai more faithful to the mathematical framing described in `data/DERMA_agent.pdf`, especially for:
- exploratory multiple-testing control
- slide-level representation from tile-level embeddings
- transparent reporting of statistical significance

---

## Core architecture

```text
DermaMind.ai
├── derma_agent/
│   ├── app.py                                # Primary Streamlit entrypoint
│   ├── derma_core/
│   │   ├── agents/
│   │   │   ├── discovery_engine.py          # Hypothesis generation + FDR-aware discovery
│   │   │   ├── orchestrator.py              # Workflow orchestration
│   │   │   └── research_assistant.py        # In-app AI assistant
│   │   ├── actions/
│   │   │   ├── code_executor.py             # Restricted sandbox executor
│   │   │   ├── critic_agent.py              # AST/code safety critic
│   │   │   └── safety_policy.py             # Execution constraints
│   │   ├── knowledge_fabric/
│   │   │   └── graph_memory.py              # Lightweight graph memory for dashboard loop
│   │   ├── memory/
│   │   │   └── research_log.py              # Episodic research narrative
│   │   └── perception/
│   │       └── wsi_engine.py                # WSI ingestion + attention-style pooling
│   └── web_interface/
│       ├── dashboard.py                     # Main scientific UI
│       └── components.py                    # Shared Streamlit UI helpers
├── tools/
│   ├── enhanced_clinical_stats.py           # DSL, KM, Cox, ML survival analysis
│   ├── enhanced_data_client.py              # GDC/local-mirror data handling
│   ├── enhanced_pathology.py                # Pathology feature extraction utilities
│   └── knowledge_fabric.py                  # Research-grade oncology knowledge graph
├── blog/                                    # Markdown source posts
├── public/blog/                             # Previewable static blog pages
├── tests/
├── walkthrough.md                           # Walkthrough of recent changes
└── README.md
```

---

## Research loop

DermaMind.ai works as a closed-loop system rather than a single prediction model:

1. **Perception** — slide metadata, tile/embedding summaries, pathology features
2. **Knowledge grounding** — graph-based prior context from genes, pathways, drugs, diseases
3. **Execution** — dynamic or DSL-routed statistical analysis in a constrained environment
4. **Validation** — survival outputs, significance assessment, logging, and UI presentation

### Mathematical alignment highlights
- **Kaplan–Meier / log-rank** for group-wise survival comparisons
- **Cox proportional hazards** modeling via `lifelines.CoxPHFitter`
- **BH/FDR correction** applied across the discovery session
- **Attention-style slide pooling** for a slide-level embedding from tile-level mock features

---

## Live links

### Static site (GitHub Pages)
These links are deterministic once GitHub Pages is enabled for the repository:
- **Homepage:** `https://sgmoorthy.github.io/DERMA-Agent/`
- **Docs hub:** `https://sgmoorthy.github.io/DERMA-Agent/docs/index.html`
- **Blog index:** `https://sgmoorthy.github.io/DERMA-Agent/blog/index.html`
- **Walkthrough mirror:** `https://sgmoorthy.github.io/DERMA-Agent/walkthrough.md`

### Interactive Streamlit app
There is currently **no verified public Streamlit runtime URL in this repository**.

That is intentional in the docs: GitHub Actions can test Streamlit startup, but **GitHub Pages cannot host a Streamlit server**. To publish the interactive app, deploy one of the Streamlit entrypoints to Streamlit Community Cloud or another Python app host, then add that returned URL here.

See: [`STREAMLIT_DEPLOYMENT.md`](STREAMLIT_DEPLOYMENT.md)

### Secret configuration paths

#### GitHub Actions secrets
If you want CI to boot the Streamlit apps with real provider keys available:
1. Open **GitHub → Settings → Secrets and variables → Actions**
2. Add repository secrets:
   - `OPENAI_API_KEY`
   - `GOOGLE_API_KEY`

#### Streamlit secrets
For Streamlit Community Cloud or local Streamlit secrets-based configuration:
- use `.streamlit/secrets.toml.example` as the template
- add the same keys in **App settings → Secrets** on Streamlit Community Cloud

---

## Installation

```bash
# Clone the repository
git clone https://github.com/sgmoorthy/DERMA-Agent.git
cd DERMA-Agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install front-end dependencies
npm install

# Optional: enable live LLM-backed hypothesis generation
export OPENAI_API_KEY="your_api_key_here"
# Windows: set OPENAI_API_KEY=your_api_key_here
```

> **No API key?** The project falls back to mock behavior for key discovery flows so you can still explore the UI and architecture locally.

---

## Running the project

### 1. Primary Streamlit dashboard
```bash
.venv\Scripts\python.exe -m streamlit run derma_agent/app.py --server.port 8507
```
Open: `http://localhost:8507`

### 2. Enhanced dashboard
```bash
.venv\Scripts\python.exe -m streamlit run app_enhanced.py --server.port 8508
```
Open: `http://localhost:8508`

### 3. Front-end landing page
```bash
npm run dev
```

### 4. Front-end production preview
```bash
npm run build
npm run preview -- --host 127.0.0.1 --port 4173
```
Open: `http://127.0.0.1:4173`

### 5. Previewable blog pages
When the Vite app is running, open:
- `/blog/index.html`
- `/blog/closed-loop-agentic-discovery.html`
- `/blog/architecture-deep-dive.html`
- `/blog/research-math-and-safety.html`
- `/blog/scaling-derma-mind-to-multiple-cancers.html`

---

## Programmatic usage

```python
from derma_agent.derma_core.agents.discovery_engine import (
    DiscoveryConfig,
    FastDiscoveryEngine,
    run_fast_discovery,
)
from tools.knowledge_fabric import create_default_knowledge_fabric

config = DiscoveryConfig(
    parallel_workers=2,
    hypothesis_per_cohort=3,
    significance_threshold=0.05,
    use_knowledge_fabric=True,
)

kg = create_default_knowledge_fabric()
engine = FastDiscoveryEngine(config=config, knowledge_fabric=kg)

report = run_fast_discovery(
    cancer_types=["Skin Cancer", "Breast Cancer"],
    config=config,
    output_dir="discoveries",
)

print(report["significant_findings"])
print(report["fdr_method"])
```

---

## Statistical and safety model

### Survival and discovery statistics
The project supports:
- Kaplan–Meier estimation
- log-rank testing
- Cox proportional hazards regression
- ML survival/discrimination summaries
- structured DSL-driven analysis requests

### Multiple-testing control
The discovery engine now stores both:
- raw `p_value`
- adjusted `adjusted_p_value` (q-value)

This is important because an agentic research loop may test many hypotheses during one session.

### Safe dynamic execution
DermaMind.ai uses multiple safeguards:
- AST validation before execution
- critic-based code review
- restricted execution context
- constrained imports and builtins
- explicit execution history / trace capture

---

## Supported cancer cohorts

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

## Documentation map

### Walkthrough
- [`walkthrough.md`](walkthrough.md) — summary of recent research-grade changes, validations, and design rationale

### Blog source posts
- [`blog/scaling-derma-mind-to-multiple-cancers.md`](blog/scaling-derma-mind-to-multiple-cancers.md)
- [`blog/closed-loop-agentic-discovery.md`](blog/closed-loop-agentic-discovery.md)
- [`blog/architecture-deep-dive.md`](blog/architecture-deep-dive.md)
- [`blog/research-math-and-safety.md`](blog/research-math-and-safety.md)

### Previewable static pages
- `public/docs/index.html`
- `public/blog/index.html`
- `public/blog/closed-loop-agentic-discovery.html`
- `public/blog/architecture-deep-dive.html`
- `public/blog/research-math-and-safety.html`
- `public/blog/scaling-derma-mind-to-multiple-cancers.html`

### Deployment guides
- [`STREAMLIT_DEPLOYMENT.md`](STREAMLIT_DEPLOYMENT.md)
- [`GITHUB_PAGES_SETUP.md`](GITHUB_PAGES_SETUP.md)
- [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)

---

## CI/CD and deployment model

### Static site deployment
The Vite-powered homepage, docs hub, walkthrough mirror, and blog pages are deployed via GitHub Pages.

Workflow:
- `.github/workflows/pages.yml`

### Streamlit CI validation
The interactive dashboards are validated in CI by booting them headlessly.

Workflow:
- `.github/workflows/streamlit-smoke.yml`

### Important hosting distinction
GitHub Actions can test Streamlit startup, but GitHub Pages cannot host a long-running Streamlit server.

Use this split:
- **GitHub Pages** → static docs/site/blog
- **Streamlit Community Cloud or another app host** → `derma_agent/app.py` or `app_enhanced.py`

---

## Validation

Commands used for recent verification:

```bash
# Python tests
.venv\Scripts\python.exe -m unittest discover tests

# Streamlit dashboards
.venv\Scripts\python.exe -m streamlit run derma_agent/app.py --server.port 8507
.venv\Scripts\python.exe -m streamlit run app_enhanced.py --server.port 8508

# Front-end build / preview
npm run build
npm run preview -- --host 127.0.0.1 --port 4173
```

Most recent test result in this workspace:
- **11 tests passed**

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

Contributions are welcome, especially in:
- additional TCGA/GDC cohort support
- stronger pathology foundation-model integration
- richer discovery benchmarks
- better export/reporting of q-values and provenance
- architecture/docs improvements

---

**DermaMind.ai** · Powered by LangGraph · OpenAI · TCGA/GDC · Lifelines · Streamlit · Vite