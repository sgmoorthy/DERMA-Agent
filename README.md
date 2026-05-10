# DERMA-Agent: Enhanced Discovery Framework

An advanced agentic framework for autonomous scientific discovery in cancer pathology, with enhanced capabilities including a Knowledge Fabric, parallel discovery, and ML-powered analysis.

## Overview

DERMA-Agent implements an intelligent "Perception-Action" loop where an AI agent autonomously:
1. **Formulates hypotheses** using clinical open datasets (20+ TCGA cohorts) and the Knowledge Fabric
2. **Perceives histopathology** with advanced segmentation (HSV, LAB, adaptive methods)
3. **Validates statistically** with Kaplan-Meier, Cox regression, and ML models (Random Forest, XGBoost)
4. **Maintains discoveries** in a Scientific Ledger with self-correction capabilities
5. **Reasons with medical knowledge** using the Knowledge Fabric (genes, drugs, pathways)

## Key Features

### 🔬 Fast Discovery Engine
- **Parallel hypothesis testing** - Test multiple hypotheses simultaneously
- **Knowledge-guided generation** - Uses medical knowledge graph for relevant hypotheses
- **Self-correcting execution** - Automatically fixes code errors with up to 3 iterations
- **ML-powered analysis** - Random Forest, Gradient Boosting for prediction

### 📚 Knowledge Fabric
- **Medical knowledge graph** with 50+ nodes (genes, proteins, drugs, pathways)
- **Graph relationships** - MUTATED_IN, TREATS, TARGETS, PART_OF, etc.
- **Semantic search** - Find similar concepts using embeddings
- **Path finding** - Discover connections between entities

### 📊 Enhanced Data Access
- **20+ TCGA cancer cohorts** - Skin, Breast, Lung, Brain, Colorectal, etc.
- **Intelligent caching** - Reduces API calls with 1-hour TTL
- **Parallel fetching** - Download multiple cohorts simultaneously
- **Automatic preparation** - Survival-ready data formatting

### 🧬 Advanced Pathology
- **Multi-method segmentation** - HSV, LAB, and adaptive approaches
- **Nuclei detection** - Individual nuclei counting and feature extraction
- **Texture analysis** - GLCM and LBP features
- **Synthetic generation** - Create test images for development

## Architecture

| Component | Description | File |
|-----------|-------------|------|
| **Fast Discovery Engine** | Parallel hypothesis testing | `agents/discovery_engine.py` |
| **Knowledge Fabric** | Medical knowledge graph | `tools/knowledge_fabric.py` |
| **Enhanced Data Client** | Multi-source data with caching | `tools/enhanced_data_client.py` |
| **Enhanced Stats Engine** | ML-powered survival analysis | `tools/enhanced_clinical_stats.py` |
| **Enhanced Pathology** | Advanced image segmentation | `tools/enhanced_pathology.py` |
| **Orchestrator (Legacy)** | Original LangGraph workflow | `agents/orchestrator.py` |
| **Enhanced Dashboard** | Streamlit visualization | `app_enhanced.py` |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DERMA-Agent.git
cd DERMA-Agent

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Download large models for better pathology analysis
# pip install opencv-python-headless cellpose
```

## Configuration

### API Keys
Set your OpenAI API key for full LLM capabilities:
```bash
export OPENAI_API_KEY="your_api_key_here"
# On Windows: set OPENAI_API_KEY=your_api_key_here
```

### Optional: Anthropic or Google
```python
# In agents/discovery_engine.py, modify _init_llm()
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-opus-20240229")
```

## Usage

### 1. Quick Start - Run Discovery
```bash
# Run fast discovery on a single cancer type
python -c "from agents.discovery_engine import run_fast_discovery; run_fast_discovery(['Breast Cancer'])"

# Or use the enhanced CLI
python -m agents.discovery_engine --cancer breast --workers 4
```

### 2. Launch Enhanced Dashboard
```bash
# Enhanced Streamlit app with all features
streamlit run app_enhanced.py

# Or use the original simple dashboard
streamlit run app.py
```

### 3. Interactive Jupyter Notebook
```bash
# Comprehensive demo with all features
jupyter notebook notebooks/comprehensive_demo.ipynb

# Original simple demo
jupyter notebook notebooks/demo_discovery.ipynb
```

### 4. Programmatic API
```python
from agents.discovery_engine import FastDiscoveryEngine, DiscoveryConfig
from tools.knowledge_fabric import create_default_knowledge_fabric

# Create knowledge fabric
kg = create_default_knowledge_fabric("data/knowledge_fabric.json")

# Configure discovery
config = DiscoveryConfig(
    parallel_workers=4,
    hypothesis_per_cohort=5,
    use_knowledge_fabric=True
)

# Run discovery
engine = FastDiscoveryEngine(config, kg)
results = engine.discover_single_cohort("TCGA-BRCA", "Breast Cancer")

# Get significant findings
significant = engine.get_significant_findings()
print(f"Found {len(significant)} significant discoveries!")
```

## Supported Cancer Types

The framework supports 20+ TCGA cancer cohorts:

| Cancer Type | TCGA Code | Cancer Type | TCGA Code |
|-------------|-----------|-------------|-----------|
| Skin (Melanoma) | TCGA-SKCM | Breast | TCGA-BRCA |
| Lung (Adenocarcinoma) | TCGA-LUAD | Lung (Squamous) | TCGA-LUSC |
| Brain (GBM) | TCGA-GBM | Brain (LGG) | TCGA-LGG |
| Colorectal | TCGA-COAD | Ovarian | TCGA-OV |
| Prostate | TCGA-PRAD | Bladder | TCGA-BLCA |
| Kidney (Clear Cell) | TCGA-KIRC | Kidney (Papillary) | TCGA-KIRP |
| Stomach | TCGA-STAD | Head and Neck | TCGA-HNSC |
| Liver | TCGA-LIHC | Pancreatic | TCGA-PAAD |

## Discovery Workflow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Data Access    │────▶│  Hypothesis Gen  │────▶│  Parallel Test  │
│  (TCGA/GDC)     │     │  (LLM + Knowledge)│     │  (Survival/ML)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
        ┌───────────────────────────────────────────────────┘
        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Results Save   │◀────│  Auto-Correction  │◀────│  Significance   │
│  (JSON Ledger)  │     │  (Error Recovery) │     │  Test (p<0.05)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Output Files

All discoveries are saved to the `discoveries/` directory:

- `discovery_ledger_*.json` - Detailed results for each hypothesis
- `discovery_report_*.json` - Summary statistics
- `knowledge_fabric.json` - Medical knowledge graph
- `dashboard.png` - Visual summary (when run from notebook)

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

## Safety & Disclaimer

⚠️ **Important**: The CodeAct environment (`tools/clinical_stats.py`) uses `exec()` to run dynamically generated code. While the environment is restricted to statistical libraries only, be aware that:

- Code executes locally on your machine
- Only use with trusted data sources
- Review generated code in production environments
- Consider sandboxing for multi-user deployments

## Performance Tips

1. **Enable caching** - First API calls are cached for 1 hour
2. **Use parallel workers** - Set `parallel_workers=4` for 4x speedup
3. **Limit hypotheses** - Start with 2-3 per cohort for faster iteration
4. **Pre-build knowledge fabric** - Run once, reuse across sessions
5. **Use mock mode** - Set `OPENAI_API_KEY=""` for testing without API costs

## Citation

If you use DERMA-Agent in your research, please cite:

```bibtex
@software{derma_agent_2024,
  title = {DERMA-Agent: Autonomous Cancer Pathology Discovery Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/DERMA-Agent}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Areas of interest:
- Additional cancer cohorts
- New ML models for survival prediction
- Enhanced segmentation methods
- Knowledge graph expansion
- UI/UX improvements

---

**DERMA-Agent** | Powered by LangGraph, OpenAI, TCGA, and Open Source
