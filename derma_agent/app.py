import sys
import os
import streamlit as st

# Insert directory paths for root resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from derma_core.knowledge_fabric.graph_memory import KnowledgeFabric
from derma_core.memory.research_log import ResearchNarrative
from web_interface.dashboard import render_dashboard

# App page configurations
st.set_page_config(
    page_title="DermaMind.ai — Autonomous AI for Cancer Pathology Discovery",
    page_icon="🧬",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .dermamind-header { padding: 0.5rem 0 0.25rem 0; }
  .dermamind-title  { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px;
                      background: linear-gradient(135deg, #4cc9f0 0%, #7c3aed 100%);
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .dermamind-sub    { font-size: 0.95rem; color: #aab4c4; margin-top: 0.15rem; }
</style>
<div class="dermamind-header">
  <div class="dermamind-title">🧬 DermaMind.ai</div>
  <div style="font-size:1rem;font-weight:600;color:#e0e0e0;margin-top:2px;">
    Autonomous AI for Cancer Pathology Discovery
  </div>
  <div class="dermamind-sub">
    From slides to survival insights: graph-powered AI that tests hypotheses,
    validates biomarkers, and accelerates oncology research.
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# Initialize session state for Graph Fabric
if 'fabric' not in st.session_state:
    st.session_state.fabric = KnowledgeFabric()

# Initialize session state for Episodic Research Narrative Log
if 'narrative' not in st.session_state:
    st.session_state.narrative = ResearchNarrative(
        agent_name="Digital_Pathologist",
        fallback_path="data/research_narrative_log.json"
    )

# Render main dashboard layout
render_dashboard()
