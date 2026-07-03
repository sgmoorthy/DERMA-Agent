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
    page_title="DERMA-Agent AILAB",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 DERMA-Agent AILAB")
st.markdown("##### Scaffolding Autonomous Cancer Pathology Discovery Engine")
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
