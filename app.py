import streamlit as st
import json
import os
import numpy as np
from PIL import Image
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

from tools.pathology_utils import generate_saliency_heatmap

st.set_page_config(page_title="SPARK Pathology Agent", layout="wide")

st.title("🔬 SPARK: Autonomous Skin Cancer Pathology Agent")
st.markdown("An agentic framework for autonomous scientific discovery in cancer pathology.")

# Define layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("The Eyes: Perception & Histopathology")
    st.markdown("Zero-shot saliency map overlay on Whole Slide Image (WSI)")
    
    # Placeholder WSI or real one if exists
    wsi_path = "sample_wsi.tif"
    if os.path.exists(wsi_path):
        # We would use tiffslide here, but for MVP we mock a tile
        image = np.ones((256, 256, 3), dtype=np.uint8) * 200
        # draw a mock cell
        image[100:150, 100:150] = [100, 100, 200]
    else:
        # Mock tissue image
        image = np.ones((512, 512, 3), dtype=np.uint8) * 240
        image[200:300, 200:300] = [150, 100, 200] # Mock blueish tissue

    if st.button("Generate Saliency Heatmap"):
        with st.spinner("Agent is scanning tissue..."):
            heatmap = generate_saliency_heatmap(image)
            st.image(heatmap, caption="Agent Attention Heatmap (Red = Nuclei/Tissue)", use_column_width=True)

with col2:
    st.header("The Brain: Scientific Ledger")
    st.markdown("Live feed of the agent's internal monologue and discoveries.")
    
    if os.path.exists("ledger.json"):
        with open("ledger.json", "r") as f:
            ledger = json.load(f)
            
        for entry in ledger:
            step = entry.get("step", "Unknown Step")
            st.markdown(f"**Step:** {step}")
            if "content" in entry:
                st.info(entry["content"])
            if "code_snippet" in entry:
                st.code(entry["code_snippet"], language="python")
            if "result" in entry:
                with st.expander("Execution Result"):
                    st.text(entry["result"])
    else:
        st.warning("No ledger found. Run `python main.py` to start the discovery loop.")

st.divider()

st.header("The Hands: Validation Dashboard")
st.markdown("Automated generation of Kaplan-Meier plots based on Agent's hypothesis.")

# Mock validation plot based on the agent's code execution
if st.button("Render Latest Validation Plot"):
    # Recreate the data for plotting in Streamlit
    df = pd.DataFrame({
        "time": [10, 20, 30, 40, 50, 60, 70, 80],
        "event": [1, 0, 1, 1, 0, 1, 0, 1],
        "group": ["High", "Low", "High", "Low", "High", "Low", "High", "Low"]
    })
    
    fig, ax = plt.subplots()
    kmf = KaplanMeierFitter()
    
    for name, grouped_df in df.groupby("group"):
        kmf.fit(grouped_df["time"], grouped_df["event"], label=name)
        kmf.plot_survival_function(ax=ax)
        
    plt.title("Kaplan-Meier Survival Curve (Discovered Feature)")
    plt.ylabel("Survival Probability")
    plt.xlabel("Timeline (Days)")
    st.pyplot(fig)
