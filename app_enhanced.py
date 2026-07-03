"""
Enhanced Streamlit Dashboard for DERMA-Agent
Comprehensive visualization of discoveries, knowledge fabric, and pathology analysis.
"""

import os
import json
import glob
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# DERMA-Agent imports
try:
    from derma_agent.derma_core.knowledge_fabric import KnowledgeFabric, create_default_knowledge_fabric
    from tools.enhanced_data_client import get_data_client, EXPANDED_CANCER_PROJECTS
    from tools.enhanced_clinical_stats import EnhancedStatsEngine, quick_survival_summary
    from tools.enhanced_pathology import (
        create_synthetic_pathology_image, NucleiSegmenter, 
        TissueAnalyzer, EnhancedWSIProcessor
    )
    from derma_agent.derma_core.derma_agent.derma_core.agents.discovery_engine import FastDiscoveryEngine, DiscoveryConfig
    ENHANCED_MODE = True
except ImportError as e:
    st.error(f"Failed to load enhanced modules: {e}")
    ENHANCED_MODE = False
    from tools.pathology_utils import generate_saliency_heatmap
    from tools.gdc_client import CANCER_PROJECTS

# Page configuration
st.set_page_config(
    page_title="DERMA-Agent: Enhanced Discovery Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .significant-finding {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insignificant-finding {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def load_knowledge_fabric():
    """Load or create the knowledge fabric."""
    kg_path = Path("data/knowledge_fabric.json")
    if kg_path.exists():
        return KnowledgeFabric.load(str(kg_path))
    else:
        return create_default_knowledge_fabric(str(kg_path))


def load_latest_discovery():
    """Load the most recent discovery results."""
    discovery_files = glob.glob("discoveries/discovery_ledger_*.json")
    if discovery_files:
        latest_file = max(discovery_files, key=os.path.getctime)
        with open(latest_file, 'r') as f:
            return json.load(f), latest_file
    return [], None


def load_latest_report():
    """Load the most recent discovery report."""
    report_files = glob.glob("discoveries/discovery_report_*.json")
    if report_files:
        latest_file = max(report_files, key=os.path.getctime)
        with open(latest_file, 'r') as f:
            return json.load(f)
    return None


def render_knowledge_fabric_tab():
    """Render the Knowledge Fabric tab."""
    st.header("📚 Knowledge Fabric")
    st.markdown("Medical knowledge graph powering discovery reasoning")
    
    if not ENHANCED_MODE:
        st.warning("Knowledge Fabric not available. Please install enhanced dependencies.")
        return
    
    try:
        kg = load_knowledge_fabric()
        stats = kg.get_statistics()
        
        # Statistics cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Nodes", stats['total_nodes'])
        with col2:
            st.metric("Total Edges", stats['total_edges'])
        with col3:
            st.metric("Node Types", len(stats['node_types']))
        with col4:
            st.metric("Graph Density", f"{stats['density']:.4f}")
        
        # Node type distribution
        st.subheader("Node Type Distribution")
        node_df = pd.DataFrame(list(stats['node_types'].items()), 
                              columns=['Node Type', 'Count'])
        fig = px.pie(node_df, values='Count', names='Node Type', 
                    title='Knowledge Graph Composition')
        st.plotly_chart(fig, use_container_width=True)
        
        # Query interface
        st.subheader("🔍 Query Knowledge Graph")
        
        query_type = st.selectbox("Query Type", [
            "Drugs for Cancer",
            "Genes for Cancer",
            "Pathway Members",
            "Drug Targets"
        ])
        
        if query_type == "Drugs for Cancer":
            cancer = st.selectbox("Select Cancer", ["Melanoma", "Breast_Carcinoma", "Lung_Adenocarcinoma", "Glioblastoma"])
            if st.button("Query"):
                drugs = []
                for edge in kg.edges:
                    if edge.relation == "TREATS" and edge.target == cancer:
                        drug_node = kg.get_node(edge.source)
                        if drug_node:
                            drugs.append({
                                "Drug": drug_node.id,
                                "Type": drug_node.properties.get('type', 'Unknown'),
                                "Target": drug_node.properties.get('target', 'Unknown'),
                                **edge.properties
                            })
                if drugs:
                    st.dataframe(pd.DataFrame(drugs))
                else:
                    st.info("No drugs found for this cancer type")
                    
        elif query_type == "Genes for Cancer":
            cancer = st.selectbox("Select Cancer", ["Melanoma", "Breast_Carcinoma", "Lung_Adenocarcinoma", "Glioblastoma"])
            if st.button("Query"):
                genes = []
                for edge in kg.edges:
                    if edge.relation == "MUTATED_IN" and cancer in edge.target:
                        gene_node = kg.get_node(edge.source)
                        if gene_node:
                            genes.append({
                                "Gene": gene_node.id,
                                "Type": gene_node.properties.get('type', 'Unknown'),
                                **edge.properties
                            })
                if genes:
                    st.dataframe(pd.DataFrame(genes))
                else:
                    st.info("No gene associations found")
                    
        elif query_type == "Pathway Members":
            pathway = st.selectbox("Select Pathway", ["MAPK_Pathway", "PI3K_AKT_mTOR", "DNA_Repair_Pathway", "Cell_Cycle_Control"])
            if st.button("Query"):
                pathway_node = kg.get_node(pathway)
                if pathway_node:
                    genes = pathway_node.properties.get('genes', [])
                    st.write(f"**Genes in {pathway}:**")
                    st.write(", ".join(genes))
                    
        elif query_type == "Drug Targets":
            drug = st.selectbox("Select Drug", ["Pembrolizumab", "Trastuzumab", "Dabrafenib", "Temozolomide"])
            if st.button("Query"):
                targets = []
                for edge in kg.edges:
                    if edge.source == drug and edge.relation == "TARGETS":
                        target_node = kg.get_node(edge.target)
                        if target_node:
                            targets.append({
                                "Target": edge.target,
                                "Relation": edge.relation,
                                **edge.properties
                            })
                if targets:
                    st.dataframe(pd.DataFrame(targets))
                    
    except Exception as e:
        st.error(f"Error loading knowledge fabric: {e}")


def render_discovery_tab():
    """Render the Discovery tab."""
    st.header("🔬 Discovery Results")
    
    # Load discovery data
    ledger, ledger_file = load_latest_discovery()
    report = load_latest_report()
    
    # 1. Run New Discovery Panel
    st.subheader("🚀 Run New Discovery Engine")
    with st.expander("Launch Discovery Run", expanded=(not ledger)):
        recipe = st.selectbox(
            "Select Guided Research Recipe",
            ["Custom Settings", 
             "Prognostic Biomarkers in Skin Melanoma (SKCM)", 
             "Breast Cancer Drug Response & Mutations (BRCA)", 
             "Pan-Cancer Mutation Landscape"]
        )
        
        # Determine defaults based on recipe selection
        default_workers = 2
        default_hyp = 3
        default_p = 0.05
        default_kf = True
        default_cohorts = ["Skin Cancer"]
        
        if recipe == "Prognostic Biomarkers in Skin Melanoma (SKCM)":
            default_workers = 2
            default_hyp = 3
            default_cohorts = ["Skin Cancer"]
            st.info("Recipe selected: Evaluates prognostic impact of gene mutations, stage, and pathology in Melanoma.")
        elif recipe == "Breast Cancer Drug Response & Mutations (BRCA)":
            default_workers = 3
            default_hyp = 4
            default_cohorts = ["Breast Cancer"]
            st.info("Recipe selected: Assesses mutation patterns, clinical features, and pathway interactions in Breast Cancer.")
        elif recipe == "Pan-Cancer Mutation Landscape":
            default_workers = 4
            default_hyp = 3
            default_cohorts = ["Skin Cancer", "Breast Cancer"]
            st.info("Recipe selected: Scans multiple cohorts to find common gene-survival linkages across cancer types.")
            
        col1, col2 = st.columns(2)
        with col1:
            workers = st.slider("Parallel Workers", 1, 8, default_workers)
            hyp_per_cohort = st.slider("Hypotheses per Cohort", 1, 10, default_hyp)
            sig_thresh = st.slider("Significance Threshold (p-value)", 0.01, 0.1, default_p, step=0.01)
        with col2:
            use_kf = st.checkbox("Use Knowledge Fabric", value=default_kf)
            # Available cohort names in EXPANDED_CANCER_PROJECTS
            cohort_names = list(EXPANDED_CANCER_PROJECTS.keys())
            selected_cohorts = st.multiselect("Select Cohorts", cohort_names, 
                                              default=[c for c in cohort_names if c in default_cohorts])
            
        if st.button("🚀 Execute Discovery Run"):
            if not selected_cohorts:
                st.error("Please select at least one cohort.")
            else:
                with st.spinner("Executing discovery engine. Running stats, ast-checking, and generating reports..."):
                    try:
                        # Translate cohort names to project IDs
                        project_ids = [EXPANDED_CANCER_PROJECTS[name] for name in selected_cohorts]
                        
                        config = DiscoveryConfig(
                            parallel_workers=workers,
                            hypothesis_per_cohort=hyp_per_cohort,
                            significance_threshold=sig_thresh,
                            use_knowledge_fabric=use_kf
                        )
                        # Map back to names for run_fast_discovery
                        cohort_names_mapped = [name.replace(" Cancer", "").replace(" Carcinoma", "") for name in selected_cohorts]
                        
                        from derma_agent.derma_core.agents.discovery_engine import run_fast_discovery
                        report = run_fast_discovery(
                            cancer_types=cohort_names_mapped,
                            config=config,
                            output_dir="discoveries"
                        )
                        st.success("Discovery completed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Discovery execution failed: {e}")
                        
    if not ledger:
        st.info("No discovery results found. Run a discovery using the generator above!")
        return
    
    # 2. Show Summary Metrics
    if report:
        st.subheader("Discovery Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Hypotheses Tested", report.get('total_hypotheses_tested', 0))
        with col2:
            st.metric("Significant Findings", report.get('significant_findings', 0))
        with col3:
            rate = report.get('significance_rate', 0) * 100
            st.metric("Significance Rate", f"{rate:.1f}%")
        with col4:
            exec_time = report.get('execution_time_seconds', 0)
            st.metric("Execution Time", f"{exec_time:.1f}s")
            
    # 3. Export Scientific Results
    st.subheader("📤 Export Scientific Results")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        # Convert ledger to CSV
        ledger_df = pd.DataFrame(ledger)
        csv = ledger_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Findings Ledger to CSV",
            data=csv,
            file_name=f"derma_agent_findings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_exp2:
        # Generate Markdown report
        report_md = f"""# DERMA-Agent Clinical Discovery Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Schema Version: {report.get('schema_version', '1.0') if report else '1.0'}

## Executive Summary
- **Total Hypotheses Tested:** {report.get('total_hypotheses_tested', 0) if report else len(ledger)}
- **Significant Findings (p <= {report.get('config', {}).get('significance_threshold', 0.05) if report else 0.05}):** {report.get('significant_findings', 0) if report else sum(1 for e in ledger if e.get('significant'))}
- **Discovery Rate:** {report.get('significance_rate', 0) * 100:.1f}%
- **Total Execution Time:** {report.get('execution_time_seconds', 0.0):.2f} seconds

## Detailed Discoveries Table
| Project ID | Hypothesis | P-Value | Hazard Ratio | Significant |
|------------|------------|---------|--------------|-------------|
"""
        for entry in ledger:
            p_val_str = f"{entry.get('p_value'):.4f}" if entry.get('p_value') is not None else "N/A"
            hr_str = f"{entry.get('hazard_ratio'):.2f}" if entry.get('hazard_ratio') is not None else "N/A"
            sig_str = "Yes" if entry.get('significant') else "No"
            report_md += f"| {entry.get('project_id')} | {entry.get('hypothesis')} | {p_val_str} | {hr_str} | {sig_str} |\n"
            
        report_md += "\n## Individual Hypotheses & Conclusions\n"
        for idx, entry in enumerate(ledger):
            report_md += f"### {idx+1}. {entry.get('hypothesis')}\n"
            report_md += f"- **Cohort:** {entry.get('project_id')}\n"
            report_md += f"- **P-Value:** {entry.get('p_value')}\n"
            if entry.get('hazard_ratio'):
                report_md += f"- **Hazard Ratio:** {entry.get('hazard_ratio'):.2f}\n"
            report_md += f"- **Conclusion:** {entry.get('conclusion')}\n"
            if entry.get('test_code'):
                report_md += f"\n**Statistical Code Run:**\n```python\n{entry.get('test_code')}\n```\n"
            report_md += "\n---\n"
            
        st.download_button(
            label="📥 Export Scientific Report to Markdown",
            data=report_md.encode('utf-8'),
            file_name=f"derma_agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    # 4. Visualizations & Analytics
    st.subheader("Discovery Analytics")
    col1, col2 = st.columns(2)
    
    with col1:
        p_values = [e.get('p_value') for e in ledger if e.get('p_value') is not None]
        if p_values:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(p_values, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
            ax.set_xlabel('P-value')
            ax.set_ylabel('Count')
            ax.set_title('P-value Distribution')
            ax.legend()
            st.pyplot(fig)
            
    with col2:
        # Forest plot of Hazard Ratios
        hr_entries = [e for e in ledger if e.get('hazard_ratio') is not None]
        if hr_entries:
            hr_df = pd.DataFrame([
                {
                    "Hypothesis": e.get('hypothesis')[:50] + "...",
                    "Hazard Ratio": e.get('hazard_ratio'),
                    "Significant": "Significant" if e.get('significant') else "Not Significant"
                }
                for e in hr_entries
            ])
            fig = px.scatter(hr_df, x="Hazard Ratio", y="Hypothesis", color="Significant",
                             title="Hazard Ratios of Discoveries",
                             symbol="Significant")
            fig.add_vline(x=1.0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        else:
            cohort_stats = {}
            for entry in ledger:
                pid = entry.get('project_id', 'unknown')
                if pid not in cohort_stats:
                    cohort_stats[pid] = {'total': 0, 'significant': 0}
                cohort_stats[pid]['total'] += 1
                if entry.get('significant'):
                    cohort_stats[pid]['significant'] += 1
            if cohort_stats:
                cohort_df = pd.DataFrame([
                    {'Cohort': k, 'Total': v['total'], 'Significant': v['significant']}
                    for k, v in cohort_stats.items()
                ])
                fig = px.bar(cohort_df, x='Cohort', y=['Total', 'Significant'],
                            title='Discoveries by Cohort', barmode='group')
                st.plotly_chart(fig, use_container_width=True)

    # 5. Interactive Kaplan-Meier Plotter (Interpretability Panel)
    st.subheader("📈 Interactive Kaplan-Meier Plotter")
    with st.expander("Generate Kaplan-Meier Curves from Cohort Data", expanded=False):
        cohort_list = list(set([entry.get('project_id') for entry in ledger if entry.get('project_id')]))
        if not cohort_list:
            cohort_list = ["TCGA-SKCM", "TCGA-BRCA"]
        
        sel_cohort = st.selectbox("Select Cohort to Plot", cohort_list)
        
        if st.button("📥 Load Cohort Data for Plotting"):
            with st.spinner("Fetching and prepping data..."):
                try:
                    data_client = get_data_client()
                    df_clin = data_client.get_survival_analysis_ready_data(sel_cohort)
                    
                    if not df_clin.empty:
                        # Try to load pathology if it exists
                        pathology_csv = f"data/pathology_{sel_cohort.lower()}_features.csv"
                        if os.path.exists(pathology_csv):
                            df_clin = data_client.merge_pathology_features(df_clin, pathology_csv)
                            
                        # Show column selector
                        cols_to_plot = [c for c in df_clin.columns if df_clin[c].nunique() > 1 and df_clin[c].nunique() < 10]
                        cont_cols = [c for c in df_clin.select_dtypes(include=[np.number]).columns 
                                     if c not in ['time', 'event', 'days_to_death', 'days_to_last_follow_up', 'age_at_diagnosis']]
                        
                        st.session_state['plot_df'] = df_clin
                        st.session_state['cols_to_plot'] = cols_to_plot
                        st.session_state['cont_cols'] = cont_cols
                        st.success(f"Loaded {len(df_clin)} samples for {sel_cohort}")
                    else:
                        st.warning("No data found for the selected cohort.")
                except Exception as e:
                    st.error(f"Error loading plot data: {e}")
                    
        if 'plot_df' in st.session_state and 'cols_to_plot' in st.session_state:
            df_plot = st.session_state['plot_df']
            cols = st.session_state['cols_to_plot']
            conts = st.session_state['cont_cols']
            
            plot_col = st.selectbox("Select Stratification Variable", cols + [f"{c} (Split by Median)" for c in conts])
            
            if st.button("Plot Kaplan-Meier Curve"):
                from lifelines import KaplanMeierFitter
                
                fig, ax = plt.subplots(figsize=(8, 5))
                kmf = KaplanMeierFitter()
                
                if "Split by Median" in plot_col:
                    base_col = plot_col.replace(" (Split by Median)", "")
                    median_val = df_plot[base_col].median()
                    df_plot['temp_group'] = df_plot[base_col].apply(lambda x: f"High (>= {median_val:.2f})" if x >= median_val else f"Low (< {median_val:.2f})")
                    group_col = 'temp_group'
                else:
                    group_col = plot_col
                    
                groups = df_plot[group_col].dropna().unique()
                
                for g in groups:
                    mask = df_plot[group_col] == g
                    if mask.sum() >= 3:
                        kmf.fit(df_plot.loc[mask, 'time'], df_plot.loc[mask, 'event'], label=str(g))
                        kmf.plot_survival_function(ax=ax, ci_show=False)
                
                ax.set_title(f'Kaplan-Meier Survival Curves by {plot_col} (Cohort: {sel_cohort})')
                ax.set_xlabel('Survival Time (Days)')
                ax.set_ylabel('Survival Probability')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

    # 6. Detailed Findings list
    st.subheader("Detailed Findings")
    show_only_significant = st.checkbox("Show only significant findings", value=False)
    
    for entry in ledger:
        if show_only_significant and not entry.get('significant'):
            continue
            
        is_sig = entry.get('significant', False)
        css_class = "significant-finding" if is_sig else "insignificant-finding"
        
        with st.container():
            st.markdown(f"<div class='{css_class}'>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{entry.get('hypothesis', 'Unknown')}**")
            with col2:
                if entry.get('p_value') is not None:
                    st.markdown(f"**p = {entry['p_value']:.4f}**")
                if is_sig:
                    st.markdown("🎯 **SIGNIFICANT**")
            
            with st.expander("Details"):
                st.write(f"**Cohort:** {entry.get('project_id', 'Unknown')}")
                st.write(f"**Execution Time:** {entry.get('execution_time', 0):.2f}s")
                st.write(f"**Conclusion:** {entry.get('conclusion', 'No conclusion')}")
                
                if entry.get('hazard_ratio'):
                    st.write(f"**Hazard Ratio:** {entry['hazard_ratio']:.2f}")
                
                if entry.get('test_code'):
                    st.code(entry['test_code'][:1000], language='python')
                
                if entry.get('execution_result'):
                    with st.expander("Execution Output"):
                        st.text(entry['execution_result'][:1000])
            
            st.markdown("</div>", unsafe_allow_html=True)


def render_data_explorer_tab():
    """Render the Data Explorer tab."""
    st.header("📊 Data Explorer")
    
    if not ENHANCED_MODE:
        st.warning("Enhanced data client not available.")
        return
    
    # Cancer type selector
    available_cancers = list(EXPANDED_CANCER_PROJECTS.items())[:10]  # Show first 10
    
    selected_cancer = st.selectbox(
        "Select Cancer Type",
        options=[c[0] for c in available_cancers],
        format_func=lambda x: f"{x} ({EXPANDED_CANCER_PROJECTS[x]})"
    )
    
    if selected_cancer:
        project_id = EXPANDED_CANCER_PROJECTS[selected_cancer]
        
        if st.button("📥 Load Data"):
            with st.spinner("Fetching data from GDC..."):
                try:
                    data_client = get_data_client()
                    df = data_client.get_survival_analysis_ready_data(project_id)
                    df = data_client.enrich_with_derived_features(df)
                    
                    # Store in session state
                    st.session_state['current_data'] = df
                    st.session_state['current_cancer'] = selected_cancer
                    st.success(f"Loaded {len(df)} samples")
                except Exception as e:
                    st.error(f"Error loading data: {e}")
    
    # Display loaded data
    if 'current_data' in st.session_state:
        df = st.session_state['current_data']
        
        st.subheader(f"Data Overview: {st.session_state['current_cancer']}")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Samples", len(df))
        with col2:
            if 'event' in df.columns:
                events = df['event'].sum()
                st.metric("Events", int(events))
        with col3:
            if 'time' in df.columns:
                median_time = df['time'].median()
                st.metric("Median Follow-up", f"{median_time:.0f} days")
        with col4:
            if 'age_years' in df.columns:
                median_age = df['age_years'].median()
                st.metric("Median Age", f"{median_age:.1f} years")
        
        # Data preview
        st.dataframe(df.head(10))
        
        # Visualizations
        st.subheader("Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Survival by stage
            if 'stage_group' in df.columns and 'time' in df.columns and 'event' in df.columns:
                from lifelines import KaplanMeierFitter
                
                fig, ax = plt.subplots()
                kmf = KaplanMeierFitter()
                
                for stage in df['stage_group'].dropna().unique()[:3]:
                    mask = df['stage_group'] == stage
                    if mask.sum() >= 5:
                        kmf.fit(df.loc[mask, 'time'], 
                               event_observed=df.loc[mask, 'event'],
                               label=stage)
                        kmf.plot_survival_function(ax=ax)
                
                ax.set_title('Survival by Stage')
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Survival Probability')
                st.pyplot(fig)
        
        with col2:
            # Age distribution
            if 'age_years' in df.columns:
                fig, ax = plt.subplots()
                df['age_years'].hist(bins=20, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_xlabel('Age (years)')
                ax.set_ylabel('Count')
                ax.set_title('Age Distribution')
                st.pyplot(fig)


def render_pathology_tab():
    """Render the Pathology tab."""
    st.header("🔬 Pathology Analysis")
    
    if not ENHANCED_MODE:
        st.warning("Enhanced pathology tools not available.")
        
        # Basic fallback
        st.subheader("Basic Saliency Analysis")
        if st.button("Generate Synthetic Image"):
            import numpy as np
            image = np.ones((256, 256, 3), dtype=np.uint8) * 200
            image[100:150, 100:150] = [100, 100, 200]
            st.image(image, caption="Synthetic Tissue Sample")
        return
    
    # Pattern selector
    pattern = st.selectbox("Tissue Pattern", ["mixed", "high_cellularity"])
    
    if st.button("🔬 Generate & Analyze"):
        with st.spinner("Analyzing tissue..."):
            # Generate synthetic image
            image = create_synthetic_pathology_image((512, 512), pattern)
            
            # Analyze
            analyzer = TissueAnalyzer()
            segmenter = NucleiSegmenter()
            
            features = analyzer.analyze_tile(image)
            mask = segmenter.segment_nuclei(image)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            
            with col2:
                st.image(mask.astype(float), caption="Nuclei Mask", use_column_width=True)
            
            with col3:
                heatmap = analyzer.generate_heatmap(image, mask, 'density')
                st.image(heatmap, caption="Density Heatmap", use_column_width=True)
            
            # Feature metrics
            st.subheader("Extracted Features")
            
            cols = st.columns(4)
            metrics = [
                ("Nuclei Count", features.nuclei_count),
                ("Nuclei Density", f"{features.nuclei_density:.2f}"),
                ("Cellularity", f"{features.cellularity:.3f}"),
                ("Tissue Area", f"{features.tissue_area_ratio:.3f}")
            ]
            
            for col, (name, value) in zip(cols, metrics):
                col.metric(name, value)
            
            # Texture features
            if features.texture_features:
                st.subheader("Texture Features")
                texture_df = pd.DataFrame(list(features.texture_features.items()),
                                        columns=['Feature', 'Value'])
                st.dataframe(texture_df)


def render_settings_tab():
    """Render the Settings tab."""
    st.header("⚙️ Settings")
    
    st.subheader("Discovery Configuration")
    
    config = {}
    config['parallel_workers'] = st.slider("Parallel Workers", 1, 8, 2)
    config['hypotheses_per_cohort'] = st.slider("Hypotheses per Cohort", 1, 10, 3)
    config['significance_threshold'] = st.slider("Significance Threshold (p-value)", 0.01, 0.1, 0.05)
    config['use_knowledge_fabric'] = st.checkbox("Use Knowledge Fabric", value=True)
    
    st.session_state['discovery_config'] = config
    
    st.subheader("API Configuration")
    openai_key = os.getenv("OPENAI_API_KEY")
    st.write(f"OpenAI API Key: {'✅ Configured' if openai_key else '❌ Not set'}")
    
    if not openai_key:
        st.info("Set OPENAI_API_KEY environment variable for full functionality")


def main():
    """Main application."""
    # Header
    st.markdown('<p class="main-header">🔬 DERMA-Agent</p>', unsafe_allow_html=True)
    st.markdown("### Autonomous Cancer Pathology Discovery Platform")
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Mode indicator
    if ENHANCED_MODE:
        st.sidebar.success("✅ Enhanced Mode")
    else:
        st.sidebar.warning("⚠️ Basic Mode")
    
    # Navigation
    tab = st.sidebar.radio(
        "Select Tab",
        ["🧠 Knowledge Fabric", "🔬 Discovery", "📊 Data Explorer", "🧬 Pathology", "⚙️ Settings"]
    )
    
    # File uploader for discovery results
    st.sidebar.markdown("---")
    st.sidebar.subheader("Import Results")
    uploaded_file = st.sidebar.file_uploader("Upload discovery JSON", type=['json'])
    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            st.sidebar.success("File loaded successfully")
        except:
            st.sidebar.error("Invalid JSON file")
    
    # Render selected tab
    if tab == "🧠 Knowledge Fabric":
        render_knowledge_fabric_tab()
    elif tab == "🔬 Discovery":
        render_discovery_tab()
    elif tab == "📊 Data Explorer":
        render_data_explorer_tab()
    elif tab == "🧬 Pathology":
        render_pathology_tab()
    elif tab == "⚙️ Settings":
        render_settings_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("<center>DERMA-Agent | Powered by LangGraph, OpenAI, and TCGA</center>", 
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
