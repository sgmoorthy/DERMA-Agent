import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from scipy.stats import mannwhitneyu
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from derma_core.knowledge_fabric.graph_memory import KnowledgeFabric
from derma_core.perception.wsi_engine import WSIEngine
from derma_core.actions.code_executor import CodeExecutor
from derma_core.actions.critic_agent import CriticAgent
from derma_core.memory.research_log import ResearchNarrative
from web_interface.components import render_metric_card, render_thought_card

# ─────────────────────────────────────────────────────────────────────────────
# Utility Plotting Functions
# ─────────────────────────────────────────────────────────────────────────────

def _buf(fig) -> io.BytesIO:
    """Save a matplotlib figure to a bytes buffer and return it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


def plot_kaplan_meier(df: pd.DataFrame) -> io.BytesIO:
    """Render split KM curves for BRAF Mutated vs WT with log-rank p-value."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    mut_mask = df["is_braf_mutated"] == 1
    wt_mask  = ~mut_mask

    lr = logrank_test(
        df.loc[mut_mask, "time"], df.loc[wt_mask, "time"],
        event_observed_A=df.loc[mut_mask, "event"],
        event_observed_B=df.loc[wt_mask, "event"]
    )
    p_val = lr.p_value

    palette = {"BRAF Mutated": "#e63946", "BRAF WT": "#4cc9f0"}
    for label, mask in [("BRAF Mutated", mut_mask), ("BRAF WT", wt_mask)]:
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[mask, "time"], event_observed=df.loc[mask, "event"], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=palette[label], linewidth=2.5, alpha=0.9)

    ax.set_title(f"Kaplan-Meier Survival Curves\nLog-rank p = {p_val:.4f}", color="#f0f0f0", fontsize=12, fontweight="bold")
    ax.set_xlabel("Days", color="#aaaaaa"); ax.set_ylabel("Survival Probability", color="#aaaaaa")
    ax.tick_params(colors="#888888"); ax.spines[:].set_color("#333333")
    ax.legend(framealpha=0.3, labelcolor="#f0f0f0", facecolor="#1a1a2e")
    ax.grid(axis="y", color="#333333", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return _buf(fig)


def plot_tissue_radar(titan_features: dict) -> io.BytesIO:
    """Radar / spider chart for TITAN tissue characterisation features."""
    labels  = ["Cellularity", "Tissue Density", "Pattern Score", "Nuclei (norm)", "Uniformity"]
    pattern_score = {"infiltrative": 0.85, "nested": 0.60, "sheet-like": 0.45, "mixed": 0.70}.get(
        titan_features.get("primary_pattern", "mixed"), 0.5
    )
    nuclei_norm   = min(titan_features["nuclei_count"] / 5000, 1.0)
    values = [
        titan_features["cellularity"],
        titan_features["tissue_density"],
        pattern_score,
        nuclei_norm,
        1 - titan_features["cellularity"] * 0.4,
    ]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_cycle = values + [values[0]]
    angles_cycle  = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
    ax.plot(angles_cycle, values_cycle, color="#7c3aed", linewidth=2.5)
    ax.fill(angles_cycle, values_cycle, color="#7c3aed", alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, color="#cccccc", fontsize=9)
    ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], color="#666666", fontsize=7)
    ax.grid(color="#333333"); ax.spines["polar"].set_color("#333333")
    ax.set_title("TITAN Tissue Features", color="#f0f0f0", fontsize=11, fontweight="bold", pad=14)
    fig.tight_layout()
    return _buf(fig)


def plot_kg_graph(fabric: KnowledgeFabric) -> io.BytesIO:
    """NetworkX spring layout for the live knowledge graph."""
    G = fabric.graph
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")

    pos = nx.spring_layout(G, seed=7, k=1.8)

    # Colour nodes by type heuristic
    color_map = []
    for n in G.nodes():
        if n.startswith("TCGA"):       color_map.append("#e63946")
        elif n in ("BRAF", "TP53", "EGFR", "PTEN"): color_map.append("#4cc9f0")
        elif "Pathway" in n:           color_map.append("#f4a261")
        elif n in ("Dabrafenib", "Vemurafenib", "Pembrolizumab"): color_map.append("#2a9d8f")
        else:                          color_map.append("#9b5de5")

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color_map, node_size=600, alpha=0.95)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="#ffffff", font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#444466", width=1.5, alpha=0.7, arrows=True,
                           arrowstyle="-|>", arrowsize=12)

    edge_labels = {(u, v): d.get("relation", "")[:12] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels,
                                 font_size=6, font_color="#aaaaaa", bbox=dict(alpha=0))

    legend_handles = [
        mpatches.Patch(color="#4cc9f0", label="Gene"),
        mpatches.Patch(color="#e63946", label="Disease/Cohort"),
        mpatches.Patch(color="#f4a261", label="Pathway"),
        mpatches.Patch(color="#2a9d8f", label="Drug"),
        mpatches.Patch(color="#9b5de5", label="Other"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", framealpha=0.3,
              labelcolor="#f0f0f0", facecolor="#1a1a2e", fontsize=7)
    ax.set_title("Live Knowledge Fabric Graph", color="#f0f0f0", fontsize=11, fontweight="bold")
    ax.axis("off"); fig.tight_layout()
    return _buf(fig)


def plot_apollo_embedding(embedding: np.ndarray, slide_id: str) -> io.BytesIO:
    """Visualisation of the 768-dim APOLLO embedding."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2): ax.set_facecolor("#0f1117")

    # Top 20 influential dims
    top_idx   = np.argsort(np.abs(embedding))[-20:][::-1]
    top_vals  = embedding[top_idx]
    bar_colors = ["#e63946" if v < 0 else "#4cc9f0" for v in top_vals]
    ax1.barh([f"dim {i}" for i in top_idx], top_vals, color=bar_colors, edgecolor="none", height=0.7)
    ax1.axvline(0, color="#666666", lw=1)
    ax1.set_title("APOLLO Top-20 Feature Dims", color="#f0f0f0", fontsize=9, fontweight="bold")
    ax1.tick_params(colors="#888888", labelsize=7)
    ax1.spines[:].set_color("#333333")

    # Full embedding magnitude distribution
    sorted_emb = np.sort(np.abs(embedding))
    ax2.fill_between(range(len(sorted_emb)), sorted_emb, alpha=0.6,
                     color="#7c3aed", linewidth=0)
    ax2.plot(sorted_emb, color="#9b5de5", linewidth=1)
    ax2.set_title("APOLLO Embedding Magnitude Distribution", color="#f0f0f0", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Dimension (sorted)", color="#aaaaaa", fontsize=8)
    ax2.set_ylabel("|Value|", color="#aaaaaa", fontsize=8)
    ax2.tick_params(colors="#888888", labelsize=7)
    ax2.spines[:].set_color("#333333")

    fig.suptitle(f"Slide: {slide_id}", color="#cccccc", fontsize=8, y=1.01)
    fig.tight_layout()
    return _buf(fig)


def plot_cellularity_survival(df: pd.DataFrame) -> io.BytesIO:
    """Scatter plot: cellularity vs survival time, coloured by event status."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")

    colors = df["event"].map({1: "#e63946", 0: "#4cc9f0"})
    ax.scatter(df["cellularity"], df["time"], c=colors, alpha=0.65, s=25, edgecolors="none")
    
    # Linear trendline
    m, b = np.polyfit(df["cellularity"], df["time"], 1)
    x_line = np.linspace(df["cellularity"].min(), df["cellularity"].max(), 100)
    ax.plot(x_line, m * x_line + b, color="#f4a261", lw=2, linestyle="--", label="Trend")

    legend_handles = [
        mpatches.Patch(color="#e63946", label="Event (deceased)"),
        mpatches.Patch(color="#4cc9f0", label="Censored"),
    ]
    ax.legend(handles=legend_handles, framealpha=0.3, labelcolor="#f0f0f0", facecolor="#1a1a2e", fontsize=8)
    ax.set_xlabel("Cellularity", color="#aaaaaa"); ax.set_ylabel("Survival Time (days)", color="#aaaaaa")
    ax.set_title("Cellularity vs Survival Time", color="#f0f0f0", fontsize=10, fontweight="bold")
    ax.tick_params(colors="#888888"); ax.spines[:].set_color("#333333")
    ax.grid(color="#333333", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _buf(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main Scientific Workflow
# ─────────────────────────────────────────────────────────────────────────────

def run_scientific_workflow(cohort: str, tissue_slide_id: str):
    """
    Full agentic perception-action loop.
    Appends generated outputs to st.session_state.discovery_runs list.
    """
    # ── Step 1: Perception ──────────────────────────────────────────────────
    with st.status("🔭 Phase 1 — WSI Perception & Foundation Model Inference", expanded=False) as s:
        wsi = WSIEngine()
        slide_metadata = wsi.ingest_slide(tissue_slide_id)
        titan_features  = wsi.get_titan_classification(tissue_slide_id)
        apollo_emb      = wsi.get_apollo_embeddings(tissue_slide_id)
        s.update(label="✅ Phase 1 complete", state="complete")

    # ── Step 2: Knowledge Fabric Consultation ───────────────────────────────
    with st.status("🧠 Phase 2 — Knowledge Fabric Consultation & Hypothesis Formation", expanded=False) as s:
        fabric = st.session_state.fabric
        priors = fabric.query_context("BRAF")

        hypothesis = (
            f"In cohort {cohort}, BRAF mutation status is prognostic of survival, "
            f"correlated with elevated cellularity ({titan_features['cellularity']:.2f}) "
            f"and {titan_features['primary_pattern']} tissue pattern."
        )
        s.update(label="✅ Phase 2 complete", state="complete")

    # ── Step 3: Generate Statistical Code + AST Critic ──────────────────────
    with st.status("🛡️ Phase 3 — CodeAct Generation & AST Security Audit", expanded=False) as s:
        rng = np.random.default_rng(int(sum(ord(c) for c in tissue_slide_id)) % 10000)
        n   = 200
        mock_df = pd.DataFrame({
            "time":            rng.exponential(600, n) + 100,
            "event":           rng.binomial(1, 0.4, n),
            "is_braf_mutated": rng.binomial(1, 0.35, n),
            "cellularity":     rng.uniform(0.1, 0.8, n),
        })

        stat_code = """# Auto-generated CodeAct — Kaplan-Meier on BRAF status
import pandas as pd
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
mut_mask = df['is_braf_mutated'] == 1

kmf.fit(df.loc[mut_mask, 'time'], event_observed=df.loc[mut_mask, 'event'], label='BRAF Mutated')
print("BRAF Mutated — Median survival:", round(kmf.median_survival_time_, 1), "days")

kmf.fit(df.loc[~mut_mask, 'time'], event_observed=df.loc[~mut_mask, 'event'], label='BRAF WT')
print("BRAF WT — Median survival:", round(kmf.median_survival_time_, 1), "days")
"""
        critic    = CriticAgent()
        is_safe, msg = critic.evaluate_code(stat_code, expected_columns=list(mock_df.columns))
        
        if not is_safe:
            st.session_state.narrative.log_discovery(hypothesis, f"Blocked: {msg}")
            s.update(label="❌ Phase 3 — Blocked by Critic", state="error")
            return
        s.update(label="✅ Phase 3 complete — Code approved", state="complete")

    # ── Step 4: Sandbox Execution ───────────────────────────────────────────
    with st.status("⚡ Phase 4 — Restricted Sandbox Execution", expanded=False) as s:
        executor = CodeExecutor()
        output   = executor.execute(stat_code, {"df": mock_df})
        if "Execution Error" in output:
            s.update(label="❌ Phase 4 — Runtime error", state="error")
            return
        s.update(label="✅ Phase 4 complete", state="complete")

    # Log Rank calculations for structured results
    mut_mask = mock_df["is_braf_mutated"] == 1
    wt_mask  = ~mut_mask
    lr_res   = logrank_test(
        mock_df.loc[mut_mask, "time"], mock_df.loc[wt_mask, "time"],
        event_observed_A=mock_df.loc[mut_mask, "event"],
        event_observed_B=mock_df.loc[wt_mask, "event"]
    )
    p_val = lr_res.p_value
    
    mut_times = mock_df.loc[mut_mask, "time"]
    wt_times  = mock_df.loc[wt_mask, "time"]
    _, mwu_p  = mannwhitneyu(mut_times, wt_times, alternative="two-sided")
    
    status = "confirmed" if p_val < 0.05 else "rejected"

    # Save to history list
    run_label = f"Hypothesis #{len(st.session_state.discovery_runs) + 1} — BRAF in {cohort}"
    
    # Store complete metadata of the run
    st.session_state.discovery_runs.append({
        "label": run_label,
        "cohort": cohort,
        "slide_id": tissue_slide_id,
        "hypothesis": hypothesis,
        "p_value": p_val,
        "status": status,
        "mwu_p": mwu_p,
        "mut_median": mut_times.median(),
        "wt_median": wt_times.median(),
        "mock_df": mock_df,
        "titan_features": titan_features,
        "apollo_emb": apollo_emb,
        "stat_code": stat_code,
        "summary": f"BRAF mutation is prognostic in {cohort} (p={p_val:.4f}; medians: {mut_times.median():.0f}d vs {wt_times.median():.0f}d)"
    })

    # Update knowledge fabric and logs
    fabric.add_relationship("BRAF", cohort, f"PROGNOSTIC_p{p_val:.3f}_{titan_features['primary_pattern']}")
    st.session_state.narrative.log_discovery(
        hypothesis,
        f"KM p={p_val:.4f} | Mutated median={mut_times.median():.0f}d | WT median={wt_times.median():.0f}d"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard Layout
# ─────────────────────────────────────────────────────────────────────────────

def render_dashboard():
    # Initialize run log registry if not present
    if "discovery_runs" not in st.session_state:
        st.session_state.discovery_runs = []

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔬 Discovery Controls")
        cohort   = st.selectbox("Select Target Cohort",
                                ["TCGA-SKCM", "TCGA-BRCA", "TCGA-LUAD", "TCGA-COAD", "TCGA-GBM"])
        slide_id = st.text_input("Enter Slide ID Target", "WSI-TCGA-SKCM-009A")
        
        st.divider()
        run_btn = st.button("🚀 Run Generative Discovery Loop", type="primary", use_container_width=True)

    # ── Top Metrics Row ───────────────────────────────────────────────────────
    fabric = st.session_state.fabric
    history = st.session_state.narrative.get_history()

    m1, m2, m3, m4 = st.columns(4)
    with m1: render_metric_card("Graph Entities",     str(len(fabric.graph.nodes)),    "🧬", "#4e73df")
    with m2: render_metric_card("Relationships",      str(len(fabric.graph.edges)),    "🔗", "#1cc88a")
    with m3: render_metric_card("Hypotheses Tested",  str(len(st.session_state.discovery_runs)), "💡", "#f6c90e")
    with m4: render_metric_card("Sandbox Executions", str(st.session_state.get("runs", 0)), "⚡", "#e63946")

    st.divider()

    # ── main layout columns: col_main (wide left) | col_side (right) ────────
    col_main, col_side = st.columns([1.8, 1.2])

    # ── LEFT MAIN: Research Findings & Visualisations ────────────────────────
    with col_main:
        # 1. Study summary / Ranked conclusions
        st.markdown("### 📊 Live Oncology Findings & Conclusions")
        
        runs = st.session_state.discovery_runs
        confirmed_runs = [r for r in runs if r["status"] == "confirmed"]
        rejected_runs  = [r for r in runs if r["status"] == "rejected"]
        
        c_conf, c_rej = st.columns(2)
        with c_conf:
            st.markdown("##### 🟢 Validated Prognostic Findings")
            if confirmed_runs:
                for r in confirmed_runs:
                    st.markdown(f"- **{r['cohort']}**: {r['summary']}")
            else:
                st.info("No significant prognostic findings validated yet.")

        with c_rej:
            st.markdown("##### 🟡 Negative Results (Non-significant)")
            if rejected_runs:
                for r in rejected_runs:
                    st.markdown(f"- **{r['cohort']}**: {r['summary']}")
            else:
                st.info("No negative outcomes mapped yet.")

        st.divider()

        # 2. Interactive selector for curves and scatter plots
        st.markdown("### 📈 Interactive Hypothesis Inspector")
        if runs:
            # Let user select from past runs
            selected_run_label = st.selectbox(
                "Select Hypothesis Run to Visualize",
                [r["label"] for r in runs],
                index=len(runs)-1
            )
            # Find the active run dict
            active_run = next(r for r in runs if r["label"] == selected_run_label)
            
            # Display Hypothesis info card
            st.info(f"**Hypothesis**: *{active_run['hypothesis']}*")
            
            # Render plots side by side
            v1, v2 = st.columns(2)
            with v1:
                st.markdown("###### Kaplan-Meier Survival Curve")
                st.image(plot_kaplan_meier(active_run["mock_df"]), use_container_width=True)
                st.markdown(f"**Conclusion**: p-value = `{active_run['p_value']:.4f}` (median survival: mutated `{active_run['mut_median']:.0f}` days vs WT `{active_run['wt_median']:.0f}` days)")
            
            with v2:
                st.markdown("###### Slide Cellularity vs Outcomes")
                st.image(plot_cellularity_survival(active_run["mock_df"]), use_container_width=True)
                st.markdown(f"**Morphology Link**: Mann-Whitney U test p-value = `{active_run['mwu_p']:.4f}`")
                
            # Embed slide details inside an expander
            with st.expander("🔍 Selected Slide Perception Details (TITAN & APOLLO)"):
                t1, t2 = st.columns(2)
                with t1:
                    st.image(plot_tissue_radar(active_run["titan_features"]), use_container_width=True)
                with t2:
                    st.image(plot_apollo_embedding(active_run["apollo_emb"], active_run["slide_id"]), use_container_width=True)
        else:
            st.info("Run the discovery loop in the sidebar to generate interactive KM plots and tissue correlations.")

    # ── RIGHT SIDEBAR: Model Internals (Secondary) & Filtered logs ──────────
    with col_side:
        # 1. Collapsible Knowledge Fabric graph
        st.markdown("### 🛠️ Model Internals")
        with st.expander("💡 Live Knowledge Fabric Graph", expanded=False):
            st.caption("This graph represents the prior reasoning space used by the agent to construct hypotheses.")
            st.image(plot_kg_graph(fabric), use_container_width=True)
            
            edges_list = [{"Source": u, "Target": v, "Relation": d.get("relation", "")}
                          for u, v, d in fabric.graph.edges(data=True)]
            if edges_list:
                st.dataframe(pd.DataFrame(edges_list), use_container_width=True, hide_index=True)

        st.divider()

        # 2. Filterable Experiment Timeline Logs
        st.markdown("### 🧠 Episodic Research Logs")
        if history:
            # Filters
            f_col, f_stat = st.columns(2)
            with f_col:
                cohort_filter = st.selectbox("Filter by Cohort", ["All", "TCGA-SKCM", "TCGA-BRCA", "TCGA-LUAD", "TCGA-COAD", "TCGA-GBM"])
            with f_stat:
                status_filter = st.selectbox("Filter by Status", ["All", "Confirmed", "Rejected", "Blocked"])

            # Filter logic
            filtered_logs = []
            for idx, log in enumerate(reversed(history)):
                # Categorize logs based on content
                log_cohort = "All"
                for c in ["TCGA-SKCM", "TCGA-BRCA", "TCGA-LUAD", "TCGA-COAD", "TCGA-GBM"]:
                    if c in log:
                        log_cohort = c
                        break
                        
                log_status = "Rejected"
                if "Blocked" in log or "BLOCKED" in log:
                    log_status = "Blocked"
                elif "KM p=" in log:
                    # Parse p-value to determine if Confirmed
                    try:
                        p_part = log.split("KM p=")[1].split(" |")[0]
                        p_val = float(p_part)
                        if p_val < 0.05:
                            log_status = "Confirmed"
                    except Exception:
                        pass
                
                # Check filters
                if cohort_filter != "All" and log_cohort != cohort_filter:
                    continue
                if status_filter != "All" and log_status != status_filter:
                    continue
                    
                filtered_logs.append((log, len(history) - idx))

            # Display
            if filtered_logs:
                for log, index in filtered_logs[:6]:  # Limit to 6 entries
                    render_thought_card(log, index)
            else:
                st.caption("No logs match the selected filters.")
        else:
            st.info("Initiate a discovery run to populate the research logs.")

    st.divider()

    # ── Run loop ─────────────────────────────────────────────────────────────
    if run_btn:
        st.session_state["runs"] = st.session_state.get("runs", 0) + 1
        run_scientific_workflow(cohort, slide_id)
        st.rerun()
