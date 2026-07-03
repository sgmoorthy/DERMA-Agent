import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
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
    return _buf(fig), p_val


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
        1 - titan_features["cellularity"] * 0.4,   # mock uniformity
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
    """
    Quick visualisation of the 768-dim APOLLO embedding:
    – Bar chart of top-20 absolute magnitude dimensions
    – Background distribution shown as fill_between
    """
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
    Full agentic perception-action loop with rich visualisations at every phase.
    """
    # ── Step 1: Perception ──────────────────────────────────────────────────
    with st.status("🔭 Phase 1 — WSI Perception & Foundation Model Inference", expanded=True) as s:
        wsi = WSIEngine()
        slide_metadata = wsi.ingest_slide(tissue_slide_id)
        titan_features  = wsi.get_titan_classification(tissue_slide_id)
        apollo_emb      = wsi.get_apollo_embeddings(tissue_slide_id)

        st.write(f"✅ Slide `{tissue_slide_id}` ingested — dims {slide_metadata['dimensions']}, "
                 f"{slide_metadata['microns_per_pixel']} µm/px")
        st.write(f"✅ TITAN classification: **{titan_features['primary_pattern']}** pattern, "
                 f"cellularity **{titan_features['cellularity']:.2f}**, "
                 f"nuclei count **{titan_features['nuclei_count']}**")
        st.write(f"✅ APOLLO embedding generated — shape {apollo_emb.shape}")
        s.update(label="✅ Phase 1 complete", state="complete")

    # Perception visualisations (side by side)
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("#### 🎯 TITAN Tissue Characterisation")
        st.image(plot_tissue_radar(titan_features), use_container_width=True)
    with v2:
        st.markdown("#### 🧬 APOLLO Foundation Model Embeddings")
        st.image(plot_apollo_embedding(apollo_emb, tissue_slide_id), use_container_width=True)

    st.divider()

    # ── Step 2: Knowledge Fabric Consultation ───────────────────────────────
    with st.status("🧠 Phase 2 — Knowledge Fabric Consultation & Hypothesis Formation", expanded=True) as s:
        fabric = st.session_state.fabric
        priors = fabric.query_context("BRAF")
        st.write(f"✅ BRAF neighbourhood: **{priors}**")

        hypothesis = (
            f"In cohort **{cohort}**, BRAF mutation status is prognostic of survival, "
            f"correlated with elevated cellularity ({titan_features['cellularity']:.2f}) "
            f"and **{titan_features['primary_pattern']}** tissue pattern."
        )
        st.markdown(f"💡 **Formulated Hypothesis**: {hypothesis}")
        s.update(label="✅ Phase 2 complete", state="complete")

    st.markdown("#### 📡 Live Knowledge Fabric Network")
    st.image(plot_kg_graph(fabric), use_container_width=True)

    st.divider()

    # ── Step 3: Generate Statistical Code + AST Critic ──────────────────────
    with st.status("🛡️ Phase 3 — CodeAct Generation & AST Security Audit", expanded=True) as s:
        rng = np.random.default_rng(int(sum(ord(c) for c in tissue_slide_id)) % 10000)
        n   = 200
        mock_df = pd.DataFrame({
            "time":           rng.exponential(600, n) + 100,
            "event":          rng.binomial(1, 0.4, n),
            "is_braf_mutated": rng.binomial(1, 0.35, n),
            "cellularity":    rng.uniform(0.1, 0.8, n),
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
        icon = "✅" if is_safe else "❌"
        st.write(f"{icon} **AST Audit**: {msg}")

        with st.expander("📄 View Auto-Generated CodeAct Script"):
            st.code(stat_code, language="python")

        if not is_safe:
            st.error(f"Execution blocked by Critic Agent: {msg}")
            st.session_state.narrative.log_discovery(
                hypothesis.replace("**", ""), f"BLOCKED — {msg}")
            s.update(label="❌ Phase 3 — Execution blocked", state="error")
            return

        s.update(label="✅ Phase 3 complete — Code approved", state="complete")

    # ── Step 4: Sandbox Execution ───────────────────────────────────────────
    with st.status("⚡ Phase 4 — Restricted Sandbox Execution", expanded=True) as s:
        executor = CodeExecutor()
        output   = executor.execute(stat_code, {"df": mock_df})
        if "Execution Error" in output:
            st.error(output)
            s.update(label="❌ Phase 4 — Runtime error", state="error")
            return
        st.write("✅ Code executed successfully inside restricted namespace.")
        st.code(output, language="bash")
        s.update(label="✅ Phase 4 complete", state="complete")

    st.divider()

    # ── Step 5: Visualisation Results ───────────────────────────────────────
    st.markdown("## 📊 Discovery Results")

    r1, r2 = st.columns(2)
    with r1:
        km_buf, p_val = plot_kaplan_meier(mock_df)
        st.markdown("#### 📈 Kaplan-Meier Survival Curves")
        st.image(km_buf, use_container_width=True)

        sig_label  = "🟢 **SIGNIFICANT** (p < 0.05)" if p_val < 0.05 else "🟡 **Not significant** (p ≥ 0.05)"
        st.markdown(f"Log-rank test: {sig_label}")

    with r2:
        st.markdown("#### 🔬 Cellularity vs Survival Time")
        st.image(plot_cellularity_survival(mock_df), use_container_width=True)

    # Statistical summary table
    mut_times = mock_df.loc[mock_df["is_braf_mutated"] == 1, "time"]
    wt_times  = mock_df.loc[mock_df["is_braf_mutated"] == 0, "time"]
    _, mwu_p  = mannwhitneyu(mut_times, wt_times, alternative="two-sided")

    stats_df = pd.DataFrame({
        "Metric": ["Cohort", "Slide ID", "Primary Pattern", "Cellularity",
                   "BRAF Mutated (n)", "BRAF WT (n)", "Median Survival — Mutated (days)",
                   "Median Survival — WT (days)", "Log-rank p-value", "Mann-Whitney U p-value"],
        "Value":  [
            cohort, tissue_slide_id, titan_features["primary_pattern"],
            f"{titan_features['cellularity']:.3f}",
            str(mock_df["is_braf_mutated"].sum()),
            str((~mock_df["is_braf_mutated"].astype(bool)).sum()),
            f"{mut_times.median():.1f}", f"{wt_times.median():.1f}",
            f"{p_val:.4f}", f"{mwu_p:.4f}",
        ],
    })
    st.markdown("#### 📋 Statistical Summary")
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # ── Step 6: Update Knowledge Fabric & Episodic Log ──────────────────────
    fabric.add_relationship("BRAF", cohort,
                            f"PROGNOSTIC_p{p_val:.3f}_{titan_features['primary_pattern']}")
    st.session_state.narrative.log_discovery(
        hypothesis.replace("**", ""),
        f"KM p={p_val:.4f} | Mutated median={mut_times.median():.0f}d | WT median={wt_times.median():.0f}d"
    )
    st.success("🔬 Discovery logged. Knowledge Fabric updated. Scroll up to see the live graph refresh.")


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard Layout
# ─────────────────────────────────────────────────────────────────────────────

def render_dashboard():
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
    with m3: render_metric_card("Hypotheses Tested",  str(len(history)),               "💡", "#f6c90e")
    with m4: render_metric_card("Sandbox Executions", str(st.session_state.get("runs", 0)), "⚡", "#e63946")

    st.divider()

    # ── Main layout: KG (left) | Narrative (right) ───────────────────────────
    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.markdown("### 📚 Live Knowledge Fabric")
        st.image(plot_kg_graph(fabric), use_container_width=True)

        st.markdown("##### Active Graph Mappings")
        edges_list = [{"Source": u, "Target": v, "Relation": d.get("relation", "")}
                      for u, v, d in fabric.graph.edges(data=True)]
        if edges_list:
            st.dataframe(pd.DataFrame(edges_list), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### 🧠 Episodic Research Narrative")
        if history:
            for idx, log in enumerate(reversed(history[-6:])):   # Show last 6
                render_thought_card(log, len(history) - idx)
        else:
            st.info("💡 Initiate a discovery run to populate the research narrative.")

    st.divider()

    # ── Run loop ─────────────────────────────────────────────────────────────
    if run_btn:
        st.session_state["runs"] = st.session_state.get("runs", 0) + 1
        run_scientific_workflow(cohort, slide_id)
        st.rerun()
