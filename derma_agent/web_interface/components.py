import streamlit as st

def render_metric_card(title: str, value: str, icon: str = "🔬", color: str = "#1f77b4"):
    """
    Renders a premium visual card for statistics and metrics.
    """
    st.markdown(f"""
    <div style="
        background-color: #f8f9fa;
        padding: 1.25rem;
        border-radius: 0.75rem;
        border-left: 5px solid {color};
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    ">
        <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
        <span style="font-size: 0.9rem; color: #6c757d; font-weight: 500; text-transform: uppercase;">{title}</span>
        <div style="font-size: 1.8rem; font-weight: 700; color: #212529; margin-top: 0.25rem;">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def render_thought_card(log_str: str, index: int):
    """
    Renders an agent's research narrative entry as a stylized timeline item.
    """
    is_error = "Error" in log_str or "Violation" in log_str
    border_color = "#dc3545" if is_error else "#28a745"
    bg_color = "#fdf3f4" if is_error else "#f4fcf6"
    
    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid {border_color};
        margin-bottom: 0.75rem;
        font-family: monospace;
        font-size: 0.9rem;
    ">
        <strong style="color: #495057;">🔬 Log Entry #{index}</strong><br/>
        {log_str}
    </div>
    """, unsafe_allow_html=True)
