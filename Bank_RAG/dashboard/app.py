import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import load_logs, format_metric

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.rag import run_rag_pipeline  # noqa: E402
st.set_page_config(page_title="BANK_RAG Observability Dashboard", layout="wide")

# Paths
LOG_PATH = BASE_DIR / "logs" / "rag_traces.jsonl"

st.title("Autonomous Self-Healing LLM Ops Dashboard")
st.caption("Semantic Validation • Trust Scoring • Self-Healing Monitoring")

# Section 1: Live Query Interface
section_live = st.container()
with section_live:
    st.subheader("Live Query Interface")
    query = st.text_input("Enter a banking question")
    submit = st.button("Submit", type="primary")

    result = None
    error_msg = None

    if submit:
        if not query:
            error_msg = "Please enter a question before submitting."
        else:
            with st.spinner("Running RAG pipeline..."):
                try:
                    result = run_rag_pipeline(query)
                except Exception as exc:  # noqa: BLE001
                    error_msg = str(exc)

    answer_value = "(placeholder)"
    retrieval_confidence_value = "(placeholder)"
    confidence_level_value = "(placeholder)"
    hallucination_risk_value = "(placeholder)"
    risk_level_value = "(placeholder)"
    self_healing_value = "(placeholder)"
    refusal_value = "(placeholder)"
    latency_retrieval = "(placeholder)"
    latency_generation = "(placeholder)"
    latency_total = "(placeholder)"

    similarity_scores = None

    if result:
        answer_value = result.get("answer", "")
        retrieval_confidence_value = format_metric(result.get("retrieval_confidence"))
        confidence_level_value = result.get("confidence_level", "")
        hallucination_risk_value = format_metric(result.get("hallucination_risk"))
        risk_level_value = result.get("risk_level", "")
        self_healing_value = "Yes" if result.get("self_healing_triggered") else "No"
        refusal_value = "Yes" if result.get("refusal_detected") else "No"
        latency = result.get("latency", {})
        latency_retrieval = format_metric(latency.get("retrieval_ms"), precision=2)
        latency_generation = format_metric(latency.get("generation_ms"), precision=2)
        latency_total = format_metric(latency.get("total_ms"), precision=2)
        similarity_scores = result.get("similarity_scores")

    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Generated Answer**")
        st.write(answer_value)
    with cols[1]:
        st.markdown("**Retrieval Confidence Score**")
        st.write(retrieval_confidence_value)
        st.markdown("**Confidence Level**")
        st.write(confidence_level_value)
    with cols[2]:
        st.markdown("**Hallucination Risk Score**")
        st.write(hallucination_risk_value)
        st.markdown("**Risk Level**")
        st.write(risk_level_value)

    cols2 = st.columns(3)
    with cols2[0]:
        st.markdown("**Self-healing Trigger**")
        st.write(self_healing_value)
    with cols2[1]:
        st.markdown("**Refusal Detection**")
        st.write(refusal_value)
    with cols2[2]:
        st.markdown("**Latency Breakdown (ms)**")
        latency_cols = st.columns(3)
        latency_cols[0].write(f"Retrieval: {latency_retrieval}")
        latency_cols[1].write(f"Generation: {latency_generation}")
        latency_cols[2].write(f"Total: {latency_total}")

    if error_msg:
        st.error(error_msg)

# Similarity Scores Visualization
st.subheader("Top-K Retrieval Similarity Scores")
if result and similarity_scores:
    ranked_scores = sorted(enumerate(similarity_scores, start=1), key=lambda kv: kv[1], reverse=True)
    ranks = [f"Rank {idx}" for idx, _ in ranked_scores]
    scores = [score for _, score in ranked_scores]

    fig_sim = px.bar(
        x=scores,
        y=ranks,
        orientation="h",
        labels={"x": "Similarity", "y": "Rank"},
        text=[f"{s:.4f}" for s in scores],
        title="Top-K Retrieval Similarity Scores",
    )
    fig_sim.update_layout(
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=80, r=40, t=60, b=40),
    )
    fig_sim.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig_sim.update_yaxes(showgrid=False)
    st.plotly_chart(fig_sim, use_container_width=True)
elif submit and not error_msg:
    st.info("No similarity scores returned for this query.")
else:
    st.write("Run a query to view similarity scores.")

# Confidence vs Risk Indicator
st.subheader("Confidence vs Risk Indicator")

def _gauge_color_conf(value: float) -> str:
    if value >= 0.75:
        return "#4caf50"  # green
    if value >= 0.50:
        return "#fbc02d"  # yellow
    return "#e53935"  # red


def _gauge_color_risk(value: float) -> str:
    if value < 0.30:
        return "#4caf50"  # green
    if value <= 0.60:
        return "#fbc02d"  # yellow
    return "#e53935"  # red


conf_val = float(result.get("retrieval_confidence", 0)) if result else 0.0
risk_val = float(result.get("hallucination_risk", 0)) if result else 0.0

col_conf, col_risk = st.columns(2)
with col_conf:
    st.caption("System Trust Confidence")
    fig_conf = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=conf_val,
            number={"valueformat": ".4f"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": _gauge_color_conf(conf_val)},
                "steps": [
                    {"range": [0, 0.5], "color": "#fdecea"},
                    {"range": [0.5, 0.75], "color": "#fff4e5"},
                    {"range": [0.75, 1], "color": "#e8f5e9"},
                ],
            },
            title={"text": "Retrieval Confidence Score"},
        )
    )
    fig_conf.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_conf, use_container_width=True)

with col_risk:
    st.caption("Hallucination Risk Index")
    fig_risk = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_val,
            number={"valueformat": ".4f"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": _gauge_color_risk(risk_val)},
                "steps": [
                    {"range": [0, 0.3], "color": "#e8f5e9"},
                    {"range": [0.3, 0.6], "color": "#fff4e5"},
                    {"range": [0.6, 1], "color": "#fdecea"},
                ],
            },
            title={"text": "Hallucination Risk Score"},
        )
    )
    fig_risk.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_risk, use_container_width=True)

# Section 2: Similarity Scores Visualization (placeholder)
st.subheader("Similarity Scores Visualization")
st.write("(placeholder for charts)")

# Section 3: Confidence vs Risk Indicator (placeholder)
st.subheader("Confidence vs Risk Indicator")
st.write("(placeholder for charts)")

# Section 4: Latency Metrics (placeholder)
st.subheader("Latency Metrics")
st.write("(placeholder for charts)")

# Section 5: Evaluation Metrics Summary (placeholder)
st.subheader("Evaluation Metrics Summary")
st.write("(placeholder for metrics)")

# Section 6: Logs Explorer
logs_container = st.container()
with logs_container:
    st.subheader("Logs Explorer")
    logs = load_logs(LOG_PATH)
    latest_logs = logs[-10:] if logs else []
    if latest_logs:
        st.dataframe(latest_logs, use_container_width=True)
    else:
        st.info("No logs found at logs/rag_traces.jsonl")

st.caption("Use 'streamlit run dashboard/app.py' to launch the dashboard.")
