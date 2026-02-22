import json
from pathlib import Path

import streamlit as st

from utils import load_logs, format_metric

st.set_page_config(page_title="BANK_RAG Observability Dashboard", layout="wide")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "rag_traces.jsonl"

st.title("BANK_RAG Observability Dashboard")
st.caption("Step 1: Dashboard skeleton with live query shell and log explorer.")

# Section 1: Live Query Interface
section_live = st.container()
with section_live:
    st.subheader("Live Query Interface")
    query = st.text_input("Enter a banking question")
    submit = st.button("Submit", type="primary")

    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Generated Answer**")
        st.write("(placeholder)")
    with cols[1]:
        st.markdown("**Retrieval Confidence Score**")
        st.write("(placeholder)")
        st.markdown("**Confidence Level**")
        st.write("(placeholder)")
    with cols[2]:
        st.markdown("**Hallucination Risk Score**")
        st.write("(placeholder)")
        st.markdown("**Risk Level**")
        st.write("(placeholder)")

    cols2 = st.columns(3)
    with cols2[0]:
        st.markdown("**Self-healing Trigger**")
        st.write("(placeholder)")
    with cols2[1]:
        st.markdown("**Refusal Detection**")
        st.write("(placeholder)")
    with cols2[2]:
        st.markdown("**Latency Breakdown**")
        st.write("(placeholder)")

    if submit:
        st.info("Backend execution not wired yet. This is a UI skeleton.")

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
