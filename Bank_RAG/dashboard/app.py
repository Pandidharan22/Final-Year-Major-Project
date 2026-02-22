import sys
from pathlib import Path

import streamlit as st

from utils import load_logs, format_metric

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.rag import run_rag_pipeline  # noqa: E402
st.set_page_config(page_title="BANK_RAG Observability Dashboard", layout="wide")

# Paths
LOG_PATH = BASE_DIR / "logs" / "rag_traces.jsonl"

st.title("BANK_RAG Observability Dashboard")
st.caption("Step 2: Live RAG integration with latency instrumentation.")

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
