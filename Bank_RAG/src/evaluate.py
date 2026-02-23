import json
import sys
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.rag import LOG_PATH, rag_answer  # noqa: E402
from src.retrieve import CHUNKS_DIR, load_chunks  # noqa: E402


def get_evaluation_summary() -> Dict[str, float]:
    """Compute simple evaluation summary metrics from existing logs or stored metrics.

    Returns a dictionary with accuracy, average confidence, average risk, refusal rate,
    and self-healing trigger rate. If an evaluation metrics file exists under
    evaluation/metrics.json with an `accuracy` key, it is used; otherwise, accuracy is
    estimated as (1 - avg_risk), bounded to [0, 1].
    """

    metrics_path = BASE_DIR / "evaluation" / "metrics.json"
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                stored = json.load(handle)
            accuracy_val = float(stored.get("accuracy", 0.0))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            accuracy_val = 0.0
    else:
        accuracy_val = 0.0

    logs: List[Dict[str, object]] = []
    if LOG_PATH.exists():
        with LOG_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        logs.append(obj)
                except json.JSONDecodeError:
                    continue

    if not logs:
        return {
            "accuracy": accuracy_val,
            "avg_confidence": 0.0,
            "avg_risk": 0.0,
            "refusal_rate": 0.0,
            "self_healing_trigger_rate": 0.0,
        }

    confidences = [float(item.get("retrieval_confidence_score", 0.0)) for item in logs]
    risks = [float(item.get("hallucination_risk_score", 0.0)) for item in logs]
    refusals = [bool(item.get("refusal_detected", False)) for item in logs]
    self_heal = [bool(item.get("self_healing_trigger", False)) for item in logs]

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    avg_risk = sum(risks) / len(risks) if risks else 0.0
    refusal_rate = sum(1 for r in refusals if r) / len(refusals) if refusals else 0.0
    self_heal_rate = sum(1 for s in self_heal if s) / len(self_heal) if self_heal else 0.0

    if accuracy_val == 0.0 and avg_risk:
        accuracy_val = max(0.0, min(1.0, 1.0 - avg_risk))

    return {
        "accuracy": accuracy_val,
        "avg_confidence": avg_conf,
        "avg_risk": avg_risk,
        "refusal_rate": refusal_rate,
        "self_healing_trigger_rate": self_heal_rate,
    }


def load_questions(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_chunk_document_map(chunks: List[Dict[str, object]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in chunks:
        chunk_id = item.get("chunk_id", "")
        doc_id = item.get("document_id", "")
        if chunk_id:
            mapping[chunk_id] = doc_id
    return mapping


def load_latest_trace_for_query(query: str) -> Dict[str, object]:
    if not LOG_PATH.exists():
        raise RuntimeError(f"Trace file not found at {LOG_PATH}")
    with LOG_PATH.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("query") == query:
            return payload
    raise RuntimeError(f"No trace entry found for query: {query}")


def evaluate() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    questions_path = base_dir / "evaluation" / "questions.json"
    questions = load_questions(questions_path)

    chunks = load_chunks(base_dir / CHUNKS_DIR)
    if not chunks:
        raise RuntimeError("No chunks loaded; ensure data/chunks is populated.")
    chunk_doc_map = build_chunk_document_map(chunks)

    total_queries = 0
    in_domain_queries = 0
    out_of_domain_queries = 0
    retrieval_hits = 0
    refusal_hits = 0
    in_domain_confidences: List[float] = []
    out_of_domain_confidences: List[float] = []

    for item in questions:
        query = str(item.get("query", ""))
        expected_document = item.get("expected_document")
        qtype = item.get("type", "")

        # Execute RAG to produce a trace
        rag_answer(query)

        trace = load_latest_trace_for_query(query)
        retrieved_chunk_ids = trace.get("retrieved_chunk_ids", [])
        confidence = float(trace.get("retrieval_confidence_score", 0.0))
        refusal_detected = bool(trace.get("refusal_detected", False))

        # Determine document IDs present in the retrieved chunks
        retrieved_docs = [chunk_doc_map.get(cid, "") for cid in retrieved_chunk_ids]

        if qtype == "in_domain":
            in_domain_queries += 1
            if expected_document and expected_document in retrieved_docs:
                retrieval_hits += 1
            in_domain_confidences.append(confidence)
        elif qtype == "out_of_domain":
            out_of_domain_queries += 1
            if refusal_detected:
                refusal_hits += 1
            out_of_domain_confidences.append(confidence)
        total_queries += 1

    retrieval_accuracy = (
        (retrieval_hits / in_domain_queries) * 100 if in_domain_queries else 0.0
    )
    refusal_accuracy = (
        (refusal_hits / out_of_domain_queries) * 100 if out_of_domain_queries else 0.0
    )
    mean_confidence_in_domain = (
        sum(in_domain_confidences) / len(in_domain_confidences) if in_domain_confidences else 0.0
    )
    mean_confidence_out_of_domain = (
        sum(out_of_domain_confidences) / len(out_of_domain_confidences) if out_of_domain_confidences else 0.0
    )

    print("=== Evaluation Summary ===")
    print(f"Retrieval Accuracy: {retrieval_accuracy:.2f}%")
    print(f"Refusal Accuracy: {refusal_accuracy:.2f}%")
    print(f"Mean Confidence (In-Domain): {mean_confidence_in_domain:.4f}")
    print(f"Mean Confidence (Out-of-Domain): {mean_confidence_out_of_domain:.4f}")


if __name__ == "__main__":
    evaluate()
