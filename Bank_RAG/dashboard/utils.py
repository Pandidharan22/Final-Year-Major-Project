from pathlib import Path
from typing import Iterable, List, Dict, Any
import json


def safe_jsonl_loader(filepath: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts; tolerate missing or malformed lines."""
    if not filepath.exists():
        return []

    records: List[Dict[str, Any]] = []
    with filepath.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                continue
    return records


def load_logs(filepath: Path) -> List[Dict[str, Any]]:
    """Return log entries from a JSONL file as a list of dicts."""
    return safe_jsonl_loader(filepath)


def format_metric(value: Any, precision: int = 4, default: str = "–") -> str:
    """Render a metric value with optional precision; fall back to default."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return str(value)
