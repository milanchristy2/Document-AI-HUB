from typing import Any, Dict, List
import json


def format_as_json(answer: str, evidence: List[Dict[str, Any]] = None, metadata: Dict[str, Any] = None) -> Dict:
    return {
        "answer": answer,
        "evidence": evidence or [],
        "metadata": metadata or {},
    }


def format_as_markdown(answer: str, evidence: List[Dict[str, Any]] = None) -> str:
    md = "## Answer\n\n" + (answer or "") + "\n\n"
    if evidence:
        md += "## Evidence\n\n"
        for i, e in enumerate(evidence, 1):
            heading = e.get("heading") or e.get("title") or f"Section {i}"
            text = e.get("text") or e.get("content") or ""
            md += f"### {heading}\n\n{text}\n\n"
    return md


def format_as_table(answer: str, evidence: List[Dict[str, Any]] = None) -> str:
    # Simple markdown table of evidence with snippets
    md = "| # | Heading | Snippet |\n|---|---|---|\n"
    if evidence:
        for i, e in enumerate(evidence, 1):
            heading = (e.get("heading") or e.get("title") or "")[:30]
            text = (e.get("text") or e.get("content") or "").replace("\n", " ")[:80]
            md += f"| {i} | {heading} | {text} |\n"
    return md


def format_response(answer: str, evidence: List[Dict] = None, style: str = "json"):
    if style == "json":
        return format_as_json(answer, evidence)
    if style == "markdown":
        return format_as_markdown(answer, evidence)
    if style == "table":
        return format_as_table(answer, evidence)
    # default to json dict
    return format_as_json(answer, evidence)
