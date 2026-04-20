import os
from typing import List, Dict, Optional

BASE = os.path.join(os.path.dirname(__file__), "..", "ai", "system_prompts")


def _read_prompt_file(name: str) -> str:
    path = os.path.normpath(os.path.join(BASE, f"{name}.md"))
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def load_system_prompt(mode: Optional[str], fallback: str = "viewer") -> str:
    if not mode:
        mode = fallback
    # normalize mode file name
    name = mode.strip().lower()
    txt = _read_prompt_file(name)
    if not txt:
        txt = _read_prompt_file(fallback)
    return txt


def build_rag_prompt(query: str, evidence: List[Dict], mode: Optional[str] = None, cot: bool = False) -> str:
    """Build a system + user prompt for RAG chains.

    - Loads system prompt for `mode` and appends evidence and user query.
    - Evidence should be a list of dicts with `text` and optional `source` or `id`.
    """
    system = load_system_prompt(mode)
    ev_parts = []
    for i, e in enumerate(evidence, 1):
        txt = e.get("text") or e.get("content") or ""
        src = e.get("source") or e.get("document_id") or e.get("id") or ""
        ev_parts.append(f"[{i}] {txt}\nSource: {src}")

    ev_text = "\n\n".join(ev_parts)
    cot_suffix = "\n\nThink step by step before answering." if cot else ""

    prompt = f"{system}\n\nEvidence:\n{ev_text}\n\nUser Query:\n{query}{cot_suffix}\n\nAnswer using only the evidence above and cite evidence numbers when making factual claims."
    return prompt


__all__ = ["load_system_prompt", "build_rag_prompt"]
