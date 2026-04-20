from functools import lru_cache
from typing import Dict

_GENERAL = "Answer ONLY using EVIDENCE. If not found: 'NOT FOUND IN DOCUMENT'"

_ROLE_PROMPTS: Dict[str, str] = {
    "legal": "You are a legal assistant. Answer using ONLY the provided evidence. Extract clauses, cite sections, and summarize contract implications.",
    "healthcare": "You are a healthcare assistant. Use ONLY the provided evidence. Extract patient history, relevant vitals and suggest next steps conservatively.",
    "finance": "You are a finance assistant. Use ONLY the provided evidence. Answer questions about policies, loans, credits, and provide concise guidance.",
    "academic": "You are an academic summarizer. Use ONLY the provided evidence. Summarize papers, extract citations and key contributions.",
    "business": "You are a business assistant. Use ONLY the provided evidence. Transcribe meeting notes, extract action items and assign priorities.",
}

_COT_SUFFIX = "<thinking>\n1. Step one: identify relevant evidence.\n2. Step two: extract facts.\n3. Step three: reason based on evidence.\n4. Step four: produce concise answer.\n</thinking>"


@lru_cache(maxsize=32)
def get_system_prompt(mode: str | None = None, cot: bool = False) -> str:
    """Return a strict system prompt for the given mode. Cached to avoid repeated file reads."""
    parts = [_GENERAL]
    if mode:
        m = mode.lower().strip()
        role_prompt = _ROLE_PROMPTS.get(m)
        if role_prompt:
            parts.append(role_prompt)
    if cot:
        parts.append(_COT_SUFFIX)
    return "\n\n".join(parts)


__all__ = ["get_system_prompt"]
