import re
from typing import List, Dict, Optional

DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
CURRENCY_RE = re.compile(r"\b(\$|USD|EUR|£)\s?([0-9,.]+)\b")


def extract_dates(text: str) -> List[str]:
    if not text:
        return []
    return DATE_RE.findall(text)


def extract_currencies(text: str) -> List[str]:
    if not text:
        return []
    return [m[0] + m[1] for m in CURRENCY_RE.findall(text)]


def simple_key_value_parse(text: str) -> Dict[str, str]:
    """Parse simple 'Key: Value' lines into a dict.

    This is useful for forms or metadata blocks.
    """
    out: Dict[str, str] = {}
    for line in text.splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            out[k.strip().lower()] = v.strip()
    return out


__all__ = ["extract_dates", "extract_currencies", "simple_key_value_parse"]
