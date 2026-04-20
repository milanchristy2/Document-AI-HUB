"""Tiny docling helper: convenience APIs for quick document analysis.

This module is intentionally thin — it orchestrates the `nlp.document_extractor`
and offers a programmatic entrypoint for tooling and tests.
"""
from app.nlp.document_extractor import extract_from_path


def analyze_document(path: str, mime: str | None = None) -> dict:
    return extract_from_path(path, mime)


__all__ = ["analyze_document"]
