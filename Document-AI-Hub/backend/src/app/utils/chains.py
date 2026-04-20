"""
Compatibility shim for chains: re-export names from the new `app.chains.rag_chain` module.

This file keeps existing imports working by importing everything from the new
location. If the new implementation is unavailable, provide minimal fallbacks.
"""

import importlib
import logging

logger = logging.getLogger(__name__)


def _ensure_real_impls():
    """Lazily import implementations from `app.chains.rag_chain`.

    This avoids circular imports during package initialization: callers
    will attempt to load the real functions when first invoked.
    """
    try:
        mod = importlib.import_module("app.chains.rag_chain")
    except Exception as e:
        logger.debug("Could not import real rag_chain implementations: %s", e)
        return False

    for name in (
        "qa_chain",
        "qa_chain_stream",
        "combine_evidence",
        "call_llm_str",
        "call_llm_stream",
        "rag_chain",
        "summarization_chain",
        "build_evidence_blocks",
        "format_response_to_json_markdown",
        "convert_to_json_schema",
    ):
        if hasattr(mod, name):
            globals()[name] = getattr(mod, name)
    return True


def _call_or_stub(name, *args, **kwargs):
    if name not in globals() or globals()[name].__module__ == __name__:
        _ensure_real_impls()
    fn = globals().get(name)
    if not fn:
        # Return a reasonable stub depending on expected behavior
        if name.endswith("_stream"):
            async def _empty_stream(*a, **k):
                if False:
                    yield ""

            return _empty_stream(*args, **kwargs)
        async def _empty(*a, **k):
            return ""

        return _empty(*args, **kwargs)
    return fn(*args, **kwargs)


async def qa_chain(*args, **kwargs):
    return await _call_or_stub("qa_chain", *args, **kwargs)


async def qa_chain_stream(*args, **kwargs):
    return await _call_or_stub("qa_chain_stream", *args, **kwargs)


def combine_evidence(*args, **kwargs):
    return _call_or_stub("combine_evidence", *args, **kwargs)


async def call_llm_str(*args, **kwargs):
    return await _call_or_stub("call_llm_str", *args, **kwargs)


async def call_llm_stream(*args, **kwargs):
    return await _call_or_stub("call_llm_stream", *args, **kwargs)


async def rag_chain(*args, **kwargs):
    return await _call_or_stub("rag_chain", *args, **kwargs)


async def summarization_chain(*args, **kwargs):
    return await _call_or_stub("summarization_chain", *args, **kwargs)


def build_evidence_blocks(*args, **kwargs):
    return _call_or_stub("build_evidence_blocks", *args, **kwargs)


