from dataclasses import dataclass
from typing import List, Tuple
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    passed: bool
    reason: str | None = None
    cleaned: str | None = None


INJECTION_PATTERNS: List[str] = [
    "ignore previous instructions",
    "disregard earlier",
    "forget all previous",
    "follow these new instructions",
    "bypass the safety",
    "override the system",
    "ignore the above",
    "do not follow previous",
    "this is a new task",
    "pretend you are",
    "act as if",
    "you are now",
    "malicious",
    "obey this command",
    "execute the following",
    "drop all constraints",
    "no longer follow",
    "override the rules",
    "ignore the guidelines",
    "ignore instructions",
]


class InputGuardrail:
    def __init__(self):
        self.max_len = 2000
        # compile patterns
        self._inj_regex = re.compile("|".join([re.escape(p) for p in INJECTION_PATTERNS]), re.I)
        # attempt to import guardrails-ai
        try:
            import guardrails as _gr
            self._guardrails = _gr
            self._has_guardrails = True
        except Exception:
            self._guardrails = None
            self._has_guardrails = False

    def validate(self, query: str | None, role: str | None = None) -> GuardResult:
        if not query or not query.strip():
            return GuardResult(False, "empty_query", "")
        if len(query) > self.max_len:
            return GuardResult(False, "too_long", query[: self.max_len])
        if self._inj_regex.search(query):
            return GuardResult(False, "injection_detected", None)
        # minimal cleaning: strip control chars
        cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", query).strip()
        return GuardResult(True, None, cleaned)

    def validate_with_guardrails(self, query: str, role: str | None = None) -> GuardResult:
        # Placeholder for integration with guardrails-ai (ToxicLanguage, DetectPII)
        try:
            # call our simple validate first
            base = self.validate(query, role)
            if not base.passed:
                return base
            # If guardrails is available, try to call a validation entrypoint.
            if self._has_guardrails and self._guardrails is not None:
                try:
                    # try typical APIs conservatively
                    if hasattr(self._guardrails, "validate_input"):
                        res = self._guardrails.validate_input(query)
                        if isinstance(res, dict) and not res.get("valid", True):
                            return GuardResult(False, res.get("reason", "guardrails_block"), None)
                    elif hasattr(self._guardrails, "validate"):
                        res = self._guardrails.validate(query)
                        if isinstance(res, dict) and not res.get("valid", True):
                            return GuardResult(False, res.get("reason", "guardrails_block"), None)
                    elif hasattr(self._guardrails, "Guard"):
                        # fallback: instantiate a Guard if it's convenient
                        try:
                            g = self._guardrails.Guard()
                            if hasattr(g, "run"):
                                out = g.run({"input": query})
                                # assume run returns dict with 'errors' or 'valid'
                                if isinstance(out, dict) and out.get("errors"):
                                    return GuardResult(False, "guardrails_errors", None)
                        except Exception:
                            pass
                except Exception:
                    logger.debug("guardrails check failed, proceeding pass-through")
            return GuardResult(True, None, base.cleaned)
        except Exception as e:
            logger.exception("Guardrails input validation failed: %s", e)
            # fail-safe: block on unknown errors
            return GuardResult(False, "guardrails_error", None)


# singleton
input_guardrail = InputGuardrail()


__all__ = ["input_guardrail", "GuardResult"]
