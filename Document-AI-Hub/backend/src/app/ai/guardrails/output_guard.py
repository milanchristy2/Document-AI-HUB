from dataclasses import dataclass
from typing import List, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class OutputGuardResult:
    passed: bool
    output: str
    modified: bool = False
    reason: str | None = None


# basic hallucination patterns to detect common LLM hallucination phrases
HALLUCINATION_PATTERNS: List[re.Pattern] = [
    re.compile(r"I think", re.I),
    re.compile(r"I believe", re.I),
    re.compile(r"As far as I know", re.I),
    re.compile(r"It is possible", re.I),
    re.compile(r"According to my training", re.I),
    re.compile(r"I was trained", re.I),
    re.compile(r"I don't have access", re.I),
    re.compile(r"I cannot find", re.I),
]


class OutputGuardrail:
    def __init__(self):
        self.min_len = 10
        # attempt to import guardrails-ai
        try:
            import guardrails as _gr
            self._guardrails = _gr
            self._has_guardrails = True
        except Exception:
            self._guardrails = None
            self._has_guardrails = False

    def validate(self, response: str, evidence: List[Dict[str, Any]] | None = None) -> OutputGuardResult:
        if not response or len(response.strip()) < self.min_len:
            return OutputGuardResult(False, response or "", True, "too_short")

        # detect hallucination patterns
        for pat in HALLUCINATION_PATTERNS:
            if pat.search(response):
                # add disclaimer
                note = "\n\n[Disclaimer]: The response contains uncertain phrasing and may not be grounded in the provided evidence."
                return OutputGuardResult(False, response + note, True, "hallucination_pattern")

        # placeholder for guardrails-ai checks (toxicity, PII)
        try:
            if self._has_guardrails and self._guardrails is not None:
                try:
                    if hasattr(self._guardrails, "validate_output"):
                        res = self._guardrails.validate_output(response)
                        if isinstance(res, dict) and not res.get("valid", True):
                            note = "\n\n[Disclaimer]: Output failed guardrails checks."
                            return OutputGuardResult(False, response + note, True, "guardrails_block")
                    elif hasattr(self._guardrails, "Guard"):
                        try:
                            g = self._guardrails.Guard()
                            if hasattr(g, "run"):
                                out = g.run({"output": response})
                                if isinstance(out, dict) and out.get("errors"):
                                    note = "\n\n[Disclaimer]: Output failed guardrails checks."
                                    return OutputGuardResult(False, response + note, True, "guardrails_errors")
                        except Exception:
                            pass
                except Exception:
                    logger.debug("guardrails output check failed; continuing")
            return OutputGuardResult(True, response, False, None)
        except Exception as e:
            logger.exception("Output guardrails validation failed: %s", e)
            return OutputGuardResult(False, response, False, "guardrails_error")

    def validate_with_guardrails(self, response: str, mode: str | None = None) -> OutputGuardResult:
        # For now delegate to validate
        return self.validate(response, None)


# singleton
output_guardrail = OutputGuardrail()


__all__ = ["output_guardrail", "OutputGuardResult"]
