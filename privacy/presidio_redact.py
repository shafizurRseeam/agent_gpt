"""
privacy/presidio_redact.py

Baseline 3 — PRESIDIO-REDACT

PII detection and redaction using Microsoft Presidio.
Detects a broad set of entity types (PERSON, EMAIL_ADDRESS, PHONE_NUMBER,
LOCATION, DATE_TIME, US_SSN, CREDIT_CARD, NRP, etc.) and replaces each
with a typed [REDACTED_<TYPE>] placeholder.

Install:
    pip install presidio-analyzer presidio-anonymizer spacy
    python -m spacy download en_core_web_sm

Interface:
    sanitize(payload, user_profile, task, memory_traces)
        -> redacted_text
    sanitize_with_trace(payload, user_profile, task, memory_traces)
        -> (redacted_text, {"spans": [(original, entity_type), ...],
                            "method": "presidio"})
"""

from __future__ import annotations

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


_analyzer  = None
_anonymizer = None


def _get_engines():
    global _analyzer, _anonymizer
    if _analyzer is None:
        _analyzer  = AnalyzerEngine()
        _anonymizer = AnonymizerEngine()
    return _analyzer, _anonymizer


def sanitize(
    payload:       str,
    user_profile:  dict = None,
    task:          str  = "",
    memory_traces: list = None,
    score_threshold: float = 0.35,
) -> str:
    redacted, _ = sanitize_with_trace(
        payload, user_profile, task, memory_traces, score_threshold
    )
    return redacted


def sanitize_with_trace(
    payload:       str,
    user_profile:  dict = None,
    task:          str  = "",
    memory_traces: list = None,
    score_threshold: float = 0.35,
):
    """
    Returns (redacted_text, trace_dict).
    trace_dict = {"spans": [(original, entity_type), ...], "method": "presidio"}
    """
    if not payload:
        return payload, {"spans": [], "method": "presidio"}

    analyzer, anonymizer = _get_engines()

    results = analyzer.analyze(
        text=payload,
        language="en",
        score_threshold=score_threshold,
    )

    entity_types = sorted(set(r.entity_type for r in results))

    operators = {
        ent: OperatorConfig("replace", {"new_value": f"[REDACTED_{ent}]"})
        for ent in entity_types
    }

    anonymized = anonymizer.anonymize(
        text=payload,
        analyzer_results=results,
        operators=operators,
    )

    # Build span trace: sort by start offset so trace reads left-to-right
    spans = [
        (payload[r.start:r.end], r.entity_type)
        for r in sorted(results, key=lambda x: x.start)
    ]

    return anonymized.text, {"spans": spans, "method": "presidio"}
