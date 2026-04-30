"""
privacy/presidio.py

Baseline — PRESIDIO-REDACT

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
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


_analyzer   = None
_anonymizer = None

_NLP_CONFIG = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}


def _get_engines():
    global _analyzer, _anonymizer
    if _analyzer is None:
        nlp_engine  = NlpEngineProvider(nlp_configuration=_NLP_CONFIG).create_engine()
        _analyzer   = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        _anonymizer = AnonymizerEngine()
    return _analyzer, _anonymizer


def sanitize(
    payload:         str,
    user_profile:    dict  = None,
    task:            str   = "",
    memory_traces:   list  = None,
    score_threshold: float = 0.35,
) -> str:
    redacted, _ = sanitize_with_trace(
        payload, user_profile, task, memory_traces, score_threshold
    )
    return redacted


def sanitize_with_trace(
    payload:         str,
    user_profile:    dict  = None,
    task:            str   = "",
    memory_traces:   list  = None,
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


# ── Standalone — runs Presidio baseline on the dentist example payload ─────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    p_t = (
        "Hi, I'm Bob Smith and you can reach me at 585-555-1212 or bob@example.com. "
        "I need a dentist near 12 ABC St, Rochester, NY. I have BlueCross Dental Plus. "
        "My insurance ID is BC-123456-A9. I am available on March 18 and March 19, "
        "preferably before 10:30 AM. If possible, book for 2 people. "
        "I previously visited Bright Smile Dental and Lake Dental Care. "
        "I have tooth pain and bleeding gums. My ZIP is 14623 and I want to keep "
        "the cost under $500."
    )

    W = 70

    print(f"\n{'═' * W}")
    print(f"  BASELINE: PRESIDIO REDACT")
    print(f"{'═' * W}")

    print(f"\n{'═' * W}")
    print(f"  INPUT  (LC-enriched payload p_t)")
    print(f"{'═' * W}")
    print(f"  {p_t}")

    redacted, trace = sanitize_with_trace(p_t)

    print(f"\n{'═' * W}")
    print(f"  DETECTED PII SPANS  ({len(trace['spans'])} entities)")
    print(f"{'═' * W}")
    for text, etype in trace["spans"]:
        print(f"    [{etype:<25}] {text!r}")

    print(f"\n{'═' * W}")
    print(f"  REDACTED OUTPUT")
    print(f"{'═' * W}")
    print(f"  {redacted}")
