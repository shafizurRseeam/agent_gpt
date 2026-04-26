"""
privacy/ner_redact.py

Baseline 3 — NER-REDACT

Surface-level PII masking using spaCy NER + structured regex patterns.
Similar to traditional anonymization / redaction tools.

Pipeline:
  1. Regex sweeps for structured PII (phone, email, SSN, credit card,
     dates, ZIP codes)
  2. spaCy NER for named entities (PERSON, ORG, GPE, DATE, CARDINAL …)
  3. Profile-bound exact matches for fields in user_profile

All detected spans are replaced with [REDACTED_<TYPE>] tokens.
No LLM is involved; this is a fully deterministic, rule-based baseline.

Interface:
    sanitize(payload, user_profile, task, memory_traces)
        → redacted_text
    sanitize_with_trace(...)
        → (redacted_text, {"spans": [...], "method": "ner_redact"})
"""

import re
from typing import Dict, List, Set, Tuple

import spacy


_nlp_model = None

def _nlp():
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = spacy.load("en_core_web_sm")
    return _nlp_model


# ── Structured-pattern regexes ────────────────────────────────────────────────

_PHONE_RE = re.compile(r'\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}')
_SSN_RE   = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
_CC_RE    = re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')
_EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b')
_ZIP_RE   = re.compile(r'\b\d{5}(?:-\d{4})?\b')
_INS_ID_RE = re.compile(r'\b[A-Z]{2,5}-\d{4,8}-[A-Z0-9]{2,6}\b')
_ISO_DATE_RE = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')
_NL_DATE_RE  = re.compile(
    r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
    r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
    r'\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?',
    re.IGNORECASE
)

# spaCy NER label → redaction tag
_NER_TAG = {
    "PERSON":   "PERSON",
    "ORG":      "ORG",
    "GPE":      "LOCATION",
    "LOC":      "LOCATION",
    "FAC":      "LOCATION",
    "DATE":     "DATE",
    "TIME":     "DATE",
    "CARDINAL": "NUMBER",
    "QUANTITY": "NUMBER",
    "MONEY":    "MONEY",
}


def sanitize(payload: str,
             user_profile: dict = None,
             task: str = "",
             memory_traces: list = None) -> str:
    redacted, _ = sanitize_with_trace(payload, user_profile, task, memory_traces)
    return redacted


def sanitize_with_trace(payload: str,
                         user_profile: dict = None,
                         task: str = "",
                         memory_traces: list = None):
    """
    Returns (redacted_text, trace_dict).
    trace_dict = {"spans": [(original, tag), ...], "method": "ner_redact"}
    """
    profile = user_profile or {}
    text    = payload
    spans_found: List[Tuple[int, int, str, str]] = []  # (start, end, original, tag)

    # ── 1. Profile-bound exact matches ───────────────────────────────────────
    _PROFILE_FIELDS = [
        ("name",           "NAME"),
        ("dob",            "DOB"),
        ("address",        "ADDRESS"),
        ("phone",          "PHONE"),
        ("email",          "EMAIL"),
        ("ssn",            "SSN"),
        ("insurance",      "INSURANCE"),
        ("insurance_id",   "INS_ID"),
        ("credit_card",    "CREDIT_CARD"),
        ("driver_license", "DRIVER_LICENSE"),
    ]
    for field, tag in _PROFILE_FIELDS:
        val = profile.get(field, "")
        if not val:
            continue
        for m in re.finditer(re.escape(str(val)), text, re.IGNORECASE):
            spans_found.append((m.start(), m.end(), m.group(0), tag))

    # ── 2. Structured regex patterns ─────────────────────────────────────────
    _REGEX_RULES = [
        (_PHONE_RE,    "PHONE"),
        (_EMAIL_RE,    "EMAIL"),
        (_SSN_RE,      "SSN"),
        (_CC_RE,       "CREDIT_CARD"),
        (_INS_ID_RE,   "INS_ID"),
        (_ZIP_RE,      "ZIP"),
        (_ISO_DATE_RE, "DATE"),
        (_NL_DATE_RE,  "DATE"),
    ]
    for pat, tag in _REGEX_RULES:
        for m in pat.finditer(text):
            spans_found.append((m.start(), m.end(), m.group(0), tag))

    # ── 3. spaCy NER ─────────────────────────────────────────────────────────
    # Age expressions like "32-year-old" or "32 years old" are demographic
    # descriptions, not PII dates — skip them to preserve task utility.
    _AGE_RE = re.compile(r'\b\d{1,3}[-\s]years?[-\s]old\b', re.IGNORECASE)
    age_spans = {(m.start(), m.end()) for m in _AGE_RE.finditer(text)}

    doc = _nlp()(text)
    for ent in doc.ents:
        tag = _NER_TAG.get(ent.label_)
        if tag == "DATE" and (ent.start_char, ent.end_char) in age_spans:
            continue  # don't redact age descriptions
        if tag:
            spans_found.append((ent.start_char, ent.end_char, ent.text, tag))

    # ── Merge overlaps: longest span wins ────────────────────────────────────
    spans_found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    merged: List[Tuple[int, int, str, str]] = []
    for start, end, orig, tag in spans_found:
        if merged and start < merged[-1][1]:
            # Overlap: keep whichever span is longer
            if (end - start) > (merged[-1][1] - merged[-1][0]):
                merged[-1] = (start, end, orig, tag)
            # else keep existing
        else:
            merged.append((start, end, orig, tag))

    # ── Replace spans right-to-left to preserve offsets ──────────────────────
    tag_counters: Dict[str, int] = {}
    result = text
    trace  = []
    for start, end, orig, tag in reversed(merged):
        tag_counters[tag] = tag_counters.get(tag, 0) + 1
        placeholder = f"[REDACTED_{tag}]"
        result = result[:start] + placeholder + result[end:]
        trace.append((orig, tag))

    return result, {"spans": list(reversed(trace)), "method": "ner_redact"}
