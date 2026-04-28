"""
privacy/span_extractor.py

Stage 1 of PrivScope — Span Extraction.

Partitions a candidate cloud-bound payload P_t into three disjoint components:

    P_t = U_loc  ∪  U_med  ∪  C_t

  U_loc  — direct identifiers (name, phone, SSN, passport, IDs, credit card…).
            Withheld from the cloud pipeline; stored in a private binding table.
            Detected exclusively via profile exact-match (Layer 1a).
  U_med  — mediation candidates eligible for downstream scope control.
            Includes soft profile values (address, insurance plan name,
            preferences — Layer 1b) plus regex and spaCy spans (Layers 2–3).
  C_t    — surrounding text (connectives, framing) that passes through unchanged.

Profile matching uses two disjoint field sets:

  _PROFILE_DIRECT_IDS  — hard identifiers → U_loc  (always withheld)
  _PROFILE_SOFT_FIELDS — contextual values → U_med  (evaluated by scope control)

Extraction layers in order:

  1a. Profile direct IDs  — exact match on _PROFILE_DIRECT_IDS  → U_loc
  1b. Profile soft fields — exact match on _PROFILE_SOFT_FIELDS → U_med
  2.  Structured patterns — regex for stable surface forms        → U_med
  3.  spaCy NER + chunks  — less regular spans                    → U_med

Overlap resolution gives priority to U_loc spans, then to longer spans.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List

import spacy


_nlp_model = None

def _nlp():
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = spacy.load("en_core_web_sm")
    return _nlp_model


@dataclass
class Span:
    text:      str
    start:     int
    end:       int
    span_type: str
    source:    str       # profile | structured_pattern | spacy
    bucket:    str       # U_loc | U_med
    subsource: str = "" # ner | noun_chunk | regex | exact_match


# ── Profile field sets ────────────────────────────────────────────────────────
#
# Two disjoint sets. The split is based on disclosure risk, not detection method:
#
#   _PROFILE_DIRECT_IDS  — hard identifiers: always withheld (→ U_loc)
#                           Includes anything that directly re-identifies the user
#                           regardless of task context.
#
#   _PROFILE_SOFT_FIELDS — contextual values: evaluated by scope control (→ U_med)
#                           User-set at profile construction time but may or may
#                           not be needed for a given task (address, insurance plan
#                           name, soft preferences). The scope control gate decides
#                           whether each value reaches the cloud.

_PROFILE_DIRECT_IDS = {
    "name":                  "name",
    "phone":                 "phone",
    "email":                 "email",
    "dob":                   "dob",
    "ssn":                   "ssn",
    "passport_number":       "passport_number",
    "driver_license":        "driver_license",
    "insurance_id":          "insurance_id",
    "patient_id":            "patient_id",
    "credit_card":           "credit_card",
    "membership_id":         "membership_id",
    "frequent_flyer_number": "frequent_flyer_number",
    "hotel_loyalty_id":      "hotel_loyalty_id",
    "vehicle_plate":         "vehicle_plate",
}

_PROFILE_SOFT_FIELDS = {
    "address":                  "address",
    "insurance":                "insurance_name",
    "doctor_gender_preference": "preference",
    "dietary_preference":       "preference",
    "seat_preference":          "preference",
}


# ── Structured-pattern rules ──────────────────────────────────────────────────

_PATTERNS: List[tuple[str, re.Pattern]] = [
    ("date", re.compile(
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|'
        r'Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?\b',
        re.IGNORECASE,
    )),
    ("date",       re.compile(r'\b\d{4}-\d{2}-\d{2}\b')),
    ("date",       re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b')),
    ("time",       re.compile(r'\b\d{1,2}(?::\d{2})?\s?(?:AM|PM|am|pm)\b')),
    ("time",       re.compile(r'\b\d{1,2}:\d{2}\b')),
    ("phone",      re.compile(r'\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b')),
    ("email",      re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')),
    ("zip",        re.compile(r'\b\d{5}(?:-\d{4})?\b')),
    ("money",      re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b')),
    ("party_size", re.compile(r'\bfor\s+\d+\s+people\b', re.IGNORECASE)),
]


# ── spaCy NER label → span type ───────────────────────────────────────────────

_NER_MAP = {
    "PERSON":   "person",
    "ORG":      "organization",
    "GPE":      "location",
    "LOC":      "location",
    "FAC":      "facility",
    "DATE":     "date",
    "TIME":     "time",
    "MONEY":    "money",
    "NORP":     "group",
}

_BAD_CHUNK_PREFIXES = {"a ", "an ", "the ", "my ", "your ", "our ", "if "}
_BAD_CHUNK_EXACT    = {
    "a dentist", "if possible, book", "my insurance id",
    "my zip", "the cost",
}
_GOOD_CHUNK_EXCEPTIONS = {"my home", "my work", "my office", "my insurance"}


# ── Main extractor ────────────────────────────────────────────────────────────

class SpanExtractor:
    """
    Stage 1 extractor. Call extract(payload, profile) → dict with keys:
        P_t, U_loc, U_med, C_t, all_spans
    """

    def extract(self, payload: str, profile: Dict) -> Dict:
        candidates: List[Span] = []

        # Layer 1a — direct identifiers (hard profile fields) → U_loc
        for field, span_type in _PROFILE_DIRECT_IDS.items():
            value = profile.get(field)
            if not value or not isinstance(value, str):
                continue
            for m in re.finditer(re.escape(value), payload, re.IGNORECASE):
                candidates.append(Span(
                    text=m.group(0), start=m.start(), end=m.end(),
                    span_type=span_type, source="profile",
                    bucket="U_loc", subsource="direct_id",
                ))

        # Layer 1b — soft profile fields (address, insurance plan, preferences) → U_med
        for field, span_type in _PROFILE_SOFT_FIELDS.items():
            value = profile.get(field)
            if not value or not isinstance(value, str):
                continue
            for m in re.finditer(re.escape(value), payload, re.IGNORECASE):
                candidates.append(Span(
                    text=m.group(0), start=m.start(), end=m.end(),
                    span_type=span_type, source="profile",
                    bucket="U_med", subsource="soft_field",
                ))

        # Layer 2 — structured patterns → U_med
        for span_type, pattern in _PATTERNS:
            for m in pattern.finditer(payload):
                candidates.append(Span(
                    text=m.group(0).strip(), start=m.start(), end=m.end(),
                    span_type=span_type, source="structured_pattern",
                    bucket="U_med", subsource="regex",
                ))

        # Layer 3 — spaCy NER + noun chunks → U_med
        doc = _nlp()(payload)
        for ent in doc.ents:
            mapped = _NER_MAP.get(ent.label_)
            if mapped:
                candidates.append(Span(
                    text=ent.text.strip(), start=ent.start_char, end=ent.end_char,
                    span_type=mapped, source="spacy",
                    bucket="U_med", subsource="ner",
                ))
        for chunk in doc.noun_chunks:
            txt = chunk.text.strip()
            if self._keep_noun_chunk(txt):
                candidates.append(Span(
                    text=txt, start=chunk.start_char, end=chunk.end_char,
                    span_type="noun_phrase", source="spacy",
                    bucket="U_med", subsource="noun_chunk",
                ))

        spans = _merge_overlaps(candidates)
        U_loc = [s for s in spans if s.bucket == "U_loc"]
        U_med = [s for s in spans if s.bucket == "U_med"]
        C_t     = _build_context(payload, spans)
        tagged  = _build_tagged_payload(payload, spans)
        working = _build_working_payload(payload, U_loc)

        return {
            "P_t":       payload,
            "U_loc":     U_loc,
            "U_med":     U_med,
            "C_t":       C_t,
            "tagged":    tagged,   # annotated view: [U_LOC:name], [U_MED:address], …
            "working":   working,  # Stage 2 input: U_loc withheld, U_med verbatim
            "all_spans": spans,
        }

    @staticmethod
    def _keep_noun_chunk(txt: str) -> bool:
        if len(txt.split()) < 2:
            return False
        txt_lower = txt.lower()
        if txt_lower in _BAD_CHUNK_EXACT:
            return False
        if any(txt_lower.startswith(p) for p in _BAD_CHUNK_PREFIXES):
            if txt_lower not in _GOOD_CHUNK_EXCEPTIONS:
                return False
        if not re.search(r'[A-Za-z]', txt):
            return False
        return True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _merge_overlaps(spans: List[Span]) -> List[Span]:
    # Deduplicate exact copies first
    seen: dict = {}
    for s in spans:
        key = (s.start, s.end, s.text.lower(), s.span_type, s.source, s.bucket, s.subsource)
        seen[key] = s
    spans = list(seen.values())

    # Sort: U_loc first, then longer span, then earlier start
    def _priority(s: Span):
        return (0 if s.bucket == "U_loc" else 1, -(s.end - s.start), s.start)

    spans.sort(key=_priority)

    accepted: List[Span] = []
    occupied: List[tuple[int, int]] = []
    for s in spans:
        if not any(s.start < b and s.end > a for a, b in occupied):
            accepted.append(s)
            occupied.append((s.start, s.end))

    return sorted(accepted, key=lambda s: s.start)


def _build_context(text: str, spans: List[Span]) -> str:
    """Internal C_t skeleton used by reconstruct_payload (generic [SPAN] slots)."""
    parts, cursor = [], 0
    for s in spans:
        if cursor < s.start:
            parts.append(text[cursor:s.start])
        parts.append("[SPAN]")
        cursor = s.end
    if cursor < len(text):
        parts.append(text[cursor:])
    return "".join(parts)


def _build_tagged_payload(text: str, spans: List[Span]) -> str:
    """Debug view: each span replaced with [U_LOC:type] or [U_MED:type]."""
    result = text
    for s in sorted(spans, key=lambda s: s.start, reverse=True):
        prefix = "U_LOC" if s.bucket == "U_loc" else "U_MED"
        result = result[:s.start] + f"[{prefix}:{s.span_type}]" + result[s.end:]
    return result


def _build_working_payload(text: str, u_loc: List[Span]) -> str:
    """Stage 2 input: U_loc spans withheld ([TYPE] placeholder), U_med verbatim."""
    result = text
    for s in sorted(u_loc, key=lambda s: s.start, reverse=True):
        result = result[:s.start] + f"[{s.span_type.upper()}]" + result[s.end:]
    return result


def spans_to_records(spans: List[Span]) -> List[dict]:
    return [asdict(s) for s in spans]


# ── Debug / development helper ────────────────────────────────────────────────

def _debug_show(result: Dict) -> None:
    print("P_t:")
    print(result["P_t"])
    print("\n" + "=" * 80)
    print("Annotated payload  [U_LOC:type] = withheld   [U_MED:type] = mediation candidate:")
    print(result["tagged"])
    print("\nU_loc spans:")
    for s in result["U_loc"]:
        print(f"  [{s.bucket}] [{s.span_type:<20}] {s.text!r:40}  ({s.source}/{s.subsource})")
    print("\nU_med spans:")
    for s in result["U_med"]:
        print(f"  [{s.bucket}] [{s.span_type:<20}] {s.text!r:40}  ({s.source}/{s.subsource})")
    print("\n" + "=" * 80)
    print("Stage 2 input  (U_loc withheld; U_med spans verbatim):")
    print(result["working"])


if __name__ == "__main__":
    from state.state_io import load_state
    state = load_state()
    profile = state.get("user_profile", {})

    payload = (
        "Hi, I'm Bob Smith and you can reach me at 585-555-1212 or bob@example.com. "
        "I need a dentist near 12 ABC St, Rochester, NY. I have BlueCross Dental Plus. "
        "My insurance ID is BC-123456-A9. I am available on March 18 and March 19, "
        "preferably before 10:30 AM. If possible, book for 2 people. "
        "I previously visited Bright Smile Dental and Lake Dental Care. "
        "I have tooth pain and bleeding gums. My ZIP is 14623 and I want to keep the cost under $500."
    )

    extractor = SpanExtractor()
    result    = extractor.extract(payload, profile)
    _debug_show(result)
