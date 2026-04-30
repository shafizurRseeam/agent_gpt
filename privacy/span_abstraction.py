"""
privacy/span_abstraction.py

Stage 3b of PrivScope — Span Abstraction.

Replaces each context-sensitive span u_j ∈ C_t with a less specific,
type-consistent representation before cloud release. Specifically:

  P̂_t = Assemble(R_t, D_t, {h_{T(u_j)}^{k_j}(u_j) : u_j ∈ C_t})

where:
  R_t   — residual scaffold (C_t slots)
  D_t   — PTH spans, released verbatim
  k_j   = π_ψ(u_j, T(u_j), g_t) — calibrated abstraction level
  T(u_j) — semantic abstraction type inferred from the extractor span type

Key design decisions:

  Semantic type inference
    The extractor's raw span_type (noun_phrase, organization, facility) is
    not specific enough for abstraction. Stage 3b first infers a semantic
    abstraction type — medical_symptom, prior_provider, service_need, etc.
    — using keyword matching and proper-noun detection. This prevents
    semantic drift such as abstracting a prior provider as a symptom.

  Adjacent symptom grouping
    Multiple medical_symptom spans in the same sentence are jointly
    abstracted into one phrase (e.g., "tooth pain" + "bleeding gums" →
    "dental symptoms"). The anchor span receives the grouped abstraction;
    remaining spans are set to empty and removed by the cleanup pass.

  LLM abstraction with type-constrained prompting
    The LLM prompt explicitly names the inferred abstraction type so the
    model cannot change the entity class. Constraints prevent diagnosis,
    hallucination, and over-specification.

  Post-abstraction validation
    After LLM abstraction, lightweight signal checks verify the output
    preserved its semantic role. Failures fall back to deterministic
    phrases from _FALLBACK.

Public API:
    SpanAbstractor()
    SpanAbstractor.abstract(css_spans, pth_spans, g_t, extraction, u_loc, local_llm)
        → AbstractionResult
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from privacy.span_extractor import Span
from privacy.abstraction_policy import (
    ABSTRACTION_HIERARCHIES,
    CALIBRATED_ABSTRACTION_POLICY,
    DEFAULT_ABSTRACTION_LEVEL,
)


# ── Semantic type inference ────────────────────────────────────────────────────
# Maps raw extractor span types to semantic abstraction types.
# noun_phrase, organization, and facility require keyword/pattern matching
# because the extractor type is too coarse for abstraction.

_DIRECT_TYPE_MAP: Dict[str, str] = {
    "address":        "address",
    "location":       "location",
    "zip":            "zip",
    "date":           "date",
    "time":           "time",
    "insurance_name": "insurance_name",
    "money":          "money",
    "party_size":     "party_size",
    "preference":     "preference",
    "person":         "person",
    "group":          "group",
}

_SYMPTOM_WORDS = {
    "pain", "ache", "aches", "gum", "gums", "tooth", "teeth", "bleed", "bleeding",
    "sore", "soreness", "hurt", "hurts", "swollen", "swelling", "cavity", "cavities",
    "sensitivity", "tenderness", "discomfort", "symptom", "toothache",
}
_DIETARY_WORDS = {
    "vegetarian", "vegan", "gluten", "kosher", "halal", "allergy", "allergen",
    "lactose", "shellfish", "pescatarian", "nut", "dairy",
}
_TRAVEL_WORDS = {
    "trip", "travel", "flight", "hotel", "business", "vacation", "leisure",
    "journey", "tour",
}
_SERVICE_WORDS = {
    "accepting", "emergency", "accessible", "wheelchair", "appointment", "same-day",
}


def _infer_abstraction_type(span: Span) -> str:
    """
    Infer semantic abstraction type from the extractor span type and text.

    Structured span types map directly. noun_phrase, organization, and
    facility use keyword matching and proper-noun detection.
    """
    if span.span_type in _DIRECT_TYPE_MAP:
        return _DIRECT_TYPE_MAP[span.span_type]

    if span.span_type in ("organization", "facility"):
        return "prior_provider"

    if span.span_type == "noun_phrase":
        words = set(re.findall(r'\b[a-z]+\b', span.text.lower()))

        if words & _SYMPTOM_WORDS:
            return "medical_symptom"
        if words & _DIETARY_WORDS:
            return "dietary_constraint"
        if words & _TRAVEL_WORDS:
            return "travel_purpose"
        if words & _SERVICE_WORDS:
            return "service_need"

        # Proper noun pattern → likely a named entity (provider, venue, etc.)
        alpha = [w for w in span.text.split() if w.isalpha()]
        if alpha and all(w[0].isupper() for w in alpha):
            return "prior_provider"

        return "generic_detail"

    return "generic_detail"


# ── Adjacent symptom grouping ──────────────────────────────────────────────────

def _group_symptoms(
    spans_with_types: List[Tuple[Span, str]],
    payload_text:     str,
) -> List[Tuple[str, List[Span]]]:
    """
    Group adjacent medical_symptom spans within the same sentence.

    Returns a list of (abstraction_type, [spans]) tuples. Groups with
    multiple spans are jointly abstracted; the anchor slot receives the
    joint abstraction, remaining slots are set to empty.
    Non-symptom spans are returned as single-element groups.
    """
    symptoms = [(s, t) for s, t in spans_with_types if t == "medical_symptom"]
    others   = [(s, t) for s, t in spans_with_types if t != "medical_symptom"]

    result: List[Tuple[str, List[Span]]] = []

    if len(symptoms) >= 2:
        symptoms.sort(key=lambda x: x[0].start)
        current_group = [symptoms[0][0]]

        for i in range(1, len(symptoms)):
            s = symptoms[i][0]
            between = payload_text[current_group[-1].end:s.start]
            if re.search(r'[.!?]', between):
                result.append(("medical_symptom", current_group))
                current_group = [s]
            else:
                current_group.append(s)

        result.append(("medical_symptom", current_group))
    else:
        for s, t in symptoms:
            result.append((t, [s]))

    for s, t in others:
        result.append((t, [s]))

    return result


# ── Rule-based structured abstractions ───────────────────────────────────────

_MONTH_RE = re.compile(
    r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
    r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b',
    re.IGNORECASE,
)
_MONTH_EXPAND = {
    "jan": "January",  "feb": "February", "mar": "March",     "apr": "April",
    "may": "May",      "jun": "June",     "jul": "July",      "aug": "August",
    "sep": "September","oct": "October",  "nov": "November",  "dec": "December",
}
_DAY_RE  = re.compile(r'\b(\d{1,2})(?:st|nd|rd|th)?\b')
_TIME_RE = re.compile(r'\b(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm)\b')
_HOUR_RE = re.compile(r'\b(\d{1,2}):(\d{2})\b')


def _rule_date(text: str, level: int) -> Optional[str]:
    months = _MONTH_RE.findall(text)
    if not months:
        return None
    month = _MONTH_EXPAND.get(months[0].lower()[:3], months[0].capitalize())
    days  = [int(d) for d in _DAY_RE.findall(text)]
    if level == 0:
        return f"in {month}"
    elif level == 1:
        if days:
            q = "early" if min(days) <= 10 else ("mid" if min(days) <= 20 else "late")
            return f"{q}-{month}"
        return f"in {month}"
    elif level == 2:
        return f"{month} {days[0]}" if days else f"in {month}"
    return None


def _rule_time(text: str, level: int) -> Optional[str]:
    hour: Optional[int] = None
    m = _TIME_RE.search(text)
    if m:
        hour, mer = int(m.group(1)), m.group(3).upper()
        if mer == "PM" and hour != 12:
            hour += 12
        elif mer == "AM" and hour == 12:
            hour = 0
    else:
        m2 = _HOUR_RE.search(text)
        if m2:
            hour = int(m2.group(1))
    if hour is None:
        return None
    if level == 0:
        return "morning" if hour < 12 else ("afternoon" if hour < 17 else "evening")
    elif level == 1:
        return "before noon" if hour < 12 else ("midday" if hour < 14 else "afternoon")
    elif level == 2:
        return f"{hour}:00–{hour + 1}:00"
    return None


def _rule_address(text: str, level: int) -> Optional[str]:
    parts = [p.strip() for p in text.split(",")]
    if level == 0:
        state = parts[-1].strip() if len(parts) >= 2 else ""
        return f"{state} area" if state else "local area"
    elif level == 1:
        if len(parts) >= 3:
            return f"{parts[-2].strip()}, {parts[-1].strip()}"
        return text  # already "City, State"
    return None


def _rule_abstract(span: Span, abs_type: str, level: int) -> Optional[str]:
    if abs_type == "date":
        return _rule_date(span.text, level)
    if abs_type == "time":
        return _rule_time(span.text, level)
    if abs_type in ("address", "location"):
        return _rule_address(span.text, level)
    if abs_type == "zip":
        return "local area" if level == 0 else "nearby"
    return None


# ── Fallback phrases (no LLM, rule fails) ─────────────────────────────────────

_FALLBACK: Dict[str, Dict[int, str]] = {
    "address":            {0: "local area",            1: "the local area",          2: "a nearby neighborhood"},
    "location":           {0: "local area",            1: "the local area"},
    "zip":                {0: "local area",             1: "nearby"},
    "insurance_name":     {0: "insurance",             1: "dental insurance",        2: "a dental plan"},
    "medical_symptom":    {0: "a health concern",      1: "a dental care need",      2: "a dental symptom"},
    "prior_provider":     {0: "a prior provider",      1: "a prior dental provider", 2: "a local dental office"},
    "service_need":       {0: "a service requirement", 1: "a provider capability"},
    "dietary_constraint": {0: "a dining constraint",   1: "a dietary preference",    2: "a dietary constraint"},
    "travel_purpose":     {0: "travel context",        1: "personal travel",         2: "a travel purpose"},
    "preference":         {0: "a preference",          1: "a personal preference",   2: "a scheduling preference"},
    "person":             {0: "a person"},
    "group":              {0: "a group",               1: "a local group"},
    "generic_detail":     {0: "a general detail",      1: "a relevant detail"},
}

GROUP_SYMPTOM_FALLBACK = "dental symptoms"


def _fallback_abstract(abs_type: str, level: int) -> str:
    fb = _FALLBACK.get(abs_type, {})
    if level in fb:
        return fb[level]
    hier = ABSTRACTION_HIERARCHIES.get(abs_type, ["general detail"])
    return hier[min(level, len(hier) - 1)]


# ── LLM abstraction prompts ────────────────────────────────────────────────────

_ABSTRACTION_PROMPT = """\
You are generalizing a span from a user payload before it is sent to a cloud search agent.

Task frame: {g_t}
Original span: "{span_text}"
Extractor type: {extractor_type}
Abstraction type: {abstraction_type}
Target abstraction level: {level_desc}

Rewrite this span as a "{level_desc}". You must follow all rules below:
- Preserve the semantic role exactly: the output must remain {abstraction_type}-like
- Do not change the entity type (a provider stays provider-like; a symptom stays symptom-like)
- Do not diagnose, infer, or introduce facts not present in the original span
- The result must be less specific than the original
- Return a short noun phrase only (2–5 words), not a sentence
- No trailing punctuation

Reply with only the abstracted phrase."""

_GROUP_ABSTRACTION_PROMPT = """\
You are jointly generalizing multiple symptom spans before sending a payload to a cloud agent.

Task frame: {g_t}
Individual symptoms: {symptom_list}
Abstraction type: medical_symptom
Target abstraction level: {level_desc}

Generate a single short noun phrase covering all these symptoms at the "{level_desc}" level.
Rules:
- The output must be symptom-like (not a diagnosis, not a condition name, not a disease)
- Do not name specific medical conditions (e.g., periodontitis, gingivitis)
- Use natural phrasing like "dental symptoms", "oral discomfort", "gum and tooth pain"
- Return a short noun phrase only (2–5 words)
- No trailing punctuation

Reply with only the abstracted phrase."""


# ── Post-abstraction validation ────────────────────────────────────────────────

_PROVIDER_SIGNALS = {
    "provider", "clinic", "office", "practice", "dental", "medical", "health",
    "center", "specialist", "doctor", "dentist", "physician", "care", "prior", "local",
}
_DIAGNOSIS_SIGNALS = {
    "disease", "disorder", "syndrome", "diagnosis", "pathology",
    "periodontitis", "gingivitis", "abscess", "inflammation",
}


def _validate(abs_type: str, abstracted_text: str) -> str:
    """Return 'passed' or 'failed:reason'."""
    words = set(re.findall(r'\b[a-z]+\b', abstracted_text.lower()))

    if abs_type == "prior_provider":
        if not (words & _PROVIDER_SIGNALS):
            return "failed:not provider-like"

    if abs_type == "medical_symptom":
        alpha = [w for w in abstracted_text.split() if w.isalpha()]
        if alpha and all(w[0].isupper() for w in alpha):
            return "failed:appears to be a proper noun (entity type change)"
        if words & _DIAGNOSIS_SIGNALS:
            return "failed:contains diagnostic terminology"

    if len(abstracted_text.strip()) < 2:
        return "failed:empty output"

    return "passed"


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class AbstractionDecision:
    span:             Span
    original_text:    str
    abstracted_text:  str
    extractor_type:   str   # raw span_type from extractor
    abstraction_type: str   # inferred semantic type
    level:            int
    level_desc:       str
    method:           str   # "rule" | "llm" | "verbatim" | "grouped"
    validation:       str   # "passed" | "failed:reason" | "n/a"
    note:             str   # "" | "group anchor (N spans)" | "grouped with X"


@dataclass
class AbstractionResult:
    decisions:     List[AbstractionDecision]
    final_payload: str
    method:        str   # "llm:<model>" | "rule"


# ── SpanAbstractor ─────────────────────────────────────────────────────────────

class SpanAbstractor:
    """
    Stage 3b abstractor.

    For each CSS span:
      1. Infer semantic abstraction type from extractor type + text.
      2. Group adjacent medical_symptom spans within the same sentence.
      3. Look up calibrated level from CALIBRATED_ABSTRACTION_POLICY.
      4. Abstract: rule-based for structured types; LLM for free-form.
      5. Validate output preserves semantic role; fall back if not.
      6. Assemble final payload P̂_t.
    """

    def abstract(
        self,
        css_spans:  List[Span],
        pth_spans:  List[Span],
        g_t:        str,
        extraction: dict,
        u_loc:      List[Span],
        local_llm             = None,
    ) -> AbstractionResult:
        payload_text = extraction["P_t"]

        # Infer semantic abstraction types for all CSS spans
        spans_with_types = [(s, _infer_abstraction_type(s)) for s in css_spans]

        # Group adjacent symptoms
        groups = _group_symptoms(spans_with_types, payload_text)

        decisions:       List[AbstractionDecision] = []
        abstraction_map: Dict[str, str]            = {}

        for abs_type, span_group in groups:
            hier  = ABSTRACTION_HIERARCHIES.get(abs_type, ABSTRACTION_HIERARCHIES["generic_detail"])
            level = min(
                CALIBRATED_ABSTRACTION_POLICY.get(abs_type, DEFAULT_ABSTRACTION_LEVEL),
                len(hier) - 1,
            )
            level_desc = hier[level]

            if len(span_group) > 1:
                # ── Joint abstraction for grouped symptoms ────────────────────
                anchor    = min(span_group, key=lambda s: s.start)
                rest      = [s for s in span_group if s is not anchor]
                abstracted, method = self._abstract_group(
                    span_group, level_desc, g_t, local_llm
                )
                validation = _validate(abs_type, abstracted)
                if validation.startswith("failed"):
                    abstracted = _fallback_abstract(abs_type, level)
                    method     = "rule"
                    validation += " → fallback"

                abstraction_map[anchor.text.lower()] = abstracted
                decisions.append(AbstractionDecision(
                    span=anchor, original_text=anchor.text, abstracted_text=abstracted,
                    extractor_type=anchor.span_type, abstraction_type=abs_type,
                    level=level, level_desc=level_desc, method=method,
                    validation=validation,
                    note=f"group anchor ({len(span_group)} spans)",
                ))
                for s in rest:
                    abstraction_map[s.text.lower()] = ""
                    decisions.append(AbstractionDecision(
                        span=s, original_text=s.text, abstracted_text="",
                        extractor_type=s.span_type, abstraction_type=abs_type,
                        level=level, level_desc=level_desc, method="grouped",
                        validation="n/a", note=f"grouped with '{anchor.text}'",
                    ))

            else:
                # ── Individual abstraction ────────────────────────────────────
                span = span_group[0]

                if level == 3:
                    abstracted, method, validation = span.text, "verbatim", "n/a"
                else:
                    abstracted, method = self._abstract_span(
                        span, abs_type, level, level_desc, g_t, local_llm
                    )
                    validation = _validate(abs_type, abstracted)
                    if validation.startswith("failed"):
                        abstracted = _fallback_abstract(abs_type, level)
                        method     = "rule"
                        validation += " → fallback"

                abstraction_map[span.text.lower()] = abstracted
                decisions.append(AbstractionDecision(
                    span=span, original_text=span.text, abstracted_text=abstracted,
                    extractor_type=span.span_type, abstraction_type=abs_type,
                    level=level, level_desc=level_desc, method=method,
                    validation=validation, note="",
                ))

        final_payload = _assemble_final_payload(extraction, pth_spans, abstraction_map, u_loc)
        method        = f"llm:{local_llm.model}" if local_llm else "rule"

        return AbstractionResult(decisions=decisions, final_payload=final_payload, method=method)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _abstract_span(
        self,
        span:       Span,
        abs_type:   str,
        level:      int,
        level_desc: str,
        g_t:        str,
        local_llm,
    ) -> Tuple[str, str]:
        rule = _rule_abstract(span, abs_type, level)
        if rule is not None:
            return rule, "rule"

        if local_llm is not None:
            try:
                prompt = _ABSTRACTION_PROMPT.format(
                    g_t=g_t.strip(),
                    span_text=span.text,
                    extractor_type=span.span_type,
                    abstraction_type=abs_type,
                    level_desc=level_desc,
                )
                raw    = local_llm.generate(prompt).strip()
                result = raw.splitlines()[0].strip().rstrip(".,;")
                if result:
                    return result, "llm"
            except Exception:
                pass

        return _fallback_abstract(abs_type, level), "rule"

    def _abstract_group(
        self,
        spans:      List[Span],
        level_desc: str,
        g_t:        str,
        local_llm,
    ) -> Tuple[str, str]:
        if local_llm is not None:
            try:
                symptom_list = ", ".join(f'"{s.text}"' for s in spans)
                prompt = _GROUP_ABSTRACTION_PROMPT.format(
                    g_t=g_t.strip(),
                    symptom_list=symptom_list,
                    level_desc=level_desc,
                )
                raw    = local_llm.generate(prompt).strip()
                result = raw.splitlines()[0].strip().rstrip(".,;")
                if result:
                    return result, "llm"
            except Exception:
                pass
        return GROUP_SYMPTOM_FALLBACK, "rule"


# ── Final payload assembly ─────────────────────────────────────────────────────

def _assemble_final_payload(
    extraction:  dict,
    pth_spans:   List[Span],
    abstracted:  Dict[str, str],
    u_loc:       List[Span],
) -> str:
    """
    Fill C_t skeleton slots:
      U_loc  → [TYPE]
      PTH    → verbatim span text
      CSS    → abstracted text  (or "" for grouped-away spans)
      dropped → ""  (triggers cleanup below)
    """
    c_t       = extraction["C_t"]
    all_spans = extraction["all_spans"]

    pth_texts = {s.text.lower() for s in pth_spans}
    u_loc_map = {s.text.lower(): f"[{s.span_type.upper()}]" for s in (u_loc or [])}

    result = c_t
    for span in all_spans:
        tl = span.text.lower()
        if tl in u_loc_map:
            replacement = u_loc_map[tl]
        elif tl in pth_texts:
            replacement = span.text
        elif tl in abstracted:
            replacement = abstracted[tl]
        else:
            replacement = ""
        result = result.replace("[SPAN]", replacement, 1)

    # Multi-pass cleanup of empty-slot artefacts
    for _ in range(3):
        result = re.sub(r'\bMy\s+\w+\s+is\s+and\s+', '', result, flags=re.IGNORECASE)
        result = re.sub(r'\s{2,}and\s+', ' ', result)
        result = re.sub(
            r'\b(visited|seen at|been to|tried|used)\s+and\s+',
            r'\1 ', result, flags=re.IGNORECASE,
        )
        result = re.sub(r'\b(\w+)\s+is\s+and\b', r'\1 and', result, flags=re.IGNORECASE)
        result = re.sub(r'\b(\w+)\s+is\s*([,.])', r'\2', result, flags=re.IGNORECASE)
        result = re.sub(
            r'\b(on|at|before|after|for|near|is|are|and|or|of|in|by)\s*([,.])',
            r'\2', result, flags=re.IGNORECASE,
        )
        result = re.sub(r',\s*\.', '.', result)
        result = re.sub(r',\s*,', ',', result)
        result = re.sub(r'\.\s*\.', '.', result)
        result = re.sub(r'\s+([,.])', r'\1', result)
        result = re.sub(r' {2,}', ' ', result)
        result = re.sub(
            r'\bI want to keep the cost\s*[,.]?\s*', '', result, flags=re.IGNORECASE
        )

    _SW = {
        'i', 'me', 'my', 'hi', 'is', 'am', 'are', 'was', 'were', 'be',
        'the', 'a', 'an', 'if', 'it', 'its', 'and', 'or', 'but', 'so',
        'for', 'in', 'on', 'at', 'to', 'of', 'by', 'not', 'do', 'did',
        'have', 'has', 'had', 'can', 'will', 'would', 'may', 'might',
        'possible', 'preferably', 'previously', 'visited', 'book', 'keep',
        'want', 'need', 'get', 'also', 'just', 'that', 'this', 'with',
    }

    def _has_content(sent: str) -> bool:
        if '[' in sent:
            return True
        words = re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower())
        return sum(1 for w in words if w not in _SW) >= 2

    sentences = re.split(r'(?<=[.!?])\s+', result.strip())
    kept      = [s.strip() for s in sentences if s.strip() and _has_content(s)]
    return " ".join(kept).strip()


# ── Debug display ──────────────────────────────────────────────────────────────

def _debug_show(result: AbstractionResult) -> None:
    print(f"\n  Abstraction method : {result.method}")
    print(f"\n  Decisions:")
    for d in result.decisions:
        if d.method == "grouped":
            print(f"    [GROUPED ] {d.note}")
            continue
        print(
            f"    [CSS]\n"
            f"      original:         {d.original_text!r}\n"
            f"      extractor_type:   {d.extractor_type}\n"
            f"      abstraction_type: {d.abstraction_type}\n"
            f"      level:            l{d.level} / {d.level_desc}\n"
            f"      replacement:      {d.abstracted_text!r}\n"
            f"      method:           {d.method}\n"
            f"      validation:       {d.validation}"
            + (f"\n      note:             {d.note}" if d.note else "")
        )


# ── Standalone — runs Stage 1 → 2 → 3a → 3b ──────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from state.state_io import load_state
    from privacy.span_extractor      import SpanExtractor,    _debug_show as _show_spans
    from privacy.scope_control       import ScopeController,  reconstruct_payload, _debug_show as _show_scope
    from privacy.span_classification import SpanClassifier,   _debug_show as _show_class
    from llm.local_llm import LocalLLM

    state   = load_state()
    profile = state.get("user_profile", {})

    try:
        local_llm = LocalLLM()
        local_llm.generate("ping")
        print(f"\n  Local LLM: connected ({local_llm.model})")
    except Exception:
        local_llm = None
        print(f"\n  Local LLM: unavailable — rule/fallback for g_t, TaskGain, classification, abstraction")

    r_t = (
        "I have tooth pain and bleeding gums, "
        "book me a dentist appointment at the earliest."
    )
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
    print(f"  ORIGINAL USER REQUEST  (r_t)")
    print(f"{'═' * W}")
    print(f"  {r_t}")

    print(f"\n{'═' * W}")
    print(f"  LC-ENRICHED PAYLOAD  (p_t)  — input to sanitization pipeline")
    print(f"{'═' * W}")
    print(f"  {p_t}")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  STAGE 1 — SPAN EXTRACTION  (operating on p_t)")
    print(f"{'═' * W}")
    extractor  = SpanExtractor()
    extraction = extractor.extract(p_t, profile)
    _show_spans(extraction)

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  STAGE 2 — SCOPE CONTROL  (g_t derived from r_t)")
    print(f"{'═' * W}")
    controller   = ScopeController()
    scope_result = controller.filter(
        u_med=extraction["U_med"],
        task=r_t,
        payload=extraction["working"],
        local_llm=local_llm,
    )
    _show_scope(scope_result)

    print(f"\n{'═' * W}")
    print(f"  STAGE 2 OUTPUT  — payload entering Stage 3a")
    print(f"{'═' * W}")
    stage2_out = reconstruct_payload(extraction, scope_result.retained, extraction["U_loc"])
    print(f"  {stage2_out}")

    # ── Stage 3a ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  STAGE 3a — SENSITIVITY CLASSIFICATION  (joint over retained spans)")
    print(f"{'═' * W}")
    classifier   = SpanClassifier()
    class_result = classifier.classify(
        retained_spans=scope_result.retained,
        g_t=scope_result.task_frame,
        local_llm=local_llm,
    )
    _show_class(class_result)

    print(f"\n{'═' * W}")
    print(f"  STAGE 3a OUTPUT  — PTH verbatim; [css:...] pending abstraction")
    print(f"{'═' * W}")
    stage3a_out = reconstruct_payload(extraction, scope_result.retained, extraction["U_loc"])
    for span in class_result.context_sensitive:
        stage3a_out = stage3a_out.replace(span.text, f"[css:{span.text}]", 1)
    print(f"  {stage3a_out}")

    # ── Stage 3b ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  STAGE 3b — SPAN ABSTRACTION  (CSS spans → calibrated semantic level)")
    print(f"{'═' * W}")
    abstractor         = SpanAbstractor()
    abstraction_result = abstractor.abstract(
        css_spans  = class_result.context_sensitive,
        pth_spans  = class_result.passthrough,
        g_t        = class_result.task_frame,
        extraction = extraction,
        u_loc      = extraction["U_loc"],
        local_llm  = local_llm,
    )
    _debug_show(abstraction_result)

    print(f"\n{'═' * W}")
    print(f"  FINAL SANITIZED PAYLOAD  P̂_t  — sent to cloud CLM")
    print(f"{'═' * W}")
    print(f"  {abstraction_result.final_payload}")
