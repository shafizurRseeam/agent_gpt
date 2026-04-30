"""
privacy/span_abstraction.py

Stage 3b of PrivScope — Span Abstraction.

Replaces each context-sensitive span u_j ∈ C_t with a less specific,
type-consistent representation before cloud release:

  P̂_t = Assemble(R_t, D_t, {h_{T(u_j)}^{k_j}(u_j) : u_j ∈ C_t})

where:
  R_t    — residual scaffold (C_t slots)
  D_t    — PTH spans, released verbatim
  k_j    = π_ψ(u_j, T(u_j), g_t) — calibrated abstraction level
  T(u_j) — semantic abstraction type inferred from extractor span type

Semantic type inference
  Raw extractor types (noun_phrase, organization, facility) are too coarse.
  Stage 3b first infers a semantic abstraction type — medical_symptom,
  prior_provider, dietary_preference, occasion_relationship, etc. — using
  priority-ordered keyword matching, then looks up the type's hierarchy and
  calibrated policy level.

Adjacent span grouping
  Spans of the same groupable type (medical_symptom, medical_condition) in
  the same sentence are jointly abstracted. The anchor span gets the joint
  abstraction; remaining spans are collapsed to empty.

LLM abstraction with type-constrained prompting
  The prompt explicitly names the inferred abstraction_type to prevent entity
  class changes. Constraints prevent diagnosis, hallucination, over-specificity.

Post-abstraction validation
  Lightweight signal checks verify the output preserved its semantic role.
  Failures fall back to deterministic phrases from _FALLBACK.

Supported semantic types (35+):
  address, location, zip, date, time, distance_proximity, budget_cost,
  party_size, provider_preference, provider_requirement, insurance_name,
  service_need, accessibility_need, medical_condition, medical_symptom,
  care_history, medication, medical_trip, pregnancy_status,
  mental_health_concern, travel_itinerary, travel_purpose,
  transport_constraint, baggage_constraint, ticket_flexibility,
  child_travel, dietary_preference, allergy, restaurant_atmosphere,
  seating_preference, occasion_relationship, prior_provider, prior_venue,
  person, generic_detail.

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


# ── U_loc placeholder registry ────────────────────────────────────────────────
# All bracket placeholders that Stage 1 inserts for direct identifiers.
# Used by prepare_cloud_payload() to strip them before CLM release.

U_LOC_PLACEHOLDERS = {
    "[NAME]", "[PHONE]", "[EMAIL]", "[DOB]", "[SSN]",
    "[PASSPORT_NUMBER]", "[DRIVER_LICENSE]", "[INSURANCE_ID]",
    "[PATIENT_ID]", "[CREDIT_CARD]", "[MEMBERSHIP_ID]",
    "[FREQUENT_FLYER_NUMBER]", "[HOTEL_LOYALTY_ID]",
    "[VEHICLE_PLATE]",
}

# Stopwords for the cloud-payload content filter (no [bracket] shortcut)
_CLOUD_SW = {
    'i', 'me', 'my', 'hi', 'is', 'am', 'are', 'was', 'were', 'be',
    'the', 'a', 'an', 'if', 'it', 'its', 'and', 'or', 'but', 'so',
    'for', 'in', 'on', 'at', 'to', 'of', 'by', 'not', 'do', 'did',
    'have', 'has', 'had', 'can', 'will', 'would', 'may', 'might',
    'possible', 'preferably', 'previously', 'visited', 'book', 'keep',
    'want', 'need', 'get', 'also', 'just', 'that', 'this', 'with',
    'you', 'your', 'we', 'they', 'them', 'their', 'him', 'her',
    'reach', 'contact', 'call', 'send', 'tell', 'please',
}


# ── Span types that support adjacent-span joint abstraction ───────────────────

_GROUPABLE_TYPES = {"medical_symptom", "medical_condition"}


# ── Direct type map: structured extractor types ───────────────────────────────
# Types whose semantic meaning is already captured by the extractor type.
# "preference" and "group" require sub-classification; excluded here.

_DIRECT_TYPE_MAP: Dict[str, str] = {
    "address":        "address",
    "location":       "location",
    "zip":            "zip",
    "date":           "date",
    "time":           "time",
    "insurance_name": "insurance_name",
    "money":          "budget_cost",
    "party_size":     "party_size",
    "person":         "person",
}


# ── Keyword sets for semantic type inference (noun_phrase classification) ─────
# Priority order matters: more specific types are checked first.

_PREGNANCY_WORDS = {
    "pregnant", "pregnancy", "trimester", "prenatal", "postnatal", "postpartum",
    "expecting", "maternity", "obgyn", "obstetrician", "midwife", "gestational",
}

_MEDICATION_WORDS = {
    "medication", "medicine", "drug", "prescription", "pill", "tablet", "capsule",
    "dose", "dosage", "antibiotic", "insulin", "inhaler", "ointment", "cream",
    "supplement", "vitamin", "aspirin", "ibuprofen", "acetaminophen", "metformin",
    "statin", "antidepressant", "anxiolytic", "refill", "pharmacy",
}

_MENTAL_HEALTH_WORDS = {
    "anxiety", "depression", "stress", "mental", "psychiatric", "therapy",
    "therapist", "counseling", "counselor", "psychologist", "psychiatrist",
    "ptsd", "adhd", "bipolar", "panic", "phobia", "ocd", "schizophrenia",
}

_CARE_HISTORY_WORDS = {
    "history", "previously", "previous", "former", "past", "surgery", "operation",
    "hospitalized", "procedure", "treated", "diagnosed", "prescribed", "removed",
    "hospitalization", "implant", "biopsy", "extraction",
}

_MEDICAL_CONDITION_WORDS = {
    "diabetes", "hypertension", "asthma", "cancer", "arthritis", "epilepsy",
    "condition", "disease", "disorder", "chronic", "syndrome", "infection",
    "eczema", "psoriasis", "thyroid", "cardiac", "heart", "stroke", "autism",
    "sclerosis", "fibromyalgia", "colitis", "crohn",
}

_SYMPTOM_WORDS = {
    "pain", "ache", "aches", "gum", "gums", "tooth", "teeth", "bleed", "bleeding",
    "sore", "soreness", "hurt", "hurts", "swollen", "swelling", "cavity", "cavities",
    "sensitivity", "tenderness", "discomfort", "symptom", "toothache", "nausea",
    "fatigue", "headache", "dizziness", "rash", "itch", "itchy", "cough", "fever",
    "vomiting", "shortness", "stiffness", "cramping", "bruising", "lesion",
}

_MEDICAL_TRIP_WORDS = {
    "treatment", "specialist", "follow-up", "checkup", "scan", "mri", "biopsy",
    "procedure", "surgery", "hospital",
}
_TRAVEL_GENERAL_WORDS = {
    "trip", "travel", "flight", "journey", "abroad", "overseas", "destination",
    "hotel", "vacation", "holiday",
}

_ACCESSIBILITY_WORDS = {
    "wheelchair", "accessible", "accessibility", "mobility", "disabled", "disability",
    "hearing", "visual", "blind", "deaf", "elevator", "ramp", "handicap", "ada",
    "assistance", "impaired",
}

_CHILD_TRAVEL_WORDS = {
    "child", "children", "kid", "kids", "baby", "infant", "toddler", "stroller",
    "minor", "pediatric", "nursery", "lap", "unaccompanied",
}

_ALLERGY_WORDS = {
    "allergy", "allergic", "allergen", "peanut", "tree nut", "shellfish", "seafood",
    "gluten", "celiac", "lactose", "soy", "wheat", "sesame", "egg", "intolerance",
    "anaphylactic",
}

_DIETARY_PREF_WORDS = {
    "vegetarian", "vegan", "pescatarian", "kosher", "halal", "omnivore",
    "plant-based", "organic", "low-carb", "keto", "paleo", "mediterranean",
    "low-sodium", "low-fat", "gluten-free", "dairy-free", "lactose-free",
}

_TICKET_FLEX_WORDS = {
    "refundable", "non-refundable", "flexible", "changeable", "cancel",
    "cancellation", "exchange", "rebook", "open-ticket", "standby", "upgrade",
}

_BAGGAGE_WORDS = {
    "luggage", "baggage", "bag", "suitcase", "carry-on", "checked", "oversize",
    "sports equipment", "golf", "ski", "stroller", "extra bag",
}

_TRANSPORT_WORDS = {
    "nonstop", "direct", "layover", "stopover", "connection", "airline",
    "business class", "economy", "first class", "window seat", "aisle seat",
    "seat preference", "flight number", "departure",
}

_TRAVEL_PURPOSE_WORDS = {
    "business", "conference", "meeting", "work trip", "vacation", "holiday",
    "leisure", "tourism", "sightseeing", "backpacking", "honeymoon",
}

_OCCASION_WORDS = {
    "birthday", "anniversary", "wedding", "engagement", "graduation", "promotion",
    "romantic", "proposal", "celebration", "reunion", "memorial", "retirement",
    "bachelorette", "bachelor", "client dinner", "colleague",
}

_ATMOSPHERE_WORDS = {
    "quiet", "lively", "cozy", "romantic", "trendy", "casual", "formal",
    "upscale", "rustic", "modern", "traditional", "vibrant", "intimate",
    "atmosphere", "ambiance", "ambience", "vibe", "family-friendly",
}

_SEATING_WORDS = {
    "outdoor", "indoor", "patio", "terrace", "booth", "bar seating", "counter",
    "window seat", "high chair", "seating", "corner table", "private room",
    "balcony",
}

_BUDGET_WORDS = {
    "budget", "affordable", "cheap", "inexpensive", "mid-range", "luxury",
    "economical", "splurge", "cost-effective", "upscale", "under", "limit",
}

_SERVICE_WORDS = {
    "accepting", "accepting new", "emergency", "same-day", "walk-in",
    "appointment", "telehealth", "online", "multilingual", "late hours",
    "weekend", "house call",
}

_VENUE_WORDS = {
    "restaurant", "cafe", "bistro", "bar", "pub", "grill", "diner", "eatery",
    "kitchen", "lounge", "brasserie", "tavern", "inn", "hotel", "resort",
    "spa", "theater", "theatre", "museum", "gallery", "buffet",
}


# ── Semantic type inference ────────────────────────────────────────────────────

def _infer_abstraction_type(span: Span) -> str:
    """
    Infer semantic abstraction type from extractor span type and text content.

    Structured types map directly via _DIRECT_TYPE_MAP.
    noun_phrase uses priority-ordered keyword matching across all domains.
    organization / facility are distinguished as prior_provider vs prior_venue.
    preference is sub-classified into dietary_preference, seating_preference,
    or provider_preference.
    """
    if span.span_type in _DIRECT_TYPE_MAP:
        return _DIRECT_TYPE_MAP[span.span_type]

    if span.span_type == "group":
        return "party_size"

    if span.span_type == "preference":
        words = set(re.findall(r'\b[a-z]+\b', span.text.lower()))
        if words & _DIETARY_PREF_WORDS:
            return "dietary_preference"
        if words & _SEATING_WORDS:
            return "seating_preference"
        return "provider_preference"

    if span.span_type in ("organization", "facility"):
        words = set(re.findall(r'\b[a-z]+\b', span.text.lower()))
        if words & _VENUE_WORDS:
            return "prior_venue"
        return "prior_provider"

    if span.span_type == "noun_phrase":
        text_lower = span.text.lower()
        words      = set(re.findall(r'\b[a-z]+\b', text_lower))

        # ── Medical (highest specificity first) ───────────────────────────────
        if words & _PREGNANCY_WORDS:
            return "pregnancy_status"
        if words & _MEDICATION_WORDS:
            return "medication"
        if words & _MENTAL_HEALTH_WORDS:
            return "mental_health_concern"
        if words & _CARE_HISTORY_WORDS:
            return "care_history"
        if words & _MEDICAL_CONDITION_WORDS:
            return "medical_condition"
        if words & _SYMPTOM_WORDS:
            return "medical_symptom"
        if words & _ACCESSIBILITY_WORDS:
            return "accessibility_need"
        # Medical trip: needs BOTH a medical purpose word AND a travel word
        if words & _MEDICAL_TRIP_WORDS and words & _TRAVEL_GENERAL_WORDS:
            return "medical_trip"

        # ── Travel sub-types ──────────────────────────────────────────────────
        if words & _CHILD_TRAVEL_WORDS:
            return "child_travel"
        if words & _BAGGAGE_WORDS:
            return "baggage_constraint"
        if words & _TICKET_FLEX_WORDS:
            return "ticket_flexibility"
        if words & _TRANSPORT_WORDS:
            return "transport_constraint"
        if words & _TRAVEL_PURPOSE_WORDS:
            return "travel_purpose"
        if words & _TRAVEL_GENERAL_WORDS:
            return "travel_itinerary"

        # ── Dining sub-types ──────────────────────────────────────────────────
        if words & _ALLERGY_WORDS:
            return "allergy"
        if words & _DIETARY_PREF_WORDS:
            return "dietary_preference"
        if words & _OCCASION_WORDS:
            return "occasion_relationship"
        if words & _ATMOSPHERE_WORDS:
            return "restaurant_atmosphere"
        if words & _SEATING_WORDS:
            return "seating_preference"

        # ── Generic service / budget ──────────────────────────────────────────
        if words & _BUDGET_WORDS:
            return "budget_cost"
        if words & _SERVICE_WORDS:
            return "service_need"

        # ── Proper noun → named entity (provider or venue) ────────────────────
        alpha = [w for w in span.text.split() if w.isalpha()]
        if alpha and all(w[0].isupper() for w in alpha):
            if any(w.lower() in _VENUE_WORDS for w in alpha):
                return "prior_venue"
            return "prior_provider"

    return "generic_detail"


# ── Adjacent span grouping (same type, same sentence) ─────────────────────────

def _group_spans(
    spans_with_types: List[Tuple[Span, str]],
    payload_text:     str,
) -> List[Tuple[str, List[Span]]]:
    """
    Group adjacent same-type spans within the same sentence for all
    types in _GROUPABLE_TYPES (medical_symptom, medical_condition).

    Returns a list of (abstraction_type, [spans]) tuples.
    Groups of 2+ spans yield one anchor entry + N-1 empty entries.
    Non-groupable spans are returned as single-element groups.
    """
    groupable = [(s, t) for s, t in spans_with_types if t in _GROUPABLE_TYPES]
    others    = [(s, t) for s, t in spans_with_types if t not in _GROUPABLE_TYPES]

    result: List[Tuple[str, List[Span]]] = []

    # Group per type separately (medical_symptom stays separate from medical_condition)
    by_type: Dict[str, List[Span]] = {}
    for s, t in groupable:
        by_type.setdefault(t, []).append(s)

    for abs_type, type_spans in by_type.items():
        if len(type_spans) >= 2:
            type_spans.sort(key=lambda s: s.start)
            current_group = [type_spans[0]]
            for i in range(1, len(type_spans)):
                s       = type_spans[i]
                between = payload_text[current_group[-1].end:s.start]
                if re.search(r'[.!?]', between):
                    result.append((abs_type, current_group))
                    current_group = [s]
                else:
                    current_group.append(s)
            result.append((abs_type, current_group))
        else:
            for s in type_spans:
                result.append((abs_type, [s]))

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
        return text
    return None


def _rule_abstract(span: Span, abs_type: str, level: int) -> Optional[str]:
    if abs_type == "date":
        return _rule_date(span.text, level)
    if abs_type == "time":
        return _rule_time(span.text, level)
    if abs_type in ("address", "location"):
        return _rule_address(span.text, level)
    if abs_type == "zip":
        return "local area" if level == 0 else "nearby area"
    return None


# ── Fallback phrases ──────────────────────────────────────────────────────────

_FALLBACK: Dict[str, Dict[int, str]] = {
    # Structured
    "address":               {0: "local area",                  1: "the local area",                2: "a nearby neighborhood"},
    "location":              {0: "local area",                  1: "the local area",                2: "a nearby area"},
    "zip":                   {0: "local area",                  1: "nearby area",                   2: "the local zip area"},
    "date":                  {0: "a month",                     1: "a week",                        2: "a day"},
    "time":                  {0: "part of day",                 1: "a time window",                 2: "an hour block"},
    "distance_proximity":    {0: "nearby",                      1: "within a broad area",           2: "within a travel-time range"},
    "budget_cost":           {0: "a budget preference",         1: "a budget tier",                 2: "an approximate limit"},
    "party_size":            {0: "a group",                     1: "an approximate group size",     2: "a group size"},

    # Provider / service
    "provider_preference":   {0: "a provider constraint",       1: "a provider attribute",          2: "a specific requirement"},
    "provider_requirement":  {0: "a service requirement",       1: "a provider capability",         2: "a specific service need"},
    "insurance_name":        {0: "insurance context",           1: "a coverage category",           2: "an insurance plan category"},
    "service_need":          {0: "a service need",              1: "a service category",            2: "a specific service type"},
    "accessibility_need":    {0: "an assistance need",          1: "an accessibility need",         2: "a specific accommodation type"},

    # Medical
    "medical_condition":     {0: "a health concern",            1: "a condition category",          2: "a specific condition"},
    "medical_symptom":       {0: "a health concern",            1: "a symptom category",            2: "a specific symptom"},
    "care_history":          {0: "prior care context",          1: "a care history category",       2: "a prior care event"},
    "medication":            {0: "medication context",          1: "a medication class",            2: "a medication category"},
    "medical_trip":          {0: "travel context",              1: "health-related travel",         2: "a medical travel need"},
    "pregnancy_status":      {0: "a health context",            1: "a pregnancy-related context",   2: "pregnancy status"},
    "mental_health_concern": {0: "a health concern",            1: "a mental health category",      2: "a mental health concern"},

    # Travel
    "travel_itinerary":      {0: "a travel plan",               1: "a transport or lodging need",   2: "a route or trip constraint"},
    "travel_purpose":        {0: "travel context",              1: "a purpose category",            2: "a specific travel purpose"},
    "transport_constraint":  {0: "a transport constraint",      1: "a flight or route preference",  2: "a specific travel constraint"},
    "baggage_constraint":    {0: "a baggage constraint",        1: "a baggage category",            2: "a specific baggage need"},
    "ticket_flexibility":    {0: "ticket flexibility",          1: "a booking flexibility need",    2: "a specific ticket requirement"},
    "child_travel":          {0: "family travel context",       1: "a child travel need",           2: "a child-related travel constraint"},

    # Dining
    "dietary_preference":    {0: "a food preference",           1: "a dietary category",            2: "a specific diet"},
    "allergy":               {0: "a food constraint",           1: "an allergy constraint",         2: "a specific allergen category"},
    "restaurant_atmosphere": {0: "a dining atmosphere",         1: "an ambience category",          2: "a specific atmosphere preference"},
    "seating_preference":    {0: "a seating preference",        1: "a seating category",            2: "a specific seating type"},
    "occasion_relationship": {0: "a social context",            1: "an occasion category",          2: "a specific occasion"},

    # Named entities / fallback
    "prior_provider":        {0: "a prior provider",            1: "a prior provider category",     2: "a local provider type"},
    "prior_venue":           {0: "a prior venue",               1: "a prior venue category",        2: "a local venue type"},
    "person":                {0: "a person reference",          1: "a role or relation"},
    "generic_detail":        {0: "general context",             1: "a relevant detail",             2: "a more specific detail"},
}


def _fallback_abstract(abs_type: str, level: int) -> str:
    fb = _FALLBACK.get(abs_type, {})
    if level in fb:
        return fb[level]
    hier = ABSTRACTION_HIERARCHIES.get(abs_type, ["general context"])
    return hier[min(level, len(hier) - 1)]


# ── LLM abstraction prompts ────────────────────────────────────────────────────

_ABSTRACTION_PROMPT = """\
You are generalizing a span from a user payload before it is sent to a cloud search agent.

Task frame: {g_t}
Original span: "{span_text}"
Extractor type: {extractor_type}
Abstraction type: {abstraction_type}
Target abstraction level: {level_desc}

Rewrite this span as a "{level_desc}". Follow all rules:
- Preserve the semantic role exactly: the output must remain {abstraction_type}-like
- Do not change the entity type (a prior provider stays provider-like; a symptom stays symptom-like)
- Do not diagnose, infer, or introduce facts not in the original span
- The result must be less specific than the original
- Return a short noun phrase only (2–5 words), not a sentence
- No trailing punctuation

Reply with only the abstracted phrase."""

_GROUP_ABSTRACTION_PROMPT = """\
You are jointly generalizing multiple co-occurring spans before sending a payload to a cloud agent.

Task frame: {g_t}
Spans (all of type "{abstraction_type}"): {span_list}
Abstraction type: {abstraction_type}
Target abstraction level: {level_desc}

Generate a single short noun phrase covering all these spans at the "{level_desc}" level.
Rules:
- The output must be {abstraction_type}-like (do not change the entity class)
- Do not diagnose, name specific entities, or introduce facts not in the originals
- Use natural phrasing appropriate to the abstraction type
- Return a short noun phrase only (2–5 words)
- No trailing punctuation

Reply with only the abstracted phrase."""


# ── Post-abstraction validation ────────────────────────────────────────────────

_VALIDATION_SIGNALS: Dict[str, Dict] = {
    "prior_provider": {
        "required_positive": {
            "provider", "clinic", "office", "practice", "dental", "medical",
            "health", "center", "specialist", "doctor", "dentist", "physician",
            "care", "prior", "local",
        },
    },
    "prior_venue": {
        "required_positive": {
            "restaurant", "venue", "bar", "cafe", "bistro", "diner", "eatery",
            "place", "spot", "local", "prior", "dining",
        },
    },
    "medical_symptom": {
        "prohibited": {
            "disease", "disorder", "syndrome", "diagnosis", "pathology",
            "periodontitis", "gingivitis", "abscess", "inflammation",
        },
        "no_proper_noun": True,
    },
    "medical_condition": {
        "no_proper_noun": True,
    },
    "mental_health_concern": {
        "no_proper_noun": True,
    },
    "allergy": {
        "required_positive": {
            "allergy", "allergen", "allergic", "constraint", "food",
            "dietary", "intolerance", "health",
        },
    },
    "dietary_preference": {
        "required_positive": {
            "food", "diet", "dietary", "preference", "eating", "vegan",
            "vegetarian", "kosher", "halal", "plant", "organic",
        },
    },
}


def _validate(abs_type: str, abstracted_text: str) -> str:
    """Return 'passed' or 'failed:reason'."""
    if len(abstracted_text.strip()) < 2:
        return "failed:empty output"

    words   = set(re.findall(r'\b[a-z]+\b', abstracted_text.lower()))
    signals = _VALIDATION_SIGNALS.get(abs_type, {})

    required = signals.get("required_positive", set())
    if required and not (words & required):
        return f"failed:not {abs_type}-like"

    prohibited = signals.get("prohibited", set())
    if prohibited and (words & prohibited):
        return "failed:contains prohibited terms"

    if signals.get("no_proper_noun"):
        alpha = [w for w in abstracted_text.split() if w.isalpha()]
        if alpha and all(w[0].isupper() for w in alpha):
            return "failed:appears to be a proper noun (entity type change)"

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
    decisions:           List[AbstractionDecision]
    final_payload:       str   # internal: C_t with [TYPE] placeholders (debug / trace only)
    final_cloud_payload: str   # cleaned: U_loc stripped, ready for CLM
    method:              str   # "llm:<model>" | "rule"


# ── SpanAbstractor ─────────────────────────────────────────────────────────────

class SpanAbstractor:
    """
    Stage 3b abstractor.

    For each CSS span:
      1. Infer semantic abstraction type from extractor type + text.
      2. Group adjacent spans of groupable types (medical_symptom, medical_condition).
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

        spans_with_types = [(s, _infer_abstraction_type(s)) for s in css_spans]
        groups           = _group_spans(spans_with_types, payload_text)

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
                # ── Joint abstraction for grouped spans ───────────────────────
                anchor = min(span_group, key=lambda s: s.start)
                rest   = [s for s in span_group if s is not anchor]

                abstracted, method = self._abstract_group(
                    span_group, abs_type, level, level_desc, g_t, local_llm
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

        final_payload       = _assemble_final_payload(extraction, pth_spans, abstraction_map, u_loc)
        final_cloud_payload = prepare_cloud_payload(final_payload)
        method              = f"llm:{local_llm.model}" if local_llm else "rule"

        return AbstractionResult(
            decisions=decisions,
            final_payload=final_payload,
            final_cloud_payload=final_cloud_payload,
            method=method,
        )

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
        abs_type:   str,
        level:      int,
        level_desc: str,
        g_t:        str,
        local_llm,
    ) -> Tuple[str, str]:
        if local_llm is not None:
            try:
                span_list = ", ".join(f'"{s.text}"' for s in spans)
                prompt = _GROUP_ABSTRACTION_PROMPT.format(
                    g_t=g_t.strip(),
                    span_list=span_list,
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


# ── Final payload assembly ─────────────────────────────────────────────────────

def _assemble_final_payload(
    extraction:  dict,
    pth_spans:   List[Span],
    abstracted:  Dict[str, str],
    u_loc:       List[Span],
) -> str:
    """
    Fill C_t skeleton slots:
      U_loc   → [TYPE]
      PTH     → verbatim span text
      CSS     → abstracted text (or "" for grouped-away spans)
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


# ── Cloud payload preparation ─────────────────────────────────────────────────

def prepare_cloud_payload(internal_payload: str) -> str:
    """
    Strip U_loc placeholders from the internally reconstructed sanitized
    payload and remove the empty identifier-only fragments they leave behind.

    The internal payload uses [NAME], [PHONE], etc. for debug traceability.
    This function produces the cleaned text actually sent to the cloud CLM.

    Steps:
      1. Remove all direct-identifier bracket placeholders.
      2. Eliminate known identifier-introduction fragments
         (greeting/contact lines, "My X is ." artefacts).
      3. Multi-pass punctuation and whitespace cleanup.
      4. Drop sentences that become content-free after removal.
    """
    text = internal_payload

    # Step 1 — Remove U_loc placeholders
    for ph in U_LOC_PLACEHOLDERS:
        text = text.replace(ph, "")

    # Step 2 — Remove identifier-introduction fragments (multi-pass)
    for _ in range(3):
        # Greeting + contact intro: "Hi, I'm  and you can reach me at  or ."
        text = re.sub(r"Hi,?\s+I'?m\b[^.!?]*[.!?]", "", text, flags=re.IGNORECASE)
        # Residual contact fragment: "you can reach me at  or ."
        text = re.sub(r"[Yy]ou can reach me\b[^.!?]*[.!?]?", "", text)
        # "My X is ." where X is a 1–3-word field name (value was a placeholder)
        text = re.sub(
            r"\bMy\s+\w+(?:\s+\w+){0,2}\s+is\s*[,.]",
            "", text, flags=re.IGNORECASE,
        )
        # Empty preposition slots: "near ." "at ." "before ." etc.
        text = re.sub(
            r'\b(on|at|before|after|for|near|is|are|and|or|of|in|by)\s*([,.])',
            r'\2', text, flags=re.IGNORECASE,
        )
        text = re.sub(r'\b(\w+)\s+is\s*([,.])', r'\2', text, flags=re.IGNORECASE)
        text = re.sub(r'\bMy\s+\w+\s+is\s+and\s+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s{2,}and\s+', ' ', text)
        text = re.sub(r',\s*\.', '.', text)
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r'\.\s*\.', '.', text)
        text = re.sub(r'\s+([,.])', r'\1', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()

    # Step 3 — Drop sentences that are content-free after removal
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    kept = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        words    = re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower())
        meaningful = sum(1 for w in words if w not in _CLOUD_SW)
        if meaningful >= 2:
            kept.append(sent)

    return ' '.join(kept).strip()


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
            f"      original_text:    {d.original_text!r}\n"
            f"      extractor_type:   {d.extractor_type}\n"
            f"      abstraction_type: {d.abstraction_type}\n"
            f"      selected_level:   l{d.level} / {d.level_desc}\n"
            f"      abstracted_text:  {d.abstracted_text!r}\n"
            f"      method:           {d.method}\n"
            f"      validation:       {d.validation}"
            + (f"\n      grouping_note:    {d.note}" if d.note else "")
        )


# ── Standalone — runs Stage 1 → 2 → 3a → 3b ──────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from state.state_io import load_state
    from privacy.span_extractor      import SpanExtractor,   _debug_show as _show_spans
    from privacy.scope_control       import ScopeController, reconstruct_payload, _debug_show as _show_scope
    from privacy.span_classification import SpanClassifier,  _debug_show as _show_class
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
    print(f"  STAGE 2 OUTPUT  — payload entering Stage 3a (retained spans verbatim; U_loc withheld)")
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
    print(f"  STAGE 3a OUTPUT  — PTH verbatim; [css:...] pending Stage 3b abstraction")
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
    print(f"  INTERNAL SANITIZED PAYLOAD  — U_loc as [TYPE] placeholders (debug / trace only)")
    print(f"{'═' * W}")
    print(f"  {abstraction_result.final_payload}")

    print(f"\n{'═' * W}")
    print(f"  FINAL CLOUD PAYLOAD  P̂_t  — U_loc stripped and cleaned, sent to CLM")
    print(f"{'═' * W}")
    print(f"  {abstraction_result.final_cloud_payload}")
