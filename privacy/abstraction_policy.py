"""
privacy/abstraction_policy.py

Runtime materialization of the offline-calibrated abstraction policy π_ψ.

In the PrivScope framework, π_ψ is calibrated offline on representative
task–type pairs by searching from coarsest to finest abstraction levels
and recording the least specific level that still allows the CLM response
to be grounded locally into an actionable result. This module stores the
output of that calibration process for runtime use by Stage 3b.

Semantic abstraction types are inferred from raw extractor span types by
Stage 3b. Unlike raw types (noun_phrase, organization, facility), semantic
types capture the role of a span in context: medical_condition,
prior_provider, dietary_preference, occasion_relationship, etc. This lets
Stage 3b select type-appropriate hierarchies and calibrated levels
regardless of which extraction layer produced the span.

Supported domains:
  Medical booking   — conditions, symptoms, care history, medication,
                      pregnancy, mental health, accessibility, medical travel
  Travel booking    — purpose, itinerary, transport, baggage, flexibility,
                      child travel, budget, proximity
  Restaurant booking — dietary preferences, allergies, atmosphere, seating,
                       occasion / relationship context
"""

from typing import Dict, List

# ── Type-specific abstraction hierarchies H_T = (h^0, h^1, h^2, h^3) ─────────
# h^0 = coarsest (maximum privacy reduction), h^3 = finest (verbatim).
# Hierarchy level strings are used as target-level descriptions in LLM prompts.

ABSTRACTION_HIERARCHIES: Dict[str, List[str]] = {

    # ── Structured / semi-structured ──────────────────────────────────────────
    "age": [
        "adult",
        "decade range",
        "5-year range",
        "exact age",
    ],
    "address": [
        "region",
        "city",
        "neighborhood",
        "exact address",
    ],
    "location": [
        "region",
        "city",
        "neighborhood",
        "exact location",
    ],
    "zip": [
        "region",
        "city",
        "local area",
        "exact zip code",
    ],
    "date": [
        "month",
        "week",
        "day range",
        "exact date",
    ],
    "time": [
        "day part",
        "hour block",
        "time window",
        "exact time",
    ],
    "distance_proximity": [
        "nearby",
        "broad distance range",
        "travel-time range",
        "exact distance or travel time",
    ],
    "budget_cost": [
        "cost preference",
        "budget tier",
        "approximate limit",
        "exact amount",
    ],
    "party_size": [
        "group size category",
        "approximate party size",
        "group size",
        "exact party size",
    ],

    # ── Provider / service constraints ────────────────────────────────────────
    "provider_preference": [
        "provider constraint",
        "provider attribute",
        "specific requirement",
        "exact provider preference",
    ],
    "provider_requirement": [
        "service requirement",
        "provider capability",
        "specific service requirement",
        "exact requirement",
    ],
    "insurance_name": [
        "insurance context",
        "coverage category",
        "plan category",
        "exact plan name",
    ],
    "service_need": [
        "service need",
        "service category",
        "specific service type",
        "exact service request",
    ],
    "accessibility_need": [
        "assistance need",
        "accessibility category",
        "specific accommodation type",
        "exact accessibility request",
    ],

    # ── Medical ───────────────────────────────────────────────────────────────
    "medical_condition": [
        "health concern",
        "condition category",
        "specific condition",
        "exact condition phrase",
    ],
    "medical_symptom": [
        "health concern",
        "symptom category",
        "specific symptom category",
        "exact symptom",
    ],
    "care_history": [
        "prior care context",
        "care-history category",
        "specific care event",
        "exact care-history detail",
    ],
    "medication": [
        "medication context",
        "medication class",
        "specific medication category",
        "exact medication name",
    ],
    "medical_trip": [
        "travel context",
        "health-related travel",
        "medical travel purpose",
        "exact medical trip detail",
    ],
    "pregnancy_status": [
        "health context",
        "pregnancy-related context",
        "pregnancy status",
        "exact pregnancy statement",
    ],
    "mental_health_concern": [
        "health concern",
        "mental-health category",
        "specific mental-health concern",
        "exact mental-health phrase",
    ],

    # ── Travel ────────────────────────────────────────────────────────────────
    "travel_itinerary": [
        "travel plan",
        "transport or lodging need",
        "route or trip constraint",
        "exact itinerary detail",
    ],
    "travel_purpose": [
        "travel context",
        "purpose category",
        "specific purpose category",
        "exact travel purpose",
    ],
    "transport_constraint": [
        "transport constraint",
        "flight or route preference",
        "specific travel constraint",
        "exact transport detail",
    ],
    "baggage_constraint": [
        "baggage constraint",
        "baggage category",
        "specific baggage limit",
        "exact baggage detail",
    ],
    "ticket_flexibility": [
        "ticket flexibility",
        "booking flexibility category",
        "specific ticket flexibility need",
        "exact ticket requirement",
    ],
    "child_travel": [
        "family travel context",
        "child-related travel need",
        "specific child-travel constraint",
        "exact child-travel detail",
    ],

    # ── Restaurant / dining ───────────────────────────────────────────────────
    "dietary_preference": [
        "food preference",
        "dietary category",
        "specific diet",
        "exact dietary preference",
    ],
    "allergy": [
        "health-related food constraint",
        "allergy constraint",
        "specific allergen category",
        "exact allergen",
    ],
    "restaurant_atmosphere": [
        "dining atmosphere",
        "ambience category",
        "specific atmosphere preference",
        "exact atmosphere phrase",
    ],
    "seating_preference": [
        "seating preference",
        "seating category",
        "specific seating type",
        "exact seating request",
    ],
    "occasion_relationship": [
        "social context",
        "occasion category",
        "specific occasion or relationship",
        "exact occasion or relationship",
    ],

    # ── Named entities / fallback ─────────────────────────────────────────────
    "prior_provider": [
        "prior provider",
        "prior provider category",
        "local provider type",
        "specific provider name",
    ],
    "prior_venue": [
        "prior venue",
        "prior venue category",
        "local venue type",
        "specific venue name",
    ],
    "person": [
        "person reference",
        "role or relation",
        "name initial",
        "full name",
    ],
    "generic_detail": [
        "general context",
        "type-consistent category",
        "more specific abstraction",
        "original span",
    ],
}

# ── Runtime policy table — output of offline calibration ──────────────────────
# Each entry k gives the calibrated abstraction level for spans of that type.
# Calibration criterion: least specific level that still allows the CLM to
# ground its response into an actionable result for the relevant task class.
#
#   0 = coarsest (maximum privacy reduction)
#   3 = finest   (verbatim — use only when exact value is task-required)
#
# Rationale by group:
#   date              → 2  day range sufficient for availability search (H_date: month ≺ week ≺ day range ≺ exact)
#   time              → 2  time window sufficient for scheduling (H_time: day part ≺ hour block ≺ time window ≺ exact)
#   address / location → 1  city sufficient for provider or venue discovery (H_loca: region ≺ city ≺ neighborhood ≺ exact)
#   budget_cost       → 2  approximate limit sufficient; exact amount over-specified
#   medical (all)     → 1  condition / medication / pregnancy category sufficient
#   accessibility     → 2  functional type (wheelchair) must be retained for service match
#   dietary / allergy → 2  specific type usually needed for menu filtering
#   occasion          → 1  occasion category sufficient; relationship detail unnecessary
#   prior providers   → 0  named provider history dropped to category

CALIBRATED_ABSTRACTION_POLICY: Dict[str, int] = {
    # Structured constraints
    "age":                   1,   # decade range (e.g. "30s") sufficient for provider matching
    "address":               1,
    "location":              1,
    "zip":                   1,
    "date":                  3,   # verbatim — exact scheduling values required
    "time":                  3,   # verbatim — exact scheduling values required
    "distance_proximity":    2,
    "budget_cost":           2,
    "party_size":            2,

    # Provider / service
    "provider_preference":   2,
    "provider_requirement":  2,
    "insurance_name":        1,
    "service_need":          2,
    "accessibility_need":    2,

    # Medical
    "medical_condition":     1,
    "medical_symptom":       1,
    "care_history":          1,
    "medication":            1,
    "medical_trip":          1,
    "pregnancy_status":      1,
    "mental_health_concern": 1,

    # Travel
    "travel_itinerary":      2,
    "travel_purpose":        1,
    "transport_constraint":  2,
    "baggage_constraint":    2,
    "ticket_flexibility":    2,
    "child_travel":          1,

    # Restaurant / dining
    "dietary_preference":    2,
    "allergy":               2,
    "restaurant_atmosphere": 2,
    "seating_preference":    2,
    "occasion_relationship": 1,

    # Named entities / fallback
    "prior_provider":        0,
    "prior_venue":           0,
    "person":                0,
    "generic_detail":        1,
}

# Default level for types absent from the policy table
DEFAULT_ABSTRACTION_LEVEL = 1
