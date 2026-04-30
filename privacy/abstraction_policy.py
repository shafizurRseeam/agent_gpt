"""
privacy/abstraction_policy.py

Runtime materialization of the offline-calibrated abstraction policy π_ψ.

In the PrivScope framework, π_ψ is calibrated offline on representative
task–type pairs by searching from coarsest to finest abstraction levels
and recording the least specific level that still allows the CLM response
to be grounded locally into an actionable result. This module stores the
output of that calibration process for runtime use by Stage 3b.

Abstraction types are semantic types inferred from extractor span types.
Unlike raw span types (noun_phrase, organization, facility), semantic
abstraction types capture the role of a span in context: medical_symptom,
prior_provider, service_need, etc. This lets Stage 3b select type-
appropriate hierarchies and calibrated levels regardless of which
extraction layer produced the span.
"""

from typing import Dict, List

# ── Type-specific abstraction hierarchies H_T = (h^0, h^1, h^2, h^3) ─────────
# h^0 = coarsest, h^3 = finest (verbatim).
# Hierarchy strings serve as target-level descriptions in LLM prompts.

ABSTRACTION_HIERARCHIES: Dict[str, List[str]] = {
    # Structured span types
    "address":            ["region",                      "city or area",             "neighborhood",              "exact address"],
    "location":           ["region",                      "city or area",             "neighborhood",              "exact location"],
    "zip":                ["region",                      "state",                    "city",                      "zip code"],
    "date":               ["month",                       "week range",               "specific day",              "exact date"],
    "time":               ["part of day",                 "time window",              "hour block",                "exact time"],
    "money":              ["cost constraint",             "budget range",             "approximate budget",        "exact budget"],
    "party_size":         ["group size context",          "approximate group size",   "group size",                "exact count"],

    # Account-linked
    "insurance_name":     ["type of insurance",           "coverage category",        "plan category",             "exact plan name"],

    # Semantic types inferred from noun_phrase / organization / facility
    "medical_symptom":    ["health concern",              "care category",            "symptom category",          "specific symptom"],
    "prior_provider":     ["prior provider",              "prior dental provider",    "local dental office",       "specific provider name"],
    "service_need":       ["service requirement",         "provider capability",      "service category",          "specific service need"],
    "dietary_constraint": ["dining constraint",           "diet or allergy type",     "constraint category",       "specific diet or allergen"],
    "travel_purpose":     ["travel context",              "personal or work travel",  "purpose category",          "specific travel purpose"],
    "preference":         ["general preference",          "preference category",      "preference type",           "exact preference"],

    # Identity / social
    "person":             ["person reference",            "role or relation",         "name initial",              "full name"],
    "group":              ["group type",                  "group category",           "group name",                "exact group"],

    # Catch-all
    "generic_detail":     ["general category",            "subcategory",              "specific detail",           "exact value"],
}

# ── Calibrated abstraction level per semantic type ─────────────────────────────
# Runtime policy table representing the output of offline calibration.
# Key: semantic abstraction type.
# Value: integer level k ∈ {0, 1, 2, 3} from ABSTRACTION_HIERARCHIES[type][k].
#
# Calibration rationale (least specific level that preserves task utility):
#
#   address/location → 1  city/area is sufficient for provider discovery
#   zip              → 0  zip + health + schedule combination is linkable
#   date/time        → 3  exact values required for calendar-grounded booking
#   money/party_size → 3  exact values required as hard task constraints
#   insurance_name   → 1  coverage category sufficient; plan name reveals account
#   medical_symptom  → 2  symptom category reduces specificity while preserving
#                         care-type signal needed for provider matching
#   prior_provider   → 0  category only; named providers are high-linkability
#                         history unless task requires continuity or follow-up
#   service_need     → 1  capability sufficient; specific service over-specified
#   dietary/travel   → 2  constraint category sufficient
#   preference       → 2  preference type sufficient; exact form is over-specified
#   person           → 0  named persons outside U_loc still linkable
#   group/generic    → 1

CALIBRATED_ABSTRACTION_POLICY: Dict[str, int] = {
    "address":            1,
    "location":           1,
    "zip":                0,

    "date":               3,
    "time":               3,

    "money":              3,
    "party_size":         3,

    "insurance_name":     1,

    "medical_symptom":    2,
    "prior_provider":     0,
    "service_need":       1,
    "dietary_constraint": 2,
    "travel_purpose":     2,
    "preference":         2,

    "person":             0,
    "group":              1,

    "generic_detail":     1,
}

# Default for types not listed above
DEFAULT_ABSTRACTION_LEVEL = 1
