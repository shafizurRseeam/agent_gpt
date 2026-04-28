"""
privacy/span_classification.py

Stage 3a of PrivScope — Sensitivity Classification.

Classifies each retained span from Stage 2 (scope control) as either:

  PTH (passthrough)       — released verbatim; the exact value is required
                            for the delegated cloud-side reasoning step.
  CSS (context-sensitive) — forwarded to Stage 3b (abstraction); the verbatim
                            form is more specific than the task requires, or
                            becomes linkable when co-released with other
                            retained spans.

Classification is performed jointly over all retained spans in a single
local-LM call, rather than span-by-span. This captures the co-release
effect: a span that appears innocuous in isolation (e.g. a symptom) may
become CSS when evaluated alongside a specific date, location, and
insurance plan that together narrow the payload toward a specific personal
episode.

Formally, given the retained set Ū_t^med and the task frame g_t:

    (β_1, …, β_M) = f_φ(Ū_t^med, g_t),   β_j ∈ {PTH, CSS}

Fallback (no LLM): conservative type-based rule assigns PTH to functional
values whose exact form is required by any task (date, time, money,
party_size, preference) and CSS to all other types.

Public API:
    SpanClassifier()
    SpanClassifier.classify(retained_spans, g_t, local_llm) → ClassificationResult
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from privacy.span_extractor import Span


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class ClassificationDecision:
    span:   Span
    label:  str   # "PTH" | "CSS"
    method: str   # "llm" | "rule"


@dataclass
class ClassificationResult:
    passthrough:       List[Span]               # β = PTH — released verbatim
    context_sensitive: List[Span]               # β = CSS — forwarded to abstraction
    decisions:         List[ClassificationDecision]
    task_frame:        str
    method:            str                      # "llm:<model>" | "rule"


# ── Span type risk hints ───────────────────────────────────────────────────────
# Passed into the classification prompt as structured context per span.
# These are risk priors and abstraction options, not final labels.
# The LLM still decides PTH vs CSS relative to task frame and co-release context.

_RISK_HINTS = {
    "address":        ("high-resolution location; usually abstractable to city, ZIP, "
                       "neighborhood, or distance constraint for provider discovery"),
    "zip":            ("coarse location; may be passthrough for provider discovery, but "
                       "can be context-sensitive when combined with health or schedule details"),
    "insurance_name": ("account-linked service context; often abstractable to insurance "
                       "category or accepted-insurance requirement"),
    "noun_phrase":    ("semantic content; judge based on task need and co-release context"),
    "organization":   ("named entity; prior provider history is usually context-sensitive "
                       "unless the task explicitly requires continuity or comparison"),
    "facility":       ("named facility; treat like organization"),
    "date":           ("scheduling constraint; may be passthrough if exact availability is "
                       "needed, or abstractable to a broader time window"),
    "time":           ("scheduling constraint; may be passthrough if exact time matters, "
                       "or abstractable to morning/afternoon/window"),
    "money":          ("budget limit; usually passthrough as an exact constraint"),
    "party_size":     ("group size; usually passthrough"),
    "preference":     ("personal preference; usually passthrough as a soft constraint"),
    "location":       ("location reference; assess resolution and co-release context"),
    "person":         ("person name; high linkability when combined with health or location"),
}


# ── Classification prompt ──────────────────────────────────────────────────────
# Joint prompt: all retained spans are shown together so the LM can reason
# about co-release risk. A single round-trip returns one label per span.

_CLASSIFICATION_PROMPT = """\
You are classifying retained spans before sending a payload to a cloud search agent.

Task frame:
{g_t}

Retained spans that may be jointly released:
{span_list}

Assign each span one label:

PTH — passthrough. Use PTH only when the cloud needs the exact verbatim value \
for the delegated reasoning step, and replacing it with a less specific form \
would materially reduce task utility.

CSS — context-sensitive. Use CSS when a less specific representation would \
preserve the information needed by the cloud, or when the exact value becomes \
more linkable when combined with the other retained spans.

Decision rule:
For each span, decide whether the exact surface form is necessary, not merely \
useful. Use the span type and hint as risk priors, but make the final decision \
from the task frame and the full retained co-release context.

Reply with exactly one label per line, numbered to match:
1. CSS
2. PTH
..."""

# ── Rule-based fallback ────────────────────────────────────────────────────────
# Used when no LLM is available. Cannot capture co-release context, so errs
# toward CSS for privacy. PTH only for types whose exact form is required by
# any task (scheduling constraints, budget limits, preferences).

_PTH_TYPES = {"date", "time", "money", "party_size", "preference"}

def _rule_label(span: Span) -> str:
    return "PTH" if span.span_type in _PTH_TYPES else "CSS"


# ── SpanClassifier ─────────────────────────────────────────────────────────────

class SpanClassifier:
    """
    Stage 3a classifier.

    classify() takes the retained span set from Stage 2 and the task frame
    g_t, and returns PTH/CSS labels for each span.

    With a local LLM, the classification is joint: all spans are presented
    together in one prompt so the model can reason about co-release linkability.

    Without a local LLM, falls back to conservative type-based rules.
    """

    def classify(
        self,
        retained_spans: List[Span],
        g_t:            str,
        local_llm               = None,
    ) -> ClassificationResult:
        """
        Classify each retained span as PTH or CSS.

        Parameters
        ----------
        retained_spans : output of ScopeController.filter().retained
        g_t            : task frame string (derived from r_t)
        local_llm      : LocalLLM instance; falls back to rule table if None
        """
        if not retained_spans:
            return ClassificationResult(
                passthrough=[], context_sensitive=[], decisions=[],
                task_frame=g_t,
                method="llm:" + local_llm.model if local_llm else "rule",
            )

        decisions = self._llm_classify(retained_spans, g_t, local_llm)

        passthrough       = [d.span for d in decisions if d.label == "PTH"]
        context_sensitive = [d.span for d in decisions if d.label == "CSS"]
        method = decisions[0].method if decisions else "rule"
        if local_llm and method == "llm":
            method = f"llm:{local_llm.model}"

        return ClassificationResult(
            passthrough=passthrough,
            context_sensitive=context_sensitive,
            decisions=decisions,
            task_frame=g_t,
            method=method,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _llm_classify(
        self,
        spans:     List[Span],
        g_t:       str,
        local_llm,
    ) -> List[ClassificationDecision]:
        if local_llm is None:
            return [ClassificationDecision(s, _rule_label(s), "rule") for s in spans]

        span_list = "\n".join(
            f'{i+1}. "{s.text}" '
            f'(type: {s.span_type}; hint: {_RISK_HINTS.get(s.span_type, "no special prior")})'
            for i, s in enumerate(spans)
        )
        prompt = _CLASSIFICATION_PROMPT.format(
            g_t=g_t.strip(),
            span_list=span_list,
        )

        try:
            raw    = local_llm.generate(prompt)
            labels = _parse_labels(raw, len(spans))
            if labels is not None:
                return [
                    ClassificationDecision(s, lbl, "llm")
                    for s, lbl in zip(spans, labels)
                ]
            print(f"\n[Stage 3a] Could not parse LLM classification output ({len(spans)} spans expected):")
            print(raw)
        except Exception as e:
            print(f"\n[Stage 3a] LLM classification call failed: {e}")

        # Fallback: rule-based for all spans
        return [ClassificationDecision(s, _rule_label(s), "rule") for s in spans]


# ── Response parser ────────────────────────────────────────────────────────────

def _parse_labels(raw: str, expected: int) -> List[str] | None:
    """
    Extract ordered PTH/CSS labels from the LLM response.
    Handles varied formats:
      1. CSS
      1) PTH
      1: "span text" - CSS
      1. address: PTH
    Returns None if the count does not match expected (triggers debug + fallback).
    """
    labels = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^\s*(\d+)[\.\):\-]?\s*.*?\b(PTH|CSS)\b', line, re.IGNORECASE)
        if m:
            labels.append((int(m.group(1)), m.group(2).upper()))

    if labels:
        ordered = [lbl for _, lbl in sorted(labels)]
        if len(ordered) == expected:
            return ordered

    # Also accept bare one-label-per-line output (no numbering)
    bare = re.findall(r'^\s*(PTH|CSS)\s*$', raw, flags=re.IGNORECASE | re.MULTILINE)
    if len(bare) == expected:
        return [x.upper() for x in bare]

    return None


# ── Debug display ──────────────────────────────────────────────────────────────

def _debug_show(result: ClassificationResult) -> None:
    print(f"\n  Task frame g_t:")
    print(f"    {result.task_frame}")
    print(f"\n  Classification method : {result.method}")
    print(f"  PTH (passthrough)     : {len(result.passthrough)}")
    print(f"  CSS (context-sensitive): {len(result.context_sensitive)}")

    print(f"\n  Decisions:")
    for d in result.decisions:
        tag = "[PTH]" if d.label == "PTH" else "[CSS]"
        print(
            f"    {tag}  [{d.span.span_type:<20}] "
            f"{d.span.text!r:<40}  ({d.method})"
        )


# ── Standalone — runs Stage 1 → Stage 2 → Stage 3a ────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from state.state_io import load_state
    from privacy.span_extractor import SpanExtractor, _debug_show as _show_spans
    from privacy.scope_control import (
        ScopeController, reconstruct_payload,
        _debug_show as _show_scope,
    )
    from llm.local_llm import LocalLLM

    state   = load_state()
    profile = state.get("user_profile", {})

    try:
        local_llm = LocalLLM()
        local_llm.generate("ping")
        print(f"\n  Local LLM: connected ({local_llm.model})")
    except Exception:
        local_llm = None
        print(f"\n  Local LLM: unavailable — rule-table fallback for g_t, TaskGain, and classification")

    # ── r_t and p_t ───────────────────────────────────────────────────────────
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

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  ORIGINAL USER REQUEST  (r_t)")
    print(f"{'═' * W}")
    print(f"  {r_t}")

    print(f"\n{'═' * W}")
    print(f"  LC-ENRICHED PAYLOAD  (p_t)  — input to sanitization pipeline")
    print(f"{'═' * W}")
    print(f"  {p_t}")

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
    controller  = ScopeController()
    scope_result = controller.filter(
        u_med=extraction["U_med"],
        task=r_t,
        payload=extraction["working"],
        local_llm=local_llm,
    )
    _show_scope(scope_result)

    # ── Stage 2 output → Stage 3a input ──────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  STAGE 2 OUTPUT  — payload entering Stage 3a (retained spans verbatim; U_loc withheld)")
    print(f"{'═' * W}")
    stage2_out = reconstruct_payload(extraction, scope_result.retained, extraction["U_loc"])
    print(f"  {stage2_out}")

    # ── Stage 3a ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  STAGE 3a — SENSITIVITY CLASSIFICATION  (joint over retained spans)")
    print(f"{'═' * W}")
    classifier  = SpanClassifier()
    class_result = classifier.classify(
        retained_spans=scope_result.retained,
        g_t=scope_result.task_frame,
        local_llm=local_llm,
    )
    _debug_show(class_result)

    # ── Stage 3a output → Stage 3b input ─────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  STAGE 3a OUTPUT  — PTH verbatim; [css:...] pending Stage 3b abstraction")
    print(f"{'═' * W}")
    stage3a_out = reconstruct_payload(extraction, scope_result.retained, extraction["U_loc"])
    for span in class_result.context_sensitive:
        stage3a_out = stage3a_out.replace(span.text, f"[css:{span.text}]", 1)
    print(f"  {stage3a_out}")
