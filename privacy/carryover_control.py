"""
privacy/carryover_control.py

Stage 2 (v2) of PrivScope — Carryover Control.

Replaces the two-step scope_control (per-span Rel + TaskGain) and
span_classification (PTH/CSS joint call) with a single local-LLM call
that jointly decides which extracted spans are genuinely required for
cloud-side task completion.

Design principle (from paper):
    A unit should reach the CLM only when some representation of it is
    useful for the current information-seeking delegation, not merely
    because it is present in the payload p_t.

The LLM is given:
  - g_t : task frame summarizing what the cloud needs to accomplish
  - All U_med spans extracted by Stage 1 as a numbered list

In one call it decides for each span: KEEP or DROP.
  KEEP  — some representation of the span (even coarse/abstracted) is
           genuinely needed; dropping it would prevent task success.
  DROP  — the cloud completes the task equally well without this span.

Kept spans proceed directly to Stage 3b (SpanAbstractor).
U_loc spans are always withheld by Stage 1 and never reach this stage.

Rule-based fallback (no LLM): type-based keep/drop table.

Public API:
    CarryoverController()
    CarryoverController.filter(u_med, task, extraction, local_llm)
        -> CarryoverResult
    reconstruct_kept_payload(extraction, kept, u_loc) -> str
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from privacy.span_extractor import Span


# ── Task frame ─────────────────────────────────────────────────────────────────
# Derived from r_t (the original user request), not from p_t.
# Same derivation logic as scope_control; duplicated here for self-containment.

_TASK_FRAME_PROMPT = """\
You are describing what a cloud search agent needs to know to complete a task.

Task: {task}

Write a concise constraint-schema description of the information required to \
complete this task. Capture: task type, provider or venue requirements, \
scheduling preferences, personal constraints, and any special requirements.

Format exactly like this example:
dentist appointment search; constraints: date availability, time preference, \
budget, location, insurance; requirements: accepting new patients; symptoms: \
tooth pain, bleeding gums

Reply with only the schema. No explanation. One line."""

_FRAME_CONSTRAINTS = [
    ({"date"},                         "date availability"),
    ({"time"},                         "time preference"),
    ({"address", "location", "zip"},   "location"),
    ({"insurance_name"},               "insurance coverage"),
    ({"money"},                        "budget limit"),
    ({"party_size"},                   "party size"),
]
_SYMPTOM_KWS = {
    "pain", "ache", "gum", "tooth", "teeth", "bleed", "symptom",
    "concern", "injury", "fever", "sore", "swollen",
}


def _derive_task_frame(task: str, u_med: List[Span] = None, local_llm=None) -> str:
    if local_llm is not None:
        try:
            prompt = _TASK_FRAME_PROMPT.format(task=task.strip())
            frame  = local_llm.generate(prompt).strip()
            frame  = next((l.strip() for l in frame.splitlines() if l.strip()), frame)
            if frame:
                return frame
        except Exception:
            pass

    constraints: List[str] = []
    symptoms:    List[str] = []
    if u_med:
        present = {s.span_type for s in u_med}
        for type_set, label in _FRAME_CONSTRAINTS:
            if present & type_set:
                constraints.append(label)
        symptoms = [
            s.text for s in u_med
            if s.span_type == "noun_phrase"
            and any(kw in s.text.lower() for kw in _SYMPTOM_KWS)
        ]
    frame = task.strip()
    if constraints:
        frame += "; constraints: " + ", ".join(constraints)
    if symptoms:
        frame += "; symptoms: " + ", ".join(symptoms)
    return frame


# ── LLM prompt ─────────────────────────────────────────────────────────────────

_CARRYOVER_PROMPT = """\
You are a privacy filter for a delegated cloud search task.

Task frame (what the cloud needs to accomplish):
{g_t}

Below are information spans extracted from the payload that is about to be sent \
to the cloud. For each span decide: KEEP or DROP.

KEEP — some representation of this span (even coarse or abstracted) is genuinely \
required for the cloud to complete the task. Without it, the task cannot succeed.

DROP — the cloud can complete the task equally well without this span. It is \
irrelevant, redundant, or carries personal history that does not improve the result.

Extracted spans:
{span_list}

Reply with ONLY one decision per line in this exact format:
1. KEEP
2. DROP
3. KEEP
...

One line per span, numbered to match the list above. No explanations. No extra text."""


# ── Rule-based fallback ────────────────────────────────────────────────────────

_RULE_KEEP_TYPES = {
    "date", "time", "address", "location", "zip",
    "insurance_name", "money", "party_size", "preference",
    "noun_phrase",
}
_RULE_DROP_TYPES = {
    "organization", "facility", "person", "group",
}


def _rule_decision(span: Span) -> bool:
    if span.span_type in _RULE_KEEP_TYPES:
        return True
    if span.span_type in _RULE_DROP_TYPES:
        return False
    return True


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class CarryoverDecision:
    span:   Span
    kept:   bool
    reason: str   # "llm:keep" | "llm:drop" | "rule:keep" | "rule:drop" | "rule:fallback:keep/drop"


@dataclass
class CarryoverResult:
    kept:       List[Span]
    dropped:    List[Span]
    decisions:  List[CarryoverDecision]
    task_frame: str
    method:     str   # "llm:<model>" | "rule"


# ── CarryoverController ────────────────────────────────────────────────────────

class CarryoverController:
    """
    Stage 2 (v2) filter: single LLM call over all U_med spans.

    U_loc spans are always withheld (binding table, Stage 1). This filter
    operates only on U_med. All kept spans proceed to Stage 3b abstraction.
    """

    def filter(
        self,
        u_med:        List[Span],
        task:         str,
        extraction:   dict,
        local_llm               = None,
        g_t_override: str       = None,
    ) -> CarryoverResult:
        g_t    = g_t_override if g_t_override is not None else _derive_task_frame(task, u_med, local_llm)
        method = f"llm:{local_llm.model}" if local_llm is not None else "rule"

        if not u_med:
            return CarryoverResult(kept=[], dropped=[], decisions=[], task_frame=g_t, method=method)

        if local_llm is not None:
            decisions = self._llm_filter(u_med, g_t, local_llm)
        else:
            decisions = self._rule_filter(u_med, source="rule")

        kept    = [d.span for d in decisions if d.kept]
        dropped = [d.span for d in decisions if not d.kept]
        return CarryoverResult(kept=kept, dropped=dropped, decisions=decisions, task_frame=g_t, method=method)

    def _llm_filter(self, u_med: List[Span], g_t: str, local_llm) -> List[CarryoverDecision]:
        span_list = "\n".join(
            f"{i + 1}. [{s.span_type}] \"{s.text}\""
            for i, s in enumerate(u_med)
        )
        prompt = _CARRYOVER_PROMPT.format(g_t=g_t.strip(), span_list=span_list)
        try:
            raw    = local_llm.generate(prompt).strip()
            parsed = _parse_llm_decisions(raw, len(u_med))
            if parsed is not None:
                return [
                    CarryoverDecision(
                        span=span,
                        kept=decision,
                        reason=f"llm:{'keep' if decision else 'drop'}",
                    )
                    for span, decision in zip(u_med, parsed)
                ]
        except Exception:
            pass
        return self._rule_filter(u_med, source="rule:fallback")

    def _rule_filter(self, u_med: List[Span], source: str = "rule") -> List[CarryoverDecision]:
        return [
            CarryoverDecision(
                span=s,
                kept=_rule_decision(s),
                reason=f"{source}:{'keep' if _rule_decision(s) else 'drop'}",
            )
            for s in u_med
        ]


# ── LLM response parser ────────────────────────────────────────────────────────

_LINE_RE = re.compile(r'^\s*\d+[.)]\s*(KEEP|DROP)\b', re.IGNORECASE)


def _parse_llm_decisions(raw: str, expected: int) -> Optional[List[bool]]:
    """
    Parse numbered KEEP/DROP lines.
    Returns list of booleans (True=KEEP) of length `expected`, or None on failure.
    """
    decisions = []
    for line in raw.splitlines():
        m = _LINE_RE.match(line)
        if m:
            decisions.append(m.group(1).upper() == "KEEP")
    return decisions if len(decisions) == expected else None


# ── Payload reconstruction ─────────────────────────────────────────────────────

_RECONSTRUCT_SW = {
    'i', 'me', 'my', 'hi', 'is', 'am', 'are', 'was', 'were', 'be',
    'the', 'a', 'an', 'if', 'it', 'its', 'and', 'or', 'but', 'so',
    'for', 'in', 'on', 'at', 'to', 'of', 'by', 'not', 'do', 'did',
    'have', 'has', 'had', 'can', 'will', 'would', 'may', 'might',
    'possible', 'preferably', 'previously', 'visited', 'book', 'keep',
    'want', 'need', 'get', 'also', 'just', 'that', 'this', 'with',
}


def reconstruct_kept_payload(extraction: dict, kept: List[Span], u_loc: List[Span]) -> str:
    """Reconstruct payload with only kept spans; U_loc as [TYPE] placeholders."""
    c_t       = extraction["C_t"]
    all_spans = extraction["all_spans"]

    kept_texts = {s.text.lower() for s in kept}
    u_loc_map  = {s.text.lower(): f"[{s.span_type.upper()}]" for s in (u_loc or [])}

    result = c_t
    for span in all_spans:
        tl = span.text.lower()
        if tl in u_loc_map:
            replacement = u_loc_map[tl]
        elif tl in kept_texts:
            replacement = span.text
        else:
            replacement = ""
        result = result.replace("[SPAN]", replacement, 1)

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

    def _has_content(sent: str) -> bool:
        if '[' in sent:
            return True
        words = re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower())
        return sum(1 for w in words if w not in _RECONSTRUCT_SW) >= 2

    sentences = re.split(r'(?<=[.!?])\s+', result.strip())
    kept_sents = [s.strip() for s in sentences if s.strip() and _has_content(s)]
    return " ".join(kept_sents).strip()


# ── Debug display ──────────────────────────────────────────────────────────────

def _debug_show(result: CarryoverResult) -> None:
    print(f"\n  Task frame g_t:")
    print(f"    {result.task_frame}")
    print(f"\n  Method: {result.method}")
    print(f"\n  Kept ({len(result.kept)}):")
    for d in result.decisions:
        if d.kept:
            print(f"    [KEEP]  [{d.span.span_type:<20}] {d.span.text!r:40}  ({d.reason})")
    print(f"\n  Dropped ({len(result.dropped)}):")
    for d in result.decisions:
        if not d.kept:
            print(f"    [DROP]  [{d.span.span_type:<20}] {d.span.text!r:40}  ({d.reason})")


# ── Standalone ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from state.state_io import load_state
    from privacy.span_extractor import SpanExtractor, _debug_show as _show_spans
    from llm.local_llm import LocalLLM

    state   = load_state()
    profile = state.get("user_profile", {})

    try:
        local_llm = LocalLLM()
        local_llm.generate("ping")
        print(f"\n  Local LLM: connected ({local_llm.model})")
    except Exception:
        local_llm = None
        print(f"\n  Local LLM: unavailable — rule-based fallback")

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

    # ── Stage 1: Span Extraction ───────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  STAGE 1 — SPAN EXTRACTION  (operating on p_t)")
    print(f"{'═' * W}")
    extractor  = SpanExtractor()
    extraction = extractor.extract(p_t, profile)
    _show_spans(extraction)

    # ── Stage 2 (v2): Carryover Control ───────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  STAGE 2 (v2) — CARRYOVER CONTROL  (g_t derived from r_t)")
    print(f"{'═' * W}")
    controller = CarryoverController()
    result     = controller.filter(
        u_med=extraction["U_med"],
        task=r_t,
        extraction=extraction,
        local_llm=local_llm,
    )
    _debug_show(result)

    print(f"\n{'═' * W}")
    print(f"  CARRYOVER CONTROL OUTPUT  — payload entering Stage 3b (kept spans verbatim; U_loc withheld)")
    print(f"{'═' * W}")
    out = reconstruct_kept_payload(extraction, result.kept, extraction["U_loc"])
    print(f"  {out}")
