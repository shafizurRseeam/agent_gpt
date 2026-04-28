"""
privacy/scope_control.py

Stage 2 of PrivScope — Scope Control.

Filters the U_med spans produced by Stage 1, retaining only those
justified by the current delegated task.

Each span u_j is evaluated against a task frame g_t using provenance-
specific signals:

  Rel(u_j, g_t) ∈ [-1, 1]   — embedding cosine similarity to the task frame;
                               used only for spans expressed in r_t
  κ(u_j) ∈ {0, 1}            — provenance status:
                               0 = span text found in r_t (user-provided)
                               1 = LC-injected from working state
  TaskGain(u_j, g_t) ∈ [0,1] — local-LM estimate of task utility;
                               used only for LC-injected spans

Retention rules:

  User-provided span (κ=0):
    keep if  Rel(u_j, g_t) >= ρ

  LC-injected span (κ=1):
    keep if  TaskGain(u_j, g_t) >= γ

LC-injected spans bypass the relevance gate because short structured
values such as addresses, dates, times, ZIP codes, and insurance names
may be task-critical despite weak embedding similarity to the task frame.

Payload reconstruction:
  reconstruct_payload() fills C_t slots with retained span text (or
  [TYPE] for U_loc), drops empty slots, and cleans up orphaned fragments.

Public API:
    ScopeController(rho_low, gamma)
    ScopeController.filter(u_med, task, payload, local_llm) → ScopeResult
    reconstruct_payload(extraction, retained_u_med, u_loc) → str
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util

from privacy.span_extractor import Span


# ── Lazy singleton ─────────────────────────────────────────────────────────────

_st_model = None

def _st() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class ScopeDecision:
    span:         Span
    rel:          Optional[float]  # set for κ=0 spans only; None for κ=1 (not used in decision)
    kappa:        int              # 0 = in r_t, 1 = LC-injected
    kappa_reason: str              # "" (in r_t) | "lc_injected"
    task_gain:    float            # TaskGain score; 0.0 if not computed (κ=0)
    kept:         bool
    reason:       str


@dataclass
class ScopeResult:
    retained:        List[Span]
    dropped:         List[Span]
    decisions:       List[ScopeDecision]
    task_frame:      str
    rho_low:         float
    gamma:           float
    task_gain_source: str   # "llm:<model>" | "rule" — what scored carryover TaskGain


# ── Task frame ─────────────────────────────────────────────────────────────────
# Primary: local LLM generates a constraint-schema summary of the delegated task,
# capturing intent, provider type, preferences, and requirements that a fixed
# label set would miss (urgency, dietary needs, accessibility, gender preference…).
# Fallback (no LLM): span-type-derived constraint labels + current symptom phrases.

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

_FRAME_CONSTRAINTS: List[Tuple[Set[str], str]] = [
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
            # Take only the first non-empty line in case the model adds extras
            frame  = next((l.strip() for l in frame.splitlines() if l.strip()), frame)
            if frame:
                return frame
        except Exception:
            pass

    # Fallback: span-type-derived constraint labels
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


# ── Type-conditioned span representation for Rel computation ───────────────────
# Naked span strings ("10:30 AM", "under $500") are poor standalone semantic
# objects for sentence embeddings. Rewriting them into role-aware phrases
# improves cosine alignment with the task frame without changing the policy.

_SPAN_REPR_TEMPLATE: Dict[str, str] = {
    "date":           "available date: {text}",
    "time":           "preferred appointment time: {text}",
    "address":        "user location: {text}",
    "zip":            "zip code: {text}",
    "location":       "location: {text}",
    "insurance_name": "health insurance: {text}",
    "money":          "budget limit: {text}",
    "party_size":     "number of people: {text}",
    "phone":          "contact phone number: {text}",
    "email":          "contact email address: {text}",
    "person":         "person: {text}",
    "organization":   "service provider organization: {text}",
    "facility":       "facility name: {text}",
    "noun_phrase":    "relevant detail: {text}",
    "preference":     "user preference: {text}",
    "group":          "group or affiliation: {text}",
}

def _span_repr(span: Span) -> str:
    """Return a role-aware semantic string for embedding, not the raw span text."""
    template = _SPAN_REPR_TEMPLATE.get(span.span_type, "information: {text}")
    return template.format(text=span.text)


# ── TaskGain: LLM-based (primary) + rule-based fallback ───────────────────────

_RULE_GAIN: Dict[str, float] = {
    "date":           1.00,
    "time":           1.00,
    "address":        0.85,
    "location":       0.85,
    "insurance_name": 0.80,
    "money":          0.75,
    "noun_phrase":    0.60,
    "zip":            0.50,
    "party_size":     0.30,
    "preference":     0.60,
    "organization":   0.05,
    "facility":       0.05,
    "group":          0.20,
    "person":         0.05,
}

_LLM_GAIN_PROMPT = """\
You are estimating whether an LC-injected span should be included in a cloud-bound payload.

Task frame: {g_t}
Sentence containing the span: {context}
Span: "{span_text}" (type: {span_type})

Return a TaskGain score as a decimal between 0.00 and 1.00, where higher means \
retaining the span would more strongly improve the delegated cloud-side reasoning \
step under the task frame. Use the full range and avoid defaulting to round bucket \
values. Do not assign a high score merely because the span is related to the task. \
If the span is a prior provider, prior venue, prior booking, organization, or \
historical output, assign a low score unless the task frame explicitly requires \
continuity, reuse, follow-up, avoidance, or comparison. If uncertain, assign a \
lower score.

Reply with only one decimal number, such as 0.17, 0.43, or 0.82."""

_FLOAT_RE = re.compile(r'\b(0\.\d+|1\.0+|0|1)\b')


def _containing_sentence(payload: str, span_text: str) -> str:
    """Return the first sentence in payload that contains span_text (case-insensitive)."""
    tl = span_text.lower()
    for sent in _split_sentences(payload):
        if tl in sent.lower():
            return sent.strip()
    return span_text


def _llm_task_gain(local_llm, g_t: str, span: Span, payload: str = "") -> Tuple[float, str]:
    """
    Score TaskGain for an LC-injected span using g_t and the containing sentence.
    Returns a 0.0-1.0 float parsed from the LLM response.
    Falls back to rule table on parse error or exception.
    """
    context = _containing_sentence(payload, span.text) if payload else span.text
    prompt = _LLM_GAIN_PROMPT.format(
        g_t=g_t.strip(),
        context=context,
        span_text=span.text,
        span_type=span.span_type,
    )
    try:
        raw = local_llm.generate(prompt).strip()
        m = _FLOAT_RE.search(raw)
        if m:
            return float(m.group(0)), "llm"
    except Exception:
        pass
    return _RULE_GAIN.get(span.span_type, 0.30), "rule"


def _rule_based_task_gain(span: Span) -> Tuple[float, str]:
    return _RULE_GAIN.get(span.span_type, 0.30), "rule"


# ── Carryover detection ────────────────────────────────────────────────────────
# Provenance-based: κ=0 if the span text appears in the original user request
# r_t; κ=1 ("lc_injected") if it was added by the LC from working state.
# This captures all LC-injected context uniformly — profile data, calendar
# preferences, and prior task history all get the stricter carryover gate.

def _carryover_status(span: Span, r_t: str) -> Tuple[int, str]:
    """
    Returns (κ, reason).
      κ=0, ""            — span text found in r_t (user stated it explicitly)
      κ=1, "lc_injected" — span was injected by LC from working state
    """
    if span.text.lower() in r_t.lower():
        return 0, ""
    return 1, "lc_injected"


# ── Payload reconstruction ────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', text.strip())


# ──────────────────────────────────────────────────────────────────────────────

def reconstruct_payload(
    extraction:     dict,
    retained_u_med: List[Span],
    u_loc:          List[Span] = None,
) -> str:
    """
    Rebuild sanitized payload from C_t skeleton + retained spans.

    [SPAN] slots filled left-to-right matching all_spans order:
      U_loc  → [TYPE] placeholder
      Retained U_med → original span text
      Dropped U_med  → empty string

    Artifact cleanup removes orphaned connectives and content-free sentences.
    """
    c_t       = extraction["C_t"]
    all_spans = extraction["all_spans"]

    retained_texts = {s.text.lower() for s in retained_u_med}
    u_loc_map      = {
        s.text.lower(): f"[{s.span_type.upper()}]"
        for s in (u_loc or [])
    }

    result = c_t
    for span in all_spans:
        tl = span.text.lower()
        if tl in u_loc_map:
            replacement = u_loc_map[tl]
        elif tl in retained_texts:
            replacement = span.text
        else:
            replacement = ""
        result = result.replace("[SPAN]", replacement, 1)

    # ── Multi-pass cleanup of empty-slot artefacts ────────────────────────────
    for _ in range(3):
        # Fix "My X is [empty] and Y" → "Y" (must precede the space+and cleanup)
        result = re.sub(r'\bMy\s+\w+\s+is\s+and\s+', '', result, flags=re.IGNORECASE)
        # Fix "  and X" (two spaces before "and" = empty preceding slot)
        result = re.sub(r'\s{2,}and\s+', ' ', result)
        # Fix "visited and X" / "seen at and X" type (verb lost its object)
        result = re.sub(
            r'\b(visited|seen at|been to|tried|used)\s+and\s+',
            r'\1 ', result, flags=re.IGNORECASE,
        )
        # Fix "VERB is/are and ..." (noun lost its value)
        result = re.sub(r'\b(\w+)\s+is\s+and\b', r'\1 and', result, flags=re.IGNORECASE)
        result = re.sub(r'\b(\w+)\s+is\s*([,.])', r'\2', result, flags=re.IGNORECASE)
        # Orphaned prepositions before punctuation
        result = re.sub(
            r'\b(on|at|before|after|for|near|is|are|and|or|of|in|by)\s*([,.])',
            r'\2', result, flags=re.IGNORECASE,
        )
        result = re.sub(r',\s*\.', '.', result)
        result = re.sub(r',\s*,', ',', result)
        result = re.sub(r'\.\s*\.', '.', result)
        result = re.sub(r'\s+([,.])', r'\1', result)
        result = re.sub(r' {2,}', ' ', result)
        # Fix "I want to keep the cost" when the value was dropped
        result = re.sub(
            r'\bI want to keep the cost\s*[,.]?\s*',
            '', result, flags=re.IGNORECASE,
        )

    # ── Drop content-free sentences ───────────────────────────────────────────
    _SW = {
        'i', 'me', 'my', 'hi', 'is', 'am', 'are', 'was', 'were', 'be',
        'the', 'a', 'an', 'if', 'it', 'its', 'and', 'or', 'but', 'so',
        'for', 'in', 'on', 'at', 'to', 'of', 'by', 'not', 'do', 'did',
        'have', 'has', 'had', 'can', 'will', 'would', 'may', 'might',
        'possible', 'preferably', 'previously', 'visited', 'book', 'keep',
        'want', 'need', 'get', 'also', 'just', 'that', 'this', 'with',
    }

    def _has_content(sent: str) -> bool:
        if '[' in sent:                          # has a [PLACEHOLDER]
            return True
        words = re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower())
        return sum(1 for w in words if w not in _SW) >= 2

    sentences  = _split_sentences(result)
    kept_sents = [s.strip() for s in sentences if s.strip() and _has_content(s)]
    return " ".join(kept_sents).strip()


# ── ScopeController ────────────────────────────────────────────────────────────

class ScopeController:
    """
    Stage 2 filter.

    Two-path retention rule:
      κ=0 (span in r_t)       : keep if  Rel(u_j, g_t) >= rho_low
      κ=1 (LC-injected span)  : keep if  TaskGain(u_j, g_t) >= gamma

    LC-injected spans bypass the relevance gate entirely — the LLM judges
    directly whether the span is useful for the delegated task. This handles
    structurally necessary spans (e.g. address for provider search) that
    score low on embedding similarity but high on task utility.

    Parameters
    ----------
    rho_low : Rel threshold for r_t spans (default 0.10)
    gamma   : TaskGain threshold for LC-injected spans (default 0.50)
    """

    def __init__(
        self,
        rho_low: float = 0.10,
        gamma:   float = 0.50,
    ):
        self.rho_low = rho_low
        self.gamma   = gamma

    def filter(
        self,
        u_med:        List[Span],
        task:         str,
        payload:      str  = "",
        local_llm           = None,
        g_t_override: str  = None,
    ) -> ScopeResult:
        """
        Apply scope control to the U_med span set.

        κ=0 spans (in r_t): filtered by Rel >= rho_low.
        κ=1 spans (LC-injected): filtered by TaskGain >= gamma — no Rel gate.
          This allows structurally necessary spans (e.g. address for provider
          search) to pass even when embedding similarity to g_t is low.

        local_llm is used for g_t construction and TaskGain scoring.
        Falls back to rule table if None.
        """
        g_t         = g_t_override if g_t_override is not None else _derive_task_frame(task, u_med, local_llm)
        gain_source = f"llm:{local_llm.model}" if local_llm is not None else "rule"

        decisions: List[ScopeDecision] = []

        if u_med:
            # Determine provenance first so we only encode κ=0 spans
            kappas = [_carryover_status(span, task) for span in u_med]
            k0_spans = [s for s, (k, _) in zip(u_med, kappas) if k == 0]

            # Embed g_t and κ=0 spans only
            if k0_spans:
                st      = _st()
                g_t_emb = st.encode(g_t, convert_to_tensor=True)
                k0_embs = st.encode([_span_repr(s) for s in k0_spans], convert_to_tensor=True)

            k0_idx = 0
            for span, (kappa, k_reason) in zip(u_med, kappas):
                if kappa == 0:
                    # ── In r_t (κ=0): keep if Rel >= ρ_low ───────────────
                    rel       = float(st_util.cos_sim(k0_embs[k0_idx], g_t_emb).item())
                    k0_idx   += 1
                    task_gain = 0.0
                    kept      = rel >= self.rho_low
                    reason    = (
                        f"Rel={rel:.2f} >= ρ_low={self.rho_low:.2f}" if kept
                        else f"Rel={rel:.2f} < ρ_low={self.rho_low:.2f}"
                    )
                else:
                    # ── LC-injected (κ=1): TaskGain >= γ, Rel not computed ─
                    rel = None
                    if local_llm is not None:
                        task_gain, gain_src = _llm_task_gain(local_llm, g_t, span, payload)
                    else:
                        task_gain, gain_src = _rule_based_task_gain(span)
                    kept   = task_gain >= self.gamma
                    reason = (
                        f"lc_injected, TaskGain={task_gain:.2f} ({gain_src}) >= γ={self.gamma:.2f}" if kept
                        else
                        f"lc_injected, TaskGain={task_gain:.2f} ({gain_src}) < γ={self.gamma:.2f}"
                    )

                decisions.append(ScopeDecision(
                    span=span, rel=rel,
                    kappa=kappa, kappa_reason=k_reason,
                    task_gain=task_gain,
                    kept=kept, reason=reason,
                ))

        retained = [d.span for d in decisions if d.kept]
        dropped  = [d.span for d in decisions if not d.kept]

        return ScopeResult(
            retained=retained, dropped=dropped,
            decisions=decisions, task_frame=g_t,
            rho_low=self.rho_low, gamma=self.gamma,
            task_gain_source=gain_source,
        )


# ── Debug display ──────────────────────────────────────────────────────────────

def _debug_show(result: ScopeResult) -> None:
    print(f"\n  Task frame g_t:")
    print(f"    {result.task_frame}")
    print(f"\n  Thresholds : ρ_low={result.rho_low}  γ={result.gamma}")
    print(f"  TaskGain   : {result.task_gain_source}")

    def _fmt(d: ScopeDecision) -> str:
        k_s     = f"κ={d.kappa}({'current' if d.kappa == 0 else 'lc_injected'})"
        rel_s   = f"  Rel={d.rel:+.2f}" if d.rel is not None else ""
        return f"  [{d.span.span_type:<20}] {d.span.text!r:40}{rel_s}  {k_s}  → {d.reason}"

    print(f"\n  Retained ({len(result.retained)}):")
    for d in result.decisions:
        if d.kept:
            print(f"    [KEEP]{_fmt(d)}")

    print(f"\n  Dropped ({len(result.dropped)}):")
    for d in result.decisions:
        if not d.kept:
            print(f"    [DROP]{_fmt(d)}")


# ── Standalone ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from state.state_io import load_state
    from privacy.span_extractor import SpanExtractor, _debug_show as _show_spans
    from llm.local_llm import LocalLLM

    state   = load_state()
    profile = state.get("user_profile", {})
    traces  = state.get("memory_traces", [])

    print(f"\n  Memory traces loaded: {len(traces)} entries")
    booked = [t for t in traces if t.get("source") == "tool:book_appointment"]
    print(f"  Prior bookings in trace: {len(booked)}")
    for b in booked:
        d = b.get("data", {})
        print(f"    -> {d.get('booked_at', '?')}  ({b.get('from_workflow','')[:55]}...)")

    # Try to connect to local LLM (Ollama); fall back gracefully if unavailable
    try:
        local_llm = LocalLLM()
        local_llm.generate("ping")   # quick connectivity check
        print(f"\n  Local LLM: connected ({local_llm.model})")
    except Exception:
        local_llm = None
        print(f"\n  Local LLM: unavailable — g_t and TaskGain will use rule-table fallback")

    # ── r_t: original user request ─────────────────────────────────────────────
    r_t = (
        "I have tooth pain and bleeding gums, "
        "book me a dentist appointment at the earliest."
    )

    # ── p_t: LC-enriched payload (profile data injected, context expanded) ─────
    p_t = (
        "Hi, I'm Bob Smith and you can reach me at 585-555-1212 or bob@example.com. "
        "I need a dentist near 12 ABC St, Rochester, NY. I have BlueCross Dental Plus. "
        "My insurance ID is BC-123456-A9. I am available on March 18 and March 19, "
        "preferably before 10:30 AM. If possible, book for 2 people. "
        "I previously visited Bright Smile Dental and Lake Dental Care. "
        "I have tooth pain and bleeding gums. My ZIP is 14623 and I want to keep "
        "the cost under $500."
    )

    print(f"\n{'═' * 70}")
    print(f"  ORIGINAL USER REQUEST  (r_t)")
    print(f"{'═' * 70}")
    print(f"  {r_t}")

    print(f"\n{'═' * 70}")
    print(f"  LC-ENRICHED PAYLOAD  (p_t)  — input to sanitization pipeline")
    print(f"{'═' * 70}")
    print(f"  {p_t}")

    # ── Stage 1: Span Extraction on p_t ───────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  STAGE 1 — SPAN EXTRACTION  (operating on p_t)")
    print(f"{'═' * 70}")
    extractor  = SpanExtractor()
    extraction = extractor.extract(p_t, profile)
    _show_spans(extraction)

    # ── Stage 2: Scope Control — g_t derived from r_t, not p_t ────────────────
    print(f"\n{'═' * 70}")
    print(f"  STAGE 2 — SCOPE CONTROL  (g_t derived from r_t)")
    print(f"{'═' * 70}")
    controller = ScopeController()
    result     = controller.filter(
        u_med=extraction["U_med"],
        task=r_t,
        payload=extraction["working"],
        local_llm=local_llm,
    )
    _debug_show(result)

    # ── Stage 2 output → Stage 3a input ───────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  STAGE 2 OUTPUT  — payload entering Stage 3a (retained spans verbatim; U_loc withheld)")
    print(f"{'═' * 70}")
    out = reconstruct_payload(extraction, result.retained, extraction["U_loc"])
    print(f"  {out}")

