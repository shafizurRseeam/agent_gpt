"""
privacy/scope_control.py

Stage 2 of PrivScope — Scope Control.

Filters the U_med spans produced by Stage 1, retaining only those
justified by the current delegated task.

Each span u_j is evaluated against a task frame g_t (enriched description
of the subtask offloaded to the CLM) using two signals:

  Rel(u_j, g_t) ∈ [-1, 1]  — embedding cosine similarity to the task frame
  κ(u_j) ∈ {0, 1}           — carryover status (0=current, 1=prior history)

Retention rules (matching paper formulation exactly):

  Current-workflow (κ=0):
    keep if  Rel(u_j, g_t) >= ρ_low

  Carryover (κ=1):
    keep if  Rel(u_j, g_t) >= ρ_high
         AND TaskGain(u_j, g_t) >= γ

    ρ_high > ρ_low ensures weakly-related carryover never reaches the
    utility check. TaskGain is scored by the local LM; falls back to a
    rule table if no LM is supplied.

Carryover detection — two independent sources (both → κ=1):
  1. Span text appears in memory_traces data from prior tasks  → "trace"
  2. Span appears inside a historical-context sentence         → "historical_sentence"
  kappa_reason records which source fired (for debug/trace only).

Payload reconstruction:
  reconstruct_payload() fills C_t slots with retained span text (or
  [TYPE] for U_loc), drops empty slots, and cleans up orphaned fragments.

Public API:
    ScopeController(rho_low, rho_high, gamma)
    ScopeController.filter(u_med, task, memory_traces, payload, local_llm) → ScopeResult
    reconstruct_payload(extraction, retained_u_med, u_loc) → str
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

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
    rel:          float
    kappa:        int          # 0=current-workflow, 1=carryover
    kappa_reason: str          # "trace" | "historical_sentence" | ""
    task_gain:    float        # TaskGain score from LLM or rule table (0 if not computed)
    kept:         bool
    reason:       str          # human-readable explanation always shown


@dataclass
class ScopeResult:
    retained:        List[Span]
    dropped:         List[Span]
    decisions:       List[ScopeDecision]
    task_frame:      str
    rho_low:         float
    rho_high:        float
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
    "organization":   0.05,
    "facility":       0.05,
    "group":          0.20,
    "person":         0.05,
}

_LLM_GAIN_PROMPT = """\
You are evaluating whether a piece of information from a user's prior history \
is genuinely useful for a new task.

Task: {task}
Information: "{span_text}" (type: {span_type})

This value comes from the user's memory of a past workflow. \
Rate how much retaining it would help accomplish the current task \
(not just related in topic — actually needed to complete the task).

Reply with ONLY a decimal number between 0.0 and 1.0. No explanation."""

_FLOAT_RE = re.compile(r'\b(0(?:\.\d+)?|1(?:\.0+)?)\b')


def _llm_task_gain(local_llm, task: str, span: Span) -> Tuple[float, str]:
    """
    Ask local LLM to score task gain for a trace-carryover span.
    Returns (score, "llm") on success, falls back to (_rule_based_task_gain, "rule") on error.
    """
    prompt = _LLM_GAIN_PROMPT.format(
        task=task.strip(),
        span_text=span.text,
        span_type=span.span_type,
    )
    try:
        raw = local_llm.generate(prompt).strip()
        m   = _FLOAT_RE.search(raw)
        if m:
            return float(m.group(0)), "llm"
    except Exception:
        pass
    return _RULE_GAIN.get(span.span_type, 0.30), "rule"


def _rule_based_task_gain(span: Span) -> Tuple[float, str]:
    return _RULE_GAIN.get(span.span_type, 0.30), "rule"


# ── Carryover detection ────────────────────────────────────────────────────────

_HISTORY_RE = re.compile(
    r"\b(previously|i'?ve had|prior to|in the past|had appointments? at|"
    r"been to|seen at|visited|used to go)\b",
    re.IGNORECASE,
)


def _build_carryover_set(memory_traces: list) -> Set[str]:
    """Extract lower-cased values from prior-task memory_traces."""
    carryover: Set[str] = set()
    for trace in memory_traces:
        data = trace.get("data", {})
        if isinstance(data, dict):
            for val in data.values():
                if isinstance(val, str) and len(val) > 3:
                    carryover.add(val.lower())
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    for val in item.values():
                        if isinstance(val, str) and len(val) > 3:
                            carryover.add(val.lower())
    return carryover


def _split_sentences(text: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', text.strip())


def _is_in_historical_sentence(span: Span, payload: str) -> bool:
    """True if span text appears in any _HISTORY_RE-matched sentence."""
    for sent in _split_sentences(payload):
        if _HISTORY_RE.search(sent):
            if re.search(re.escape(span.text), sent, re.IGNORECASE):
                return True
    return False


def _carryover_status(
    span: Span, carryover: Set[str], payload: str
) -> Tuple[int, str]:
    """
    Returns (κ, reason).
      κ=0, ""                   — current workflow
      κ=1, "trace"              — value found in memory_traces
      κ=1, "historical_sentence"— span inside a historical-context sentence
    """
    if span.text.lower() in carryover:
        return 1, "trace"
    if _is_in_historical_sentence(span, payload):
        return 1, "historical_sentence"
    return 0, ""


# ── Payload reconstruction ─────────────────────────────────────────────────────

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
    Stage 2 filter. Hybrid: deterministic for current-workflow and
    historical-sentence carryover; LLM-scored for trace-carryover spans
    that pass the Rel gate (falls back to rule table if local_llm is None).

    Parameters
    ----------
    rho_low  : Rel threshold for current-workflow spans (default 0.15)
    rho_high : Rel threshold for carryover (trace) spans (default 0.30)
    gamma    : TaskGain threshold for trace-carryover spans (default 0.50)
    """

    def __init__(
        self,
        rho_low:  float = 0.10,
        rho_high: float = 0.30,
        gamma:    float = 0.50,
    ):
        assert rho_high > rho_low
        self.rho_low  = rho_low
        self.rho_high = rho_high
        self.gamma    = gamma

    def filter(
        self,
        u_med:         List[Span],
        task:          str,
        memory_traces: list = None,
        payload:       str  = "",
        local_llm             = None,   # used for g_t generation and TaskGain scoring
        g_t_override:  str  = None,    # if set, skip _derive_task_frame and use this directly
    ) -> ScopeResult:
        """
        Apply scope control to the U_med span set.
        local_llm drives both g_t construction (richer task frame) and TaskGain
        scoring for carryover spans. Falls back to span-type heuristics if None.
        g_t_override bypasses all task-frame construction — useful for comparison runs.
        Rel is computed on type-aware span representations (not raw span text).
        """
        traces     = memory_traces or []
        g_t        = g_t_override if g_t_override is not None else _derive_task_frame(task, u_med, local_llm)
        carryover  = _build_carryover_set(traces)
        gain_source = f"llm:{local_llm.model}" if local_llm is not None else "rule"

        st      = _st()
        g_t_emb = st.encode(g_t, convert_to_tensor=True)

        decisions: List[ScopeDecision] = []

        if u_med:
            span_embs = st.encode([_span_repr(s) for s in u_med], convert_to_tensor=True)

            for i, span in enumerate(u_med):
                rel            = float(st_util.cos_sim(span_embs[i], g_t_emb).item())
                kappa, k_reason = _carryover_status(span, carryover, payload)

                if kappa == 0:
                    # ── Current-workflow (κ=0): keep if Rel >= ρ_low ──────
                    task_gain = 0.0
                    kept      = rel >= self.rho_low
                    reason    = (
                        f"Rel={rel:.2f} >= ρ_low={self.rho_low:.2f}" if kept
                        else f"Rel={rel:.2f} < ρ_low={self.rho_low:.2f}"
                    )
                else:
                    # ── Carryover (κ=1): Rel >= ρ_high AND TaskGain >= γ ──
                    # kappa_reason ("trace" | "historical_sentence") is logged
                    # for debug transparency but does not change the policy.
                    if rel < self.rho_high:
                        task_gain = 0.0
                        kept      = False
                        reason    = (
                            f"carryover [{k_reason}], "
                            f"Rel={rel:.2f} < ρ_high={self.rho_high:.2f}"
                        )
                    else:
                        # Rel passes stricter gate — score TaskGain
                        if local_llm is not None:
                            task_gain, gain_src = _llm_task_gain(local_llm, task, span)
                        else:
                            task_gain, gain_src = _rule_based_task_gain(span)
                        kept   = task_gain >= self.gamma
                        reason = (
                            f"carryover [{k_reason}], "
                            f"TaskGain={task_gain:.2f} ({gain_src}) >= γ={self.gamma:.2f}" if kept
                            else
                            f"carryover [{k_reason}], "
                            f"TaskGain={task_gain:.2f} ({gain_src}) < γ={self.gamma:.2f}"
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
            rho_low=self.rho_low, rho_high=self.rho_high, gamma=self.gamma,
            task_gain_source=gain_source,
        )


# ── Debug display ──────────────────────────────────────────────────────────────

def _debug_show(result: ScopeResult) -> None:
    print(f"\n  Task frame g_t:")
    print(f"    {result.task_frame}")
    print(f"\n  Thresholds : ρ_low={result.rho_low}  ρ_high={result.rho_high}  γ={result.gamma}")
    print(f"  TaskGain   : {result.task_gain_source}")

    print(f"\n  Retained ({len(result.retained)}):")
    for d in result.decisions:
        if not d.kept:
            continue
        k_s = f"κ={d.kappa}({d.kappa_reason or 'current'})"
        print(f"    [KEEP] [{d.span.span_type:<20}] {d.span.text!r:40}"
              f"  Rel={d.rel:+.2f}  {k_s}  → {d.reason}")

    print(f"\n  Dropped ({len(result.dropped)}):")
    for d in result.decisions:
        if d.kept:
            continue
        k_s = f"κ={d.kappa}({d.kappa_reason or 'current'})"
        print(f"    [DROP] [{d.span.span_type:<20}] {d.span.text!r:40}"
              f"  Rel={d.rel:+.2f}  {k_s}  → {d.reason}")


# ── Standalone ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
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
        print(f"    → {d.get('booked_at', '?')}  ({b.get('from_workflow','')[:55]}…)")

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

    # Strip U_loc from working payload before Stage 2
    working = p_t
    for span in sorted(extraction["U_loc"], key=lambda s: s.start, reverse=True):
        working = working[:span.start] + f"[{span.span_type.upper()}]" + working[span.end:]

    # ── Stage 2: Scope Control — g_t derived from r_t, not p_t ────────────────
    print(f"\n{'═' * 70}")
    print(f"  STAGE 2 — SCOPE CONTROL  (g_t derived from r_t)")
    print(f"{'═' * 70}")
    controller = ScopeController()
    result     = controller.filter(
        u_med=extraction["U_med"],
        task=r_t,                   # task frame built from original request
        memory_traces=traces,
        payload=working,
        local_llm=local_llm,
    )
    _debug_show(result)

    # ── Sanitized output ───────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  SANITIZED PAYLOAD  — sent to cloud")
    print(f"{'═' * 70}")
    out = reconstruct_payload(extraction, result.retained, extraction["U_loc"])
    print(f"  {out}")

    # ── Comparison: g_t = r_t directly (no LLM expansion, no heuristics) ──────
    print(f"\n{'═' * 70}")
    print(f"  COMPARISON — g_t = r_t (raw request, no expansion)")
    print(f"{'═' * 70}")
    result_rt = controller.filter(
        u_med=extraction["U_med"],
        task=r_t,
        memory_traces=traces,
        payload=working,
        local_llm=local_llm,
        g_t_override=r_t,           # bypass _derive_task_frame entirely
    )
    _debug_show(result_rt)
    out_rt = reconstruct_payload(extraction, result_rt.retained, extraction["U_loc"])
    print(f"\n  Sanitized payload (g_t = r_t):")
    print(f"  {out_rt}")
