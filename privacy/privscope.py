"""
privacy/privscope.py  —  PrivScope v2

On-device payload governor for hybrid agent execution.

Core principle: task-scoped disclosure. The cloud receives only
information justified by the current delegated task, in the least
revealing form that preserves task utility.

Pipeline (v2):

  Stage 1 — Span Extraction      (privacy/span_extractor.py)
  Stage 2 — Carryover Control    (privacy/carryover_control.py)
             Single LLM call that jointly decides which extracted
             spans are genuinely required for cloud-side task
             completion. Replaces the v1 two-step scope_control
             (per-span Rel + TaskGain) + span_classification (PTH/CSS).
  Stage 3b — Span Abstraction    (privacy/span_abstraction.py)
             All kept spans treated as context-sensitive (CSS) and
             abstracted to their calibrated policy level.
  Stage 4  — Local Restoration   [TODO]

Public API (same as v1 for drop-in replacement):
    sanitize(payload, user_profile, task, memory_traces) -> str
    sanitize_with_trace(...)  -> (sanitized_str, trace_dict)

v1 archived at: privacy/legacy/privscope_v1.py

Run standalone:
    uv run python privacy/privscope.py
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

from privacy.span_extractor    import SpanExtractor, Span, spans_to_records
from privacy.carryover_control import CarryoverController
from privacy.span_abstraction  import SpanAbstractor


class PrivScope:
    """
    PrivScope v2 governor. Instantiate once and call sanitize_with_trace()
    per payload.

    Parameters
    ----------
    local_llm : optional LocalLLM instance used by Stage 2 (carryover control)
                for joint span-necessity scoring. Falls back to rule table if None.
    """

    def __init__(self, local_llm=None):
        self._extractor  = SpanExtractor()
        self._carryover  = CarryoverController()
        self._abstractor = SpanAbstractor()
        self._local_llm  = local_llm

    # ── Public API ────────────────────────────────────────────────────────────

    def sanitize(
        self,
        payload:       str,
        user_profile:  dict = None,
        task:          str  = "",
        memory_traces: list = None,
    ) -> str:
        sanitized, _ = self.sanitize_with_trace(
            payload, user_profile, task, memory_traces
        )
        return sanitized

    def sanitize_with_trace(
        self,
        payload:       str,
        user_profile:  dict = None,
        task:          str  = "",
        memory_traces: list = None,
    ) -> Tuple[str, dict]:
        """
        Run payload through all active PrivScope v2 stages.

        Returns (sanitized_payload, trace_dict).

        trace_dict keys:
            stage1        — span extraction output
            stage2        — carryover control output (kept / dropped + g_t)
            stage3b       — abstraction decisions
            binding_table — {span_text: span_type} for U_loc (never sent to cloud)
            sanitized_internal_payload — P̂_t with [TYPE] placeholders (debug only)
            final_cloud_payload        — U_loc stripped, ready for CLM
            method        — "privscope_v2"
        """
        profile = user_profile or {}

        # ── Stage 1: Span Extraction ──────────────────────────────────────────
        t0_s1 = time.perf_counter()
        extraction = self._extractor.extract(payload, profile)
        t_s1 = time.perf_counter() - t0_s1
        u_loc: List[Span] = extraction["U_loc"]
        u_med: List[Span] = extraction["U_med"]

        binding_table: Dict[str, str] = {s.text: s.span_type for s in u_loc}

        # ── Stage 2: Carryover Control ────────────────────────────────────────
        t0_s2 = time.perf_counter()
        carryover_result = self._carryover.filter(
            u_med=u_med,
            task=task,
            extraction=extraction,
            local_llm=self._local_llm,
        )
        t_s2 = time.perf_counter() - t0_s2

        # ── Stage 3b: Span Abstraction ────────────────────────────────────────
        # All kept spans treated as CSS — no PTH/CSS split in v2.
        # Abstraction policy handles verbatim release for level-3 types internally.
        t0_s3b = time.perf_counter()
        abstraction_result = self._abstractor.abstract(
            css_spans  = carryover_result.kept,
            pth_spans  = [],
            g_t        = carryover_result.task_frame,
            extraction = extraction,
            u_loc      = u_loc,
            local_llm  = self._local_llm,
        )
        t_s3b = time.perf_counter() - t0_s3b

        # ── Stage 4 placeholder ───────────────────────────────────────────────
        # Local restoration of U_loc spans will be wired here.

        trace = {
            "stage1": {
                "u_loc": spans_to_records(u_loc),
                "u_med": spans_to_records(u_med),
            },
            "stage2": {
                "task_frame": carryover_result.task_frame,
                "method":     carryover_result.method,
                "kept":       [s.text for s in carryover_result.kept],
                "dropped": [
                    {"text": d.span.text, "reason": d.reason}
                    for d in carryover_result.decisions if not d.kept
                ],
            },
            "stage3b": {
                "method": abstraction_result.method,
                "decisions": [
                    {
                        "original":   d.original_text,
                        "abstracted": d.abstracted_text,
                        "level":      d.level,
                        "level_desc": d.level_desc,
                        "method":     d.method,
                    }
                    for d in abstraction_result.decisions
                ],
            },
            "sanitized_internal_payload": abstraction_result.final_payload,
            "final_cloud_payload":        abstraction_result.final_cloud_payload,
            "binding_table": binding_table,
            "method":        "privscope_v2",
            "stage_timings": {
                "stage1_s":  round(t_s1,  4),
                "stage2_s":  round(t_s2,  4),
                "stage3b_s": round(t_s3b, 4),
                "total_s":   round(t_s1 + t_s2 + t_s3b, 4),
            },
        }

        return abstraction_result.final_cloud_payload, trace


# ── Standalone ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from state.state_io import load_state
    from llm.local_llm  import LocalLLM

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

    print(f"\n{'=' * W}")
    print(f"  ORIGINAL USER REQUEST  (r_t)")
    print(f"{'=' * W}")
    print(f"  {r_t}")

    print(f"\n{'=' * W}")
    print(f"  LC-ENRICHED PAYLOAD  (p_t)  -- input to sanitization pipeline")
    print(f"{'=' * W}")
    print(f"  {p_t}")

    privscope            = PrivScope(local_llm=local_llm)
    cloud_payload, trace = privscope.sanitize_with_trace(
        payload=p_t, user_profile=profile, task=r_t,
    )

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  STAGE 1 -- SPAN EXTRACTION")
    print(f"{'=' * W}")
    s1 = trace["stage1"]
    print(f"  U_loc withheld  ({len(s1['u_loc'])} spans):")
    for s in s1["u_loc"]:
        print(f"    [{s['span_type']:<20}] {s['text']!r}")
    print(f"  U_med candidates ({len(s1['u_med'])} spans):")
    for s in s1["u_med"]:
        print(f"    [{s['span_type']:<20}] {s['text']!r}")

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  STAGE 2 -- CARRYOVER CONTROL  [{trace['stage2']['method']}]")
    print(f"{'=' * W}")
    s2 = trace["stage2"]
    print(f"  Task frame g_t : {s2['task_frame']}")
    print(f"  Kept    ({len(s2['kept'])}):")
    for t in s2["kept"]:
        print(f"    [KEEP]  {t!r}")
    print(f"  Dropped ({len(s2['dropped'])}):")
    for d in s2["dropped"]:
        print(f"    [DROP]  {d['text']!r}  ({d['reason']})")

    # ── Stage 3b ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  STAGE 3b -- SPAN ABSTRACTION  [{trace['stage3b']['method']}]")
    print(f"{'=' * W}")
    for d in trace["stage3b"]["decisions"]:
        if d["method"] == "grouped":
            continue
        print(
            f"    {d['original']!r:<35}  l{d['level']} ({d['level_desc']})  "
            f"->  {d['abstracted']!r}  [{d['method']}]"
        )

    # ── Payloads ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  INTERNAL SANITIZED PAYLOAD  -- U_loc as [TYPE] placeholders (debug/trace only)")
    print(f"{'=' * W}")
    print(f"  {trace['sanitized_internal_payload']}")

    print(f"\n{'=' * W}")
    print(f"  FINAL CLOUD PAYLOAD  P_t_hat  -- U_loc stripped, sent to CLM")
    print(f"{'=' * W}")
    print(f"  {cloud_payload}")
