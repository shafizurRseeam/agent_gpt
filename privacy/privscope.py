"""
privacy/privscope.py

PrivScope — on-device payload governor for hybrid agent execution.

Core principle: task-scoped disclosure. The cloud receives only
information justified by the current delegated task, in the least
revealing form that preserves task utility.

Pipeline (stages added incrementally):

  Stage 1 — Span Extraction      (privacy/span_extractor.py)       ← ACTIVE
  Stage 2 — Scope Control        (privacy/scope_control.py)        ← ACTIVE
  Stage 3a — Sensitivity Classification (privacy/span_classification.py) ← ACTIVE
  Stage 3b — Abstraction / Transformation (privacy/span_abstraction.py)  ← ACTIVE
  Stage 4  — Local Restoration                                      [TODO]

Public API (mirrors privacyscope.py for drop-in replacement):
    sanitize(payload, user_profile, task, memory_traces) -> str
    sanitize_with_trace(...)
        -> (sanitized_str, trace_dict)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from privacy.span_extractor      import SpanExtractor, Span, spans_to_records
from privacy.scope_control       import ScopeController
from privacy.span_classification import SpanClassifier
from privacy.span_abstraction    import SpanAbstractor


class PrivScope:
    """
    Main PrivScope governor. Instantiate once and call sanitize_with_trace()
    per payload.

    Parameters
    ----------
    local_llm : optional LocalLLM instance used by Stage 2 for TaskGain scoring.
                If None, scope control falls back to rule table.
    rho_low   : Rel threshold for r_t spans (Stage 2)
    gamma     : TaskGain threshold for LC-injected spans (Stage 2)
    """

    def __init__(
        self,
        local_llm = None,
        rho_low:  float = 0.10,
        gamma:    float = 0.50,
    ):
        self._extractor   = SpanExtractor()
        self._scope       = ScopeController(rho_low=rho_low, gamma=gamma)
        self._classifier  = SpanClassifier()
        self._abstractor  = SpanAbstractor()
        self._local_llm   = local_llm

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
        Run payload through all active PrivScope stages.

        Returns (sanitized_payload, trace_dict).

        trace_dict keys:
            stage1        — span extraction output
            stage2        — scope control output
            stage3a       — PTH/CSS classification output
            stage3b       — abstraction decisions
            binding_table — {span_text: span_type} for U_loc (never sent to cloud)
            method        — "privscope"
        """
        profile = user_profile or {}
        traces  = memory_traces or []

        # ── Stage 1: Span Extraction ──────────────────────────────────────────
        extraction = self._extractor.extract(payload, profile)

        u_loc: List[Span] = extraction["U_loc"]
        u_med: List[Span] = extraction["U_med"]

        # Private binding table — U_loc identifiers withheld from cloud
        binding_table: Dict[str, str] = {s.text: s.span_type for s in u_loc}

        # ── Stage 2: Scope Control ────────────────────────────────────────────
        scope_result = self._scope.filter(
            u_med=u_med,
            task=task,
            payload=extraction["working"],
            local_llm=self._local_llm,
        )

        # ── Stage 3a: Sensitivity Classification ─────────────────────────────
        class_result = self._classifier.classify(
            retained_spans=scope_result.retained,
            g_t=scope_result.task_frame,
            local_llm=self._local_llm,
        )

        # ── Stage 3b: Span Abstraction ────────────────────────────────────────
        abstraction_result = self._abstractor.abstract(
            css_spans  = class_result.context_sensitive,
            pth_spans  = class_result.passthrough,
            g_t        = class_result.task_frame,
            extraction = extraction,
            u_loc      = u_loc,
            local_llm  = self._local_llm,
        )

        # ── Stage 4 placeholder ───────────────────────────────────────────────
        # Local restoration of U_loc spans will be wired here.

        trace = {
            "stage1": {
                "u_loc": spans_to_records(u_loc),
                "u_med": spans_to_records(u_med),
            },
            "stage2": {
                "task_frame": scope_result.task_frame,
                "rho_low":    scope_result.rho_low,
                "gamma":      scope_result.gamma,
                "retained":   [s.text for s in scope_result.retained],
                "dropped": [
                    {"text": d.span.text, "reason": d.reason,
                     "rel": round(d.rel, 3) if d.rel is not None else None,
                     "kappa": d.kappa}
                    for d in scope_result.decisions if not d.kept
                ],
            },
            "stage3a": {
                "method":            class_result.method,
                "passthrough":       [s.text for s in class_result.passthrough],
                "context_sensitive": [s.text for s in class_result.context_sensitive],
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
            "method":        "privscope",
        }

        return abstraction_result.final_cloud_payload, trace


# ── Standalone — runs the full PrivScope governor on the dentist example ───────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from state.state_io import load_state
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

    privscope            = PrivScope(local_llm=local_llm)
    cloud_payload, trace = privscope.sanitize_with_trace(
        payload=p_t, user_profile=profile, task=r_t,
    )

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  STAGE 1 — SPAN EXTRACTION")
    print(f"{'═' * W}")
    s1 = trace["stage1"]
    print(f"  U_loc withheld  ({len(s1['u_loc'])} spans):")
    for s in s1["u_loc"]:
        print(f"    [{s['span_type']:<20}] {s['text']!r}")
    print(f"  U_med mediation ({len(s1['u_med'])} spans):")
    for s in s1["u_med"]:
        print(f"    [{s['span_type']:<20}] {s['text']!r}")

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  STAGE 2 — SCOPE CONTROL")
    print(f"{'═' * W}")
    s2 = trace["stage2"]
    print(f"  Task frame g_t : {s2['task_frame']}")
    print(f"  Thresholds     : ρ_low={s2['rho_low']}  γ={s2['gamma']}")
    print(f"  Retained ({len(s2['retained'])}):")
    for t in s2["retained"]:
        print(f"    [KEEP]  {t!r}")
    print(f"  Dropped  ({len(s2['dropped'])}):")
    for d in s2["dropped"]:
        print(f"    [DROP]  {d['text']!r}  ({d['reason']})")

    # ── Stage 3a ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  STAGE 3a — SENSITIVITY CLASSIFICATION  [{trace['stage3a']['method']}]")
    print(f"{'═' * W}")
    s3a = trace["stage3a"]
    print(f"  PTH (passthrough)      : {s3a['passthrough']}")
    print(f"  CSS (context-sensitive): {s3a['context_sensitive']}")

    # ── Stage 3b ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  STAGE 3b — SPAN ABSTRACTION  [{trace['stage3b']['method']}]")
    print(f"{'═' * W}")
    for d in trace["stage3b"]["decisions"]:
        if d["method"] == "grouped":
            continue
        print(
            f"    {d['original']!r:<35} → "
            f"l{d['level']} ({d['level_desc']}) → "
            f"{d['abstracted']!r}  [{d['method']}]"
        )

    # ── Payloads ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  INTERNAL SANITIZED PAYLOAD  — U_loc as [TYPE] placeholders (debug/trace only)")
    print(f"{'═' * W}")
    print(f"  {trace['sanitized_internal_payload']}")

    print(f"\n{'═' * W}")
    print(f"  FINAL CLOUD PAYLOAD  P̂_t  — U_loc stripped and cleaned, sent to CLM")
    print(f"{'═' * W}")
    print(f"  {cloud_payload}")
