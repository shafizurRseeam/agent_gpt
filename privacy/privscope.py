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

        sanitized = abstraction_result.final_payload

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
            "binding_table": binding_table,
            "method":        "privscope",
        }

        return sanitized, trace

