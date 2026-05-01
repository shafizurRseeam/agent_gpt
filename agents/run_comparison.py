"""
run_comparison.py
════════════════════════════════════════════════════════════════════════
Given a task, builds the LC-enriched naive payload, applies all three
privacy baselines (PrivScope, Presidio, PEP), sends each payload to the
cloud LLM, and displays a comparison of payloads and CLM responses.

Usage:
    uv run python run_comparison.py
    uv run python run_comparison.py --task "book a dentist appointment"
════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import sys
import textwrap

sys.stdout.reconfigure(encoding="utf-8")

from agents.hybrid_agent import HybridAgent
from privacy.privscope import PrivScope
import privacy.presidio as presidio_baseline
import privacy.pep as pep_baseline

W = 72


# ── Display helpers ───────────────────────────────────────────────────────────

def _div(label="", char="─"):
    if label:
        pad = max(0, W - len(label) - 4)
        print(f"\n  {char * 2}  {label}  {char * pad}")
    else:
        print(f"\n  {char * W}")


def _box(title, text, indent=4):
    """Print a titled box with word-wrapped content."""
    print(f"\n  ┌─ {title}")
    for line in (text or "").strip().splitlines():
        for seg in (textwrap.wrap(line, width=W - indent - 2) if line.strip() else [""]):
            print(f"  │  {' ' * (indent - 4)}{seg}")
    print(f"  └{'─' * (W - 2)}")


def _header(title):
    print(f"\n{'═' * W}")
    print(f"  {title}")
    print(f"{'═' * W}")


def _section(label):
    print(f"\n{'─' * W}")
    print(f"  {label}")
    print(f"{'─' * W}")


def _payload_stats(payloads: dict[str, str]) -> None:
    """Print a compact summary table of payload lengths and reduction."""
    naive_len = len(payloads.get("naive", ""))
    print(f"\n  {'Baseline':<14}  {'Chars':>6}  {'Reduction':>10}")
    print(f"  {'─' * 14}  {'─' * 6}  {'─' * 10}")
    for label, text in payloads.items():
        n = len(text or "")
        if label == "naive":
            pct = "—"
        else:
            pct = f"{(1 - n / naive_len) * 100:+.1f}%" if naive_len else "—"
        print(f"  {label:<14}  {n:>6}  {pct:>10}")


# ── Core comparison ───────────────────────────────────────────────────────────

def run_comparison(task: str) -> None:
    _header(f"PAYLOAD COMPARISON  —  all baselines")
    print(f"  Task: {task}")
    print(f"  LC model: {_get_model_name()}")

    # ── Instantiate components ─────────────────────────────────────────────────
    print(f"\n  Initialising models …")
    agent   = HybridAgent()
    profile = agent.state.get("user_profile", {})
    traces  = agent.state.get("memory_traces", [])

    # New PrivScope governor with local LLM for TaskGain and abstraction
    ps = PrivScope(local_llm=agent.local)

    # ── Phase 1: LC builds naive payload ──────────────────────────────────────
    _section("PHASE 1  —  LC reasoning  →  naive payload")

    inferred_prefs = agent._lc_infer_preferences(task)
    if inferred_prefs:
        _box("LC Inferred Context", inferred_prefs)

    reasoning, naive_payload = agent._lc_reason_cloud_query(task, inferred_prefs)
    _box("LC Reasoning", reasoning)
    _box("Naive Payload  (p_t — sent unmodified as baseline 1)", naive_payload)

    # ── Phase 2: Privacy baselines ────────────────────────────────────────────
    _section("PHASE 2  —  Privacy baselines applied to p_t")

    # PrivScope (new governor — Stages 1–3b)
    print("\n  [PrivScope] Running Stages 1–3b …")
    privscope_payload, ps_trace = ps.sanitize_with_trace(
        naive_payload, profile, task, traces
    )
    _box("PrivScope Output  (new governor, Stages 1–3b)", privscope_payload)

    # Presidio
    print("  [Presidio] Running NER redaction …")
    presidio_payload, presidio_trace = presidio_baseline.sanitize_with_trace(
        naive_payload, profile, task, traces
    )
    span_summary = (
        ", ".join(f"{t}" for _, t in presidio_trace["spans"][:6])
        + ("…" if len(presidio_trace["spans"]) > 6 else "")
    ) if presidio_trace["spans"] else "none detected"
    print(f"  [Presidio] Entities redacted: {len(presidio_trace['spans'])}  ({span_summary})")
    _box("Presidio Output  (NER redaction baseline)", presidio_payload)

    # PEP
    print("  [PEP] Running local-LLM light-touch filter …")
    pep_payload, _pep_trace = pep_baseline.sanitize_with_trace(
        naive_payload, profile, task, traces, agent.local
    )
    _box("PEP Output  (privacy-enhancing prompt baseline)", pep_payload)

    # ── Phase 3: CLM responses ─────────────────────────────────────────────────
    _section("PHASE 3  —  Cloud LLM responses  (one per baseline payload)")

    print("\n  Querying cloud LLM …")

    clm_naive     = agent._cloud_search(naive_payload)
    clm_privscope = agent._cloud_search(privscope_payload)
    clm_presidio  = agent._cloud_search(presidio_payload)
    clm_pep       = agent._cloud_search(pep_payload)

    _box("CLM Response  →  Naive payload", clm_naive)
    _box("CLM Response  →  PrivScope payload", clm_privscope)
    _box("CLM Response  →  Presidio payload", clm_presidio)
    _box("CLM Response  →  PEP payload", clm_pep)

    # ── Summary table ──────────────────────────────────────────────────────────
    _section("SUMMARY  —  payload lengths and reduction vs naive")
    _payload_stats({
        "naive":      naive_payload,
        "privscope":  privscope_payload,
        "presidio":   presidio_payload,
        "pep":        pep_payload,
    })

    # PrivScope stage detail
    if ps_trace:
        s1 = ps_trace.get("stage1", {})
        s2 = ps_trace.get("stage2", {})
        s3a = ps_trace.get("stage3a", {})
        s3b = ps_trace.get("stage3b", {})
        print(f"\n  PrivScope stage detail:")
        print(f"    Stage 1  — U_loc: {len(s1.get('u_loc', []))} spans withheld  |  "
              f"U_med: {len(s1.get('u_med', []))} mediation candidates")
        print(f"    Stage 2  — retained: {len(s2.get('retained', []))}  |  "
              f"dropped: {len(s2.get('dropped', []))}")
        print(f"    Stage 3a — PTH: {len(s3a.get('passthrough', []))}  |  "
              f"CSS: {len(s3a.get('context_sensitive', []))}")
        print(f"    Stage 3b — {len(s3b.get('decisions', []))} abstraction decisions  "
              f"(method: {s3b.get('method', '?')})")
        if ps_trace.get("sanitized_internal_payload"):
            _box("PrivScope Internal Payload  (debug — [TYPE] placeholders for U_loc)",
                 ps_trace["sanitized_internal_payload"])

    print(f"\n{'═' * W}\n")


def _get_model_name() -> str:
    try:
        from llm.local_llm import LocalLLM
        return LocalLLM().model
    except Exception:
        return "unknown"


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run all privacy baselines on a task and compare payloads + CLM responses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--task", "-t", default=None,
        help="Task string (e.g. 'book a dentist appointment'). "
             "If omitted, you will be prompted interactively.",
    )
    args = ap.parse_args()

    task = args.task
    if not task:
        print("\n  Enter the user task (or press Enter for the dentist example):")
        task = input("  Task: ").strip()
        if not task:
            task = "I have tooth pain and bleeding gums, book me a dentist appointment at the earliest."
            print(f"  Using default: {task}")

    run_comparison(task)


if __name__ == "__main__":
    main()
