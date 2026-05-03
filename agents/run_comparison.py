"""
agents/run_comparison.py
========================================================================
Given a task, builds the LC-enriched naive payload, applies all three
privacy methods, sends each payload to all three CLMs, and displays a
comparison of payloads, CLM responses, and a token accounting table.

Methods compared:
  naive       -- raw LC-enriched payload, no privacy filter
  privscope   -- PrivScope v2 (carryover control + abstraction)
  presidio    -- Presidio NER redaction baseline
  pep         -- PEP local-LLM filter baseline

Cloud models:
  OpenAI  -- gpt-4o-mini-2024-07-18
  Claude  -- claude-sonnet-4-6
  Gemini  -- gemini-2.5-flash

Usage (from project root agent_gpt/):
    uv run python agents/run_comparison.py
    uv run python agents/run_comparison.py --task "book a dentist appointment"
    uv run python agents/run_comparison.py --local-model qwen2.5:7b
========================================================================
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from typing import Dict, Tuple

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from agents.hybrid_agent import HybridAgent
from privacy.privscope   import PrivScope
import privacy.presidio  as presidio_baseline
import privacy.pep       as pep_baseline
from llm.cloud_router    import CloudLLM

W = 72

_CLM_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Return a numbered list of up to 5 local service providers matching the request. "
    "For each provider include exactly: name, address, phone. Be concise. "
    "Format each entry as:\n"
    "1. <Name>\n   Address: <address>\n   Phone: <phone>\n"
)

_PROVIDERS = ("openai", "claude", "gemini")


# ── Display helpers (original) ────────────────────────────────────────────────

def _div(label="", char="-"):
    if label:
        pad = max(0, W - len(label) - 4)
        print(f"\n  {char * 2}  {label}  {char * pad}")
    else:
        print(f"\n  {char * W}")


def _box(title, text, indent=4):
    print(f"\n  +-- {title}")
    for line in (text or "").strip().splitlines():
        for seg in (textwrap.wrap(line, width=W - indent - 2) if line.strip() else [""]):
            print(f"  |  {' ' * (indent - 4)}{seg}")
    print(f"  +{'-' * (W - 2)}")


def _header(title):
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")


def _section(label):
    print(f"\n{'-' * W}")
    print(f"  {label}")
    print(f"{'-' * W}")


def _payload_stats(payloads: dict) -> None:
    """Original compact summary: payload lengths and reduction vs naive."""
    naive_len = len(payloads.get("naive", ""))
    print(f"\n  {'Baseline':<14}  {'Chars':>6}  {'Reduction':>10}")
    print(f"  {'-' * 14}  {'-' * 6}  {'-' * 10}")
    for label, text in payloads.items():
        n = len(text or "")
        pct = "-" if label == "naive" else (
            f"{(1 - n / naive_len) * 100:+.1f}%" if naive_len else "-"
        )
        print(f"  {label:<14}  {n:>6}  {pct:>10}")


def _token_table(rows: list) -> None:
    """
    Token accounting table.
    rows: list of dicts — method, chars, in_tok, oai_out, cld_out, gmn_out
    """
    c0, c1, c2, c3, c4, c5 = 14, 7, 8, 9, 9, 9

    def _row(m, ch, it, oo, co, go):
        print(f"  {str(m):<{c0}}  {str(ch):>{c1}}  {str(it):>{c2}}  "
              f"{str(oo):>{c3}}  {str(co):>{c4}}  {str(go):>{c5}}")

    print()
    _row("Method", "Chars", "In tok", "OAI out", "CLD out", "GMN out")
    _row("-" * c0, "-" * c1, "-" * c2, "-" * c3, "-" * c4, "-" * c5)

    naive_chars = next((r["chars"] for r in rows if r["method"] == "naive"), 1) or 1
    for r in rows:
        label = r["method"]
        if r["method"] != "naive" and naive_chars:
            pct = (1 - r["chars"] / naive_chars) * 100
            label = f"{r['method']} ({pct:+.0f}%)"
        _row(label, r["chars"], r["in_tok"], r["oai_out"], r["cld_out"], r["gmn_out"])


# ── CLM query helper ──────────────────────────────────────────────────────────

def _query_clm(clm: CloudLLM, payload: str) -> Tuple[str, int, int]:
    """Returns (response_text, input_tokens, output_tokens)."""
    try:
        messages = [
            {"role": "system", "content": _CLM_SYSTEM_PROMPT},
            {"role": "user",   "content": payload},
        ]
        return clm.chat_with_usage(messages)
    except Exception as e:
        return f"[error: {e}]", 0, 0


# ── Core comparison ───────────────────────────────────────────────────────────

def run_comparison(task: str, local_model: str = None) -> None:
    _header(f"PAYLOAD COMPARISON  --  all baselines  x  all CLMs")
    print(f"  Task: {task}")

    # ── Instantiate components ─────────────────────────────────────────────────
    print(f"\n  Initialising models ...")
    agent = HybridAgent(local_model=local_model) if local_model else HybridAgent()
    print(f"  Local model : {agent.local.model}")

    clms: Dict[str, CloudLLM] = {}
    for p in _PROVIDERS:
        try:
            clms[p] = CloudLLM(provider=p)
            print(f"  CLM [{p:<6}] : {clms[p].model}")
        except Exception as e:
            print(f"  CLM [{p:<6}] : unavailable ({e})")

    profile = agent.state.get("user_profile", {})
    traces  = agent.state.get("memory_traces", [])
    ps      = PrivScope(local_llm=agent.local)

    # ── Phase 1: LC builds naive payload ──────────────────────────────────────
    _section("PHASE 1  --  LC reasoning  ->  naive payload")

    inferred_prefs = agent._lc_infer_preferences(task)
    if inferred_prefs:
        _box("LC Inferred Context", inferred_prefs)

    reasoning, naive_payload = agent._lc_reason_cloud_query(task, inferred_prefs)
    _box("LC Reasoning", reasoning)
    _box("Naive Payload  (p_t -- sent unmodified as baseline)", naive_payload)

    # ── Phase 2: Privacy baselines ────────────────────────────────────────────
    _section("PHASE 2  --  Privacy baselines applied to p_t")

    print("\n  [PrivScope v2] Running Stage 1 -> Carryover Control -> Stage 3b ...")
    ps_payload, ps_trace = ps.sanitize_with_trace(naive_payload, profile, task, traces)
    _box("PrivScope v2 Output  (Stages 1, 2, 3b)", ps_payload)

    print("  [Presidio] Running NER redaction ...")
    presidio_payload, presidio_trace = presidio_baseline.sanitize_with_trace(
        naive_payload, profile, task, traces
    )
    span_summary = (
        ", ".join(f"{t}" for _, t in presidio_trace["spans"][:6])
        + ("..." if len(presidio_trace["spans"]) > 6 else "")
    ) if presidio_trace["spans"] else "none detected"
    print(f"  [Presidio] Entities redacted: {len(presidio_trace['spans'])}  ({span_summary})")
    _box("Presidio Output  (NER redaction baseline)", presidio_payload)

    print("  [PEP] Running local-LLM light-touch filter ...")
    pep_payload, _pep_trace = pep_baseline.sanitize_with_trace(
        naive_payload, profile, task, traces, agent.local
    )
    _box("PEP Output  (privacy-enhancing prompt baseline)", pep_payload)

    payloads = {
        "naive":     naive_payload,
        "privscope": ps_payload,
        "presidio":  presidio_payload,
        "pep":       pep_payload,
    }

    # ── Phase 3: CLM responses (all methods x all CLMs) ───────────────────────
    _section("PHASE 3  --  Cloud LLM responses  (all methods x all CLMs)")

    # results[method][provider] = (text, in_tok, out_tok)
    results: Dict[str, Dict[str, Tuple[str, int, int]]] = {}

    for method, payload in payloads.items():
        results[method] = {}
        print(f"\n  Querying all CLMs for [{method}] ...")
        for p, clm in clms.items():
            text, in_tok, out_tok = _query_clm(clm, payload)
            results[method][p] = (text, in_tok, out_tok)
            print(f"    [{p:<6}]  in_tok={in_tok:>4}  out_tok={out_tok:>4}")

    # Show CLM response boxes per method
    for method in payloads:
        _div(f"CLM responses  ->  {method} payload", char="-")
        for p in _PROVIDERS:
            if p in results[method]:
                text, _, _ = results[method][p]
                _box(f"[{p}]", text)

    # ── Summary: payload sizes (original) ─────────────────────────────────────
    _section("SUMMARY  --  payload lengths and reduction vs naive")
    _payload_stats(payloads)

    # ── Summary: token accounting (new) ───────────────────────────────────────
    _section("TOKEN ACCOUNTING  --  input tokens  |  output tokens per CLM")
    print(f"  In tok  = tokens in full request (system prompt + payload) per OpenAI tokenizer")
    print(f"  Out tok = tokens in CLM response")

    token_rows = []
    for method, payload in payloads.items():
        r = results[method]
        in_tok  = r.get("openai", ("", 0, 0))[1]
        oai_out = r.get("openai", ("", 0, 0))[2]
        cld_out = r.get("claude", ("", 0, 0))[2]
        gmn_out = r.get("gemini", ("", 0, 0))[2]
        token_rows.append({
            "method":  method,
            "chars":   len(payload or ""),
            "in_tok":  in_tok,
            "oai_out": oai_out,
            "cld_out": cld_out,
            "gmn_out": gmn_out,
        })
    _token_table(token_rows)

    # ── PrivScope stage detail (updated for v2) ────────────────────────────────
    if ps_trace:
        s1  = ps_trace.get("stage1", {})
        s2  = ps_trace.get("stage2", {})
        s3b = ps_trace.get("stage3b", {})
        print(f"\n  PrivScope v2 stage detail:")
        print(f"    Stage 1  -- U_loc withheld: {len(s1.get('u_loc', []))} spans  |  "
              f"U_med candidates: {len(s1.get('u_med', []))} spans")
        print(f"    Stage 2  -- kept: {len(s2.get('kept', []))}  |  "
              f"dropped: {len(s2.get('dropped', []))}")
        print(f"    Stage 3b -- {len(s3b.get('decisions', []))} abstraction decisions  "
              f"(method: {s3b.get('method', '?')})")
        if ps_trace.get("sanitized_internal_payload"):
            _box("PrivScope Internal Payload  (debug -- [TYPE] placeholders for U_loc)",
                 ps_trace["sanitized_internal_payload"])

    print(f"\n{'=' * W}\n")


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
        help="Task string. If omitted, prompted interactively.",
    )
    ap.add_argument(
        "--local-model", default=None,
        help="Local Ollama model (default: LOCAL_MODEL from config.py). "
             "Options: llama3.2, phi3, mistral, qwen2.5:7b, llama3.1:8b",
    )
    args = ap.parse_args()

    task = args.task
    if not task:
        print("\n  Enter the user task (or press Enter for the dentist example):")
        task = input("  Task: ").strip()
        if not task:
            task = "I have tooth pain and bleeding gums, book me a dentist appointment at the earliest."
            print(f"  Using default: {task}")

    run_comparison(task, local_model=args.local_model)


if __name__ == "__main__":
    main()
