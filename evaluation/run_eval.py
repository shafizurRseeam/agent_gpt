"""
evaluation/run_eval.py
════════════════════════════════════════════════════════════════════════════════
Standalone privacy evaluation runner.

Picks N tasks from task_prompts.json, generates payloads via the LC (local
model only — NO cloud calls), applies PrivacyScope and PRESIDIO, then runs
all four privacy metrics and saves results to a JSON file.

Baselines evaluated:
  naive        — raw LC output (what would be sent to cloud)
  privacyscope — after PrivScope sanitization
  presidio   — after PRESIDIO redaction

Does NOT modify working_trace.json (read-only w.r.t. state).

════════════════════════════════════════════════════════════════════════════════
HOW TO RUN  (from project root: agent_gpt/)
════════════════════════════════════════════════════════════════════════════════

  # 3 tasks (default), save to evaluation/eval_results.json
  uv run python evaluation/run_eval.py

  # 10 tasks
  uv run python evaluation/run_eval.py --n-tasks 10

  # 10 tasks, only medical domain
  uv run python evaluation/run_eval.py --n-tasks 10 --domain medical_booking

  # Custom output file
  uv run python evaluation/run_eval.py --n-tasks 5 --output evaluation/my_results.json

  # Fixed random seed for reproducibility
  uv run python evaluation/run_eval.py --n-tasks 5 --seed 42

════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import textwrap
import time
from datetime import datetime
from pathlib import Path

# ── Load .env from project root ───────────────────────────────────────────────
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    for _line in _ENV_PATH.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).resolve().parent.parent
_TASKS_PATH   = _ROOT / "task_generated" / "task_prompts.json"
_PROFILE_PATH = _ROOT / "state" / "profile_state.json"
_DEFAULT_OUT  = Path(__file__).resolve().parent / "eval_results.json"

_BASELINES = ("naive", "privacyscope", "presidio")


# ── Payload generator (LC logic, no cloud) ───────────────────────────────────

class PayloadGenerator:
    """
    Generates the naive cloud-bound payload using the local LLM, then
    applies PrivacyScope and PRESIDIO to produce sanitized variants.

    Mirrors the LC phases in HybridAgent without any cloud calls.
    """

    def __init__(self):
        from llm.local_llm import LocalLLM
        from privacy.privacyscope import PrivacyScope
        import privacy.presidio as presidio_mod
        from state.state_io import load_state

        self.local      = LocalLLM()
        self.ps         = PrivacyScope()
        self.presidio = presidio_mod
        self.state      = load_state()

    # ── Internal LC helpers (mirrors hybrid_agent) ────────────────────────────

    def _infer_preferences(self, task: str) -> str:
        traces = self.state.get("memory_traces", [])
        p      = self.state.get("user_profile", {})

        if not traces:
            return ""

        activity_lines = []
        for t in traces:
            src  = t.get("source", "")
            wf   = t.get("from_workflow", "")
            data = t.get("data", {})

            if src == "tool:book_appointment" and isinstance(data, dict):
                activity_lines.append(f'- Booked "{data.get("booked_at")}" for task: "{wf}"')
            elif src == "tool:get_location":
                activity_lines.append(f'- Location recorded: {data} (task: "{wf}")')
            elif src == "tool:get_calendar":
                free = data.get("free_slots", [])
                if free:
                    activity_lines.append(f'- Calendar: free on {", ".join(free[:2])}')
            elif isinstance(data, list):
                for item in data[:2]:
                    activity_lines.append(f'- Past data from "{wf}": {str(item)[:80]}')

        if not activity_lines:
            return ""

        activity = "\n".join(activity_lines)
        prompt = f"""You are a Local Controller scanning the user's past activity to find context relevant to the current task.

Current task: "{task}"
User location: {p.get('address', 'unknown')}

Past activity from memory:
{activity}

Identify what type of task this is, then extract ONLY facts from the activity log that directly help complete it.
- Medical/injury/appointment task → relevant: past relevant medical bookings, insurance, location, date availability
- Dental task → relevant: past dental bookings, dentist name, insurance
- Restaurant/dining task → relevant: cuisine preferences, party size, past restaurants visited
- Car/garage task → relevant: past car service bookings, vehicle info
- Any task → always relevant: user's location, calendar availability

Ignore history that does not match the task type. Be concise. Do not mention unrelated past activity.

Output 2-4 bullet points (or write "None" if nothing is relevant to this specific task type).
INFERRED CONTEXT:"""

        raw = self.local.generate(prompt).strip()
        if "INFERRED CONTEXT:" in raw:
            prefs = raw.split("INFERRED CONTEXT:")[-1].strip()
        else:
            prefs = raw

        if len(prefs) < 10 or prefs.lower().startswith("none"):
            return ""
        return prefs

    def _reason_cloud_query(self, task: str, inferred_prefs: str = "") -> tuple[str, str]:
        p      = self.state.get("user_profile", {})
        traces = self.state.get("memory_traces", [])

        availability = ""
        for t in traces:
            if t["source"] == "tool:get_calendar":
                free = t["data"].get("free_slots", [])
                availability = f"User is free on: {', '.join(free)}" if free else ""

        pref_section = (
            f"\nInferred preferences from memory:\n{inferred_prefs}"
            if inferred_prefs else ""
        )

        lc_prompt = f"""You are the Local Controller (LC) of a hybrid agent running on-device.
Your role is to be a context packer: read all available local data and assemble the richest possible cloud-bound message so the cloud assistant can complete the task end-to-end.
You are NOT a privacy filter. Privacy is handled downstream by PrivScope. Your job is to maximise task utility by over-including rather than under-including.

User task: "{task}"

Local data available:
  Name:         {p.get('name', '')}
  Age:          {p.get('age', '')}
  DOB:          {p.get('dob', '')}
  Address:      {p.get('address', '')}
  Phone:        {p.get('phone', '')}
  Email:        {p.get('email', '')}
  Insurance:    {p.get('insurance', '')} (ID: {p.get('insurance_id', '')})
  {availability}{pref_section}

Rules for the CLOUD QUERY — you MUST follow all of these:
1. Copy the user's exact symptoms and health complaints into the query verbatim. Never omit them.
2. Include full name, age, DOB, address, phone, insurance name and ID.
3. Include any past relevant bookings or medical history from the inferred context above.
4. Include availability dates so the cloud can check scheduling.
5. Do not write any privacy disclaimer or say you are withholding information. Just pack everything.

REASONING: <think through what local data is most useful to include so the cloud can complete this task>
CLOUD QUERY: <rich natural-language message to the cloud packed with symptoms, personal details, past history, and availability — include everything>"""

        response = self.local.generate(lc_prompt).strip()

        upper_resp = response.upper()
        cq_pos     = upper_resp.rfind("CLOUD QUERY:")

        if cq_pos != -1:
            reasoning_raw = response[:cq_pos].strip()
            cloud_query   = response[cq_pos + 12:].strip().strip('"').strip()
        else:
            reasoning_raw = response
            cloud_query   = ""

        blank_line = re.search(r'\n\s*\n', cloud_query)
        if blank_line:
            cloud_query = cloud_query[:blank_line.start()].strip()
        cloud_query = re.sub(
            r'\s*\n?\s*(?:This (?:query|message|briefing|request|information|context|'
            r'cloud query|natural.language)|Note:|P\.?S\.?:?|Please note)[^"]*$',
            '', cloud_query, flags=re.IGNORECASE | re.DOTALL
        ).strip().strip('"').strip()

        r_pos = reasoning_raw.upper().find("REASONING:")
        reasoning = reasoning_raw[r_pos + 10:].strip() if r_pos != -1 else reasoning_raw
        reasoning = re.sub(r'^Step\s*1\s*[-—:]+\s*Reason\s*[:\-—]?\s*', '', reasoning, flags=re.IGNORECASE).strip()
        step2 = re.search(r'\bStep\s*2\b', reasoning, re.IGNORECASE)
        if step2:
            reasoning = reasoning[:step2.start()].strip()

        if not cloud_query:
            brief_match = re.search(r'briefing[:\s]*\n*\s*"([^"]{40,})"', response, re.DOTALL | re.IGNORECASE)
            if brief_match:
                cloud_query = brief_match.group(1).strip()

        if not cloud_query:
            quote_match = re.search(r'"([^"]{100,})"', reasoning, re.DOTALL)
            if quote_match:
                cloud_query = quote_match.group(1).strip()

        if not cloud_query:
            cloud_query = (
                f"{p.get('name', 'User')} at {p.get('address', '')} "
                f"needs help: {task}. {availability}."
            )
        if not reasoning:
            reasoning = response

        return reasoning, cloud_query

    # ── Public: generate all three payloads for one task ─────────────────────

    def generate(self, task: str) -> dict:
        """
        Returns a dict with:
          naive           — raw LC cloud query
          privacyscope    — after PrivScope sanitization
          presidio      — after PRESIDIO redaction
          lc_reasoning    — LC reasoning text
          latency_lc_s    — seconds spent on LC payload generation
          latency_ps_s    — seconds spent on PrivacyScope sanitization
          latency_ner_s   — seconds spent on PRESIDIO sanitization
          latency_total_s — total wall-clock seconds for the full pipeline
        """
        p      = self.state.get("user_profile", {})
        traces = self.state.get("memory_traces", [])

        t0 = time.perf_counter()
        inferred  = self._infer_preferences(task)
        reasoning, naive = self._reason_cloud_query(task, inferred)
        t_lc = time.perf_counter() - t0

        t1 = time.perf_counter()
        ps_sanitized, _ = self.ps.sanitize_with_trace(naive, p, task, traces)
        t_ps = time.perf_counter() - t1

        t2 = time.perf_counter()
        ner_sanitized, _ = self.presidio.sanitize_with_trace(naive, p, task, traces)
        t_ner = time.perf_counter() - t2

        return {
            "naive":            naive,
            "privacyscope":     ps_sanitized,
            "presidio":       ner_sanitized,
            "lc_reasoning":     reasoning,
            "latency_lc_s":     round(t_lc,  3),
            "latency_ps_s":     round(t_ps,  3),
            "latency_ner_s":    round(t_ner, 3),
            "latency_total_s":  round(t_lc + t_ps + t_ner, 3),
        }


# ── Token counting ───────────────────────────────────────────────────────────

def _token_count(text: str) -> int:
    """Whitespace-split word count as a proxy for token count."""
    return len(text.split()) if text else 0


def _payload_reduction(naive: str, sanitized: str) -> float | None:
    """
    PR = (tokens(naive) - tokens(sanitized)) / tokens(naive)
    Returns None if naive is empty.
    """
    n = _token_count(naive)
    if n == 0:
        return None
    s = _token_count(sanitized)
    return (n - s) / n


# ── Privacy metrics ───────────────────────────────────────────────────────────

def _leaking_facts(facts: list[str], payload: str) -> list[str]:
    """Return subset of facts that appear (case-insensitive substring) in payload."""
    if not payload:
        return []
    pl = payload.lower()
    return [f for f in facts if f.lower() in pl]


def _extract_profile_facts(profile_data: dict) -> list[str]:
    profile = profile_data.get("user_profile", {}) if isinstance(profile_data, dict) else {}
    facts = []
    for v in profile.values():
        s = str(v).strip() if v is not None else ""
        if len(s) >= 3:
            facts.append(s)
    return facts


def compute_metrics(
    seed_facts:  list[str],
    resid_facts: list[str],
    payload:     str,
) -> dict:
    leaked_seed  = _leaking_facts(seed_facts,  payload)
    leaked_resid = _leaking_facts(resid_facts, payload)

    lr      = (1 if leaked_seed  else 0) if seed_facts  else None
    lratio  = len(leaked_seed)  / len(seed_facts)  if seed_facts  else None
    rlr     = 1 if leaked_resid else 0
    rlratio = len(leaked_resid) / len(resid_facts) if resid_facts else 0.0

    return {
        "LR":           lr,
        "LRatio":       lratio,
        "RLR":          rlr,
        "RLRatio":      rlratio,
        "leaked_seed":  leaked_seed,
        "leaked_resid": leaked_resid,
    }


# ── Task selection ────────────────────────────────────────────────────────────

def select_tasks(
    tasks_path: Path,
    n_tasks:    int,
    domain:     str | None,
    rng:        random.Random,
) -> list[dict]:
    data = json.loads(tasks_path.read_text())
    pool = data["task_prompts"]

    if domain:
        pool = [t for t in pool if t.get("domain") == domain]
        if not pool:
            raise SystemExit(f"No tasks found for domain '{domain}'")

    return rng.sample(pool, k=min(n_tasks, len(pool)))


# ── Main runner ───────────────────────────────────────────────────────────────

def _box(title: str, lines: list[str]) -> None:
    print(f"\n  ┌─ {title} {'─' * max(0, 50 - len(title))}")
    for line in lines:
        for seg in (textwrap.wrap(line, width=70) if line.strip() else [""]):
            print(f"  │  {seg}")
    print(f"  └{'─' * 54}")


def run(
    tasks_path: Path,
    n_tasks:    int,
    domain:     str | None,
    rng_seed:   int,
    output:     Path,
) -> None:

    print(f"\n{'═' * 65}")
    print(f"  PRIVACY EVALUATION  —  {n_tasks} tasks  |  domain: {domain or 'all'}")
    print(f"  Baselines: naive, privacyscope, presidio")
    print(f"  Matching : case-insensitive substring")
    print(f"{'═' * 65}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    rng = random.Random(rng_seed)
    tasks = select_tasks(tasks_path, n_tasks, domain, rng)

    profile_data  = json.loads(_PROFILE_PATH.read_text()) if _PROFILE_PATH.exists() else {}
    profile_facts = _extract_profile_facts(profile_data)

    print(f"  Loaded {len(tasks)} tasks from {tasks_path}")
    print(f"  Profile facts: {len(profile_facts)} values from user_profile")

    # ── Init generator (loads local LLM) ─────────────────────────────────────
    print("\n  Initialising local model…")
    gen = PayloadGenerator()
    print(f"  Local model: {gen.local.model}\n")

    # ── Per-task accumulators ─────────────────────────────────────────────────
    trace_facts: list[str] = []   # S_t^trace, grows after each task

    accum = {
        b: {"lr_hits": 0, "lr_ratio_sum": 0.0, "lr_n": 0,
            "rlr_hits": 0, "rlr_ratio_sum": 0.0, "rlr_n": 0,
            "pr_sum": 0.0, "pr_n": 0,
            "san_lat_sum": 0.0, "san_lat_n": 0}
        for b in _BASELINES
    }

    task_records = []

    for i, task_entry in enumerate(tasks, 1):
        task_prompt  = task_entry["prompt"]
        seed_facts   = [str(x).strip() for x in task_entry.get("sensitive_info", []) if str(x).strip()]
        resid_facts  = profile_facts + trace_facts   # S^profile ∪ S_t^trace

        print(f"{'─' * 65}")
        print(f"  Task {i}/{len(tasks)}  [{task_entry.get('domain', '?')}]  {task_entry.get('seed_id', '?')}")
        print(f"  Prompt: {task_prompt[:80]}")
        print(f"  Seed facts ({len(seed_facts)}): {seed_facts}")
        print(f"  Residual pool: {len(resid_facts)} facts ({len(profile_facts)} profile + {len(trace_facts)} trace)")

        # ── Generate payloads ─────────────────────────────────────────────────
        print("\n  Generating payloads via LC…")
        payloads = gen.generate(task_prompt)

        _box("Naive payload", payloads["naive"].splitlines())
        _box("PrivacyScope payload", payloads["privacyscope"].splitlines())
        _box("PRESIDIO payload", payloads["presidio"].splitlines())

        # ── Compute metrics ───────────────────────────────────────────────────
        metrics_per_baseline = {}
        for b in _BASELINES:
            m = compute_metrics(seed_facts, resid_facts, payloads[b])

            # Payload reduction vs naive (naive vs itself = 0%)
            pr = _payload_reduction(payloads["naive"], payloads[b])
            m["PR"]            = pr
            m["tokens_naive"]  = _token_count(payloads["naive"])
            m["tokens_payload"] = _token_count(payloads[b])
            metrics_per_baseline[b] = m

            if seed_facts:
                accum[b]["lr_hits"]      += (m["LR"] or 0)
                accum[b]["lr_ratio_sum"] += (m["LRatio"] or 0.0)
                accum[b]["lr_n"]         += 1

            accum[b]["rlr_hits"]      += m["RLR"]
            accum[b]["rlr_ratio_sum"] += m["RLRatio"]
            accum[b]["rlr_n"]         += 1

            if pr is not None:
                accum[b]["pr_sum"] += pr
                accum[b]["pr_n"]   += 1

            san_lat = {
                "naive":        0.0,
                "privacyscope": payloads["latency_ps_s"],
                "presidio":   payloads["latency_ner_s"],
            }.get(b, 0.0)
            accum[b]["san_lat_sum"] += san_lat
            accum[b]["san_lat_n"]   += 1

        # Print per-task metric table
        print(f"\n  Latency — LC: {payloads['latency_lc_s']:.2f}s  "
              f"PS: {payloads['latency_ps_s']:.2f}s  "
              f"NER: {payloads['latency_ner_s']:.2f}s  "
              f"total: {payloads['latency_total_s']:.2f}s")
        print(f"\n  {'Baseline':<14}  {'LR':>4}  {'LRatio':>7}  {'RLR':>4}  {'RLRatio':>8}  {'PR':>7}  Tok(naive→out)")
        print(f"  {'─'*14}  {'─'*4}  {'─'*7}  {'─'*4}  {'─'*8}  {'─'*7}  {'─'*13}")
        for b in _BASELINES:
            m = metrics_per_baseline[b]
            lr_s = f"{m['LR']}"      if m['LR']      is not None else " —"
            lr_r = f"{m['LRatio']*100:5.1f}%" if m['LRatio'] is not None else "   —   "
            pr_s = f"{m['PR']*100:5.1f}%" if m['PR'] is not None else "  —  "
            print(
                f"  {b:<14}  {lr_s:>4}  {lr_r:>7}  "
                f"{m['RLR']:>4}  {m['RLRatio']*100:>7.1f}%  "
                f"{pr_s:>7}  "
                f"{m['tokens_naive']} → {m['tokens_payload']}"
            )

        task_records.append({
            "task_id":        i,
            "seed_id":        task_entry.get("seed_id", ""),
            "variant_id":     task_entry.get("variant_id", ""),
            "domain":         task_entry.get("domain", ""),
            "prompt":         task_prompt,
            "sensitive_info": seed_facts,
            "n_resid_facts":  len(resid_facts),
            "latency": {
                "lc_s":    payloads["latency_lc_s"],
                "ps_s":    payloads["latency_ps_s"],
                "ner_s":   payloads["latency_ner_s"],
                "total_s": payloads["latency_total_s"],
            },
            "payloads": {
                "naive":        payloads["naive"],
                "privacyscope": payloads["privacyscope"],
                "presidio":   payloads["presidio"],
                "lc_reasoning": payloads["lc_reasoning"],
            },
            "metrics": metrics_per_baseline,
        })

        # Accumulate trace for next task
        trace_facts.extend(seed_facts)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  SUMMARY  —  {len(tasks)} tasks evaluated")
    print(f"{'═' * 65}")

    summary: dict[str, dict] = {}

    def _f(v): return f"{v*100:5.1f}%" if v is not None else "  —  "

    print(f"\n  {'Baseline':<14}  {'LR':>7}  {'LRatio':>7}  {'RLR':>7}  {'RLRatio':>8}  {'N(LR)':>5}  {'N(RLR)':>6}  {'PR':>7}  {'Lat(s)':>7}")
    print(f"  {'─'*14}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*7}")

    for b in _BASELINES:
        a      = accum[b]
        lr_n   = a["lr_n"]
        rlr_n  = a["rlr_n"]
        pr_n   = a["pr_n"]
        lat_n  = a["san_lat_n"]
        lr     = a["lr_hits"]      / lr_n   if lr_n   else None
        lratio = a["lr_ratio_sum"] / lr_n   if lr_n   else None
        rlr    = a["rlr_hits"]     / rlr_n  if rlr_n  else None
        rr     = a["rlr_ratio_sum"]/ rlr_n  if rlr_n  else None
        pr     = a["pr_sum"]       / pr_n   if pr_n   else None
        lat    = a["san_lat_sum"]  / lat_n  if lat_n  else 0.0

        summary[b] = {
            "LR": lr, "LRatio": lratio, "RLR": rlr, "RLRatio": rr,
            "PR": pr, "mean_san_latency_s": round(lat, 3),
            "n_lr": lr_n, "n_rlr": rlr_n,
        }

        lat_s = f"{lat:.2f}s"
        print(
            f"  {b:<14}  {_f(lr):>7}  {_f(lratio):>7}  "
            f"{_f(rlr):>7}  {_f(rr):>8}  {lr_n:>5}  {rlr_n:>6}  "
            f"{_f(pr):>7}  {lat_s:>7}"
        )

    print()

    # ── Save results ──────────────────────────────────────────────────────────
    result_doc = {
        "generated_at":       datetime.now().isoformat(),
        "n_tasks":            len(tasks),
        "domain_filter":      domain,
        "rng_seed":           rng_seed,
        "baselines":          list(_BASELINES),
        "matching":           "case-insensitive substring",
        "token_counting":     "whitespace-split word count",
        "summary":            summary,
        "tasks":              task_records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result_doc, indent=2, ensure_ascii=False))
    print(f"  Results saved to: {output}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run privacy evaluation: generate payloads, compute metrics, save results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--n-tasks", type=int, default=3,
        help="Number of tasks to evaluate (default: 3)",
    )
    ap.add_argument(
        "--domain", default=None,
        help="Filter tasks to a specific domain (e.g. medical_booking)",
    )
    ap.add_argument(
        "--tasks", default=str(_TASKS_PATH),
        help=f"Path to task_prompts.json (default: {_TASKS_PATH})",
    )
    ap.add_argument(
        "--output", default=str(_DEFAULT_OUT),
        help=f"Output JSON file (default: {_DEFAULT_OUT})",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for task selection (default: 42)",
    )
    args = ap.parse_args()

    run(
        tasks_path = Path(args.tasks),
        n_tasks    = args.n_tasks,
        domain     = args.domain,
        rng_seed   = args.seed,
        output     = Path(args.output),
    )


if __name__ == "__main__":
    main()
