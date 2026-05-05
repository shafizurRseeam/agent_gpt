"""
evaluation/run_evaluation.py
════════════════════════════════════════════════════════════════════════════════
Full privacy + utility evaluation pipeline for PrivScope.

Implements every metric from the paper's metrics section:

  LR           — Leakage Rate                  (current-task facts)
  LRatio       — Leakage Ratio                 (current-task facts)
  RLR          — Residual Leakage Rate         (S^prof ∪ S^hist)
  RLRatio      — Residual Leakage Ratio        (S^prof ∪ S^hist)
  LRatio^cur   — Source-specific leakage ratio  current-task facts
  LRatio^prof  — Source-specific leakage ratio  profile facts
  LRatio^hist  — Source-specific leakage ratio  history facts
  URR          — Utility Retention Rate         (candidate overlap vs naive)
  TSR          — Task Success Rate              (GPT-4o-mini judge, domain match binary)
  PR           — Payload Reduction              (fractional token reduction)
  Latency      — Sanitization latency (s)       (transform time only, 0 for naive)

Sensitive fact sets (per paper notation):
  S^cur_t   — sensitive_info labels from the current task seed
  S^prof    — user_profile values from state/profile_state.json
  S^hist_t  — ∪_{k<t} S^cur_k  (accumulated from prior tasks in order)
              ∪ data-field values extracted from state/working_trace.json
  S^res_t   — S^prof ∪ S^hist_t

Usage (from project root agent_gpt/):
    uv run python evaluation/run_evaluation.py
    uv run python evaluation/run_evaluation.py --n-tasks 20
    uv run python evaluation/run_evaluation.py --n-tasks 10 --out evaluation/my_results.json

Results are written to evaluation/run_evaluation.json (or --out path).
════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.stdout.reconfigure(encoding="utf-8")

# ── Add project root to path so relative imports resolve ─────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from agents.hybrid_agent import HybridAgent
from privacy.privscope   import PrivScope
import privacy.presidio  as presidio_baseline
import privacy.pep       as pep_baseline
from llm.cloud_router    import CloudLLM

# ── Canonical file paths ──────────────────────────────────────────────────────
_TASKS_FILE        = _ROOT / "task_generated" / "task_prompts.json"
_PROFILE_FILE      = _ROOT / "state" / "profile_state.json"
_BACKSTORY_FILE    = _ROOT / "state" / "memory_backstory.json"   # fixed pre-memory, never reset
_TRACE_FILE        = _ROOT / "state" / "working_trace.json"      # grows during eval, reset between runs
_RESULTS_FILE      = Path(__file__).resolve().parent / "run_evaluation.json"

_METHODS          = ("naive", "privscope", "presidio", "pep")
_CLOUD_PROVIDERS  = ("openai", "claude", "gemini")

_CLM_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Return a numbered list of up to 5 local service providers matching the request. "
    "For each provider include exactly: name, address, phone. Be concise. "
    "Format each entry as:\n"
    "1. <Name>\n   Address: <address>\n   Phone: <phone>\n"
)


def _cloud_call(clm: CloudLLM, payload: str) -> Tuple[str, int]:
    """Send payload to a CLM and return (response_text, output_token_count)."""
    try:
        text, _, out_tok = clm.chat_with_usage([
            {"role": "system", "content": _CLM_SYSTEM_PROMPT},
            {"role": "user",   "content": payload},
        ])
        return text, out_tok
    except Exception as e:
        return f"[CLM error: {e}]", 0


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════════

def _load_json(path: Path, required: bool = True):
    if not path.exists():
        if required:
            sys.exit(f"[ERROR] Required file not found: {path}")
        return None
    raw = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            raw = path.read_text(encoding=enc)
            break
        except (UnicodeDecodeError, OSError):
            continue
    if raw is None:
        sys.exit(f"[ERROR] Could not read {path} with any supported encoding.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.exit(f"[ERROR] Cannot parse JSON in {path}: {exc}")


def load_tasks(path: Path) -> List[dict]:
    """Load task_prompts.json; expect {"task_prompts": [...]}."""
    data = _load_json(path)
    if not isinstance(data, dict) or "task_prompts" not in data:
        sys.exit(f"[ERROR] {path} must contain a top-level 'task_prompts' list.")
    tasks = data["task_prompts"]
    if not tasks:
        sys.exit(f"[ERROR] 'task_prompts' is empty in {path}")
    return tasks


_TIER1_FIELDS = {
    "name", "dob", "phone", "email", "ssn", "passport_number",
    "driver_license", "insurance_id", "patient_id", "credit_card",
    "membership_id", "frequent_flyer_number", "hotel_loyalty_id", "vehicle_plate",
}
_TIER2_FIELDS = {"address", "insurance"}


def load_profile_facts(path: Path):
    """
    Extract all non-trivial string values from user_profile.
    Returns (facts: List[str], weight_map: Dict[str, float]).
    Tier 1 direct identifiers → w=1.0, Tier 2 quasi-identifiers → w=0.6,
    Tier 3 contextual attributes → w=0.3.
    """
    data    = _load_json(path)
    profile = data.get("user_profile", {}) if isinstance(data, dict) else {}
    facts, weight_map = [], {}
    for k, v in profile.items():
        s = str(v).strip() if v is not None else ""
        if len(s) >= 3:
            facts.append(s)
            k_lower = k.lower()
            if k_lower in _TIER1_FIELDS:
                weight_map[s] = 1.0
            elif k_lower in _TIER2_FIELDS:
                weight_map[s] = 0.6
            else:
                weight_map[s] = 0.3
    return facts, weight_map


def load_trace_facts() -> List[str]:
    """
    Extract sensitive fact strings from memory traces.
    Loads backstory first (fixed, never reset), then working_trace (grows during eval).
    Only 'data' payload of each entry is extracted to avoid circular contamination.
    """
    def _facts_from_file(p: Path) -> List[str]:
        data = _load_json(p, required=False)
        if data is None:
            return []
        entries = data.get("memory_traces", []) if isinstance(data, dict) else []
        out = []
        for entry in entries:
            if isinstance(entry, dict):
                out.extend(_extract_strings(entry.get("data")))
        return out

    all_facts = _facts_from_file(_BACKSTORY_FILE)   # pre-existing history — never reset
    all_facts += _facts_from_file(_TRACE_FILE)       # runtime accumulation — reset between runs

    seen: set = set()
    deduped: List[str] = []
    for f in all_facts:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    return deduped


def _extract_strings(obj, min_len: int = 3) -> List[str]:
    """Recursively extract non-trivial strings from any JSON-compatible object."""
    out = []
    if isinstance(obj, str):
        s = obj.strip()
        if len(s) >= min_len:
            out.append(s)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_extract_strings(v, min_len))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_extract_strings(item, min_len))
    return out


# ════════════════════════════════════════════════════════════════════════════════
# TRACE FILE HELPERS  (path-aware; do not rely on state_io hardcoded paths)
# ════════════════════════════════════════════════════════════════════════════════

def _append_trace(entry: dict) -> None:
    """Append one entry to working_trace.json (runtime accumulation only)."""
    data: dict = {"memory_traces": []}
    if _TRACE_FILE.exists():
        try:
            data = json.loads(_TRACE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    data.setdefault("memory_traces", []).append(entry)
    _TRACE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _reset_working_trace() -> None:
    """Clear working_trace.json only. Backstory is never touched."""
    _TRACE_FILE.write_text(json.dumps({"memory_traces": []}, indent=2), encoding="utf-8")


def _reload_agent_traces(agent: HybridAgent) -> None:
    """
    Refresh agent.state['memory_traces'] from:
      1. memory_backstory.json  — fixed pre-existing history, never reset
      2. working_trace.json     — runtime accumulation, reset between eval runs
    """
    traces: list = []
    backstory = _load_json(_BACKSTORY_FILE, required=False)
    if backstory:
        traces.extend(backstory.get("memory_traces", []))
    runtime = _load_json(_TRACE_FILE, required=False)
    if runtime:
        traces.extend(runtime.get("memory_traces", []))
    agent.state["memory_traces"] = traces


# ════════════════════════════════════════════════════════════════════════════════
# TOKEN COUNTING  (for PR = Payload Reduction)
# ════════════════════════════════════════════════════════════════════════════════

try:
    import tiktoken as _tiktoken
    _enc = _tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text)) if text else 0
    _TOK_SOURCE = "tiktoken cl100k_base"
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text.split()) if text else 0
    _TOK_SOURCE = "word count (install tiktoken for accurate counts)"


# ════════════════════════════════════════════════════════════════════════════════
# LEAKAGE DETECTION
# ════════════════════════════════════════════════════════════════════════════════

def leaking_facts(facts: List[str], payload: str) -> List[str]:
    """
    Return the subset of facts that appear verbatim (case-insensitive
    substring) in the payload.

    Per the paper: coarse abstractions count as leakage only when they
    reveal the protected fact at the annotated specificity level.
    Substring matching correctly handles this for PrivScope — an
    abstraction like "skin condition" does NOT match "eczema", so
    PrivScope correctly avoids leakage credit for well-abstracted spans.
    """
    if not payload or not facts:
        return []
    pl = payload.lower()
    return [f for f in facts if len(f) >= 3 and f.lower() in pl]


# ════════════════════════════════════════════════════════════════════════════════
# ILS AND WLS
# ════════════════════════════════════════════════════════════════════════════════

def injection_leakage_severity(
    state_facts: List[str],
    payload:     str,
    request:     str,
) -> Optional[float]:
    """ILS = |Reveal(S^state, P^m) \ Reveal(S^state, r_t)| / |S^state|"""
    if not state_facts:
        return None
    already  = set(leaking_facts(state_facts, request))
    in_pay   = set(leaking_facts(state_facts, payload))
    injected = in_pay - already
    return round(len(injected) / len(state_facts), 4)


def weighted_leakage_severity(
    facts:      List[str],
    payload:    str,
    weight_map: Dict[str, float],
    default_w:  float = 0.6,
) -> Optional[float]:
    """WLS = Σ w(f) for f in Reveal(S,P) / Σ w(f) for f in S"""
    if not facts:
        return None
    leaked    = leaking_facts(facts, payload)
    numerator = sum(weight_map.get(f, default_w) for f in leaked)
    denom     = sum(weight_map.get(f, default_w) for f in facts)
    return round(numerator / denom, 4) if denom else 0.0


# ════════════════════════════════════════════════════════════════════════════════
# CANDIDATE PARSING  (for URR = Utility Retention Rate)
# ════════════════════════════════════════════════════════════════════════════════

def parse_candidates(clm_response: str) -> List[str]:
    """
    Parse CLM numbered-list response into a list of candidate/provider names.
    Handles: "1. Name", "1) **Name**", "**1. Name**", "**1. **Name****" variants.
    Some CLMs (e.g. Gemini) bold the entire list entry including the number,
    so the leading \*{0,2} is required to match those lines.
    Returns lower-cased names for fuzzy comparison.
    """
    names: List[str] = []
    for line in clm_response.splitlines():
        m = re.match(r"^\*{0,2}\d+[.)]\s+\*{0,2}(.+?)\*{0,2}\s*$", line.strip())
        if m:
            name = m.group(1).strip(" -:").lower()
            if name:
                names.append(name)
    return names


def match_candidates(naive: List[str], method: List[str]) -> int:
    """
    Count how many naive candidates have a match in the method candidate list.
    Uses a 10-character prefix key to tolerate minor surface variations
    (punctuation, honorifics) in provider names.
    Match(Y_naive, Y_method) in paper notation.
    """
    if not naive:
        return 0
    count = 0
    for n in naive:
        key = n[:10].strip()
        if key and any(key in m or m[:10] in n for m in method):
            count += 1
    return count


# ════════════════════════════════════════════════════════════════════════════════
# TSR JUDGE  (majority-vote ensemble: GPT-4o-mini + Claude + Gemini)
# ════════════════════════════════════════════════════════════════════════════════

_TSR_SYSTEM = """\
You are a task success evaluator for an AI agent system.

You will receive:
  1. The original user request.
  2. A cloud assistant's response containing a list of candidates (e.g. providers,
     venues, services, products, or options).

Your job: determine whether the returned candidates are in the correct domain and
are actionable for completing the original user request.

Domain match is strict — every returned candidate must belong to the same category
the user asked for. Examples:
  - User asks for flower shops → candidates must be flower shops, NOT chocolate shops.
  - User asks for a water park → candidates must be water parks, NOT amusement parks.
  - User asks for a cardiologist → candidates must be cardiologists, NOT general practitioners.

Output exactly the digit 1 or 0 — nothing else, no explanation."""

_TSR_PROMPT = """\
Original user request:
{r_t}

Cloud assistant response:
{response}

Are the candidates returned by the cloud assistant in the correct domain and \
actionable for the original user request?

  1 = yes: all candidates are in the correct domain and can be used to complete the task
  0 = no:  candidates are off-domain, wrong category, empty, or too generic to be useful

Answer with 1 or 0:"""


def _single_judge_call(clm, messages: list) -> int:
    """Query one judge LLM and return 0 or 1. Returns 1 on any failure."""
    try:
        raw, _, _ = clm.chat_with_usage(messages)
        m = re.search(r"[01]", raw.strip())
        return int(m.group()) if m else 1
    except Exception:
        return 1


def judge_tsr(judge_instances: dict, r_t: str, clm_response: str) -> int:
    """
    J_judge(r_t, Y_t^m) → {0, 1}.

    Queries all available judge LLMs independently (GPT-4o-mini, Claude, Gemini)
    and returns 1 if a strict majority (≥2 of 3) vote 1, else 0.
    This eliminates self-judging bias — GPT's response is judged by Claude and
    Gemini as well as GPT itself.
    Falls back to 1 (optimistic) if no judges are available.
    """
    if not judge_instances:
        return 1

    messages = [
        {"role": "system", "content": _TSR_SYSTEM},
        {"role": "user",   "content": _TSR_PROMPT.format(
            r_t      = r_t.strip(),
            response = clm_response.strip()[:2000],
        )},
    ]

    votes = [_single_judge_call(clm, messages) for clm in judge_instances.values()]
    return 1 if sum(votes) >= 2 else 0


# ════════════════════════════════════════════════════════════════════════════════
# NAIVE PAYLOAD CONSTRUCTION
# ════════════════════════════════════════════════════════════════════════════════

def build_naive_payload(agent: HybridAgent, task_prompt: str) -> Tuple[str, str]:
    """
    Construct P_t: the LC-enriched naive cloud-bound payload.
    Returns (naive_payload, lc_reasoning).
    Sanitization latency for the naive method is defined as 0 because
    no transformation is applied — P_t^naive = P_t.
    """
    inferred_prefs = agent._lc_infer_preferences(task_prompt)
    reasoning, naive_payload = agent._lc_reason_cloud_query(task_prompt, inferred_prefs)
    return naive_payload, reasoning


# ════════════════════════════════════════════════════════════════════════════════
# METHOD APPLICATION  (sanitization — latency measured here only)
# ════════════════════════════════════════════════════════════════════════════════

def apply_method(
    method:        str,
    naive_payload: str,
    task_prompt:   str,
    profile:       dict,
    traces:        list,
    ps:            PrivScope,
    local_llm,
) -> Tuple[str, float, dict]:
    """
    Transform naive_payload P_t into P_t^m using sanitization method m.
    Returns (sanitized_payload, latency_seconds, trace_dict).

    Latency is measured as wall-clock time for the transform only.
    It excludes: (a) naive payload construction, (b) CLM response time.
    Naive method returns latency=0.0 and passes payload through unchanged.
    """
    if method == "naive":
        return naive_payload, 0.0, {"method": "naive"}

    t0 = time.perf_counter()

    if method == "privscope":
        payload, trace = ps.sanitize_with_trace(naive_payload, profile, task_prompt, traces)
    elif method == "presidio":
        payload, trace = presidio_baseline.sanitize_with_trace(naive_payload, profile, task_prompt, traces)
    elif method == "pep":
        payload, trace = pep_baseline.sanitize_with_trace(naive_payload, profile, task_prompt, traces, local_llm)
    else:
        raise ValueError(f"Unknown method: {method}")

    return payload, time.perf_counter() - t0, trace


# ════════════════════════════════════════════════════════════════════════════════
# PER-TASK EVALUATION
# ════════════════════════════════════════════════════════════════════════════════

def evaluate_task(
    task_idx:        int,
    task:            dict,
    agent:           HybridAgent,
    ps:              PrivScope,
    profile_facts:   List[str],
    history_facts:   List[str],
    trace_facts:     List[str],
    cloud_instances: Dict[str, CloudLLM],
    cloud_mode:      str,
    judge_instances: Dict[str, "CloudLLM"],
    weight_map:      Dict[str, float],
    n_clm_calls:     int  = 2,
) -> dict:
    """
    Full evaluation for one task across all methods.

    Two-pass design:
      Pass 1 — apply all sanitization methods and collect CLM responses.
               PrivScope's g_t is captured here for reuse across methods.
      Pass 2 — compute all metrics (leakage, URR, TSR, PR) now that
               g_t and naive candidates are known.

    Fact sets constructed at evaluation time for task t:
      S^hist_t = history_facts (∪ S^cur_k, k<t) ∪ trace_facts
      S^res_t  = profile_facts ∪ S^hist_t
    """
    task_prompt = task.get("prompt", "").strip()
    cur_facts   = [str(f).strip() for f in task.get("sensitive_info", []) if str(f).strip()]
    domain      = task.get("domain", "")
    seed_id     = task.get("seed_id",    f"seed_{task_idx:04d}")
    variant_id  = task.get("variant_id", 0)

    # Build S^hist_t and S^res_t for this task
    full_history   = list(dict.fromkeys(history_facts + trace_facts))
    residual_facts = list(dict.fromkeys(profile_facts + full_history))

    print(f"  [{task_idx+1:>4}] {seed_id} v{variant_id}  {domain}", end="  ", flush=True)

    # ── Build naive payload P_t ───────────────────────────────────────────────
    naive_payload, reasoning = build_naive_payload(agent, task_prompt)
    profile_dict  = agent.state.get("user_profile", {})
    memory_traces = agent.state.get("memory_traces", [])
    naive_tokens  = count_tokens(naive_payload)

    # ── Pass 1: apply methods, collect payloads and CLM responses ─────────────
    # Methods run in order; PrivScope is run second to capture g_t early.
    raw: Dict[str, dict] = {}
    g_t = ""

    for method in _METHODS:
        payload, latency, trace = apply_method(
            method, naive_payload, task_prompt,
            profile_dict, memory_traces, ps, agent.local,
        )

        # g_t is derived from r_t (original request) by PrivScope Stage 2.
        # It is reused for TSR judging across all methods because it
        # characterises the delegated task, not the specific payload.
        if method == "privscope" and not g_t:
            g_t = (trace.get("stage2") or {}).get("task_frame", "") or ""

        # Make n_clm_calls independent CLM calls per payload for URR/TSR averaging.
        # Each call is paired with the corresponding naive call of the same index
        # so trial-level hallucination variance cancels in the average.
        call_responses_list: List[Dict[str, str]]       = []
        call_candidates_list: List[Dict[str, List[str]]] = []
        call_tokens_list: List[Dict[str, int]]           = []
        for _ in range(n_clm_calls):
            _results = {prov: _cloud_call(clm, payload) for prov, clm in cloud_instances.items()}
            cr = {prov: v[0] for prov, v in _results.items()}
            ct = {prov: v[1] for prov, v in _results.items()}
            cc = {prov: parse_candidates(r) for prov, r in cr.items()}
            call_responses_list.append(cr)
            call_candidates_list.append(cc)
            call_tokens_list.append(ct)

        primary      = next(iter(cloud_instances)) if cloud_instances else ""
        primary_resp = call_responses_list[0][primary] if call_responses_list else ""
        clm_responses  = call_responses_list[0]  if call_responses_list else {}
        clm_candidates = call_candidates_list[0] if call_candidates_list else {}
        clm_tokens     = call_tokens_list[0]     if call_tokens_list     else {}

        raw[method] = {
            "payload":              payload,
            "latency":              latency,
            "trace":                trace,
            "clm_response":         primary_resp,
            "clm_responses":        clm_responses,
            "clm_response_tokens":  clm_tokens,
            "candidates":           clm_candidates.get(primary, []),
            "clm_candidates":       clm_candidates,
            "call_responses_list":  call_responses_list,
            "call_candidates_list": call_candidates_list,
            "call_tokens_list":     call_tokens_list,
        }
        print("." * (len(cloud_instances) * n_clm_calls), end="", flush=True)

    # Fall back to task_prompt if PrivScope did not produce a task frame
    if not g_t:
        g_t = task_prompt

    # naive call-candidate lists — URR is paired: naive call i vs method call i
    naive_call_candidates = raw["naive"]["call_candidates_list"]

    # ── Pass 2: compute all metrics ───────────────────────────────────────────
    method_results: Dict[str, dict] = {}

    for method, r in raw.items():
        payload      = r["payload"]
        clm_response = r["clm_response"]   # primary provider
        candidates   = r["candidates"]     # primary provider

        # ── Leakage detection (case-insensitive substring) ────────────────────
        leaked_cur  = leaking_facts(cur_facts,      payload)  # from S^cur_t
        leaked_prof = leaking_facts(profile_facts,  payload)  # from S^prof
        leaked_hist = leaking_facts(full_history,   payload)  # from S^hist_t
        leaked_res  = leaking_facts(residual_facts, payload)  # from S^res_t

        n_cur  = len(cur_facts)
        n_prof = len(profile_facts)
        n_hist = len(full_history)
        n_res  = len(residual_facts)

        # LR: 1[|LeakSet(S^cur, P_t^m)| > 0]
        LR = 1 if leaked_cur else 0

        # LRatio: |LeakSet(S^cur, P_t^m)| / |S^cur|
        LRatio = round(len(leaked_cur) / n_cur, 4) if n_cur else None

        # RLR: 1[|LeakSet(S^res, P_t^m)| > 0]
        RLR = 1 if leaked_res else 0

        # RLRatio: |LeakSet(S^res, P_t^m)| / |S^res|
        RLRatio = round(len(leaked_res) / n_res, 4) if n_res else 0.0

        # Source-specific LRatio^a for a ∈ {cur, prof, hist}
        LRatio_cur  = round(len(leaked_cur)  / n_cur,  4) if n_cur  else None
        LRatio_prof = round(len(leaked_prof) / n_prof, 4) if n_prof else None
        LRatio_hist = round(len(leaked_hist) / n_hist, 4) if n_hist else None

        # ILS: state facts injected by LC beyond what user stated in r_t
        ILS = injection_leakage_severity(residual_facts, payload, task_prompt)

        # WLS: weighted leakage over full sensitive set (Tier1=1.0, Tier2=0.6, Tier3=0.3)
        all_facts = list(dict.fromkeys(cur_facts + residual_facts))
        WLS = weighted_leakage_severity(all_facts, payload, weight_map)

        # PR: (tok(P_t) - tok(P_t^m)) / tok(P_t)
        tok_method = count_tokens(payload)
        PR = round((naive_tokens - tok_method) / naive_tokens, 4) if naive_tokens > 0 else 0.0

        # URR and TSR — averaged over n_clm_calls paired (naive_i, method_i) calls.
        # URR_i = Match(naive_cands_i, method_cands_i) / |naive_cands_i|
        # TSR_i = J_local(r_t, g_t, clm_response_i)   (binary 0/1 per call)
        # Final URR/TSR = mean over valid calls.
        URR_per: Dict[str, Optional[float]] = {}
        TSR_per: Dict[str, Optional[float]] = {}
        for prov in cloud_instances:
            urr_vals: List[float] = []
            tsr_vals: List[float] = []
            for ci in range(n_clm_calls):
                naive_cands_ci = (
                    naive_call_candidates[ci].get(prov, [])
                    if ci < len(naive_call_candidates) else []
                )
                if method == "naive":
                    urr_vals.append(1.0)
                elif naive_cands_ci:
                    method_cands_ci = (
                        r["call_candidates_list"][ci].get(prov, [])
                        if ci < len(r["call_candidates_list"]) else []
                    )
                    matched = match_candidates(naive_cands_ci, method_cands_ci)
                    urr_vals.append(round(matched / len(naive_cands_ci), 4))
                # else: naive call ci produced no parseable candidates — skip this trial

                prov_resp_ci = (
                    r["call_responses_list"][ci].get(prov, "")
                    if ci < len(r["call_responses_list"]) else ""
                )
                if prov_resp_ci:
                    tsr_vals.append(judge_tsr(judge_instances, task_prompt, prov_resp_ci))

            URR_per[prov] = round(sum(urr_vals) / len(urr_vals), 4) if urr_vals else None
            TSR_per[prov] = round(sum(tsr_vals) / len(tsr_vals), 4) if tsr_vals else None

        # Expose as scalar when single-provider, dict when multi-provider
        if cloud_mode == "all":
            URR = URR_per
            TSR = TSR_per
        else:
            prov = next(iter(cloud_instances))
            URR  = URR_per[prov]
            TSR  = TSR_per[prov]

        method_results[method] = {
            "method":                  method,
            "payload":                 payload,
            "clm_response":            clm_response,
            "clm_responses":           r.get("clm_responses", {}),
            "clm_response_tokens":     r.get("clm_response_tokens", {}),
            "sanitization_latency_s":  round(r["latency"], 6),
            "token_count":             tok_method,
            # leaked fact lists (for inspection)
            "leaked_cur":              leaked_cur,
            "leaked_prof":             leaked_prof,
            "leaked_hist":             leaked_hist,
            "leaked_res":              leaked_res,
            # per-task metric values
            "LR":                      LR,
            "LRatio":                  LRatio,
            "RLR":                     RLR,
            "RLRatio":                 RLRatio,
            "ILS":                     ILS,
            "WLS":                     WLS,
            "LRatio_cur":              LRatio_cur,
            "LRatio_prof":             LRatio_prof,
            "LRatio_hist":             LRatio_hist,
            "URR":                     URR,
            "TSR":                     TSR,
            "PR":                      PR,
        }

    print(" ✓")

    return {
        "task_idx":          task_idx,
        "seed_id":           seed_id,
        "variant_id":        variant_id,
        "domain":            domain,
        "prompt":            task_prompt,
        # sensitive fact sets at time of evaluation
        "cur_facts":         cur_facts,
        "profile_facts":     profile_facts,
        "history_facts":     full_history,
        "n_cur":             len(cur_facts),
        "n_prof":            len(profile_facts),
        "n_hist":            len(full_history),
        "n_res":             len(residual_facts),
        # naive payload and metadata
        "naive_payload":     naive_payload,
        "naive_token_count": naive_tokens,
        "lc_reasoning":      reasoning,
        "task_frame_g_t":    g_t,
        # per-method results
        "methods":           method_results,
    }


# ════════════════════════════════════════════════════════════════════════════════
# AGGREGATE METRICS  (dataset-level averages)
# ════════════════════════════════════════════════════════════════════════════════

def _mean(vals: list) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def compute_aggregates(results: List[dict]) -> dict:
    """
    Compute dataset-level averages for every metric and method.
    URR and TSR are stored per-CLM when cloud_mode='all'.
    Metrics undefined for a task are excluded rather than counted as 0.
    """
    # Detect whether URR/TSR are dicts (all-mode) or scalars (single-mode)
    _sample_method = results[0]["methods"].get("naive", {}) if results else {}
    _multi_clm = isinstance(_sample_method.get("URR"), dict)
    _active_provs = list(_sample_method["URR"].keys()) if _multi_clm else []
    # Detect CLM providers from clm_response_tokens (works for both modes)
    _token_provs = list(_sample_method.get("clm_response_tokens", {}).keys())

    agg: Dict[str, dict] = {}

    for method in _METHODS:
        acc: Dict[str, list] = {k: [] for k in (
            "LR", "LRatio", "RLR", "RLRatio",
            "ILS", "WLS",
            "LRatio_cur", "LRatio_prof", "LRatio_hist",
            "PR", "latency_s", "token_count",
        )}
        # URR/TSR accumulators — per provider if multi-CLM, else single key
        if _multi_clm:
            for prov in _active_provs:
                acc[f"URR_{prov}"] = []
                acc[f"TSR_{prov}"] = []
        else:
            acc["URR"] = []
            acc["TSR"] = []
        # CLM response token accumulators — per provider
        for prov in _token_provs:
            acc[f"clm_tokens_{prov}"] = []

        for r in results:
            m = r["methods"].get(method)
            if not m:
                continue
            for key in ("LR", "LRatio", "RLR", "RLRatio",
                        "ILS", "WLS",
                        "LRatio_cur", "LRatio_prof", "LRatio_hist", "PR"):
                if m.get(key) is not None:
                    acc[key].append(m[key])
            acc["latency_s"].append(m["sanitization_latency_s"])
            acc["token_count"].append(m["token_count"])

            if _multi_clm:
                for prov in _active_provs:
                    v = m["URR"].get(prov) if isinstance(m.get("URR"), dict) else None
                    if v is not None:
                        acc[f"URR_{prov}"].append(v)
                    t = m["TSR"].get(prov) if isinstance(m.get("TSR"), dict) else None
                    if t is not None:
                        acc[f"TSR_{prov}"].append(t)
            else:
                if m.get("URR") is not None:
                    acc["URR"].append(m["URR"])
                if m.get("TSR") is not None:
                    acc["TSR"].append(m["TSR"])

            clm_tok_dict = m.get("clm_response_tokens", {})
            for prov in _token_provs:
                tok_v = clm_tok_dict.get(prov)
                if tok_v is not None:
                    acc[f"clm_tokens_{prov}"].append(tok_v)

        entry: dict = {
            "LR":              _mean(acc["LR"]),
            "LRatio":          _mean(acc["LRatio"]),
            "RLR":             _mean(acc["RLR"]),
            "RLRatio":         _mean(acc["RLRatio"]),
            "ILS":             _mean(acc["ILS"]),
            "WLS":             _mean(acc["WLS"]),
            "LRatio_cur":      _mean(acc["LRatio_cur"]),
            "LRatio_prof":     _mean(acc["LRatio_prof"]),
            "LRatio_hist":     _mean(acc["LRatio_hist"]),
            "PR":              _mean(acc["PR"]),
            "avg_latency_s":   _mean(acc["latency_s"]),
            "avg_token_count": _mean(acc["token_count"]),
            "avg_clm_tokens":  {p: _mean(acc[f"clm_tokens_{p}"]) for p in _token_provs},
            "n_tasks":         len(acc["LR"]),
        }
        if _multi_clm:
            entry["URR"] = {p: _mean(acc[f"URR_{p}"]) for p in _active_provs}
            entry["TSR"] = {p: _mean(acc[f"TSR_{p}"]) for p in _active_provs}
            entry["URR_mean"] = _mean([v for v in entry["URR"].values() if v is not None])
            entry["TSR_mean"] = _mean([v for v in entry["TSR"].values() if v is not None])
        else:
            entry["URR"]      = _mean(acc["URR"])
            entry["TSR"]      = _mean(acc["TSR"])
            entry["URR_mean"] = entry["URR"]
            entry["TSR_mean"] = entry["TSR"]

        agg[method] = entry

    return agg


# ════════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ════════════════════════════════════════════════════════════════════════════════

def print_summary(agg: dict, n_tasks: int) -> None:
    W = 80

    def pct(v):
        return f"{v * 100:5.1f}%" if v is not None else "   —  "

    def ms_(v):
        return f"{v * 1000:6.0f}" if v is not None else "     —"

    # Detect multi-CLM mode from aggregates
    _sample = next(iter(agg.values()), {})
    _multi_clm = isinstance(_sample.get("URR"), dict)
    _active_provs = list(_sample["URR"].keys()) if _multi_clm else []

    # Short labels for column headers
    _prov_short = {"openai": "gpt", "claude": "claude", "gemini": "gemini"}

    print(f"\n{'═' * W}")
    print(f"  EVALUATION RESULTS  —  {n_tasks} tasks  —  {len(_METHODS)} methods")
    print(f"{'═' * W}")

    def tok_(v):
        return f"{int(round(v)):>6}" if v is not None else "     —"

    print(
        f"\n  {'Method':<12}  {'LR':>6}  {'LRatio':>7}  "
        f"{'RLR':>6}  {'RLRatio':>8}  {'URR':>6}  {'TSR':>6}  "
        f"{'PR':>6}  {'Lat ms':>7}  {'Tokens':>6}"
    )
    print(
        f"  {'─'*12}  {'─'*6}  {'─'*7}  "
        f"{'─'*6}  {'─'*8}  {'─'*6}  {'─'*6}  "
        f"{'─'*6}  {'─'*7}  {'─'*6}"
    )
    for method in _METHODS:
        a = agg.get(method, {})
        if a.get("n_tasks", 0) == 0:
            continue
        urr = a.get("URR_mean") if _multi_clm else a.get("URR")
        tsr = a.get("TSR_mean") if _multi_clm else a.get("TSR")
        print(
            f"  {method:<12}  "
            f"{pct(a.get('LR')):>6}  "
            f"{pct(a.get('LRatio')):>7}  "
            f"{pct(a.get('RLR')):>6}  "
            f"{pct(a.get('RLRatio')):>8}  "
            f"{pct(urr):>6}  "
            f"{pct(tsr):>6}  "
            f"{pct(a.get('PR')):>6}  "
            f"{ms_(a.get('avg_latency_s')):>7}  "
            f"{tok_(a.get('avg_token_count')):>6}"
        )

    # Per-CLM URR/TSR breakdown (only shown in all-mode)
    if _multi_clm:
        col_w = 7  # fits "100.0%" exactly
        print(f"\n  Per-CLM URR and TSR:")
        hdr = f"  {'Method':<12}  "
        for prov in _active_provs:
            label = _prov_short.get(prov, prov)
            hdr += f"  {'URR('+label+')':>{col_w+4}}  {'TSR('+label+')':>{col_w+4}}"
        print(hdr)
        sep = f"  {'─'*12}  " + "  ".join(
            f"{'─'*(col_w+4)}  {'─'*(col_w+4)}" for _ in _active_provs
        )
        print(sep)
        for method in _METHODS:
            a = agg.get(method, {})
            if a.get("n_tasks", 0) == 0:
                continue
            row = f"  {method:<12}  "
            for prov in _active_provs:
                urr_v = a["URR"].get(prov) if isinstance(a.get("URR"), dict) else None
                tsr_v = a["TSR"].get(prov) if isinstance(a.get("TSR"), dict) else None
                row += f"  {pct(urr_v):>{col_w+4}}  {pct(tsr_v):>{col_w+4}}"
            print(row)

    # CLM response token counts
    _sample_clm_toks = next(iter(agg.values()), {}).get("avg_clm_tokens", {})
    _clm_tok_provs = list(_sample_clm_toks.keys())
    if _clm_tok_provs:
        print(f"\n  Avg CLM response tokens (output tokens per CLM per method):")
        tok_col_w = 9
        hdr2 = f"  {'Method':<12}  {'Payload':>{tok_col_w}}"
        for prov in _clm_tok_provs:
            label = _prov_short.get(prov, prov)
            hdr2 += f"  {label+' resp':>{tok_col_w}}"
        print(hdr2)
        sep2 = f"  {'─'*12}  {'─'*tok_col_w}" + "  " + "  ".join(f"{'─'*tok_col_w}" for _ in _clm_tok_provs)
        print(sep2)
        for method in _METHODS:
            a = agg.get(method, {})
            if a.get("n_tasks", 0) == 0:
                continue
            payload_tok = a.get("avg_token_count")
            row2 = f"  {method:<12}  {tok_(payload_tok):>{tok_col_w}}"
            for prov in _clm_tok_provs:
                resp_tok = a.get("avg_clm_tokens", {}).get(prov)
                row2 += f"  {tok_(resp_tok):>{tok_col_w}}"
            print(row2)

    # ILS and WLS
    print(f"\n  Injection & Weighted Leakage Severity:")
    print(f"  {'Method':<12}  {'ILS':>8}  {'WLS':>8}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*8}")
    for method in _METHODS:
        a = agg.get(method, {})
        if a.get("n_tasks", 0) == 0:
            continue
        print(
            f"  {method:<12}  "
            f"{pct(a.get('ILS')):>8}  "
            f"{pct(a.get('WLS')):>8}"
        )

    # Source-specific LRatio
    print(f"\n  Source-specific LRatio (fraction of each fact type leaked):")
    print(f"  {'Method':<12}  {'LRatio^cur':>10}  {'LRatio^prof':>11}  {'LRatio^hist':>11}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*11}  {'─'*11}")
    for method in _METHODS:
        a = agg.get(method, {})
        if a.get("n_tasks", 0) == 0:
            continue
        print(
            f"  {method:<12}  "
            f"{pct(a.get('LRatio_cur')):>10}  "
            f"{pct(a.get('LRatio_prof')):>11}  "
            f"{pct(a.get('LRatio_hist')):>11}"
        )

    print(f"\n  LR       = fraction of tasks leaking ≥1 current-task sensitive fact")
    print(f"  LRatio   = mean fraction of S^cur leaked  per task")
    print(f"  RLR      = fraction of tasks leaking ≥1 residual (S^prof ∪ S^hist) fact")
    print(f"  RLRatio  = mean fraction of S^res leaked  per task")
    print(f"  ILS      = Injection Leakage Severity: state facts injected by LC beyond r_t / |S^state|")
    print(f"  WLS      = Weighted Leakage Severity: PII-tier weighted leakage (T1=1.0, T2=0.6, T3=0.3)")
    print(f"  URR      = mean fraction of naive candidates retained (averaged over N paired CLM calls)")
    print(f"  TSR      = fraction of CLM responses judged domain-correct and actionable by GPT-4o-mini (averaged over N calls)")
    print(f"  PR       = mean fractional token reduction vs naive payload  ({_TOK_SOURCE})")
    print(f"  Lat ms   = mean sanitization latency in ms  (0 for naive — no transform)")
    print(f"{'═' * W}\n")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full PrivScope privacy + utility evaluation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--n-tasks", type=int, default=None,
        help="Number of tasks to evaluate (default: all tasks in file)",
    )
    ap.add_argument(
        "--out", default=str(_RESULTS_FILE),
        help=f"Output JSON path (default: evaluation/run_evaluation.json)",
    )
    ap.add_argument(
        "--tasks-file", default=str(_TASKS_FILE),
        help="Path to task_prompts.json",
    )
    ap.add_argument(
        "--local-model", default=None,
        help=(
            "Ollama local model to use for the entire run "
            "(LC reasoning, PrivScope, PEP). "
            "Default: LOCAL_MODEL from config.py (currently llama3.2). "
            "Options: llama3.2, phi3, mistral, qwen2.5:7b, llama3.1:8b"
        ),
    )
    ap.add_argument(
        "--cloud-mode", default="openai",
        choices=["openai", "claude", "gemini", "all"],
        help=(
            "Cloud model(s) to use for CLM calls. "
            "'openai' (default) uses GPT-4o-mini only. "
            "'all' runs all three CLMs per payload and reports per-CLM URR and TSR."
        ),
    )
    ap.add_argument(
        "--n-clm-calls", type=int, default=1,
        metavar="N",
        help=(
            "Number of independent CLM calls per payload for URR/TSR averaging. "
            "Default: 1. Each call is paired with the same-index naive call so "
            "hallucination variance cancels in the mean URR."
        ),
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading evaluation data …")
    all_tasks     = load_tasks(Path(args.tasks_file))
    profile_facts, weight_map = load_profile_facts(_PROFILE_FILE)

    trace_facts = load_trace_facts()   # backstory + working_trace merged as S^hist

    n_total = len(all_tasks)
    n_eval  = min(args.n_tasks, n_total) if args.n_tasks else n_total
    tasks   = all_tasks[:n_eval]

    backstory_count = len(_load_json(_BACKSTORY_FILE, required=False).get("memory_traces", []) if _BACKSTORY_FILE.exists() else [])
    runtime_count   = len(_load_json(_TRACE_FILE,     required=False).get("memory_traces", []) if _TRACE_FILE.exists()     else [])

    print(f"  Tasks            : {n_total} in file  →  evaluating {n_eval}")
    print(f"  Profile facts    : {len(profile_facts)} values  (S^prof, constant)")
    print(f"  Backstory        : {_BACKSTORY_FILE.name}  ({backstory_count} entries, fixed — never reset)")
    print(f"  Working trace    : {_TRACE_FILE.name}  ({runtime_count} entries at start — grows each task)")
    print(f"  S^hist at start  : {len(trace_facts)} total facts  (backstory + working trace)")
    print(f"  Methods          : {', '.join(_METHODS)}")
    print(f"  CLM calls/task   : {len(_METHODS)}  ({n_eval * len(_METHODS)} total)")
    print(f"  Output           : {out_path}")

    # ── Initialise models ─────────────────────────────────────────────────────
    print("\nInitialising models …")
    agent = HybridAgent(local_model=args.local_model)   # None → uses config default
    ps    = PrivScope(local_llm=agent.local)

    active_provs = list(_CLOUD_PROVIDERS) if args.cloud_mode == "all" else [args.cloud_mode]
    cloud_instances: Dict[str, CloudLLM] = {}
    for prov in active_provs:
        try:
            cloud_instances[prov] = CloudLLM(provider=prov)
        except Exception as e:
            print(f"  WARNING: Could not initialise CLM '{prov}': {e}")
    if not cloud_instances:
        sys.exit("[ERROR] No cloud models could be initialised.")

    # TSR judges — always all three providers for majority-vote, independent of --cloud-mode
    _all_judge_provs = ["openai", "claude", "gemini"]
    judge_instances: Dict[str, CloudLLM] = {}
    for prov in _all_judge_provs:
        if prov in cloud_instances:
            judge_instances[prov] = cloud_instances[prov]
        else:
            try:
                judge_instances[prov] = CloudLLM(provider=prov)
            except Exception as e:
                print(f"  WARNING: Could not initialise TSR judge '{prov}': {e}")
    if not judge_instances:
        print("  WARNING: No TSR judges available — TSR will default to 1 (optimistic).")

    cloud_mode = args.cloud_mode
    print(f"  Local LLM  : {agent.local.model}")
    print(f"  Cloud LLM  : {', '.join(cloud_instances.keys())}")
    print(f"  TSR judges : {', '.join(judge_instances.keys())} (majority vote ≥2/3)")
    print(f"  PrivScope  : Stages 1–3b active")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"  CLM calls/payload: {args.n_clm_calls}  (URR and TSR averaged over {args.n_clm_calls} paired calls per provider)")
    print(f"\nEvaluating {n_eval} tasks  (dots = CLM calls per task) …\n")

    results:       List[dict] = []
    history_facts: List[str]  = []  # S^hist_t — grows as tasks are processed

    for i, task in enumerate(tasks):
        result = evaluate_task(
            task_idx        = i,
            task            = task,
            agent           = agent,
            ps              = ps,
            profile_facts   = profile_facts,
            history_facts   = list(history_facts),
            trace_facts     = trace_facts,
            cloud_instances = cloud_instances,
            cloud_mode      = cloud_mode,
            judge_instances = judge_instances,
            weight_map      = weight_map,
            n_clm_calls     = args.n_clm_calls,
        )
        results.append(result)

        # ── Write task trace to the active trace file ─────────────────────────
        # Appends to whichever --trace-file was supplied (default or ablation).
        # The naive CLM response is stored so future LC calls can draw
        # prior-workflow facts from the trace (carryover leakage simulation).
        naive_clm = result["methods"]["naive"]["clm_response"]
        _append_trace({
            "source":        f"eval:{result['domain']}",
            "gathered_at":   datetime.now().isoformat(),
            "from_workflow": result["prompt"][:200],
            "lc_reasoning":  result.get("lc_reasoning", ""),
            "naive_payload": result["naive_payload"],
            "data": {
                "domain":         result["domain"],
                "task":           result["prompt"],
                "result":         naive_clm[:1000],
                "sensitive_info": result["cur_facts"],
            },
        })
        # Refresh agent's in-memory traces (backstory + updated working trace)
        _reload_agent_traces(agent)

        # S^hist_{t+1} = S^hist_t ∪ S^cur_t
        history_facts.extend(result["cur_facts"])

    # ── Aggregates ────────────────────────────────────────────────────────────
    print("\nComputing aggregate metrics …")
    aggregates = compute_aggregates(results)
    print_summary(aggregates, n_eval)

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "n_tasks_evaluated": n_eval,
            "n_tasks_total":     n_total,
            "tasks_file":        Path(args.tasks_file).name,
            "backstory_file":     _BACKSTORY_FILE.name,
            "trace_file":         _TRACE_FILE.name,
            "local_model":       agent.local.model,
            "cloud_mode":        cloud_mode,
            "cloud_models":      {p: cloud_instances[p].model for p in cloud_instances},
            "methods":           list(_METHODS),
            "profile_facts_n":   len(profile_facts),
            "trace_facts_n":     len(trace_facts),
        },
        "aggregates": aggregates,
        "tasks":      results,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Results saved → {out_path}\n")


if __name__ == "__main__":
    main()
