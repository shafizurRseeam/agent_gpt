"""
agents/langgraph_demo.py

LangGraph demo — mirrors HybridAgent's 4-phase pipeline as a graph.
This is for comparison only. Delete when done.

Install first:
    uv add langgraph

Run:
    uv run python agents/langgraph_demo.py
    uv run python agents/langgraph_demo.py --task "I need a dentist appointment"
"""

import argparse
from typing import Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from llm.local_llm import LocalLLM
from llm.cloud_router import CloudLLM
from privacy.privacyscope import PrivacyScope
from state.state_io import load_state
from tools.web_form_tool import get_form_fields, submit_form
import requests as _requests

MOCK_SERVER = "http://localhost:8000"


# ── 1. Shared state ───────────────────────────────────────────────────────────
# In LangGraph every node reads from and writes to this dict.
# Equivalent to all the local variables passed between phases in HybridAgent.run()

class AgentState(TypedDict):
    task: str
    user_profile: dict
    memory_traces: list

    # Phase 1 outputs
    inferred_prefs: str
    lc_reasoning: str
    cloud_query: str            # naive payload

    # Phase 2 outputs
    sanitized_query: Optional[str]

    # Phase 3 outputs
    clm_response: str
    providers: list             # [{name, address, phone, url}]

    # Phase 4 outputs
    form_result: dict
    final_result: str


# ── 2. Nodes (one per phase) ──────────────────────────────────────────────────
# Each node is a plain function: AgentState → dict of updates.
# LangGraph merges the returned dict back into the shared state.

def node_load_state(state: AgentState) -> dict:
    """Phase 1a — load working state from disk."""
    print("\n  [Node: load_state]")
    loaded = load_state()
    return {
        "user_profile":  loaded.get("user_profile", {}),
        "memory_traces": loaded.get("memory_traces", []),
    }


def node_lc_reason(state: AgentState) -> dict:
    """Phase 1b — LC infers context from memory and builds cloud query."""
    print("  [Node: lc_reason]")
    local = LocalLLM()
    p      = state["user_profile"]
    traces = state["memory_traces"]
    task   = state["task"]

    # Pull calendar availability from traces
    availability = ""
    for t in traces:
        if t.get("source") == "tool:get_calendar":
            free = t["data"].get("free_slots", [])
            availability = f"User is free on: {', '.join(free)}" if free else ""

    prompt = f"""You are the Local Controller of a hybrid agent.

User task: "{task}"
Name: {p.get('name','')}  Age: {p.get('age','')}  Address: {p.get('address','')}
Phone: {p.get('phone','')}  Insurance: {p.get('insurance','')} (ID: {p.get('insurance_id','')})
{availability}

REASONING: <think through what to include>
CLOUD QUERY: <rich natural-language message packed with personal details>"""

    response = local.generate(prompt).strip()

    upper = response.upper()
    cq_pos = upper.rfind("CLOUD QUERY:")
    if cq_pos != -1:
        reasoning   = response[:cq_pos].strip()
        cloud_query = response[cq_pos + 12:].strip().strip('"')
    else:
        reasoning   = response
        cloud_query = f"{p.get('name','User')} needs help: {task}. Insurance: {p.get('insurance','')}."

    print(f"  LC Reasoning: {reasoning[:80]}...")
    print(f"  Cloud Query:  {cloud_query[:80]}...")

    return {"lc_reasoning": reasoning, "cloud_query": cloud_query}


def node_privacy_scope(state: AgentState) -> dict:
    """Phase 2 — sanitize the naive payload with PrivacyScope."""
    print("  [Node: privacy_scope]")
    ps = PrivacyScope()
    sanitized, _ = ps.sanitize_with_trace(
        state["cloud_query"],
        state["user_profile"],
        state["task"],
        state["memory_traces"],
    )
    print(f"  Sanitized:    {sanitized[:80]}...")
    return {"sanitized_query": sanitized}


def node_cloud_search(state: AgentState) -> dict:
    """Phase 3 — send query to cloud LLM and parse provider list."""
    print("  [Node: cloud_search]")
    cloud = CloudLLM()
    query = state["sanitized_query"] or state["cloud_query"]

    messages = [
        {
            "role": "system",
            "content": (
                "Return a numbered list of up to 5 local service providers. "
                "For each: name, address, phone.\n"
                "Format: 1. <Name>\n   Address: <address>\n   Phone: <phone>"
            ),
        },
        {"role": "user", "content": query},
    ]
    response = cloud.chat(messages)
    print(f"  CLM response: {response[:80]}...")

    # Register first parsed provider with mock server
    import re
    providers = []
    current = None
    for line in response.strip().splitlines():
        line = line.strip()
        m = re.match(r'^\d+[.)]\s*\*{0,2}(.+?)\*{0,2}\s*$', line)
        if m:
            if current:
                providers.append(current)
            current = {"name": m.group(1).strip(" -:"), "address": "", "phone": ""}
            continue
        if current is None:
            continue
        phone_m = re.search(r'(\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4})', line)
        if phone_m and not current["phone"]:
            current["phone"] = phone_m.group(1).strip()
        addr_m = re.search(
            r'(\d+\s+\w[\w\s]+(?:St|Ave|Rd|Blvd|Dr|Ln|Way|Pkwy|Plaza|Ct|Circle|Sq)[\w\s,]*)',
            line, re.I,
        )
        if addr_m and not current["address"]:
            current["address"] = addr_m.group(1).strip()
    if current:
        providers.append(current)

    # Register with mock server
    registered = []
    for p in providers[:5]:
        try:
            r = _requests.post(
                f"{MOCK_SERVER}/services/register",
                json={"name": p["name"], "address": p["address"],
                      "phone": p["phone"], "category": "general"},
                timeout=5,
            )
            if r.status_code == 200:
                registered.append({**p, "url": r.json()["url"]})
        except Exception:
            pass

    return {"clm_response": response, "providers": registered}


def node_submit_form(state: AgentState) -> dict:
    """Phase 4 — fetch form fields and submit to the chosen provider."""
    print("  [Node: submit_form]")
    providers = state["providers"]
    if not providers:
        return {"form_result": {}, "final_result": "No providers available"}

    chosen = providers[0]
    p = state["user_profile"]

    form_fields = get_form_fields(chosen["url"])
    field_map = {
        "name":             p.get("name", ""),
        "phone":            p.get("phone", ""),
        "address":          p.get("address", ""),
        "insurance":        p.get("insurance", ""),
        "dob":              p.get("dob", ""),
        "appointment_date": "2026-05-01",
        "date":             "2026-05-01",
        "time":             "10:00",
        "party_size":       "2",
    }
    form_data = {f["field"]: field_map[f["field"]] for f in form_fields if f["field"] in field_map}

    result = submit_form(chosen["url"], form_data)
    print(f"  Booked at: {result.get('service', chosen['name'])}")

    return {
        "form_result":  result,
        "final_result": f"Booked at {chosen['name']}",
    }


# ── 3. Build the graph ────────────────────────────────────────────────────────
# This is the LangGraph equivalent of HybridAgent.run()'s linear phase sequence.

def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("load_state",    node_load_state)
    graph.add_node("lc_reason",     node_lc_reason)
    graph.add_node("privacy_scope", node_privacy_scope)
    graph.add_node("cloud_search",  node_cloud_search)
    graph.add_node("submit_form",   node_submit_form)

    # Wire edges (linear pipeline — same order as HybridAgent phases)
    graph.set_entry_point("load_state")
    graph.add_edge("load_state",    "lc_reason")
    graph.add_edge("lc_reason",     "privacy_scope")
    graph.add_edge("privacy_scope", "cloud_search")
    graph.add_edge("cloud_search",  "submit_form")
    graph.add_edge("submit_form",   END)

    return graph.compile()


# ── 4. Run ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="I need to book a doctor appointment for knee pain")
    args = parser.parse_args()

    print(f"\n{'═' * 60}")
    print(f"  LANGGRAPH DEMO")
    print(f"  Task: {args.task}")
    print(f"{'═' * 60}")

    app = build_graph()

    # Invoke the graph — LangGraph runs each node in order, passing state through
    final_state = app.invoke({"task": args.task})

    print(f"\n{'─' * 60}")
    print(f"  Result: {final_state['final_result']}")
    print(f"{'─' * 60}\n")

    # Show how state looks after the full run
    print("  Final state keys populated by the graph:")
    for key, val in final_state.items():
        if val and key not in ("user_profile", "memory_traces"):
            preview = str(val)[:60] + "..." if len(str(val)) > 60 else str(val)
            print(f"    {key:<20} {preview}")


if __name__ == "__main__":
    main()
