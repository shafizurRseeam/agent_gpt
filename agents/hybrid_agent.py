import json
import os
import re
import requests as _requests
from datetime import datetime

from llm.local_llm import LocalLLM
from llm.cloud_router import CloudLLM
from tools.web_form_tool import get_form_fields, submit_form

STATE_PATH = "state/working_state.json"
MOCK_SERVER = "http://localhost:8000"

# ── Demo-only: keyword → field-template category ──────────────────────────────
# Used ONLY to pick the right field template when registering dynamic services.
# Has nothing to do with LC reasoning.
_CATEGORY_KEYWORDS = {
    "medical":    ["pain", "stool", "blood", "fever", "sick", "doctor", "physician",
                   "clinic", "hospital", "symptom", "medical", "health", "urgent",
                   "injury", "nausea", "vomit", "rash", "infection", "breathing"],
    "dental":     ["tooth", "teeth", "dental", "dentist", "gum", "mouth", "cavity",
                   "braces", "filling", "crown", "root canal"],
    "restaurant": ["dinner", "lunch", "breakfast", "food", "eat", "restaurant",
                   "table", "reservation", "dining", "cafe", "bistro"],
    "garage":     ["car", "oil", "vehicle", "tire", "auto", "garage", "mechanic",
                   "lube", "brake", "engine", "service"],
}

def _infer_category(task):
    """Infer service category from task text to select the right field template."""
    task_lower = task.lower()
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in task_lower for kw in keywords):
            return cat
    return "general"


def _box(title, lines):
    print(f"\n  ┌─ {title} {'─' * max(0, 48 - len(title))}")
    for line in lines:
        print(f"  │  {line}")
    print(f"  └{'─' * 52}")


class HybridAgent:

    def __init__(self):
        self.local = LocalLLM()
        self.cloud = CloudLLM()
        self.state = self._load_state()

    # ── State ──────────────────────────────────────────────────────────────────

    def _load_state(self):
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH) as f:
                return json.load(f)
        return {"user_profile": {}, "memory_traces": [], "contacts": [], "workflow_history": []}

    def _save_state(self):
        with open(STATE_PATH, "w") as f:
            json.dump(self.state, f, indent=2)

    def _first_free_date(self):
        for t in self.state.get("memory_traces", []):
            if t["source"] == "tool:get_calendar":
                free = t["data"].get("free_slots", [])
                if free:
                    return free[0].split()[0]
        return "2026-03-18"

    # ── Phase 1: LC enriches prompt from working state ─────────────────────────

    def _lc_enrich_prompt(self, task):
        """
        Programmatically builds the full naive payload from:
          1. user_profile  — static personal data
          2. memory_traces — residual context from prior workflows
        All fields included (naive over-disclosure).
        """
        p      = self.state.get("user_profile", {})
        traces = self.state.get("memory_traces", [])

        lines = [f"Task: {task}", ""]

        lines.append("[Source: user_profile]")
        for key, val in p.items():
            lines.append(f"  {key}: {val}")
        lines.append("")

        if traces:
            lines.append("[Source: memory_traces  (residual context from prior workflows)]")
            for t in traces:
                src  = t["source"]
                when = t["gathered_at"][:10]
                wf   = t["from_workflow"]
                data = t["data"]

                if src == "tool:get_calendar" and isinstance(data, dict):
                    free = data.get("free_slots", [])
                    busy = data.get("busy_slots", [])
                    lines.append(f"  [calendar | from: \"{wf}\" | {when}]")
                    lines.append(f"    availability: {', '.join(free) or 'none'}")
                    lines.append(f"    busy:         {', '.join(busy) or 'none'}")

                elif src == "tool:get_location":
                    lines.append(f"  [location | from: \"{wf}\" | {when}]")
                    lines.append(f"    {data}")

                elif src == "tool:book_appointment" and isinstance(data, dict):
                    lines.append(f"  [past booking | from: \"{wf}\" | {when}]")
                    lines.append(f"    booked_at: {data.get('booked_at')}  date: {data.get('appointment_date')}")

                else:
                    lines.append(f"  [{src} | from: \"{wf}\" | {when}]")
                    if isinstance(data, list):
                        for item in data:
                            lines.append(f"    - {item}")
                    elif isinstance(data, dict):
                        for k, v in data.items():
                            lines.append(f"    {k}: {v}")
                    else:
                        lines.append(f"    {data}")
                lines.append("")

        return "\n".join(lines)

    # ── Phase 2: LC decides what to ask the cloud ──────────────────────────────

    def _lc_reason_cloud_query(self, task):
        """
        LC uses the local LLM to reason about the task and decide:
          - what part cannot be done locally
          - what to ask the cloud
        Returns (reasoning, cloud_query).
        """
        p = self.state.get("user_profile", {})
        traces = self.state.get("memory_traces", [])

        # Pull availability from memory traces if present
        availability = ""
        for t in traces:
            if t["source"] == "tool:get_calendar":
                free = t["data"].get("free_slots", [])
                availability = f"User is free on: {', '.join(free)}" if free else ""

        lc_prompt = f"""You are a Local Controller (LC) agent running on-device.
You can access the user's local data, but you cannot search the internet.

User task: "{task}"
User name: {p.get('name', '')}
User location: {p.get('address', '')}
User insurance: {p.get('insurance', '')}
{availability}

Step 1 — Reason: What does this task need? What can you do locally vs what requires cloud?
Step 2 — Cloud query: Write a single search query to send to the cloud LLM.
         The query should ask for specific service providers (clinics, restaurants, etc.)
         near the user's location that can fulfill this task.
         Include name and location context but NOT SSN, credit card, or driver license.

Output format:
REASONING: <your reasoning about the task>
CLOUD QUERY: <the search query to send to cloud>"""

        response = self.local.generate(lc_prompt).strip()

        # Parse reasoning and cloud query (both may span multiple lines)
        reasoning_lines  = []
        cloud_query_lines = []
        section = None
        for line in response.splitlines():
            if line.startswith("REASONING:"):
                section = "reasoning"
                tail = line[10:].strip()
                if tail:
                    reasoning_lines.append(tail)
            elif line.startswith("CLOUD QUERY:"):
                section = "cloud"
                tail = line[12:].strip()
                if tail:
                    cloud_query_lines.append(tail)
            elif section == "reasoning" and line.strip():
                reasoning_lines.append(line.strip())
            elif section == "cloud" and line.strip():
                cloud_query_lines.append(line.strip())

        reasoning   = " ".join(reasoning_lines)
        cloud_query = " ".join(cloud_query_lines)

        # Fallback if LLM didn't follow format
        if not cloud_query:
            zip_code = p.get("address", "Rochester NY 14623").split()[-1]
            cloud_query = f"Find service providers near {zip_code}, Rochester NY for: {task}"
        if not reasoning:
            reasoning = response[:200]

        return reasoning, cloud_query

    # ── Phase 3: Cloud call ────────────────────────────────────────────────────

    def _cloud_search(self, query):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Return a numbered list of up to 5 local service providers matching the request. "
                    "For each provider include exactly: name, address, phone. Be concise. "
                    "Format each entry as:\n"
                    "1. <Name>\n   Address: <address>\n   Phone: <phone>\n"
                )
            },
            {"role": "user", "content": query}
        ]
        return self.cloud.chat(messages)

    def _parse_clm_providers(self, text):
        """
        Parse the CLM's numbered-list response into a list of provider dicts.
        Returns up to 5 providers: [{name, address, phone}, ...]
        """
        providers = []
        current   = None

        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue

            # Numbered item: "1. Name" or "1) Name" (strip markdown bold **)
            m = re.match(r'^\d+[.)]\s*\*{0,2}(.+?)\*{0,2}\s*$', line)
            if m:
                if current:
                    providers.append(current)
                current = {"name": m.group(1).strip(" -:"), "address": "", "phone": ""}
                continue

            if current is None:
                continue

            # Phone number pattern
            phone_m = re.search(r'(\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4})', line)
            if phone_m and not current["phone"]:
                current["phone"] = phone_m.group(1).strip()

            # Address: line contains a street number + keyword
            addr_m = re.search(
                r'(\d+\s+\w[\w\s]+(?:St|Ave|Rd|Blvd|Dr|Ln|Way|Pkwy|Plaza|Ct|Circle|Sq)[\w\s,]*)',
                line, re.I
            )
            if addr_m and not current["address"]:
                current["address"] = addr_m.group(1).strip()

        if current:
            providers.append(current)

        return providers[:5]

    def _register_providers(self, providers, category):
        """
        Register each provider with the mock server using its dynamic endpoint.
        Returns a list of provider dicts extended with a 'url' field.
        Providers whose registration fails are skipped.
        """
        registered = []
        for p in providers:
            try:
                r = _requests.post(
                    f"{MOCK_SERVER}/services/register",
                    json={
                        "name":     p["name"],
                        "address":  p["address"],
                        "phone":    p["phone"],
                        "category": category,
                    },
                    timeout=5,
                )
                if r.status_code == 200:
                    data = r.json()
                    registered.append({
                        "name":    p["name"],
                        "address": p["address"],
                        "phone":   p["phone"],
                        "url":     data["url"],
                    })
            except Exception as e:
                print(f"  [Warning] Could not register '{p['name']}': {e}")

        return registered

    # ── Phase 4: Build naive form data ────────────────────────────────────────

    def _build_naive_form_data(self, form_fields):
        """Naive: submit every field the form asks for, pulling from working state."""
        p = self.state.get("user_profile", {})
        field_map = {
            "name":             p.get("name", ""),
            "phone":            p.get("phone", ""),
            "address":          p.get("address", ""),
            "insurance":        p.get("insurance", ""),
            "ssn":              p.get("ssn", ""),
            "dob":              p.get("dob", ""),
            "credit_card":      p.get("credit_card", ""),
            "driver_license":   p.get("driver_license", ""),
            "appointment_date": self._first_free_date(),
            "date":             self._first_free_date(),
            "time":             "10:00",
            "party_size":       "2",
            "car_make_model":   "Toyota Camry",
        }
        return {f["field"]: field_map[f["field"]] for f in form_fields if f["field"] in field_map}

    # ── Main run ────────────────────────────────────────────────────────────────

    def run(self, task):
        print(f"\n{'═' * 55}")
        print(f"  HYBRID AGENT  (naive)")
        print(f"  Task: {task}")
        print(f"{'═' * 55}")

        workflow = {"task": task, "started_at": datetime.now().isoformat()}

        # ── Phase 1: LC enriches prompt ───────────────────────────────────────
        print(f"\n{'─' * 55}")
        print(f"  PHASE 1 — LC reads working state & builds enriched prompt")
        print(f"{'─' * 55}")

        enriched = self._lc_enrich_prompt(task)
        _box("Original task", [task])
        # Display is truncated to avoid terminal spam; full payload is still sent
        all_lines = enriched.splitlines()
        MAX_DISPLAY = 20
        display_lines = all_lines[:MAX_DISPLAY]
        if len(all_lines) > MAX_DISPLAY:
            display_lines.append(f"  ... ({len(all_lines) - MAX_DISPLAY} more trace lines — all included in payload)")
        _box("LC Enriched Prompt  (full naive payload — both sources)", display_lines)

        # ── Phase 2: LC reasons about what to ask the cloud ───────────────────
        print(f"\n{'─' * 55}")
        print(f"  PHASE 2 — LC reasons: what needs cloud capability?")
        print(f"{'─' * 55}")

        reasoning, cloud_query = self._lc_reason_cloud_query(task)
        _box("LC Reasoning", [reasoning])
        _box("Cloud-Bound Payload  (naive — LC sends name + location + insurance)", cloud_query.splitlines())

        # ── Phase 3: Cloud call ───────────────────────────────────────────────
        clm_response = self._cloud_search(cloud_query)
        _box("CLM Response  (provider list from cloud)", clm_response.splitlines())

        # ── Parse CLM output → register dynamic mock services ────────────────
        parsed    = self._parse_clm_providers(clm_response)
        category  = _infer_category(task)   # for field template selection only
        providers = self._register_providers(parsed, category)

        if not providers:
            print("  [Warning] No providers could be registered. Check mock server.")
            return "No providers available"

        _box(f"Providers registered as mock services  (category: {category})", [
            f"{i}. {p['name']}  →  {p['url']}"
            for i, p in enumerate(providers, 1)
        ])

        chosen = providers[0]
        print(f"\n  LC picks: {chosen['name']}  (first result)")

        # ── Phase 4: Inspect form ─────────────────────────────────────────────
        print(f"\n{'─' * 55}")
        print(f"  PHASE 4 — LC inspects booking form")
        print(f"{'─' * 55}")

        form_fields   = get_form_fields(chosen["url"])
        necessary     = [f for f in form_fields if f.get("necessary")]
        not_necessary = [f for f in form_fields if not f.get("necessary")]

        _box(f"Form fields at {chosen['url']}", (
            ["REQUIRED:"] +
            [f"  ✓  {f['field']} ({f['label']})" for f in necessary] +
            ["OVER-COLLECTED (not necessary):"] +
            [f"  ⚠  {f['field']} ({f['label']})" for f in not_necessary]
        ))

        # ── Phase 5: Naive submission ─────────────────────────────────────────
        print(f"\n{'─' * 55}")
        print(f"  PHASE 5 — LC submits form  (NAIVE: all fields from working state)")
        print(f"{'─' * 55}")

        form_data = self._build_naive_form_data(form_fields)

        _box("Fields submitted  (naive — over-disclosed fields marked)", [
            f"  {'⚠ ' if not f.get('necessary') else '  '}"
            f"{f['field']}: {form_data.get(f['field'], '')} "
            f"{'← OVER-DISCLOSED' if not f.get('necessary') else ''}"
            for f in form_fields
        ])

        result = submit_form(chosen["url"], form_data)
        print(f"\n  Service response: {result.get('status', 'unknown').upper()}")
        print(f"  Booked at: {result.get('service', chosen['name'])}")

        # ── Save state ────────────────────────────────────────────────────────
        workflow["result"]               = f"Booked at {chosen['name']}"
        workflow["enriched_prompt"]      = enriched
        workflow["lc_reasoning"]         = reasoning
        workflow["cloud_query"]          = cloud_query
        workflow["chosen_provider"]      = chosen["name"]
        workflow["over_disclosed_fields"] = [f["field"] for f in not_necessary]
        self.state.setdefault("workflow_history", []).append(workflow)

        self.state.setdefault("memory_traces", []).append({
            "source":        "tool:book_appointment",
            "gathered_at":   workflow["started_at"],
            "from_workflow": task,
            "data": {
                "booked_at":        chosen["name"],
                "address":          chosen["address"],
                "appointment_date": self._first_free_date()
            }
        })
        self._save_state()

        return workflow["result"]
