import json
import os
import re
import textwrap
import requests as _requests
from datetime import datetime

from llm.local_llm import LocalLLM
from llm.cloud_router import CloudLLM
from tools.web_form_tool import get_form_fields, submit_form
from privacy.privacyscope import PrivacyScope

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
        for seg in (textwrap.wrap(line, width=68) if line.strip() else [""]):
            print(f"  │  {seg}")
    print(f"  └{'─' * 52}")


class HybridAgent:

    def __init__(self, local_model=None):
        self.local = LocalLLM(model=local_model) if local_model else LocalLLM()
        self.cloud = CloudLLM()
        self.state = self._load_state()
        self.ps    = PrivacyScope()

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

    # ── Phase 1.5: LC infers preferences from memory ──────────────────────────

    def _lc_infer_preferences(self, task):
        """
        LC scans memory_traces to surface preferences/context relevant to the task.
        For example: past restaurant bookings reveal cuisine preference; past location
        traces constrain the search area.
        Returns a short string of bullet-point inferences, or "" if nothing found.
        """
        traces = self.state.get("memory_traces", [])
        p      = self.state.get("user_profile", {})

        if not traces:
            return ""

        # Build a compact summary of past activity for the LLM to scan
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

        # Extract section after the label
        if "INFERRED CONTEXT:" in raw:
            prefs = raw.split("INFERRED CONTEXT:")[-1].strip()
        else:
            prefs = raw

        # Discard if the model just said nothing useful
        if len(prefs) < 10 or prefs.lower().startswith("none"):
            return ""

        return prefs

    # ── Phase 2: LC decides what to ask the cloud ──────────────────────────────

    def _lc_reason_cloud_query(self, task, inferred_prefs=""):
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

        # Split on the LAST occurrence of CLOUD QUERY: (case-insensitive).
        # Everything before it = reasoning, everything after = cloud query.
        # Using rfind avoids false splits when the LLM mentions the label mid-reasoning.
        upper_resp = response.upper()
        cq_pos     = upper_resp.rfind("CLOUD QUERY:")

        if cq_pos != -1:
            reasoning_raw = response[:cq_pos].strip()
            cloud_query   = response[cq_pos + 12:].strip().strip('"').strip()
        else:
            reasoning_raw = response
            cloud_query   = ""

        # Strip LLM meta-commentary that sometimes follows the cloud query.
        # The commentary always begins in a new paragraph (blank line) and
        # typically starts with "This query/message/briefing/note..."
        # Approach 1: cut at the first blank line (double newline).
        blank_line = re.search(r'\n\s*\n', cloud_query)
        if blank_line:
            cloud_query = cloud_query[:blank_line.start()].strip()
        # Approach 2: strip trailing sentence(s) starting with known meta phrases.
        cloud_query = re.sub(
            r'\s*\n?\s*(?:This (?:query|message|briefing|request|information|context|'
            r'cloud query|natural.language)|Note:|P\.?S\.?:?|Please note)[^"]*$',
            '', cloud_query, flags=re.IGNORECASE | re.DOTALL
        ).strip().strip('"').strip()

        # Strip the REASONING: label from the front of the reasoning block
        r_pos = reasoning_raw.upper().find("REASONING:")
        if r_pos != -1:
            reasoning = reasoning_raw[r_pos + 10:].strip()
        else:
            reasoning = reasoning_raw

        # Strip Step 1 label if the LLM echoed it, and cut at Step 2 if present
        reasoning = re.sub(r'^Step\s*1\s*[-—:]+\s*Reason\s*[:\-—]?\s*', '', reasoning, flags=re.IGNORECASE).strip()
        step2 = re.search(r'\bStep\s*2\b', reasoning, re.IGNORECASE)
        if step2:
            reasoning = reasoning[:step2.start()].strip()

        # Fallback A: quoted text after "briefing:" keyword
        if not cloud_query:
            brief_match = re.search(
                r'briefing[:\s]*\n*\s*"([^"]{40,})"',
                response, re.DOTALL | re.IGNORECASE
            )
            if brief_match:
                cloud_query = brief_match.group(1).strip()
                cut = re.search(r"here'?s? a? ?natural.language briefing", reasoning, re.IGNORECASE)
                if cut:
                    reasoning = reasoning[:cut.start()].strip()

        # Fallback B: any long quoted string embedded anywhere in the reasoning.
        # This fires when the LLM places the full cloud query in quotes inside
        # its reasoning block instead of after the CLOUD QUERY: label.
        if not cloud_query:
            quote_match = re.search(r'"([^"]{100,})"', reasoning, re.DOTALL)
            if quote_match:
                cloud_query = quote_match.group(1).strip()
                # Trim reasoning: keep only what came before the opening quote,
                # dropping any trailing intro line ("Here's the query:", etc.)
                anchor = reasoning.find('"' + cloud_query[:30])
                if anchor > 0:
                    pre = reasoning[:anchor].strip()
                    pre = re.sub(
                        r'\n?[^\n]*(?:here\'?s?|crafted|is the query|'
                        r'following query|packed it)[^\n]*$',
                        '', pre, flags=re.IGNORECASE
                    ).strip()
                    reasoning = pre or reasoning

        # Trim any trailing "This query includes…" explanation the LLM added
        # after the quoted cloud query (still inside the reasoning block)
        if reasoning:
            reasoning = re.sub(
                r'\n+\s*(?:This (?:query|message|request)|By over-including|'
                r'\d+\.\s+User\'?s? exact).*$',
                '', reasoning, flags=re.IGNORECASE | re.DOTALL
            ).strip()

        # Hard fallback if nothing was extracted
        if not cloud_query:
            task_clean = task.strip().rstrip(".")
            task_clean = task_clean[:1].upper() + task_clean[1:]
            cloud_query = (
                f"{p.get('name', 'User')} at {p.get('address', 'Rochester NY')} "
                f"is looking for help — {task_clean}. "
                f"{availability}. Insurance: {p.get('insurance', '')}."
            )
        if not reasoning:
            reasoning = response

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
        print(f"  LC model : {self.local.model}")
        print(f"  Task: {task}")
        print(f"{'═' * 55}")

        workflow = {"task": task, "started_at": datetime.now().isoformat()}

        # ── Phase 1: LC reads working state & infers task-relevant context ───────
        print(f"\n{'─' * 55}")
        print(f"  PHASE 1 — LC reads working state, infers context & reasons")
        print(f"{'─' * 55}")

        # Build enriched payload internally (still logged for research)
        enriched = self._lc_enrich_prompt(task)

        # Show a compact summary of what the LC has access to (not a full dump)
        p = self.state.get("user_profile", {})
        traces = self.state.get("memory_traces", [])
        _box("LC local state (available on-device)", [
            f"user_profile : {len(p)} fields  ({', '.join(list(p.keys())[:5])}{'...' if len(p) > 5 else ''})",
            f"memory_traces: {len(traces)} entries  "
            f"({', '.join(set(t['source'] for t in traces))})" if traces else "memory_traces: 0 entries",
        ])

        # LC scans memory to infer what's relevant for this specific task
        inferred_prefs = self._lc_infer_preferences(task)
        if inferred_prefs:
            _box("LC Inferred Context (task-relevant preferences from memory)", inferred_prefs.splitlines())
        else:
            print("  No relevant preferences found in memory.")

        # LC reasons over the task and inferred context → produces naive payload
        reasoning, cloud_query = self._lc_reason_cloud_query(task, inferred_prefs)
        _box("LC Reasoning", reasoning.splitlines() or [reasoning])
        _box("Naive Cloud-Bound Payload  (to be sent to cloud)", cloud_query.splitlines())

        # ── Phase 2: PrivacyScope Sanitization ────────────────────────────────
        print(f"\n{'─' * 55}")
        print(f"  PHASE 2 — PrivacyScope Sanitization")
        print(f"{'─' * 55}")

        sanitized_query, ps_stages = self.ps.sanitize_with_trace(
            cloud_query, p, task, traces
        )
        ps_spans    = ps_stages["spans"]
        ps_tasktype = ps_stages["task_type"]

        def _trunc(s, n=42):
            return (s[:n] + "…") if len(s) > n else s

        # ── Stage 1: Span Extraction ───────────────────────────────────────────
        _box("Stage 1 — Span Extraction", [
            f"  {s.span_type:<18}  \"{_trunc(s.text)}\""
            for s in ps_spans
        ] or ["  (no candidate spans found)"])

        # ── Stage 2: Scope Control ─────────────────────────────────────────────
        scope_lines = [f"  inferred task type: {ps_tasktype}", ""]
        for s in ps_spans:
            tag    = "kept   " if s.kept else "removed"
            reason = f"  ← {s.removal_reason}" if s.removal_reason else ""
            scope_lines.append(
                f"  {tag}  {s.span_type:<18}  \"{_trunc(s.text, 35)}\"{reason}"
            )
        _box("Stage 2 — Scope Control", scope_lines)

        # ── Stage 3: Span Classification ──────────────────────────────────────
        classify_lines = []
        for s in ps_spans:
            if not s.kept:
                continue
            classify_lines.append(
                f"  {s.span_class:<5}  {s.span_type:<18}  \"{_trunc(s.text, 35)}\""
            )
        classify_lines.append("  BEN    [remaining task text]")
        _box("Stage 3 — Span Classification", classify_lines)

        # ── Stage 4: Transformation ────────────────────────────────────────────
        transform_lines = []
        for s in ps_spans:
            if not s.kept:
                transform_lines.append(
                    f"  ---   \"{_trunc(s.text, 30)}\"  →  [removed — scope control]"
                )
            elif s.span_class in ("DI", "CSS"):
                transform_lines.append(
                    f"  {s.span_class:<5}  \"{_trunc(s.text, 30)}\""
                    f"  →  {s.result}"
                )
        transform_lines.append("  BEN    [remaining task text]               →  unchanged")
        _box("Stage 4 — Transformation", transform_lines)

        _box("Sanitized Cloud-Bound Payload  (PrivacyScope output)", sanitized_query.splitlines())

        # ── Phase 3: CLM Responses ────────────────────────────────────────────
        print(f"\n{'─' * 55}")
        print(f"  PHASE 3 — CLM Response")
        print(f"{'─' * 55}")

        clm_response           = self._cloud_search(cloud_query)
        clm_response_sanitized = self._cloud_search(sanitized_query)

        _box("CLM Response  (naive payload)", clm_response.splitlines())
        _box("CLM Response  (PrivacyScope payload)", clm_response_sanitized.splitlines())

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

        # ── Phase 4: Submission ───────────────────────────────────────────────
        print(f"\n{'─' * 55}")
        print(f"  PHASE 4 — Submission")
        print(f"{'─' * 55}")

        form_fields = get_form_fields(chosen["url"])

        form_data = self._build_naive_form_data(form_fields)

        _box("Fields submitted", [
            f"  {f['field']}: {form_data.get(f['field'], '')}"
            for f in form_fields
        ])

        result = submit_form(chosen["url"], form_data)
        print(f"\n  Service response: {result.get('status', 'unknown').upper()}")
        print(f"  Booked at: {result.get('service', chosen['name'])}")

        # ── Save state ────────────────────────────────────────────────────────
        workflow["result"]                = f"Booked at {chosen['name']}"
        workflow["enriched_prompt"]      = enriched
        workflow["lc_reasoning"]         = reasoning
        workflow["cloud_query_naive"]    = cloud_query
        workflow["cloud_query_sanitized"] = sanitized_query
        workflow["chosen_provider"]      = chosen["name"]
        workflow["over_disclosed_fields"] = [f["field"] for f in form_fields if not f.get("necessary")]
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
