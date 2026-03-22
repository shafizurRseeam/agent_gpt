"""
privacy/privacyscope.py

PrivScope — on-device payload sanitization pipeline (simplified, no DP).

Four displayed stages (§3 of the paper):
  Stage 1  Span Extraction     — identify semantically coherent disclosure units
  Stage 2  Scope Control       — drop spans not justified for the current subtask
  Stage 3  Span Classification — DI / CSS / BEN
  Stage 4  Transformation      — DI → placeholder, CSS → abstraction, BEN → unchanged

Public surface:
  sanitize_with_trace(payload, user_profile, task, memory_traces)
      → (sanitized_text, stages_dict)   # stages_dict used by display layer
  sanitize(...)
      → sanitized_text                  # simple wrapper
"""

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Set, Tuple


# ── Span record ───────────────────────────────────────────────────────────────

@dataclass
class Span:
    """One candidate disclosure unit tracked through the pipeline."""
    text: str
    span_type: str          # 'name' | 'phone' | 'address' | 'symptom' | 'date' | ...
    span_class: str = ""    # 'DI' | 'CSS' | 'BEN'  (set at Stage 3)
    kept: bool = True       # False if dropped at Stage 2
    removal_reason: str = ""
    result: str = ""        # value after Stage 4 transformation


# ── PrivacyScope ──────────────────────────────────────────────────────────────

class PrivacyScope:

    # ── Regex patterns ────────────────────────────────────────────────────────
    _PHONE_RE    = re.compile(r'\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}')
    _SSN_RE      = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    _CC_RE       = re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')
    _INS_ID_RE   = re.compile(r'\b[A-Z]{2,5}-\d{4,8}-[A-Z0-9]{2,6}\b')
    _EMAIL_RE    = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b')
    _ISO_DATE_RE = re.compile(r'\b(20\d{2}-\d{2}-\d{2}(?:\s*\([^)]+\))?)')
    _NL_DATE_RE  = re.compile(
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        r'\s+\d{1,2}(?:st|nd|rd|th)?'
        r'(?:\s+and\s+\d{1,2}(?:st|nd|rd|th)?)?'
        r'(?:,?\s*(?:of\s+)?\d{4})?',
        re.IGNORECASE
    )

    # ── Symptom classification map ────────────────────────────────────────────
    # More-specific entries MUST come before less-specific ones.
    _SYMPTOM_MAP: List[Tuple[FrozenSet, str]] = [
        (frozenset({"tooth", "teeth", "gum", "gums", "cavity", "crown",
                    "root canal", "dental", "dentist", "oral", "mouth"}),
         "dental concern"),
        (frozenset({"knee", "hip", "shoulder", "joint", "sprain", "fracture",
                    "ligament", "tendon", "ortho", "strain", "musculoskeletal"}),
         "orthopedic concern"),
        (frozenset({"chest pain", "cardiac", "heart attack", "palpitation"}),
         "cardiac concern"),
        (frozenset({"anxiety", "depression", "mental health",
                    "stress", "therapy", "psychiatr"}),
         "mental health concern"),
        (frozenset({"pain", "injury", "bleed", "bleeding", "fever", "sick",
                    "symptom", "nausea", "rash", "infection", "hurt",
                    "sore", "ache", "swollen", "swelling", "stool", "stomach"}),
         "medical concern"),
    ]

    # Detects "experiencing/for/about <symptom phrase>" in natural language
    _SYMPTOM_CONTEXT_RE = re.compile(
        r'(?:'
        r'(?:is\s+)?(?:experiencing|having|suffering\s+from|reporting|'
        r'complaining\s+of|presents?\s+with|dealing\s+with)'
        r'|\bfor|\babout'
        r')\s+([^,.;:\n]{5,80})',
        re.IGNORECASE
    )

    # ── Insurance generalization map ──────────────────────────────────────────
    _INSURANCE_CATEGORY = [
        ({"dental", "teeth", "oral"},   "dental insurance"),
        ({"vision", "eye", "optical"},  "vision insurance"),
        ({"mental", "behav", "psych"},  "mental health insurance"),
        ({"medicare", "medicaid"},      "government health insurance"),
    ]

    # ── Classification lookups ────────────────────────────────────────────────
    _DI_TYPES  = frozenset({"name", "phone", "ssn", "cc", "email",
                             "dob", "dl", "ins_id"})
    _CSS_TYPES = frozenset({"address", "insurance_name", "date", "symptom"})

    _DI_KIND = {
        "name":   "NAME",  "phone": "PHONE", "ssn":    "SSN",
        "cc":     "CC",    "email": "EMAIL", "dob":    "DOB",
        "dl":     "DL",    "ins_id": "INS_ID",
    }

    # ── Task-type keywords (scope control) ────────────────────────────────────
    _TASK_TYPES = {
        "dental":     {"tooth", "teeth", "dental", "dentist", "gum", "oral", "cavity"},
        "medical":    {"pain", "injury", "sick", "fever", "doctor", "clinic",
                       "hospital", "stool", "blood", "swelling", "knee"},
        "restaurant": {"restaurant", "dinner", "lunch", "eat", "dining",
                       "food", "reservation"},
        "garage":     {"car", "oil", "tire", "brake", "mechanic", "garage",
                       "vehicle", "auto"},
    }

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self):
        self.bindings: Dict[str, str] = {}
        self._counter = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def sanitize(self, payload: str, user_profile: dict = None,
                 task: str = "", memory_traces: list = None) -> str:
        """Simple wrapper — returns only the sanitized text."""
        text, _ = self.sanitize_with_trace(payload, user_profile, task, memory_traces)
        return text

    def sanitize_with_trace(self, payload: str, user_profile: dict = None,
                             task: str = "", memory_traces: list = None):
        """
        Full four-stage pipeline.

        Returns
        -------
        (sanitized_text, stages)
        where stages = {"task_type": str, "spans": [Span, ...]}
        self.bindings is populated with {placeholder → true_value} after this call.
        """
        self.bindings = {}
        self._counter = 0
        p = user_profile or {}
        traces = memory_traces or []

        # ── Stage 1: Span Extraction ──────────────────────────────────────────
        spans = self._extract_spans(payload, p)

        # ── Stage 2: Scope Control ────────────────────────────────────────────
        current_type = self._infer_task_type(task)
        residue = self._collect_residue_values(traces, current_type)
        self._mark_scope_removals(spans, payload, residue)

        # Remove sentences containing dropped spans
        working = payload
        for s in spans:
            if not s.kept:
                working = self._drop_sentence_with(working, s.text)
        working = re.sub(r'  +', ' ', working).strip()

        # ── Stage 3: Classification ───────────────────────────────────────────
        for s in spans:
            if not s.kept:
                continue
            if s.span_type in self._DI_TYPES:
                s.span_class = "DI"
            elif s.span_type in self._CSS_TYPES:
                s.span_class = "CSS"
            else:
                s.span_class = "BEN"

        # ── Stage 4: Transformation ───────────────────────────────────────────
        result_text = working
        for s in spans:
            if not s.kept:
                s.result = "removed"
                continue

            if s.span_class == "DI":
                kind = self._DI_KIND.get(s.span_type, s.span_type.upper())
                ph = self._placeholder(kind)
                self.bindings[ph] = s.text
                s.result = ph
                result_text = re.sub(re.escape(s.text), ph,
                                     result_text, flags=re.IGNORECASE)
                # Catch first/last name parts appearing alone
                if s.span_type == "name":
                    for part in s.text.split():
                        if len(part) > 2:
                            result_text = re.sub(
                                r'\b' + re.escape(part) + r'\b', ph,
                                result_text, flags=re.IGNORECASE
                            )

            elif s.span_class == "CSS":
                abstracted = self._abstract_css(s, p)
                s.result = abstracted
                if s.span_type == "insurance_name":
                    # Prevent "dental insurance insurance" when text already
                    # has "BlueCross Dental Plus insurance"
                    result_text = re.sub(
                        re.escape(s.text) + r'(\s+insurance\b)?',
                        abstracted, result_text, flags=re.IGNORECASE
                    )
                else:
                    result_text = re.sub(re.escape(s.text), abstracted,
                                         result_text, flags=re.IGNORECASE)
                # For symptoms: clean up bare keywords that survived phrase replacement
                if s.span_type == "symptom":
                    result_text = self._clean_residual_symptom_kws(
                        result_text, abstracted)

            else:  # BEN
                s.result = s.text

        # ── Pattern-based fallback (DI + CSS not caught by span extraction) ───
        result_text = self._replace_pattern(result_text, self._PHONE_RE,  "PHONE")
        result_text = self._replace_pattern(result_text, self._SSN_RE,    "SSN")
        result_text = self._replace_pattern(result_text, self._CC_RE,     "CC")
        result_text = self._replace_pattern(result_text, self._INS_ID_RE, "INS_ID")
        result_text = self._replace_pattern(result_text, self._EMAIL_RE,  "EMAIL")
        result_text = self._ISO_DATE_RE.sub(self._date_repl, result_text)
        result_text = self._NL_DATE_RE.sub("this week", result_text)

        stages = {"task_type": current_type, "spans": spans}
        return result_text.strip(), stages

    # ── Stage 1 helpers ───────────────────────────────────────────────────────

    def _extract_spans(self, text: str, profile: dict) -> List[Span]:
        spans: List[Span] = []
        seen: Set[str] = set()

        def add(t: str, typ: str):
            key = t.strip().lower()
            if key and key not in seen:
                seen.add(key)
                spans.append(Span(text=t.strip(), span_type=typ))

        # Profile-bound — exact value matches (highest precision)
        for field, typ in [("name",           "name"),
                            ("dob",            "dob"),
                            ("address",        "address"),
                            ("insurance",      "insurance_name"),
                            ("ssn",            "ssn"),
                            ("credit_card",    "cc"),
                            ("driver_license", "dl"),
                            ("insurance_id",   "ins_id")]:
            val = profile.get(field, "")
            if val and re.search(re.escape(val), text, re.IGNORECASE):
                add(val, typ)

        # Pattern-based — catches values not in profile or partially present
        for pat, typ in [(self._PHONE_RE,  "phone"),
                          (self._EMAIL_RE,  "email"),
                          (self._SSN_RE,    "ssn"),
                          (self._CC_RE,     "cc"),
                          (self._INS_ID_RE, "ins_id")]:
            for m in pat.finditer(text):
                add(m.group(0), typ)

        # Dates (ISO + natural language)
        for pat in (self._ISO_DATE_RE, self._NL_DATE_RE):
            for m in pat.finditer(text):
                add(m.group(0), "date")

        # Symptom phrase (contextual detection)
        sym = self._find_symptom_span(text)
        if sym:
            add(sym, "symptom")

        return spans

    def _find_symptom_span(self, text: str) -> str:
        """Return the first symptom-bearing phrase found in text, or ''."""
        for m in self._SYMPTOM_CONTEXT_RE.finditer(text):
            phrase = m.group(1).strip().rstrip(".,;:")
            for keywords, _ in self._SYMPTOM_MAP:
                if any(kw in phrase.lower() for kw in keywords):
                    return phrase
        return ""

    # ── Stage 2 helpers ───────────────────────────────────────────────────────

    def _infer_task_type(self, task: str) -> str:
        t = task.lower()
        for typ, kws in self._TASK_TYPES.items():
            if any(kw in t for kw in kws):
                return typ
        return "general"

    def _collect_residue_values(self, traces: list, current_type: str) -> Set[str]:
        """Values from traces whose source workflow is a different task type."""
        residue: Set[str] = set()
        for trace in traces:
            source = trace.get("source", "")
            wf     = trace.get("from_workflow", "")
            data   = trace.get("data", {})

            if source in ("tool:get_calendar", "tool:get_location"):
                continue  # availability and location are always relevant

            wf_type = self._infer_task_type(wf)
            if wf_type in ("general", current_type):
                continue  # same task type — keep

            if isinstance(data, dict):
                for key in ("booked_at", "name", "address"):
                    val = data.get(key, "")
                    if val and len(val) > 3:
                        residue.add(val.lower())
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        val = item.get("name", "")
                        if val and len(val) > 3:
                            residue.add(val.lower())
        return residue

    def _mark_scope_removals(self, spans: List[Span],
                              text: str, residue: Set[str]) -> None:
        """Mark spans as removed if they match a residue value. Mutates spans in-place."""
        seen = {s.text.lower() for s in spans}

        for s in spans:
            if s.text.lower() in residue:
                s.kept = False
                s.removal_reason = "cross-workflow residue"

        # Residue values not yet present as spans — add them so they show in display
        for val in residue:
            if val not in seen:
                m = re.search(re.escape(val), text, re.IGNORECASE)
                if m:
                    new = Span(text=m.group(0), span_type="booking_residue",
                               kept=False, removal_reason="cross-workflow residue")
                    spans.append(new)
                    seen.add(val)

    def _drop_sentence_with(self, text: str, value: str) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        kept = [s for s in sentences
                if not re.search(re.escape(value), s, re.IGNORECASE)]
        return " ".join(kept)

    # ── Stage 4 helpers ───────────────────────────────────────────────────────

    def _abstract_css(self, span: Span, profile: dict) -> str:
        """Return the coarsened representation for a CSS span."""
        if span.span_type == "address":
            return self._coarsen_address(profile.get("address", span.text))
        if span.span_type == "insurance_name":
            return self._generalize_insurance(span.text)
        if span.span_type == "date":
            return ("this week (all day)" if re.search(r'all day', span.text, re.I)
                    else "this week")
        if span.span_type == "symptom":
            for keywords, label in self._SYMPTOM_MAP:
                if any(kw in span.text.lower() for kw in keywords):
                    return label
            return "health concern"
        return span.text

    def _coarsen_address(self, full_address: str) -> str:
        parts = [s.strip() for s in full_address.split(",")]
        city_state_raw = parts[1] if len(parts) > 1 else ""
        tokens = city_state_raw.split()
        return (f"{tokens[0]}, {tokens[1]} area"
                if len(tokens) >= 2 else "local area")

    def _generalize_insurance(self, name: str) -> str:
        lower = name.lower()
        for kws, label in self._INSURANCE_CATEGORY:
            if any(kw in lower for kw in kws):
                return label
        return "health insurance"

    def _clean_residual_symptom_kws(self, text: str, applied_label: str) -> str:
        """After phrase-level replacement, clear bare keywords from the same category."""
        _TRAILER = re.compile(
            r'\b(?:concern|insurance|coverage|care|appointment|service)\b',
            re.IGNORECASE
        )
        for keywords, label in self._SYMPTOM_MAP:
            if label != applied_label:
                continue
            for kw in sorted(keywords, key=len, reverse=True):
                m = re.search(r'\b' + re.escape(kw) + r'\b', text, re.IGNORECASE)
                if m:
                    after = text[m.end():m.end() + 20].strip()
                    if not _TRAILER.match(after):
                        text = text[:m.start()] + label + text[m.end():]
                        break
        return text

    @staticmethod
    def _date_repl(m: re.Match) -> str:
        raw = m.group(0)
        return "this week (all day)" if "(all day)" in raw else "upcoming"

    def _placeholder(self, kind: str) -> str:
        self._counter += 1
        return f"<{kind}_{self._counter}>"

    def _replace_pattern(self, text: str, pattern: re.Pattern, kind: str) -> str:
        def repl(m):
            ph = self._placeholder(kind)
            self.bindings[ph] = m.group(0)
            return ph
        return pattern.sub(repl, text)
