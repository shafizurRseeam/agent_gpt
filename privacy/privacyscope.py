"""
privacy/privacyscope.py

PrivScope — on-device payload sanitization pipeline.

Four-stage pipeline (§3 of the paper):
  Stage 1  Span Extraction     — three-layer extraction (regex + NER + noun chunks)
  Stage 2  Scope Control       — three signals: SemRel, Resid, Keep → retention rule
  Stage 3  Span Classification — DI / CSS / BEN
  Stage 4  Transformation      — DI → placeholder, CSS → abstraction, BEN → unchanged

Scope-control retention rule (Eq. 1 in the paper):
  K_t = { u_j | Keep(u_j) = 1  OR  ( SemRel(u_j, τ_t) >= ρ  AND  Resid(u_j) = 0 ) }

Public API:
  sanitize_with_trace(payload, user_profile, task, memory_traces)
      → (sanitized_text, stages_dict)
  sanitize(...)
      → sanitized_text
"""

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Tuple

import spacy
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util


# ── Lazy model singletons (loaded once, shared across calls) ──────────────────

_nlp_model       = None
_st_model        = None

def _nlp():
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = spacy.load("en_core_web_sm")
    return _nlp_model

def _st():
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


# ── Span record ───────────────────────────────────────────────────────────────

@dataclass
class Span:
    """One candidate disclosure unit tracked through the pipeline."""
    text: str
    span_type: str          # 'name' | 'phone' | 'address' | 'symptom' | ...
    source: str = ""        # extraction layer: 'regex' | 'ner' | 'noun_chunk' | 'profile'

    # Stage 3
    span_class: str = ""    # 'DI' | 'CSS' | 'BEN'

    # Stage 2 signals
    sem_rel: float = 0.0    # SemRel(u_j, τ_t) ∈ [0, 1]
    resid:   int   = 0      # Resid(u_j) ∈ {0, 1}   — cross-workflow provenance
    keep:    int   = 0      # Keep(u_j) ∈ {0, 1}    — local necessity

    # Stage 2 decision
    kept: bool = True
    removal_reason: str = ""

    # Stage 4 result
    result: str = ""


# ── PrivacyScope ──────────────────────────────────────────────────────────────

class PrivacyScope:

    # ── Layer 1: structured-pattern regexes ──────────────────────────────────
    _PHONE_RE    = re.compile(r'\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}')
    _SSN_RE      = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    _CC_RE       = re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')
    _INS_ID_RE   = re.compile(r'\b[A-Z]{2,5}-\d{4,8}-[A-Z0-9]{2,6}\b')
    _EMAIL_RE    = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b')
    _ZIP_RE      = re.compile(r'\b\d{5}(?:-\d{4})?\b')
    _ISO_DATE_RE = re.compile(r'\b(20\d{2}-\d{2}-\d{2}(?:\s*\([^)]+\))?)')
    _NL_DATE_RE  = re.compile(
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        r'\s+\d{1,2}(?:st|nd|rd|th)?'
        r'(?:\s+and\s+\d{1,2}(?:st|nd|rd|th)?)?'
        r'(?:,?\s*(?:of\s+)?\d{4})?',
        re.IGNORECASE
    )

    # ── Layer 2: spaCy NER label → span type ─────────────────────────────────
    _NER_TYPE_MAP = {
        "PERSON":   "person_ner",
        "ORG":      "org_ner",
        "GPE":      "location_ner",
        "LOC":      "location_ner",
        "FAC":      "location_ner",
        "DATE":     "date",
        "TIME":     "date",
        "CARDINAL": "quantity_ner",
        "QUANTITY": "quantity_ner",
        "MONEY":    "quantity_ner",
    }

    # ── Symptom / clinical category map (for CSS abstraction) ─────────────────
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

    # ── Insurance generalization ───────────────────────────────────────────────
    _INSURANCE_CATEGORY = [
        ({"dental", "teeth", "oral"},   "dental insurance"),
        ({"vision", "eye", "optical"},  "vision insurance"),
        ({"mental", "behav", "psych"},  "mental health insurance"),
        ({"medicare", "medicaid"},      "government health insurance"),
    ]

    # ── Span classification ───────────────────────────────────────────────────
    # DI: direct identifiers — always replaced with placeholders
    _DI_TYPES  = frozenset({"name", "phone", "ssn", "cc", "email",
                             "dob", "dl", "ins_id", "person_ner"})
    # CSS: context-sensitive — abstracted to coarser form
    _CSS_TYPES = frozenset({"address", "insurance_name", "date",
                             "zip", "location_ner", "noun_phrase",
                             "org_ner", "quantity_ner"})

    _DI_KIND = {
        "name":       "NAME",  "person_ner": "NAME",
        "phone":      "PHONE", "ssn":        "SSN",
        "cc":         "CC",    "email":      "EMAIL",
        "dob":        "DOB",   "dl":         "DL",
        "ins_id":     "INS_ID",
    }

    # ── Keep=1 types — needed locally for form-fill / local resolution ─────────
    # These always survive scope control (they appear as placeholders, not raw text)
    _KEEP_TYPES = frozenset({"name", "phone", "email", "dob", "ssn", "cc",
                              "dl", "ins_id", "person_ner",
                              "address", "insurance_name", "ins_id",
                              "date"})   # dates for appointment scheduling

    # ── Task-type keywords (scope-control provenance) ─────────────────────────
    _TASK_TYPES = {
        "dental":     {"tooth", "teeth", "dental", "dentist", "gum", "oral", "cavity"},
        "medical":    {"pain", "injury", "sick", "fever", "doctor", "clinic",
                       "hospital", "stool", "blood", "swelling", "knee", "treatment"},
        "restaurant": {"restaurant", "dinner", "lunch", "eat", "dining",
                       "food", "reservation"},
        "garage":     {"car", "oil", "tire", "brake", "mechanic",
                       "garage", "vehicle", "auto"},
    }

    # Relevance threshold ρ
    _RHO = 0.10

    # Signals that a sentence describes historical/past activity
    _HISTORY_RE = re.compile(
        r"\b(previously|i'?ve had|prior|past|had appointments? at|"
        r"been to|seen at|visited|before)\b",
        re.IGNORECASE,
    )

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self):
        self.bindings: Dict[str, str] = {}
        self._counter = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def sanitize(self, payload: str, user_profile: dict = None,
                 task: str = "", memory_traces: list = None) -> str:
        text, _ = self.sanitize_with_trace(payload, user_profile, task, memory_traces)
        return text

    def sanitize_with_trace(self, payload: str, user_profile: dict = None,
                             task: str = "", memory_traces: list = None):
        """
        Full pipeline. Returns (sanitized_text, stages).
        stages = {"task_type": str, "rho": float, "spans": [Span, ...]}
        self.bindings is populated after this call.
        """
        self.bindings = {}
        self._counter = 0
        p      = user_profile or {}
        traces = memory_traces or []

        # ── Stage 1: Span Extraction ──────────────────────────────────────────
        spans = self._extract_spans(payload, p)

        # ── Stage 2: Scope Control ────────────────────────────────────────────
        current_type   = self._infer_task_type(task)
        residue        = self._collect_residue_values(traces, current_type)
        historical_org = self._historical_org_names(payload)
        task_emb       = _st().encode(task, convert_to_tensor=True)
        self._compute_scope_signals(spans, payload, task_emb, residue, historical_org)

        # Remove sentences containing dropped spans from the working text
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
                ph   = self._placeholder(kind)
                self.bindings[ph] = s.text
                s.result = ph
                result_text = re.sub(re.escape(s.text), ph,
                                     result_text, flags=re.IGNORECASE)
                if s.span_type in ("name", "person_ner"):
                    # Temporarily protect email addresses from name-part substitution
                    _email_prot: Dict[str, str] = {}
                    def _prot(m, _ep=_email_prot):
                        k = f"__EP{len(_ep)}__"
                        _ep[k] = m.group(0)
                        return k
                    guarded = self._EMAIL_RE.sub(_prot, result_text)
                    for part in s.text.split():
                        if len(part) > 2:
                            guarded = re.sub(
                                r'\b' + re.escape(part) + r'\b', ph,
                                guarded, flags=re.IGNORECASE
                            )
                    for k, v in _email_prot.items():
                        guarded = guarded.replace(k, v)
                    result_text = guarded

            elif s.span_class == "CSS":
                abstracted = self._abstract_css(s, p)
                s.result   = abstracted
                if s.span_type == "insurance_name":
                    result_text = re.sub(
                        re.escape(s.text) + r'(\s+insurance\b)?',
                        abstracted, result_text, flags=re.IGNORECASE
                    )
                else:
                    result_text = re.sub(re.escape(s.text), abstracted,
                                         result_text, flags=re.IGNORECASE)
                if s.span_type == "noun_phrase":
                    result_text = self._clean_residual_symptom_kws(
                        result_text, abstracted)

            else:  # BEN
                s.result = s.text

        # Pattern-based fallback for anything not caught by span extraction
        result_text = self._replace_pattern(result_text, self._PHONE_RE,  "PHONE")
        result_text = self._replace_pattern(result_text, self._SSN_RE,    "SSN")
        result_text = self._replace_pattern(result_text, self._CC_RE,     "CC")
        result_text = self._replace_pattern(result_text, self._INS_ID_RE, "INS_ID")
        result_text = self._replace_pattern(result_text, self._EMAIL_RE,  "EMAIL")
        result_text = self._ISO_DATE_RE.sub(self._date_repl, result_text)
        result_text = self._NL_DATE_RE.sub(self._nl_date_repl, result_text)

        stages = {
            "task_type": current_type,
            "rho":       self._RHO,
            "spans":     spans,
        }
        return result_text.strip(), stages

    # ── Stage 1: Span Extraction ──────────────────────────────────────────────

    def _extract_spans(self, text: str, profile: dict) -> List[Span]:
        candidates: List[Tuple[int, int, Span]] = []   # (start, end, Span)
        seen: Set[str] = set()

        def try_add(matched_text: str, typ: str, source: str,
                    start: int, end: int):
            key = matched_text.strip().lower()
            if not key or len(key) < 2:
                return
            if key in seen:
                return
            seen.add(key)
            candidates.append((start, end,
                                Span(text=matched_text.strip(),
                                     span_type=typ, source=source)))

        # ── Layer 0: profile-bound exact matches ─────────────────────────────
        for pfield, typ in [("name",           "name"),
                             ("dob",            "dob"),
                             ("address",        "address"),
                             ("insurance",      "insurance_name"),
                             ("ssn",            "ssn"),
                             ("credit_card",    "cc"),
                             ("driver_license", "dl"),
                             ("insurance_id",   "ins_id")]:
            val = profile.get(pfield, "")
            if not val:
                continue
            m = re.search(re.escape(val), text, re.IGNORECASE)
            if m:
                try_add(val, typ, "profile", m.start(), m.end())

        # ── Layer 1: structured-pattern rules ────────────────────────────────
        for pat, typ in [
            (self._PHONE_RE,    "phone"),
            (self._EMAIL_RE,    "email"),
            (self._SSN_RE,      "ssn"),
            (self._CC_RE,       "cc"),
            (self._INS_ID_RE,   "ins_id"),
            (self._ZIP_RE,      "zip"),
            (self._ISO_DATE_RE, "date"),
            (self._NL_DATE_RE,  "date"),
        ]:
            for m in pat.finditer(text):
                try_add(m.group(0), typ, "regex", m.start(), m.end())

        # ── Layer 2: spaCy NER ────────────────────────────────────────────────
        doc = _nlp()(text)
        for ent in doc.ents:
            typ = self._NER_TYPE_MAP.get(ent.label_)
            if typ:
                try_add(ent.text, typ, "ner", ent.start_char, ent.end_char)

        # ── Layer 3: spaCy noun chunks ────────────────────────────────────────
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:
                try_add(chunk.text, "noun_phrase", "noun_chunk",
                        chunk.start_char, chunk.end_char)

        # ── Merge: longest-span precedence ───────────────────────────────────
        # Sort by span length descending, then by start position
        candidates.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
        merged: List[Tuple[int, int, Span]] = []
        for start, end, span in candidates:
            # Accept if it does not overlap with any already-accepted span
            if not any(s <= start < e or s < end <= e or
                       (start <= s and end >= e)
                       for s, e, _ in merged):
                merged.append((start, end, span))

        # Return spans sorted by position in original text
        merged.sort(key=lambda x: x[0])
        return [sp for _, _, sp in merged]

    # ── Stage 2: Scope Control ────────────────────────────────────────────────

    def _compute_scope_signals(self, spans: List[Span], text: str,
                                task_emb, residue: Set[str],
                                historical_org: Set[str]) -> None:
        """
        Compute SemRel, Resid, Keep for every span and apply the retention rule:
          kept ↔  Keep = 1  OR  (SemRel >= ρ AND Resid = 0)
        Mutates spans in-place.
        """
        st = _st()

        # Batch-encode all span texts for efficiency
        if spans:
            span_texts = [s.text for s in spans]
            span_embs  = st.encode(span_texts, convert_to_tensor=True)
        else:
            return

        for i, s in enumerate(spans):
            # ── SemRel: cosine similarity to the task ─────────────────────────
            s.sem_rel = float(
                st_util.cos_sim(span_embs[i], task_emb).item()
            )

            # ── Resid: cross-workflow provenance ──────────────────────────────
            # Two provenance sources:
            #   1. memory_traces from past bookings / cross-type workflows
            #   2. org names appearing in historical-context sentences in the payload
            #      (e.g. "Previously, I've had appointments at UR Medicine …")
            in_trace_residue   = s.text.lower() in residue
            in_history_context = (
                s.span_type == "org_ner"
                and s.text.lower() in historical_org
            )
            s.resid = 1 if (in_trace_residue or in_history_context) else 0

            # ── Keep: local necessity ─────────────────────────────────────────
            # Spans needed for local resolution / form filling even if cloud
            # doesn't need them (e.g. exact address, phone, insurance for actuation)
            s.keep = 1 if s.span_type in self._KEEP_TYPES else 0

            # ── Retention rule ────────────────────────────────────────────────
            keep_by_rule = (
                s.keep == 1
                or (s.sem_rel >= self._RHO and s.resid == 0)
            )
            if not keep_by_rule:
                s.kept = False
                s.removal_reason = (
                    "cross-workflow residue" if s.resid
                    else f"SemRel {s.sem_rel:.2f} < ρ={self._RHO:.2f}"
                )

        # Also add residue values from traces that didn't appear as extracted spans
        seen_texts = {s.text.lower() for s in spans}
        for val in residue:
            if val not in seen_texts:
                m = re.search(re.escape(val), text, re.IGNORECASE)
                if m:
                    new = Span(text=m.group(0), span_type="booking_residue",
                               source="trace_provenance",
                               sem_rel=0.0, resid=1, keep=0,
                               kept=False,
                               removal_reason="cross-workflow residue")
                    spans.append(new)

    def _collect_residue_values(self, traces: list, current_type: str) -> Set[str]:
        residue: Set[str] = set()
        for trace in traces:
            source = trace.get("source", "")
            wf     = trace.get("from_workflow", "")
            data   = trace.get("data", {})
            if source in ("tool:get_calendar", "tool:get_location"):
                continue
            # Past bookings are ALWAYS residue — a previous appointment location
            # is never needed when making a new booking request to the cloud.
            if source == "tool:book_appointment":
                if isinstance(data, dict):
                    for key in ("booked_at", "name", "address"):
                        val = data.get(key, "")
                        if val and len(val) > 3:
                            residue.add(val.lower())
                continue
            # Other tool sources: only flag if from a different workflow type
            wf_type = self._infer_task_type(wf)
            if wf_type in ("general", current_type):
                continue
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

    def _historical_org_names(self, text: str) -> Set[str]:
        """
        Extract org names mentioned in historical-context sentences
        (e.g. "Previously, I've had appointments at UR Medicine …").
        Flags LC-generated past-provider names as Resid=1 even when they
        are absent from memory_traces.
        """
        historical: Set[str] = set()
        for sent in re.split(r'(?<=[.!?])\s+', text):
            if self._HISTORY_RE.search(sent):
                doc = _nlp()(sent)
                for ent in doc.ents:
                    if ent.label_ == "ORG":
                        historical.add(ent.text.lower())
        return historical

    def _infer_task_type(self, task: str) -> str:
        t = task.lower()
        for typ, kws in self._TASK_TYPES.items():
            if any(kw in t for kw in kws):
                return typ
        return "general"

    def _drop_sentence_with(self, text: str, value: str) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        kept = [s for s in sentences
                if not re.search(re.escape(value), s, re.IGNORECASE)]
        return " ".join(kept)

    # ── Stage 4: Transformation helpers ──────────────────────────────────────

    def _abstract_css(self, span: Span, profile: dict) -> str:
        if span.span_type == "address":
            return self._coarsen_address(profile.get("address", span.text))
        if span.span_type == "insurance_name":
            return self._generalize_insurance(span.text)
        if span.span_type == "date":
            return self._abstract_date(span.text)
        if span.span_type == "zip":
            # Extract city from profile address
            addr = profile.get("address", "")
            parts = [s.strip() for s in addr.split(",")]
            city = parts[1].split()[0] if len(parts) > 1 else "local"
            return f"{city} area"
        if span.span_type in ("location_ner",):
            return "local area"
        if span.span_type == "noun_phrase":
            # Try to generalise as symptom
            for kws, label in self._SYMPTOM_MAP:
                if any(kw in span.text.lower() for kw in kws):
                    return label
            return span.text    # benign noun phrase — unchanged
        if span.span_type == "org_ner":
            return "a local provider"
        return span.text

    def _abstract_date(self, text: str) -> str:
        """
        Distinguish date semantics:
          • Age expression  ("32 years old")         → "in their 30s"
          • Past year / DOB ("July 14th, 1993")       → "1990s"
          • Scheduling date ("March 18th, 2026 …")   → "this week"
        """
        if re.search(r'all\s+day', text, re.I):
            return "this week (all day)"
        # Age expression
        age_m = re.search(r'\b(\d{1,3})\s+years?\s+old\b', text, re.I)
        if age_m:
            age = int(age_m.group(1))
            return f"in their {(age // 10) * 10}s"
        # Past year (DOB-like): e.g., "July 14th, 1993" or "1993-07-14"
        year_m = re.search(r'\b(19\d{2}|200\d|201\d)\b', text)
        if year_m:
            year = int(year_m.group(1))
            return f"{(year // 10) * 10}s"
        # Future / scheduling date
        return "this week"

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
        _TRAILER = re.compile(
            r'\b(?:concern|insurance|coverage|care|appointment|service)\b',
            re.IGNORECASE
        )
        for kws, label in self._SYMPTOM_MAP:
            if label != applied_label:
                continue
            for kw in sorted(kws, key=len, reverse=True):
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

    @staticmethod
    def _nl_date_repl(m: re.Match) -> str:
        """
        Fallback replacement for NL dates not caught by span extraction.
        Distinguish past years (DOB) from scheduling/future dates.
        """
        raw = m.group(0)
        year_m = re.search(r'\b(19\d{2}|200\d|201\d)\b', raw)
        if year_m:
            year = int(year_m.group(1))
            return f"{(year // 10) * 10}s"
        return "this week"

    def _placeholder(self, kind: str) -> str:
        self._counter += 1
        return f"<{kind}_{self._counter}>"

    def _replace_pattern(self, text: str, pattern: re.Pattern, kind: str) -> str:
        def repl(m):
            ph = self._placeholder(kind)
            self.bindings[ph] = m.group(0)
            return ph
        return pattern.sub(repl, text)
