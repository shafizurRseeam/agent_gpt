"""
privacy/pep.py

Baseline — Privacy-Enhancing Prompt (PEP)

Given the LC-enriched payload p_t, the local LLM acts as a light-touch
privacy filter: it removes or generalises information it judges to be
sensitive for the delegated cloud search while preserving everything
that is needed to complete the task.

No preprocessing pipeline; no span extraction; no scope control.
The model relies entirely on its own judgment about what is sensitive,
trading privacy reduction against task utility with a single prompt.

This is the minimal-intervention baseline: it represents what you get
by asking a capable local LLM to "be a bit careful" before forwarding a
payload to the cloud, without any structured filtering machinery.

Interface:
    sanitize(payload, user_profile, task, memory_traces, local_llm)
        → sanitized_text
    sanitize_with_trace(...)
        → (sanitized_text, {"method": "pep", "raw_output": sanitized_text})
"""

from __future__ import annotations

_SYSTEM_PROMPT = (
    "You are a helpful privacy-conscious assistant preparing a message for a cloud search service. "
    "Rewrite the user's message so it is clear and useful for finding suitable options. "
    "Keep details that may help the cloud service return relevant results, and add reasonable "
    "search-relevant details if they are necessary to make the request actionable. "
    "Do not over-redact; the rewritten request should preserve enough context for the service to be useful."
)

_USER_TEMPLATE = """\
The user wants to: {task}

Full message:
{payload}

Rewrite this as a search request for a cloud search service.

Guidelines:
- Keep details that may help find relevant options.
- Add reasonable search-relevant details if needed to make the request actionable.
- Keep the request natural, specific, and actionable.
- Phrase it as finding suitable options, not completing the final booking.
- Preserve the full length and detail of the original message — do not summarise or shorten it.
- Return only the rewritten request. No explanation.
"""


def sanitize(
    payload:       str,
    user_profile:  dict = None,
    task:          str  = "",
    memory_traces: list = None,
    local_llm             = None,
) -> str:
    result, _ = sanitize_with_trace(payload, user_profile, task, memory_traces, local_llm)
    return result


def sanitize_with_trace(
    payload:       str,
    user_profile:  dict = None,
    task:          str  = "",
    memory_traces: list = None,
    local_llm             = None,
):
    """
    Returns (sanitized_text, trace_dict).
    trace_dict = {"method": "pep", "raw_output": sanitized_text}
    """
    if local_llm is None:
        raise ValueError("pep.sanitize() requires a local_llm instance")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _USER_TEMPLATE.format(
            task=task.strip() or "complete the user's request",
            payload=payload,
        )},
    ]

    raw = local_llm.chat(messages).strip()

    # Strip common LLM preamble labels
    for marker in (
        "Cleaned message:", "Sanitized message:", "Sanitized:", "Rewritten:",
        "Output:", "Here is", "Here's", "Sure,", "Certainly,", "Of course,",
    ):
        if raw.lower().startswith(marker.lower()):
            rest = raw[len(marker):].strip()
            if len(rest) > 20:
                raw = rest
            break

    return raw, {"method": "pep", "raw_output": raw}


# ── Standalone — runs PEP baseline on the dentist example payload ──────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from llm.local_llm import LocalLLM

    r_t = (
        "I have tooth pain and bleeding gums, "
        "book me a dentist appointment at the earliest."
    )
    p_t = (
        "Hi, I'm Bob Smith and you can reach me at 585-555-1212 or bob@example.com. "
        "I need a dentist near 12 ABC St, Rochester, NY. I have BlueCross Dental Plus. "
        "My insurance ID is BC-123456-A9. I am available on March 18 and March 19, "
        "preferably before 10:30 AM. If possible, book for 2 people. "
        "I previously visited Bright Smile Dental and Lake Dental Care. "
        "I have tooth pain and bleeding gums. My ZIP is 14623 and I want to keep "
        "the cost under $500."
    )

    W = 70

    print(f"\n{'═' * W}")
    print(f"  BASELINE: PEP  (Privacy-Enhancing Prompt)")
    print(f"{'═' * W}")

    print(f"\n{'═' * W}")
    print(f"  ORIGINAL USER REQUEST  (r_t)")
    print(f"{'═' * W}")
    print(f"  {r_t}")

    print(f"\n{'═' * W}")
    print(f"  INPUT  (LC-enriched payload p_t)")
    print(f"{'═' * W}")
    print(f"  {p_t}")

    try:
        local_llm = LocalLLM()
        local_llm.generate("ping")
        print(f"\n  Local LLM: connected ({local_llm.model})")
    except Exception as e:
        print(f"\n  Local LLM: unavailable ({e})")
        sys.exit(1)

    result, trace = sanitize_with_trace(p_t, task=r_t, local_llm=local_llm)

    print(f"\n{'═' * W}")
    print(f"  PEP OUTPUT  — sent to CLM")
    print(f"{'═' * W}")
    print(f"  {result}")
