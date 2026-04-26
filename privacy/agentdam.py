"""
privacy/agentdam.py

Baseline 2 — AGENTDAM

Stronger prompt-level behavioral steering with explicit data-minimization
instructions, a clear definition of sensitive information, and a few
chain-of-thought demonstrations.

Distinction from Privacy-Enhancing Prompt:
  • Explicit data-minimization principle (only use what the task requires)
  • Enumerated list of sensitive-data categories
  • Three CoT examples showing how to reason and strip irrelevant PII

No LLM pre-filter or post-filter is implemented; this is purely
prompt-level steering.

Interface:
    sanitize(payload, user_profile, task, memory_traces, local_llm)
        → sanitized_text
"""

_SYSTEM_PROMPT = """\
You are a message sanitizer for a privacy-preserving AI Agent. Your job \
is to rewrite cloud-bound messages so they follow the principle of data \
minimization: only keep information that is directly necessary to \
complete the current task. You MUST always produce a rewritten message. \
Never refuse.

Sensitive information to minimize or remove (unless strictly required):
- Full name, date of birth, national ID / SSN, driver's license number
- Home or work address beyond what is needed for geolocation
- Phone numbers and email addresses when not needed for contact
- Financial details: credit card numbers, bank accounts
- Insurance policy numbers and IDs
- Records from previous, unrelated tasks or workflows
- Any personal data that goes beyond the minimum needed for the task

Always keep: the user's main complaint or task request, general location \
(city/region), relevant insurance type (not ID), and availability dates.\
"""

_EXAMPLES = """\
---
Example 1
Task: "Book a table at an Italian restaurant for 2 people tonight."
Original message: "Hi, I'm Alice Johnson, DOB 1988-04-12, SSN 987-65-4321, \
at 12 Maple Ave, Boston MA. My phone is 617-555-0199 and email is \
alice@email.com. I need a table for 2 tonight at an Italian restaurant. \
Last month I also visited City Dental for a tooth extraction."

Reasoning: The restaurant only needs the request (2 people, Italian, \
tonight) and possibly a contact for the reservation confirmation. \
Name is useful, phone or email is useful for confirmation. Address, \
SSN, DOB, dental history are entirely unnecessary.

Rewritten: "I'd like to book a table for 2 at an Italian restaurant \
tonight. Contact: Alice, 617-555-0199."

---
Example 2
Task: "Find an orthopedic clinic for knee pain."
Original message: "This is Bob Smith, born July 14 1993, SSN 123-45-6789, \
living at 45 Elm St Rochester NY 14623, phone 585-555-0192, \
email bob@example.com, insurance BlueCross Dental Plus ID BC-1234. \
I have knee pain and swelling. Previously I booked Bright Smile Dental \
for a tooth issue and Tony's Italian Kitchen for dinner."

Reasoning: Finding an orthopedic clinic requires the symptom (knee pain \
and swelling), a general location (Rochester NY area), and insurance \
type for eligibility. Full address, SSN, DOB, credit card, past dental \
and restaurant bookings are not needed.

Rewritten: "I'm looking for an orthopedic clinic in the Rochester, NY \
area for knee pain and swelling. I have BlueCross health insurance. \
Available on March 18 or 19."

---
Example 3
Task: "Reserve a slot at an auto garage for an oil change."
Original message: "I'm Carol White, 33 years old, at 88 Oak Rd, \
Seattle WA 98101, SSN 456-78-9012, phone 206-555-0177, \
credit card 4111-1111-1111-1111. I need an oil change for my Toyota Camry. \
I also recently visited Riverside Medical for a checkup."

Reasoning: Garage needs vehicle info (Toyota Camry, oil change), \
location (Seattle area), and possibly a contact. SSN, credit card, \
medical visit are irrelevant.

Rewritten: "I need an oil change for a Toyota Camry in the Seattle, WA \
area. Contact: Carol, 206-555-0177."
---
"""


def sanitize(payload: str,
             user_profile: dict = None,
             task: str = "",
             memory_traces: list = None,
             local_llm=None) -> str:
    """
    Rewrite *payload* using explicit data-minimization steering + CoT demos.
    Requires a LocalLLM instance via *local_llm*.
    """
    if local_llm is None:
        raise ValueError("agentdam.sanitize() requires a local_llm instance")

    user_msg = (
        f"{_EXAMPLES}\n"
        f"Now apply the same reasoning to the following. "
        f"You MUST produce a Rewritten output — do not refuse.\n\n"
        f"Task: \"{task}\"\n"
        f"Original message:\n{payload}\n\n"
        f"Reasoning: <think step by step about what information is strictly "
        f"needed for this task and what should be omitted>\n"
        f"Rewritten: <the minimized cloud message keeping only what is necessary "
        f"— include the main complaint, general location, insurance type, "
        f"and availability>"
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    raw = local_llm.chat(messages).strip()

    # Extract only the "Rewritten:" section from the CoT output
    rewritten_match = None
    for marker in ("Rewritten:", "REWRITTEN:", "rewritten:"):
        idx = raw.rfind(marker)
        if idx != -1:
            candidate = raw[idx + len(marker):].strip().strip('"').strip()
            if len(candidate) > 20:
                rewritten_match = candidate
                break

    if rewritten_match:
        # Drop any trailing "Reasoning:" block that sometimes leaks through
        trailing = rewritten_match.find("\nReasoning:")
        if trailing != -1:
            rewritten_match = rewritten_match[:trailing].strip()
        return rewritten_match

    # Fallback: if the LLM did not use the label, return the last paragraph
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    return paragraphs[-1] if paragraphs else raw
