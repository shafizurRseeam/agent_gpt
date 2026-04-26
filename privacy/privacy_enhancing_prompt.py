"""
privacy/privacy_enhancing_prompt.py

Baseline 1 — Privacy-Enhancing Prompt (PEP)

Prompt-level behavioral steering: the local LLM is given a short
privacy-conscious system prompt and asked to rewrite the cloud-bound
payload.  No explicit filtering module; no preprocessing stage; no
few-shot examples.  The agent is trusted to self-censor based on the
high-level instruction alone.

Interface matches PrivacyScope.sanitize():
    sanitize(payload, user_profile, task, memory_traces, local_llm)
        → sanitized_text
"""

SYSTEM_PROMPT = (
    "You are a helpful and privacy-conscious AI Agent who can utilize "
    "external tools to help the user accomplish tasks following their "
    "instructions. Your primary objective is to provide accurate and "
    "efficient assistance while maintaining the highest level of discretion "
    "and security. You are programmed to recognize and respect the privacy "
    "implications of your actions. "
    "When asked to sanitize a message, you MUST output the sanitized version. "
    "Never refuse — always produce the rewritten message."
)


def sanitize(payload: str,
             user_profile: dict = None,
             task: str = "",
             memory_traces: list = None,
             local_llm=None) -> str:
    """
    Rewrite *payload* with privacy-conscious behavioral steering only.
    Requires a LocalLLM instance via *local_llm*.
    """
    if local_llm is None:
        raise ValueError("privacy_enhancing_prompt.sanitize() requires a local_llm instance")

    user_msg = (
        f"Sanitize the following message before it is sent to an external cloud "
        f"service. The task is: \"{task}\"\n\n"
        f"Instructions:\n"
        f"- Keep all symptoms, complaints, and information directly needed to "
        f"complete the task.\n"
        f"- Remove or generalize personal identifiers that the cloud does not "
        f"need (e.g. full name, exact DOB, SSN, credit card, exact address).\n"
        f"- You MUST output the sanitized message. Do not refuse or explain.\n\n"
        f"Original message:\n{payload}\n\n"
        f"Sanitized message:"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    result = local_llm.chat(messages).strip()

    # If the LLM output starts with a "Sanitized message:" label, strip it
    for marker in ("Sanitized message:", "Sanitized:", "Rewritten:", "Output:"):
        if result.lower().startswith(marker.lower()):
            result = result[len(marker):].strip()
            break

    # Drop any intro sentence ("Here is the sanitized version:", etc.)
    for prefix in ("Here is", "Here's", "Sure,", "Certainly,", "Of course,"):
        if result.lower().startswith(prefix.lower()):
            first_newline = result.find("\n")
            if first_newline != -1:
                candidate = result[first_newline:].strip()
                if len(candidate) > 30:
                    result = candidate
                    break

    return result
