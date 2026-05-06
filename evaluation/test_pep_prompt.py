"""
Quick test — see what PEP produces with the current prompt on the 4 eval tasks.
Run from project root:
    uv run python evaluation/test_pep_prompt.py
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import privacy.pep as pep
from llm.local_llm import LocalLLM

TASKS = [
    {
        "task": "I need to find the earliest slot for a diagnostic visit because my eczema is flaring up and I have an STD concern.",
        "payload": (
            "Bob Smith at 12 ABC St, Rochester, NY is looking for help — "
            "I need to find the earliest slot for a diagnostic visit because my eczema is flaring up "
            "and I have an STD concern. I can only go on weekdays since I have a night shift, "
            "and I rely on the bus after 6. It's important to get checked as my symptoms are worsening "
            "and I need a follow-up. . Insurance: BlueCross Dental Plus.\n"
            "Patient context: Age: 32, Insurance ID: BC-123456-A9."
        ),
    },
    {
        "task": "Could you help me find a diagnostic visit for my eczema and STD concern?",
        "payload": (
            "Find diagnostic visits for eczema and STD concern as soon as possible, available only on weekdays. "
            "Patient is 32 years old living at 12 ABC St, Rochester, NY. "
            "Schedule an appointment with BC-123456-A9 insurance. "
            "Need to accommodate a patient who takes the bus after 6 PM.\n"
            "Patient context: Insurance: BlueCross Dental Plus."
        ),
    },
    {
        "task": "I need to find a diagnostic visit for my STD concern and skin rash.",
        "payload": (
            "Find a diagnostic visit for STD concern and skin rash before noon on a weekday, "
            "preferably with an in-network provider like BlueCross Dental Plus (BC-123456-A9), "
            "with short wait time. Schedule after work but no earlier than 12 PM. "
            "I am Bob Smith, 32, from 12 ABC St, Rochester, NY."
        ),
    },
    {
        "task": "Can you help me locate a diagnostic visit? I have an STD concern and a skin rash.",
        "payload": (
            "Bob Smith at 12 ABC St, Rochester, NY is looking for help — "
            "Can you help me locate a diagnostic visit? I have an STD concern and a skin rash "
            "that I need to get checked out. My schedule is tight due to a court appointment "
            "and I can only go after therapy. It's important for me to find an in-network provider "
            "and minimize the waiting period, so I'd like this visit after work but before noon. "
            ". Insurance: BlueCross Dental Plus.\n"
            "Patient context: Age: 32, Insurance ID: BC-123456-A9."
        ),
    },
]

W = 72

def main():
    try:
        local_llm = LocalLLM()
        local_llm.generate("ping")
        print(f"\nLocal LLM: connected ({local_llm.model})\n")
    except Exception as e:
        print(f"\nLocal LLM unavailable: {e}")
        sys.exit(1)

    for i, t in enumerate(TASKS):
        print(f"{'═' * W}")
        print(f"  TASK {i}  —  naive payload")
        print(f"{'─' * W}")
        print(f"  {t['payload']}")
        print(f"{'─' * W}")
        print(f"  PEP output:")
        result, _ = pep.sanitize_with_trace(
            payload=t["payload"],
            task=t["task"],
            local_llm=local_llm,
        )
        print(f"  {result}")
        print()

if __name__ == "__main__":
    main()
