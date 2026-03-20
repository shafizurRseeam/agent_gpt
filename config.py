PROVIDER = "openai"

MODEL_CONFIG = {
    "openai": "gpt-4o-mini",
    "claude": "claude-3-sonnet-20240229",
    "gemini": "gemini-1.5-pro"
}

# ── Local model selection ──────────────────────────────────────────────────────
# Change LOCAL_MODEL to any key below to switch the LC for evaluation.
# Can also be overridden at runtime via:  python main.py --model llama3.2:1b
EVAL_LOCAL_MODELS = {
    "llama3.2":    "3B  — default, balanced",
    "llama3.2:1b": "1B  — smaller, less filtering, more over-disclosure",
    "llama3.1:8b": "8B  — larger, more capable / more conservative",
    "mistral":     "7B  — alternative architecture",
    "phi3":        "3.8B — Microsoft, instruction-tuned",
}
LOCAL_MODEL = "llama3.2"

# Token limits
LOCAL_MAX_TOKENS  = 8024
CLOUD_MAX_TOKENS  = 8024

CLOUD_TEMPERATURE = 0.7
LOCAL_TEMPERATURE = 0.9   # higher = more creative/varied decisions by the LC
