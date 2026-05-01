PROVIDER = "openai"

MODEL_CONFIG = {
    "openai": "gpt-4o-mini-2024-07-18",
    "claude": "claude-sonnet-4-6",
    "gemini": "gemini-2.5-flash",
}

# ── Local model selection ──────────────────────────────────────────────────────
# Change LOCAL_MODEL to any key below to switch the LC for evaluation.
# Can also be overridden at runtime via:  python main.py --model llama3.2:1b
EVAL_LOCAL_MODELS = {
    "llama3.2":     "3B   — default, balanced",
    "phi3":         "3.8B — Microsoft, instruction-tuned",
    "mistral":      "7B   — alternative architecture, strong generative",
    "qwen2.5:7b":   "7B   — Alibaba, strong instruction-following and classification",
    "llama3.1:8b":  "8B   — larger, more capable / more conservative",
}
LOCAL_MODEL = "llama3.2"

# Token limits
LOCAL_MAX_TOKENS  = 8024
CLOUD_MAX_TOKENS  = 8024

CLOUD_TEMPERATURE = 0.7
LOCAL_TEMPERATURE = 0.9   # higher = more creative/varied decisions by the LC
