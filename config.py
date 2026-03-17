PROVIDER = "openai"

MODEL_CONFIG = {
    "openai": "gpt-4o-mini",
    "claude": "claude-3-sonnet-20240229",
    "gemini": "gemini-1.5-pro"
}

LOCAL_MODEL = "llama3.2"   # swap to "phi3", "mistral", "llama3.1:8b", etc.

# Token limits
LOCAL_MAX_TOKENS  = 8024   # controls local LLM response length
CLOUD_MAX_TOKENS  = 8024    # controls cloud LLM response length

CLOUD_TEMPERATURE = 0.7
LOCAL_TEMPERATURE = 0.7
