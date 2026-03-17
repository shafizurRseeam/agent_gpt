import requests
from config import LOCAL_MODEL, LOCAL_MAX_TOKENS, LOCAL_TEMPERATURE


class LocalLLM:

    def __init__(self, model=LOCAL_MODEL):
        self.model        = model
        self.generate_url = "http://localhost:11434/api/generate"
        self.chat_url     = "http://localhost:11434/api/chat"

    def generate(self, prompt):
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"num_predict": LOCAL_MAX_TOKENS, "temperature": LOCAL_TEMPERATURE}
        }
        r = requests.post(self.generate_url, json=payload)
        return r.json()["response"]

    def chat(self, messages):
        """Multi-turn conversation. messages is a list of {role, content} dicts."""
        payload = {
            "model":    self.model,
            "messages": messages,
            "stream":   False,
            "options":  {"num_predict": LOCAL_MAX_TOKENS}
        }
        r = requests.post(self.chat_url, json=payload)
        return r.json()["message"]["content"]
