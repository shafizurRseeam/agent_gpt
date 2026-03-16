import requests


class LocalLLM:

    def __init__(self, model="phi3"):
        self.url = "http://localhost:11434/api/generate"
        self.model = model

    def generate(self, prompt):

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        r = requests.post(self.url, json=payload)

        return r.json()["response"]