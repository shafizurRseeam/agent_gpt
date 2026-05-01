import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from google import genai as google_genai
from google.genai import types as genai_types

from config import PROVIDER, MODEL_CONFIG, CLOUD_MAX_TOKENS, CLOUD_TEMPERATURE

load_dotenv()


class CloudLLM:

    def __init__(self, provider: str = None):
        self.provider = provider or PROVIDER
        self.model = MODEL_CONFIG[self.provider]

        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self.client = OpenAI(api_key=api_key)

        elif self.provider == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self.client = anthropic.Anthropic(api_key=api_key)

        elif self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
            self.client = google_genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])

    def chat(self, messages: list) -> str:
        """Multi-turn conversation. messages is a list of {role, content} dicts."""

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=CLOUD_MAX_TOKENS,
                temperature=CLOUD_TEMPERATURE,
            )
            return response.choices[0].message.content

        elif self.provider == "claude":
            # Anthropic API: system message is a separate top-level field
            system_content = ""
            chat_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    chat_messages.append({"role": msg["role"], "content": msg["content"]})

            kwargs = dict(
                model=self.model,
                max_tokens=CLOUD_MAX_TOKENS,
                messages=chat_messages,
            )
            if system_content:
                kwargs["system"] = system_content

            response = self.client.messages.create(**kwargs)
            return response.content[0].text

        elif self.provider == "gemini":
            # Build a single prompt string: prepend system text to first user turn
            system_text = ""
            parts = []
            for msg in messages:
                if msg["role"] == "system":
                    system_text = msg["content"]
                elif msg["role"] == "user":
                    text = (system_text + "\n\n" + msg["content"]).strip() if system_text else msg["content"]
                    parts.append(text)
                    system_text = ""
                elif msg["role"] == "assistant":
                    parts.append(msg["content"])

            prompt = "\n".join(parts)
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=CLOUD_MAX_TOKENS,
                    temperature=CLOUD_TEMPERATURE,
                ),
            )
            return response.text
