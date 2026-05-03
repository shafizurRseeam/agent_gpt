import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from google import genai as google_genai
from google.genai import types as genai_types

from config import PROVIDER, MODEL_CONFIG, CLOUD_MAX_TOKENS, CLOUD_TEMPERATURE

load_dotenv(override=True)


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
        """Multi-turn conversation. Returns response text only."""
        text, _, _ = self.chat_with_usage(messages)
        return text

    def chat_with_usage(self, messages: list) -> tuple:
        """
        Multi-turn conversation.
        Returns (text, input_tokens, output_tokens).
        Token counts come from each API's native usage object.
        """
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=CLOUD_MAX_TOKENS,
                temperature=CLOUD_TEMPERATURE,
            )
            text = response.choices[0].message.content
            in_tok  = response.usage.prompt_tokens
            out_tok = response.usage.completion_tokens
            return text, in_tok, out_tok

        elif self.provider == "claude":
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
            text    = response.content[0].text
            in_tok  = response.usage.input_tokens
            out_tok = response.usage.output_tokens
            return text, in_tok, out_tok

        elif self.provider == "gemini":
            system_text = ""
            parts = []
            for msg in messages:
                if msg["role"] == "system":
                    system_text = msg["content"]
                elif msg["role"] == "user":
                    parts.append(msg["content"])
                elif msg["role"] == "assistant":
                    parts.append(msg["content"])

            prompt = "\n".join(parts)
            config_kwargs = dict(
                max_output_tokens=CLOUD_MAX_TOKENS,
                temperature=CLOUD_TEMPERATURE,
            )
            if system_text:
                config_kwargs["system_instruction"] = system_text
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
            meta    = response.usage_metadata
            in_tok  = getattr(meta, "prompt_token_count",     0) or 0
            out_tok = getattr(meta, "candidates_token_count", 0) or 0
            return response.text, in_tok, out_tok
