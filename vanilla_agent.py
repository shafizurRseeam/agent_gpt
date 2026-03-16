from llm_router import LLMRouter


class VanillaAgent:

    def __init__(self):
        self.llm = LLMRouter()

    def run(self, task):

        payload = f"""
You are an AI assistant.

Task:
{task}

Provide the best possible response.
"""

        print("\nSending task to cloud LLM...\n")

        response = self.llm.generate(payload)

        return payload, response