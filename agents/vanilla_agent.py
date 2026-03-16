from llm.cloud_router import CloudLLM


class VanillaAgent:

    def __init__(self):
        self.llm = CloudLLM()

    def run(self, task):

        prompt = f"""You are an AI assistant.

Task:
{task}

Provide the best possible response.
"""
        response = self.llm.generate(prompt)
        return prompt, response
