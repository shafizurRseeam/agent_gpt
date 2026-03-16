from llm.local_llm import LocalLLM
from llm.cloud_router import CloudLLM

from tools.search_tool import search_dentists
from tools.location_tool import get_location
from tools.booking_service import book_appointment


class HybridAgent:

    def __init__(self):

        self.local_llm = LocalLLM()
        self.cloud_llm = CloudLLM()

    def run(self, task):

        print("\n--- Local reasoning ---")

        plan_prompt = f"""
Task: {task}

Break this into subtasks.
"""

        plan = self.local_llm.generate(plan_prompt)

        print("\nLocal Plan:")
        print(plan)

        location = get_location()

        print("\nUser Location:", location)

        cloud_prompt = f"""
Find dentists near {location}
"""

        print("\n--- Cloud call ---")

        cloud_result = self.cloud_llm.generate(cloud_prompt)

        print("\nCloud Response:")
        print(cloud_result)

        dentists = search_dentists(location)

        best = min(dentists, key=lambda x: x["distance"])

        print("\nSelected dentist:", best["name"])

        result = book_appointment(best["name"])

        return result