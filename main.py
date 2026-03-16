from dotenv import load_dotenv
load_dotenv()

from agents.vanilla_agent import VanillaAgent
from agents.hybrid_agent import HybridAgent


def main():

    print("\n===============================")
    print("         AgentGPT CLI          ")
    print("===============================")
    print("1. Vanilla Agent  (cloud only)")
    print("2. Hybrid Agent   (local + cloud + tools)")

    choice = input("\nSelect agent (1 or 2): ").strip()

    if choice == "1":

        agent = VanillaAgent()

        while True:
            task = input("\nEnter a task (or 'exit'): ")
            if task.lower() == "exit":
                break

            prompt, response = agent.run(task)

            print("\n--- Prompt Sent ---\n")
            print(prompt)
            print("\n--- Response ---\n")
            print(response)

    elif choice == "2":

        agent = HybridAgent()

        while True:
            task = input("\nEnter a task (or 'exit'): ")
            if task.lower() == "exit":
                break

            result = agent.run(task)
            print("\nResult:", result)

    else:
        print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
