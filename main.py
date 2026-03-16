from vanilla_agent import VanillaAgent
from config import PROVIDER


def main():

    print("\n===============================")
    print(" VanillaAgent Cloud Chat ")
    print("===============================")
    print(f"Provider: {PROVIDER}")
    print("===============================\n")

    agent = VanillaAgent()

    while True:

        task = input("Enter a task (or 'exit'): ")

        if task.lower() == "exit":
            break

        payload, response = agent.run(task)

        print("\n--- Payload Sent to Cloud ---")
        print(payload)

        print("\n--- Cloud Response ---")
        print(response)

        print("\n===============================\n")


if __name__ == "__main__":
    main()