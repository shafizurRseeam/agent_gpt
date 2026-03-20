import argparse
from dotenv import load_dotenv
load_dotenv()

from config import LOCAL_MODEL, EVAL_LOCAL_MODELS
from agents.vanilla_agent import VanillaAgent
from agents.hybrid_agent import HybridAgent


def main():
    parser = argparse.ArgumentParser(description="AgentGPT CLI")
    parser.add_argument(
        "--model", default=None,
        help=(
            "Override the local LC model for this run. "
            f"Available: {', '.join(EVAL_LOCAL_MODELS.keys())}. "
            f"Default: {LOCAL_MODEL}"
        )
    )
    parser.add_argument(
        "--agent", choices=["1", "2"], default=None,
        help="Pre-select agent: 1=Vanilla, 2=Hybrid (skips interactive prompt)"
    )
    args = parser.parse_args()

    active_model = args.model or LOCAL_MODEL

    print("\n===============================")
    print("         AgentGPT CLI          ")
    print("===============================")
    print("1. Vanilla Agent  (cloud only)")
    print("2. Hybrid Agent   (local + cloud + tools)")
    if args.model:
        print(f"\n  [LC model override: {active_model}]")

    choice = args.agent or input("\nSelect agent (1 or 2): ").strip()

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

        agent = HybridAgent(local_model=active_model)

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
