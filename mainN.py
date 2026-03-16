from agents.hybrid_agent import HybridAgent


def main():

    agent = HybridAgent()

    while True:

        task = input("\nEnter task (or exit): ")

        if task == "exit":
            break

        result = agent.run(task)

        print("\nResult:", result)


if __name__ == "__main__":
    main()