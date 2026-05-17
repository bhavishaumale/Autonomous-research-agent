# main.py

from agent.graph import agent


def run_agent(topics=None):
    if topics is None:
        topics = [
            "retrieval augmented generation",
            "large language model agents",
            "efficient llm inference"
        ]

    print("Starting Research Agent...")
    print(f"Topics: {topics}")
    print()

    initial_state = {
        "topics": topics,
        "fetched_papers": [],
        "new_papers": [],
        "summaries": [],
        "digest": "",
        "approved": False,
        "error": "",
        "messages": []
    }

    result = agent.invoke(initial_state)

    print("\nAgent run complete.")
    print(f"Papers found: {len(result.get('summaries', []))}")
    return result


if __name__ == "__main__":
    run_agent()