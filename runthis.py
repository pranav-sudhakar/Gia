from langchain_core.messages import HumanMessage
from build_model import react_graph

def run_question(question: str, file_path: str = None):
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "file_path": file_path,
    }

    final_state = react_graph.invoke(
        initial_state,
        config={"recursion_limit": 15},
    )

    print("\n=== FULL MESSAGE TRACE ===")
    for msg in final_state["messages"]:
        role = msg.__class__.__name__
        print(f"\n[{role}]")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  -> calling tool: {tc['name']}({tc['args']})")
        print(msg.content)

    print("\n=== FINAL ANSWER ===")
    print(final_state["messages"][-1].text)


if __name__ == "__main__":
    run_question(
    "This spreadsheet lists products, units sold, and price. What is the total "
    "revenue (units sold × price, summed across all products)?",
    file_path="test_spreadsheet.xlsx",
) 