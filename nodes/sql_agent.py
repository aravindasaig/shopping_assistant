from shopping_assistant.state import ProductSearchState
from langchain_core.messages import AIMessage

try:
    from text2sql_agent import Text2SQLAgent
    SQL_AGENT_AVAILABLE = True
except ImportError:
    print("⚠️ Text2SQL agent not found. FAQ support disabled.")
    SQL_AGENT_AVAILABLE = False

def sql_agent_node(state: ProductSearchState) -> ProductSearchState:
    print("🧾 Running SQL agent for FAQ...")

    if not SQL_AGENT_AVAILABLE:
        response = "❌ SQL agent unavailable. Please install `text2sql_agent.py`."
        state["sql_results"] = response
        state["agent_response"] = response
        state["messages"].append(AIMessage(content=response))
        return state

    try:
        sql_agent = Text2SQLAgent()
        result = sql_agent.query(state["user_input"])

        # Ensure result is always a dict
        if not isinstance(result, dict):
            result = {
                "query": "",
                "answer": str(result)
            }

        state["sql_query"] = result.get("query", "")
        state["sql_results"] = result.get("answer", "")
        state["agent_response"] = result["answer"]
        state["messages"].append(AIMessage(content=result["answer"]))
        print("✅ SQL result sent.")

    except Exception as e:
        error_msg = f"❌ SQL agent error: {e}"
        state["sql_results"] = error_msg
        state["agent_response"] = error_msg
        state["messages"].append(AIMessage(content=error_msg))
        print(error_msg)

    return state
