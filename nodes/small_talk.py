from shopping_assistant.state import ProductSearchState
from shopping_assistant.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_DEPLOYMENT_NAME, AZURE_API_VERSION
from langchain_core.messages import AIMessage
from openai import AzureOpenAI

# LLM client
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION,
)

def small_talk_node(state: ProductSearchState) -> ProductSearchState:
    print("💬 Handling small talk...")

    user_input = state["user_input"]

    prompt = f"""
    The user said: "{user_input}"

    This is small talk (greeting, thanks, casual chat). Respond:
    - In a friendly, conversational tone
    - Briefly acknowledge what they said
    - Gently shift back to shopping
    - Keep the tone warm, not robotic

    Example:
    "Hi" → "Hello! I'm here to help you find products. What are you shopping for today?"
    "Thanks" → "You're welcome! Anything else I can help you with?"
    """

    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "Generate friendly small talk replies with gentle redirection to shopping."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=80
        )

        reply = response.choices[0].message.content.strip()
        state["agent_response"] = reply
        state["messages"].append(AIMessage(content=reply))

    except Exception as e:
        print(f"⚠️ Small talk error: {e}")
        fallback = "Hi there! I'm here to help you shop. What are you looking for today?"
        state["agent_response"] = fallback
        state["messages"].append(AIMessage(content=fallback))

    return state
