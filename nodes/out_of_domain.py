import json
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

def out_of_domain_node(state: ProductSearchState) -> ProductSearchState:
    print("🚫 Handling out-of-domain query...")

    prompt = f"""
    The user asked: "{state['user_input']}"

    This question is outside a retail shopping assistant's domain.

    1. Classify the type: general_knowledge, personal_advice, entertainment, weather, technical_help, or unknown
    2. Return a friendly message that:
       - Acknowledges the question
       - Explains you're focused on helping users shop
       - Invites them to ask about products instead

    Return JSON:
    {{
        "category": "entertainment",
        "response": "That's a fun question! But I'm best at helping you shop. Want to find a product instead?"
    }}
    """

    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "Return out-of-domain classification and helpful redirect. JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=150
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()

        result = json.loads(content)
        reply = result.get("response", "I'm here to help with shopping! What would you like to browse?")
        state["agent_response"] = reply
        state["messages"].append(AIMessage(content=reply))

        print(f"📤 Redirected with category: {result.get('category')}")

    except Exception as e:
        print(f"⚠️ Out-of-domain error: {e}")
        fallback = "I’m specialized in helping you find great products. What would you like to shop for today?"
        state["agent_response"] = fallback
        state["messages"].append(AIMessage(content=fallback))

    return state
