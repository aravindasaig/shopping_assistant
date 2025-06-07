from shopping_assistant.state import ProductSearchState
from shopping_assistant.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_DEPLOYMENT_NAME, AZURE_API_VERSION
from openai import AzureOpenAI

# LLM client
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION,
)

def intent_classifier_node(state: ProductSearchState) -> ProductSearchState:
    print("🎯 Classifying user intent...")

    user_input = state["user_input"]
    turn = state["turn_count"]
    prev_context = state["conversation_memory"].active_context

    prompt = f"""
    Classify the user intent from: "{user_input}"

    Conversation Turn: {turn}
    Previous context: {prev_context}

    Possible intents:
    - product_search
    - faq
    - clarification_response
    - modification
    - continuation

    Return only the intent name.
    """

    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an intent classifier. Return only the intent name."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=20
        )

        intent = response.choices[0].message.content.strip().lower()
        state["intent"] = intent
        print(f"✅ Intent detected: {intent}")

    except Exception as e:
        print(f"⚠️ Intent classification error: {e}")
        state["intent"] = "product_search"  # safe fallback

    return state
