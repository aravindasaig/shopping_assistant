import json
from shopping_assistant.state import ProductSearchState
from shopping_assistant.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_DEPLOYMENT_NAME, AZURE_API_VERSION
from openai import AzureOpenAI

# LLM client initialization
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION,
)

def supervisor_node(state: ProductSearchState) -> ProductSearchState:
    print("🎭 Supervisor analyzing request...")

    # Get context about available search results
    current_results = state.get("search_results", [])
    memory = state["conversation_memory"]
    has_recent_search = len(current_results) > 0
    
    # Check for search results in conversation history
    has_historical_search = False
    if memory.turn_history:
        for turn in reversed(memory.turn_history):
            if turn.search_results:
                has_historical_search = True
                break

    prompt = f"""
    You are a Supervisor Agent for a retail shopping assistant. Analyze the user input and decide the next action.

    User Input: "{state['user_input']}"
    Turn Count: {state['turn_count']}
    Previous Context: {state['conversation_memory'].active_context}
    Cart Items: {len(state.get('shopping_cart', []).items)}
    
    SEARCH RESULTS CONTEXT:
    - Current search results available: {has_recent_search} ({len(current_results)} items)
    - Historical search results available: {has_historical_search}
    - Can add to cart: {has_recent_search or has_historical_search}

    DECISION CRITERIA (in priority order):

    1. SAFETY CHECK (Priority 1):
       - Toxic language, harassment, inappropriate content
       - If unsafe → "guardrails"

    2. CART ACTIONS (Priority 2):
       - EXPLICIT cart commands: "add to cart", "buy this", "I want this", "remove from cart", "show cart", "checkout"
       - For ADD actions: Route to cart_manager if ANY search results available (current OR historical)
       - For VIEW/REMOVE/CHECKOUT: always route to cart_manager
       - Only route to intent_classifier if it's an ADD action with NO search results at all
       - If explicit cart action → "cart_manager" (unless ADD with zero search results)

    3. SMALL TALK (Priority 3):
       - Greetings: "hi", "hello", "good morning"
       - Thanks: "thank you", "thanks"
       - If small talk → "small_talk"

    4. OUT OF DOMAIN (Priority 4):
       - Non-shopping topics: weather, politics, general knowledge
       - If out of domain → "out_of_domain"

    5. ALL OTHER QUERIES (Priority 5):
       - FAQ/Analytics: "price", "how many", "average", "cheapest", etc.
       - Product search: "red t-shirt", "Nike shoes", etc.
       - Clarification responses: "medium", "large", "red", etc.
       - ALL of these go to → "intent_classifier" (for context processing)

    IMPORTANT CHANGE: 
    - NO direct routing to sql_agent anymore
    - ALL queries (FAQ, product search, clarification) go to "intent_classifier"
    - The intent_classifier and context pipeline will handle routing to sql_agent or vector search

    EXAMPLES:
    
    CART ACTIONS:
    - "add to cart" + current OR historical search results → "cart_manager"
    - "add to cart" + no search results at all → "intent_classifier" (search first)
    - "show cart", "checkout", "remove" → "cart_manager" (always allow)
    
    ALL OTHER QUERIES → "intent_classifier":
    - "how many wrangler t-shirts?" → "intent_classifier" ✅
    - "average price of them?" → "intent_classifier" ✅
    - "red t-shirt" → "intent_classifier" ✅
    - "medium" → "intent_classifier" ✅

    Return JSON only:
    {{
        "action": "guardrails | cart_manager | out_of_domain | small_talk | intent_classifier",
        "reasoning": "Brief explanation including search results context",
        "is_safe": true,
        "is_in_domain": true,
        "intent": "faq | clarification_response | product_search | cart_action",
        "confidence": 0.85
    }}
    """

    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a supervisor that makes routing decisions. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()

        decision = json.loads(content)

        state["next_action"] = decision.get("action", "intent_classifier")
        state["supervisor_reasoning"] = decision.get("reasoning", "Fallback reasoning")
        state["is_safe"] = decision.get("is_safe", True)
        state["is_in_domain"] = decision.get("is_in_domain", True)
        state["intent"] = decision.get("intent", "product_search")
        state["intent_confidence"] = decision.get("confidence", 0.5)

        print(f"🎯 Decision: {state['next_action']}")
        print(f"💭 Reasoning: {state['supervisor_reasoning']}")
        print(f"🔍 Search context: Current={len(current_results)}, Historical={has_historical_search}")

    except Exception as e:
        print(f"⚠️ Supervisor error: {e}")
        state["next_action"] = "intent_classifier"
        state["supervisor_reasoning"] = "Fallback due to exception"
        state["is_safe"] = True

    return state