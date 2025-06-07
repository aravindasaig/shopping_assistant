import json
from shopping_assistant.state import ProductSearchState
from shopping_assistant.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_DEPLOYMENT_NAME, AZURE_API_VERSION
from openai import AzureOpenAI

# LLM client
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION,
)

def conversation_stitcher_node(state: ProductSearchState) -> ProductSearchState:
    print("🧵 Stitching conversation context...")

    memory = state["conversation_memory"]
    current = state["raw_entities"]
    previous = memory.active_context

    # First turn — nothing to stitch
    if state["turn_count"] == 1:
        state["stitched_entities"] = current
        memory.active_context = current.copy()
        print("🧩 First turn — no stitching needed")
        return state

    prompt = f"""
    Merge user intent context.

    Previous context: {previous}
    Current user input: "{state['user_input']}"
    Extracted entities: {current}
    Intent: {state['intent']}

    Rules:
    - If intent is "modification", override specific fields in previous context.
    - If intent is "continuation", add new fields to previous context.
    - If user uses pronouns ("this", "that", "it"), resolve from previous context.
    - Keep everything valid and minimal.

    Return only merged JSON.
    """

    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "Return only merged JSON of stitched context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=400
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()

        stitched = json.loads(content)
        state["stitched_entities"] = stitched
        memory.active_context = stitched.copy()
        print(f"🧩 Stitched context: {stitched}")

    except Exception as e:
        print(f"⚠️ Stitching error: {e}")
        merged = previous.copy()
        merged.update(current)
        state["stitched_entities"] = merged
        memory.active_context = merged

    return state
