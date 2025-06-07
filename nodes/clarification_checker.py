import json
from shopping_assistant.state import ProductSearchState
from shopping_assistant.config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_DEPLOYMENT_NAME, AZURE_API_VERSION
from openai import AzureOpenAI

# LLM client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_API_VERSION
)

def clarification_checker_node(state: ProductSearchState) -> ProductSearchState:
    print("❓ Checking if clarification is needed...")

    results = state["search_results"]
    memory = state["conversation_memory"]
    entities = state["stitched_entities"]

    filtered = []
    for res in results:
        score = None
        if 'score' in res:
            score = res['score']
        elif '_additional' in res:
            if 'certainty' in res['_additional']:
                score = res['_additional']['certainty']
            elif 'distance' in res['_additional']:
                score = 1.0 - res['_additional']['distance']
        elif 'distance' in res:
            score = 1.0 - res['distance']

        score = score if score is not None else 0.5
        if score > 0.6:
            filtered.append(res)

    state["search_results"] = filtered
    state["needs_clarification"] = False
    state["clarification_question"] = ""

    if memory.clarification_count >= 3:
        print("🛑 Max clarifications reached.")
        return state

    if len(filtered) == 0:
        state["needs_clarification"] = True
        state["clarification_question"] = "I couldn't find good matches. Can you share more details or change your criteria?"
        memory.clarification_count += 1
        return state

    if len(filtered) > 8:
        # Ask clarifying question to narrow down
        print("🔍 Too many high-confidence matches, asking for clarification...")

        prompt = f"""
        Given the context: {entities}
        And {len(filtered)} matching products found.

        Ask ONE brief clarifying question to help narrow the search.
        Examples:
        - "What's your preferred brand?"
        - "Do you have a color in mind?"
        - "Casual or formal style?"

        Return the question only and example of it from the shortlisted.
        """

        try:
            response = client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "Generate a short clarification question."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )

            q = response.choices[0].message.content.strip()
            state["needs_clarification"] = True
            state["clarification_question"] = q
            memory.clarification_count += 1
            return state

        except Exception as e:
            print(f"⚠️ Clarification LLM error: {e}")
            state["needs_clarification"] = True
            state["clarification_question"] = "Would you like to narrow by brand, price, or color?"
            memory.clarification_count += 1

    print(f"✅ Filtered results: {len(filtered)}. Clarification needed: {state['needs_clarification']}")
    return state
