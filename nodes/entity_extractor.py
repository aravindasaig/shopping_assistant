import json
import base64
from shopping_assistant.state import ProductSearchState
from shopping_assistant.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_DEPLOYMENT_NAME, AZURE_API_VERSION
from openai import AzureOpenAI

# LLM client
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION,
)

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 for GPT-4.1 vision"""
    try:
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        print(f"⚠️ Image encoding error: {e}")
        return None

def entity_extractor_node(state: ProductSearchState) -> ProductSearchState:
    print("🔎 Extracting entities...")

    prompt = f"""
    Extract product-related entities from the user input: "{state['user_input']}"
    
    Intent: {state['intent']}
    Image provided: {state['has_image']}
    
    Extract these entities if mentioned:
    - product_type (shirt, jeans, shoes, etc.)
    - brand (nike, adidas, levis, etc.)
    - color (red, blue, black, etc.)
    - material (cotton, denim, leather, etc.)
    - gender (male, female, unisex)
    - size (S, M, L, XL, etc.)
    - pattern (solid, striped, graphic, etc.)
    - theme (casual, formal, sports, etc.)
    - price_range (under 2000, between 1000-3000, etc.)
    
    IMPORTANT: If an image is provided, analyze the image first and extract visual entities (product type, color, style, pattern, etc.) from the image. Then combine with any text entities.
    
    Return as JSON object with only the entities that are clearly visible or mentioned.
    If nothing is found, return empty JSON {{}}.
    """

    try:
        # Prepare messages for multimodal input
        messages = [
            {"role": "system", "content": "You are an expert at extracting product entities from text and images. Return valid JSON only."}
        ]
        
        # Handle multimodal input if image is provided
        if state["has_image"] and state["image_path"]:
            print("🖼️ Processing multimodal input (text + image)...")
            
            # Encode image to base64
            image_base64 = encode_image_to_base64(state["image_path"])
            
            if image_base64:
                # Create multimodal message
                user_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
                messages.append(user_message)
            else:
                # Fallback to text only if image encoding fails
                messages.append({"role": "user", "content": prompt})
                print("⚠️ Image encoding failed, falling back to text-only extraction")
        else:
            # Text-only input
            print("📝 Processing text-only input...")
            messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=messages,
            temperature=0,
            max_tokens=400  # Increased for image analysis
        )

        content = response.choices[0].message.content.strip()
        
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()

        extracted = json.loads(content)
        state["raw_entities"] = extracted
        
        if state["has_image"]:
            print(f"✅ Extracted entities (multimodal): {extracted}")
        else:
            print(f"✅ Extracted entities (text): {extracted}")

    except Exception as e:
        print(f"⚠️ Entity extraction error: {e}")
        state["raw_entities"] = {}

    return state