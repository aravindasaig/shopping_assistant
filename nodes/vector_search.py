from shopping_assistant.state import ProductSearchState
from shopping_assistant.utils.embedding import get_text_embedding, get_image_embedding
from shopping_assistant.utils.search import search_products
from shopping_assistant.schema import ConversationTurn
from datetime import datetime
import json

def vector_search_node(state: ProductSearchState) -> ProductSearchState:
    """Node 4: Search Weaviate with hybrid embedding + full metadata preservation"""
    print("🔍 Searching vector database...")
    
    entities = state["stitched_entities"]
    
    # Create search text from entities
    search_parts = []
    for key, value in entities.items():
        if value:
            search_parts.append(f"{key}: {value}")
    
    search_text = ". ".join(search_parts) if search_parts else state["user_input"]
    print(f"📝 Search text: {search_text}")
    
    # Generate embeddings
    final_embedding = None
    
    if state["has_image"] and state["image_path"]:
        print("🖼️ Processing hybrid search (image + text)...")
        
        # Get both embeddings
        image_embedding = get_image_embedding(state["image_path"])
        text_embedding = get_text_embedding(search_text)
        
        if image_embedding and text_embedding:
            print("🔀 Combining embeddings: Image(80%) + Text(20%)")
            final_embedding = []
            for i in range(len(image_embedding)):
                combined_value = (image_embedding[i] * 0.8) + (text_embedding[i] * 0.2)
                final_embedding.append(combined_value)
        elif image_embedding:
            final_embedding = image_embedding
        elif text_embedding:
            final_embedding = text_embedding
    else:
        print("📝 Processing text search...")
        final_embedding = get_text_embedding(search_text)
    
    # Search vector database
    if final_embedding:
        raw_results = search_products(final_embedding, limit=20)
        
        # ENHANCED: Process and normalize metadata for all results
        processed_results = []
        for i, result in enumerate(raw_results):
            try:
                # Extract and normalize metadata
                metadata = extract_metadata_from_result(result)
                
                # Create standardized result structure
                processed_result = {
                    "score": extract_score_from_result(result),
                    "metadata": metadata,
                    "raw_result": result,  # Keep original for debugging
                    "index": i
                }
                
                processed_results.append(processed_result)
                print(f"✅ Processed result {i+1}: {metadata.get('brand', 'Unknown')} {metadata.get('product_type', 'Product')} - ₹{metadata.get('price_inr', 0)}")
                
            except Exception as e:
                print(f"⚠️ Failed to process result {i+1}: {e}")
                # Keep raw result as fallback
                processed_results.append({
                    "score": 0.5,
                    "metadata": {},
                    "raw_result": result,
                    "index": i
                })
        
        state["search_results"] = processed_results
        print(f"✅ Found and processed {len(processed_results)} products")
        
        # Update memory with processed results
        current_turn = ConversationTurn(
            turn_id=state["turn_count"],
            user_input=state["user_input"],
            extracted_entities=state["stitched_entities"],
            timestamp=datetime.now(),
            search_results=processed_results,  # Store processed results
            context_snapshot=state["stitched_entities"].copy()
        )
        state["conversation_memory"].add_turn(current_turn)
        
    else:
        print("❌ Failed to generate embedding")
        state["search_results"] = []
    
    return state

def extract_metadata_from_result(result: dict) -> dict:
    """Extract and normalize metadata from search result"""
    metadata = {}
    
    # Based on your actual data structure: result['data']['metadata']
    if 'data' in result and 'metadata' in result['data']:
        raw_metadata = result['data']['metadata']
    # Fallback methods
    elif 'properties' in result:
        props = result['properties']
        raw_metadata = props.get('metadata', {})
    elif 'metadata' in result:
        raw_metadata = result['metadata']
    else:
        raw_metadata = {}
    
    # Parse JSON string if needed
    if isinstance(raw_metadata, str):
        try:
            metadata = json.loads(raw_metadata)
        except Exception as e:
            print(f"❌ Failed to parse metadata JSON: {e}")
            metadata = {}
    else:
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    
    # Ensure all standard fields exist with defaults
    standardized_metadata = {
        "product_type": metadata.get("product_type", "Product"),
        "brand": metadata.get("brand", "Unknown"),
        "color": metadata.get("color", "N/A"),
        "material": metadata.get("material", "N/A"),
        "gender": metadata.get("gender", "Unisex"),
        "size": metadata.get("size", "N/A"),
        "pattern": metadata.get("pattern", "N/A"),
        "theme": metadata.get("theme", "N/A"),
        "price_inr": float(metadata.get("price_inr", 0.0)),
        "image_id": metadata.get("image_id", "unknown"),
        "fit": metadata.get("fit", "N/A"),
        "sleeve_type": metadata.get("sleeve_type", "N/A"),
        "neck_type": metadata.get("neck_type", "N/A"),
        "visual_tags": metadata.get("visual_tags", []),
        "category": result.get("data", {}).get("category", "Fashion"),
        "subcategory": result.get("data", {}).get("subcategory", "Clothing")
    }
    
    return standardized_metadata

def extract_score_from_result(result: dict) -> float:
    """Extract confidence score from search result"""
    # Based on your actual data: result['score']
    if 'score' in result:
        return float(result['score'])
    # Fallback methods
    elif '_additional' in result:
        if 'certainty' in result['_additional']:
            return float(result['_additional']['certainty'])
        elif 'distance' in result['_additional']:
            return 1.0 - float(result['_additional']['distance'])
    elif 'distance' in result:
        return 1.0 - float(result['distance'])
    else:
        return 0.5  # Default score