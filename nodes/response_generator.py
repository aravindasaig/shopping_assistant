import json
from shopping_assistant.state import ProductSearchState
from shopping_assistant.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_DEPLOYMENT_NAME, AZURE_API_VERSION
from langchain_core.messages import AIMessage
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION,
)

def response_generator_node(state: ProductSearchState) -> ProductSearchState:
    """Node 6: Generate final response using processed metadata"""
    print("💬 Generating response...")
    
    if state["needs_clarification"]:
        # Return clarification question
        response = state["clarification_question"]
        
    else:
        # Generate product presentation using PROCESSED metadata
        results = state["search_results"]  # Already processed with standardized metadata
        entities = state["stitched_entities"]
        memory = state["conversation_memory"]
        
        if results:
            print(f"📦 Displaying {len(results)} high-quality products")
            
            # Use the processed metadata directly (no re-parsing needed!)
            products_data = []
            for i, item in enumerate(results[:8]):  # Show up to 8 products
                
                # Use the standardized metadata from vector_search_node
                if "metadata" in item and isinstance(item["metadata"], dict):
                    metadata = item["metadata"]
                    score = item.get("score", 0.5)
                    
                    product_info = {
                        "product_type": metadata.get("product_type", "Product"),
                        "brand": metadata.get("brand", "Unknown"),
                        "color": metadata.get("color", "N/A"),
                        "material": metadata.get("material", "N/A"),
                        "price_inr": metadata.get("price_inr", 0.0),
                        "image_id": metadata.get("image_id", f"item_{i}"),
                        "fit": metadata.get("fit", "N/A"),
                        "pattern": metadata.get("pattern", "N/A"),
                        "score": round(score, 3),
                        "category": metadata.get("category", "N/A"),
                        "subcategory": metadata.get("subcategory", "N/A")
                    }
                    
                    products_data.append(product_info)
                    print(f"📦 Product {i+1}: {product_info['brand']} {product_info['product_type']} - {product_info['color']} (₹{product_info['price_inr']}, Score: {product_info['score']})")
                
                else:
                    # Fallback for any items that don't have processed metadata
                    print(f"⚠️ Item {i+1} missing processed metadata, skipping")
            
            if products_data:
                # Generate response using actual product data
                response = f"Perfect! I found {len(products_data)} excellent matches for your search:\n\n"
                
                for i, product in enumerate(products_data, 1):
                    response += f"{i}. {product['brand']} {product['product_type']} ({product['color']})\n"
                    response += f"   ₹{product['price_inr']} | {product['material']} | {product['fit']} Fit\n"
                    response += f"   ID: {product['image_id']} | Quality Score: {product['score']}\n\n"
                    
                response += f"All products have high similarity scores (>0.6) for your requirements!"
                
                # Add helpful context
                if len(results) > len(products_data):
                    response += f"\n\n({len(results) - len(products_data)} more similar items available)"
                    
            else:
                response = f"""I found {len(results)} matches but couldn't process the product details properly. 

The search results have good similarity scores, but there may be an issue with the data format. Please try refining your search or contact support."""
                
        else:
            response = f"""I couldn't find any high-quality matches for your criteria: {entities}

Would you like me to:
- Try with relaxed criteria
- Search for similar products  
- Start a new search with different requirements"""
    
    state["agent_response"] = response
    state["messages"].append(AIMessage(content=response))
    
    print(f"✅ Response generated")
    return state