import requests
from typing import List, Dict, Optional
from shopping_assistant.config import WEAVIATE_URL, COLLECTION_NAME, AUTH_TOKEN


def search_products(query_vector: List[float], limit: int = 20) -> List[Dict]:
    """Search Weaviate vector database using a query vector"""
    endpoint = f"{WEAVIATE_URL}/search"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AUTH_TOKEN}"
    }

    payload = {
        "collection_name": COLLECTION_NAME,
        "query": {"vector": query_vector},
        "columns": {
            "content_embedding": 0.8,
            "context_embedding":0.2
        },
        "output_fields": ["category", "subcategory", "metadata"],
        "top_k": limit
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            # print(result,"-------------------------------------")
            return (
                result.get("results")
                or result.get("data")
                or (result if isinstance(result, list) else [])
            )
        else:
            print(f"⚠️ Vector search failed: {response.status_code}")
    except Exception as e:
        print(f"🚨 Vector search exception: {e}")

    return []
