import requests
from typing import List, Optional
import os
from shopping_assistant.config import EMBEDDING_URL


def get_text_embedding(text: str) -> Optional[List[float]]:
    """Generate text embedding from embedding service"""
    endpoint = f"{EMBEDDING_URL}/generate-text-embedding"
    headers = {"Content-Type": "application/json"}
    payload = {"text": text}

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            return (
                result.get("embedding")
                or result.get("data")
                or (result if isinstance(result, list) else None)
            )
        else:
            print(f"⚠️ Text embedding failed: {response.status_code}")
    except Exception as e:
        print(f"🚨 Text embedding error: {e}")
    return None


def get_image_embedding(image_path: str) -> Optional[List[float]]:
    """Generate image embedding from embedding service"""
    endpoint = f"{EMBEDDING_URL}/embed-image"
    headers = {"accept": "application/json"}

    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(endpoint, headers=headers, files=files)

        if response.status_code == 200:
            result = response.json()
            return (
                result.get("embedding")
                or result.get("data")
                or (result if isinstance(result, list) else None)
            )
        else:
            print(f"⚠️ Image embedding failed: {response.status_code}")
    except Exception as e:
        print(f"🚨 Image embedding error: {e}")
    return None
