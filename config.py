import os

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("ENDPOINT_URL", "")
AZURE_DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")

AZURE_API_VERSION = "2025-01-01-preview"

# Weaviate Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8000")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ret_shp")

# Embedding Service Configuration
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://44.200.93.203:6006")

# Authorization Token
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "your_default_jwt_token")
