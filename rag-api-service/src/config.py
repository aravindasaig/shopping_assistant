import json
import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("config")

def commandline_params(param_key, default=None, required=True):
    """Get commandline parameters, if not required default will be read"""
    param_key = param_key.strip()
    if not param_key[-1] == "=":
        param_key += "="
    if any(param_key in arg for arg in sys.argv):
        for arg in sys.argv:
            if param_key in arg:
                param_value = arg[len(param_key):]
                return param_value
    elif not required:
        return default
    else:
        print("Please provide param: " + str(param_key))
        exit(0)

# Load configuration from JSON file
# Default config path is relative to the src directory
default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
CONFIG_PATH = commandline_params(param_key="config", default=default_config_path, required=False)

try:
    with open(CONFIG_PATH, "r") as fp:
        configuration = json.load(fp)
    logger.info(f"Configuration loaded from {CONFIG_PATH}")
except Exception as e:
    logger.error(f"Error loading configuration from {CONFIG_PATH}: {str(e)}")
    configuration = {}

# Azure OpenAI settings
AZURE_OPENAI_API_KEY = configuration.get("azure_openai", {}).get("api_key")
AZURE_OPENAI_ENDPOINT = configuration.get("azure_openai", {}).get("endpoint")
AZURE_OPENAI_VERSION = configuration.get("azure_openai", {}).get("api_version", "2023-05-15")

# Milvus settings
MILVUS_URI = configuration.get("milvus", {}).get("uri")
MILVUS_USER = configuration.get("milvus", {}).get("user")
MILVUS_PASSWORD = configuration.get("milvus", {}).get("password")
MILVUS_TOKEN = configuration.get("milvus", {}).get("token")

# Service settings
vector_db_config = configuration.get("vector_db", {})
VECTOR_DB_TYPE = vector_db_config.get("type", "weaviate")
service_config = configuration.get("service", {})
PORT = int(service_config.get("port", 8000))
HOST = service_config.get("host", "0.0.0.0")

# Weaviate settings
weaviate_config = vector_db_config.get("weaviate", {})
WEAVIATE_HTTP_HOST = weaviate_config.get("http_host", "weaviate.weaviate.svc.cluster.local")
WEAVIATE_HTTP_PORT = int(weaviate_config.get("http_port", 80))
WEAVIATE_GRPC_HOST = weaviate_config.get("grpc_host", "weaviate-grpc.weaviate.svc.cluster.local")
WEAVIATE_GRPC_PORT = int(weaviate_config.get("grpc_port", 50051))
WEAVIATE_HTTP_SECURE = weaviate_config.get("http_secure", False)
WEAVIATE_GRPC_SECURE = weaviate_config.get("grpc_secure", False)

# Logging configuration
logging_config = configuration.get("logging", {})
LOG_TO_FILE = logging_config.get("to_file", False)
LOG_FILE_PATH = logging_config.get("file_path", "./logs/rag_api.log")
LOG_MAX_BYTES = int(logging_config.get("max_bytes", 10485760))  # 10MB
LOG_BACKUP_COUNT = int(logging_config.get("backup_count", 5))

# Log configuration information
logger.info(f"Vector DB Type: {VECTOR_DB_TYPE}, Service: {HOST}:{PORT}")
if VECTOR_DB_TYPE == "weaviate":
    logger.info(f"Weaviate HTTP: {WEAVIATE_HTTP_HOST}:{WEAVIATE_HTTP_PORT}, gRPC: {WEAVIATE_GRPC_HOST}:{WEAVIATE_GRPC_PORT}")
elif VECTOR_DB_TYPE == "milvus":
    logger.info(f"Milvus URI: {MILVUS_URI or 'Not set'}")

# Check for required configuration
if VECTOR_DB_TYPE == "weaviate":
    if not (WEAVIATE_HTTP_HOST and WEAVIATE_GRPC_HOST):
        logger.warning("Weaviate host settings not fully configured, using defaults")
elif VECTOR_DB_TYPE == "milvus":
    if not MILVUS_URI:
        logger.warning("Milvus URI not set - Milvus connection will likely fail")

if not AZURE_OPENAI_API_KEY:
    logger.warning("Azure OpenAI API key not set - Embedding functionality will not be available") 