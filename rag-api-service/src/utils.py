from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any, Literal
from openai import AzureOpenAI
import asyncio
from src.config import *
from fastapi import HTTPException, WebSocketException
import re, json, logging
import traceback
from weaviate.classes.config import Property, DataType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Map string data types to Weaviate DataType enum
DATA_TYPE_MAP = {
    "text": DataType.TEXT,
    "text[]": DataType.TEXT_ARRAY,
    "int": DataType.INT,
    "int[]": DataType.INT_ARRAY,
    "boolean": DataType.BOOL,
    "boolean[]": DataType.BOOL_ARRAY,
    "number": DataType.NUMBER,
    "number[]": DataType.NUMBER_ARRAY,
    "date": DataType.DATE,
    "date[]": DataType.DATE_ARRAY,
    "uuid": DataType.UUID,
    "uuid[]": DataType.UUID_ARRAY,
    "geoCoordinates": DataType.GEO_COORDINATES,
    "blob": DataType.BLOB,
    "phoneNumber": DataType.PHONE_NUMBER,
    "object": DataType.OBJECT,
    "object[]": DataType.OBJECT_ARRAY,
    "[object]": DataType.OBJECT_ARRAY,
    "[text]": DataType.TEXT_ARRAY,
    "[int]": DataType.INT_ARRAY,
    "[number]": DataType.NUMBER_ARRAY,
    "[date]": DataType.DATE_ARRAY,
    "[uuid]": DataType.UUID_ARRAY,
    "[boolean]": DataType.BOOL_ARRAY
}

# Initialize Azure OpenAI client if API key is available
openai_client = None
if AZURE_OPENAI_API_KEY:
    try:
        openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        logger.info("Azure OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")

class QueryInput(BaseModel):
    vector: Optional[Union[List[float], Dict[str, List[float]]]] = None
    text: Optional[str] = None

class ChunkDescription(BaseModel):
    chunk_id: Optional[str] = ""
    heading: Optional[str] = ""

class PropertyDefinition(BaseModel):
    name: str
    data_type: str
    description: Optional[str] = None
    tokenization: Optional[str] = None
    index_filterable: Optional[bool] = True
    index_searchable: Optional[bool] = True
    nested_properties: Optional[List['PropertyDefinition']] = None

    def to_weaviate_property(self) -> Property:
        """Convert PropertyDefinition to Weaviate Property object"""
        data_type = get_weaviate_datatype(self.data_type)
        
        # Handle nested properties if this is an object type
        nested_props = None
        if data_type in [DataType.OBJECT, DataType.OBJECT_ARRAY]:
            if not self.nested_properties:
                raise ValueError(f"Property '{self.name}': At least one nested property is required for data type {self.data_type}")
            nested_props = []
            for nested_prop in self.nested_properties:
                try:
                    weaviate_nested_prop = nested_prop.to_weaviate_property()
                    logger.info(f"Created nested property for {self.name}: {weaviate_nested_prop}")
                    nested_props.append(weaviate_nested_prop)
                except Exception as e:
                    logger.error(f"Error creating nested property {nested_prop.name} for {self.name}: {str(e)}")
                    raise
        
        # Create property with only supported parameters
        property_params = {
            "name": self.name,
            "data_type": data_type,
        }
        
        # Add optional parameters if they exist
        if self.description:
            property_params["description"] = self.description
        
        if self.tokenization:
            property_params["tokenization"] = self.tokenization
        
        if self.index_filterable is not None:
            property_params["index_filterable"] = self.index_filterable
        
        if self.index_searchable is not None:
            property_params["index_searchable"] = self.index_searchable
        
        if nested_props:
            property_params["nested_properties"] = nested_props
        
        logger.info(f"Creating property {self.name} with params: {json.dumps(property_params, default=str)}")
        return Property(**property_params)

class VectorConfig(BaseModel):
    name: str
    vector_type: Literal["none", "text2vec-openai", "text2vec-huggingface"] = "none"
    dimensions: Optional[int] = None
    config: Optional[Dict[str, Any]] = None

class CollectionRequest(BaseModel):
    collection_name: str
    properties: Optional[List[PropertyDefinition]] = None
    vectors: Optional[List[VectorConfig]] = None
    description: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "collection_name": "my_collection",
                "description": "My collection for documents",
                "properties": [
                    {
                        "name": "title",
                        "data_type": "text",
                        "description": "Document title"
                    },
                    {
                        "name": "content",
                        "data_type": "text",
                        "description": "Document content"
                    }
                ],
                "vectors": [
                    {
                        "name": "content_vector",
                        "vector_type": "none", 
                        "dimensions": 1536
                    }
                ]
            }
        }

class InsertItem(BaseModel):
    properties: Dict[str, Any]
    vectors: Optional[Dict[str, List[float]]] = None
    id: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "properties": {
                    "name": "John Smith",
                    "home_address": {
                        "street": {
                            "number": 123,
                            "name": "Main Street"
                        },
                        "city": "London"
                    },
                    "office_addresses": [
                        {
                            "office_name": "London HQ",
                            "street": {
                                "number": 456,
                                "name": "Oxford Street"
                            }
                        }
                    ]
                },
                "vectors": {
                    "vector": [0.1, 0.2, 0.3, 0.4, 0.5]
                }
            }
        }

class InsertRequest(BaseModel):
    collection_name: str
    data: List[InsertItem]

# Legacy models for backward compatibility
class LegacyChunkDescription(BaseModel):
    chunk_id: Optional[str] = ""
    heading: Optional[str] = ""

class LegacyInsertItem(BaseModel):
    chunk_id: str
    chunk_text: str
    chunk_context: str
    chunk_description: Optional[LegacyChunkDescription] = LegacyChunkDescription()
    chunk_embeddings: List[float]
    context_embeddings: List[float]

class LegacyInsertRequest(BaseModel):
    collection_name: str
    data: List[LegacyInsertItem]

class FilterCriteria(BaseModel):
    property: str
    operator: str  # "Equal", "Like", "GreaterThan", "LessThan", "ContainsAny", "ContainsAll"
    valueText: Union[str, List[str], float, int]

class SearchRequest(BaseModel):
    collection_name: str
    query: QueryInput
    columns: Dict[str, float]
    filters: Optional[FilterCriteria] = None
    output_fields: List[str]
    top_k: int = 10
    db_type: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "collection_name": "articles",
                "query": {
                    "vector": {
                        "content_vector": [0.1, 0.2, 0.3],
                        "title_vector": [0.4, 0.5, 0.6]
                    }
                },
                "columns": {
                    "content_vector": 0.7,
                    "title_vector": 0.3
                },
                "filters": {
                    "property": "category",
                    "operator": "Equal",
                    "valueText": "technology"
                },
                "output_fields": ["title", "content"],
                "top_k": 5
            }
        }

class ReadRequest(BaseModel):
    collection_name: str
    include_vectors: bool = False
    object_id: Optional[str] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    filters: Optional[FilterCriteria] = None

    class Config:
        schema_extra = {
            "example": {
                "collection_name": "WineReview",
                "include_vectors": False,
                "limit": 10,
                "offset": 0,
                "filters": {
                    "property": "country",
                    "operator": "Equal",
                    "valueText": "Italy"
                }
            }
        }

class UpdatePropertiesRequest(BaseModel):
    collection_name: str
    object_id: str
    properties: Dict[str, Any]

    class Config:
        schema_extra = {
            "example": {
                "collection_name": "articles",
                "object_id": "123e4567-e89b-12d3-a456-426614174000",
                "properties": {
                    "title": "Updated Title",
                    "content": "Updated content"
                }
            }
        }

class UpdateWithVectorRequest(BaseModel):
    collection_name: str
    object_id: str
    properties: Dict[str, Any]
    vector: List[float]

    class Config:
        schema_extra = {
            "example": {
                "collection_name": "articles",
                "object_id": "123e4567-e89b-12d3-a456-426614174000",
                "properties": {
                    "title": "Updated Title",
                    "content": "Updated content"
                },
                "vector": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        }

class ReplaceObjectRequest(BaseModel):
    collection_name: str
    object_id: str
    properties: Dict[str, Any]
    vector: Optional[List[float]] = None

    class Config:
        schema_extra = {
            "example": {
                "collection_name": "articles",
                "object_id": "123e4567-e89b-12d3-a456-426614174000",
                "properties": {
                    "title": "New Title",
                    "content": "New content"
                },
                "vector": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        }

async def embed_query(query_text: str):
    try:
        response = await asyncio.to_thread(
            openai_client.embeddings.create,
            input=query_text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error embedding query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error embedding query: {str(e)}"
        )

def beautify_grpc_error(error_input) -> str:
    """
    Formats gRPC error messages in a more human-readable format,
    particularly for vector length mismatch errors.
    
    Args:
        error_input: The error object or message string to format
        
    Returns:
        Formatted error message
    """
    try:
        # Ensure we're working with a string
        raw_error = str(error_input)
        logger.debug(f"Formatting gRPC error: {raw_error[:200]}...")
        
        # Initialize result parts
        result_parts = []
        error_type = "Unknown"
        
        # Check for vector length mismatch errors
        if "vector lengths don't match" in raw_error:
            error_type = "Vector Length Mismatch"
            
            # Extract dimension information
            vector_mismatch = re.search(r'vector lengths don\'t match.*?(\d+)\s+vs\s+(\d+)', raw_error)
            expected_dim = vector_mismatch.group(1) if vector_mismatch else "unknown"
            actual_dim = vector_mismatch.group(2) if vector_mismatch else "unknown"
            
            result_parts.append("**Vector Length Mismatch Error**")
            result_parts.append(f"The vector dimensions in your request don't match what the database expects.")
            result_parts.append(f"**Expected dimension**: {expected_dim}")
            result_parts.append(f"**Actual dimension**: {actual_dim}")
            result_parts.append(f"**Solution**: Ensure your embedding model produces vectors with {expected_dim} dimensions.")
            
        # Check for connection errors
        elif "failed to connect" in raw_error.lower() or "connection refused" in raw_error.lower():
            error_type = "Connection Error"
            result_parts.append("**Database Connection Error**")
            result_parts.append("Could not establish a connection to the vector database.")
            result_parts.append("**Possible causes**:")
            result_parts.append("• Database service is not running")
            result_parts.append("• Network configuration is incorrect")
            result_parts.append("• Incorrect host or port settings")
            
        # Check for authentication errors
        elif "authentication" in raw_error.lower() or "unauthorized" in raw_error.lower():
            error_type = "Authentication Error"
            result_parts.append("**Authentication Error**")
            result_parts.append("Failed to authenticate with the vector database.")
            result_parts.append("**Solution**: Check your API keys and credentials.")
            
        # Generic error formatting
        else:
            # Extract debug error string if present
            debug_match = re.search(r'debug_error_string\s*=\s*"(.*?)"', raw_error, re.DOTALL)
            debug_info = debug_match.group(1) if debug_match else None
            
            # Extract call chain
            chain_match = re.search(r'explorer: get class:.*?vector search:.*?(?=\n)', raw_error, re.DOTALL)
            chain = chain_match.group(0) if chain_match else ""
            chain_parts = chain.split(': ') if chain else []
            
            result_parts.append(f"**gRPC Query Error**")
            result_parts.append("An error occurred during the database query.")
            
            if chain_parts:
                result_parts.append("\n**Error Trace**:")
                indent = "  "
                for part in chain_parts:
                    if part.strip():
                        result_parts.append(f"{indent}• {part.strip()}")
                        indent += "  "
        
        # Add error details
        result_parts.append("\n**Technical Details**:")
        error_lines = [line for line in raw_error.split("\n") if line.strip() and "debug_error_string" not in line]
        if len(error_lines) > 5:
            error_lines = error_lines[:5] + ["..."]
        
        for line in error_lines:
            result_parts.append(f"  {line.strip()}")
            
        logger.debug(f"Formatted {error_type} error successfully")
        return "\n".join(result_parts)

    except Exception as e:
        logger.error(f"Failed to format gRPC error: {str(e)}", exc_info=True)
        return f"Error parsing gRPC error: {str(e)}\n\nOriginal error:\n{str(error_input)[:500]}"

def get_collection_names(collections_list):
    """Convert Weaviate collection objects to a simple list of collection names"""
    try:
        # Try to extract collection names directly from the list
        if isinstance(collections_list, list):
            if all(isinstance(item, str) for item in collections_list):
                return collections_list  # Already a list of strings
            
            # Check for objects with a name attribute or method
            result = []
            for item in collections_list:
                if hasattr(item, 'name'):
                    if callable(item.name):
                        result.append(item.name())
                    else:
                        result.append(item.name)
                elif hasattr(item, '__str__'):
                    result.append(str(item))
                else:
                    result.append("unknown_collection")
            return result
        
        # If it's not a list, try to convert it to a list
        elif hasattr(collections_list, '__iter__'):
            return list(str(item) for item in collections_list)
            
        # If all else fails, return a generic message
        return ["collections_list_not_serializable"]
    except Exception as e:
        logger.error(f"Error converting collections list: {str(e)}")
        return ["error_converting_collections"]

def get_weaviate_datatype(type_str: str) -> DataType:
    """Convert string data type to Weaviate DataType enum"""
    # Remove any square brackets from the type string for lookup
    clean_type = type_str.replace("[", "").replace("]", "")
    data_type = DATA_TYPE_MAP.get(type_str.lower()) or DATA_TYPE_MAP.get(clean_type.lower())
    if not data_type:
        logger.warning(f"Unknown data type: {type_str}, using TEXT as default")
        return DataType.TEXT
    return data_type
