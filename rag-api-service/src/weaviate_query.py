from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
import weaviate
from openai import AzureOpenAI
from fastapi.responses import JSONResponse
from weaviate.classes.query import TargetVectors, MetadataQuery, Filter
from weaviate.classes.config import Configure, Property, DataType
from src.config import *
from src.utils import *
import logging, json, re, os
import traceback
import inspect
import socket
import urllib.request

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Weaviate Service", 
              description="API service for Weaviate vector database operations")

# Check if a host is reachable
def is_host_reachable(host, port):
    try:
        # Create a socket object
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Set a timeout
        s.settimeout(2)
        # Try to connect
        s.connect((host, port))
        # Close the socket
        s.close()
        return True
    except:
        return False

# Initialize Weaviate client with environment variables
client = None
try:
    # Try to connect using environment variables
    if WEAVIATE_HTTP_HOST and WEAVIATE_HTTP_PORT:
        # First, check if the specified host is reachable
        if is_host_reachable(WEAVIATE_HTTP_HOST, int(WEAVIATE_HTTP_PORT)):
            logger.info(f"Connecting to Weaviate at {WEAVIATE_HTTP_HOST}:{WEAVIATE_HTTP_PORT}")
            client = weaviate.connect_to_custom(
                http_host=WEAVIATE_HTTP_HOST,
                http_port=WEAVIATE_HTTP_PORT,
                grpc_host=WEAVIATE_GRPC_HOST,
                grpc_port=WEAVIATE_GRPC_PORT,
                http_secure=WEAVIATE_HTTP_SECURE,
                grpc_secure=WEAVIATE_GRPC_SECURE
            )
            logger.info("Connected to Weaviate using custom connection")
        else:
            logger.warning(f"Specified host {WEAVIATE_HTTP_HOST}:{WEAVIATE_HTTP_PORT} is not reachable")
            
    # If we couldn't connect using environment variables, try localhost
    if client is None:
        # Try local Weaviate instance
        if is_host_reachable('localhost', 8080):
            logger.info("Connecting to local Weaviate instance")
            client = weaviate.connect_to_local()
            logger.info("Connected to local Weaviate instance")
        
    # If we still don't have a client, try host.docker.internal (for Docker)
    if client is None and is_host_reachable('host.docker.internal', 8080):
        logger.info("Connecting to Weaviate on host.docker.internal")
        client = weaviate.connect_to_custom(
            http_host='host.docker.internal',
            http_port=8080,
            grpc_host='host.docker.internal',
            grpc_port=50051,
            http_secure=False,
            grpc_secure=False
        )
        logger.info("Connected to Weaviate on host.docker.internal")
        
    if client is None:
        logger.error("Could not connect to any Weaviate instance")
    else:
        logger.info(f"Weaviate client initialized successfully")
        
except Exception as e:
    logger.error(f"Failed to initialize Weaviate client: {str(e)}")
    # Don't raise exception here - the service will attempt to connect when needed

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

# Get available DataType values
available_datatypes = [attr for attr in dir(DataType) if not attr.startswith('_')]
logger.info(f"Available Weaviate DataTypes: {available_datatypes}")

# Check supported parameters for NamedVectors.none()
none_params = []
try:
    none_signature = inspect.signature(Configure.NamedVectors.none)
    none_params = list(none_signature.parameters.keys())
    logger.info(f"Supported parameters for NamedVectors.none(): {none_params}")
except Exception as e:
    logger.warning(f"Could not inspect NamedVectors.none(): {str(e)}")

# Check supported parameters for near_vector
near_vector_params = []
try:
    # Only try to inspect if we have a client
    if client:
        # Try to get a collection to inspect
        test_collections = client.collections.list_all()
        if test_collections:
            # Use the first available collection
            test_collection = client.collections.get(test_collections[0])
            if test_collection:
                near_vector_method = getattr(test_collection.query, "near_vector", None)
                if near_vector_method:
                    near_vector_signature = inspect.signature(near_vector_method)
                    near_vector_params = list(near_vector_signature.parameters.keys())
                    logger.info(f"Supported parameters for near_vector(): {near_vector_params}")
except Exception as e:
    logger.warning(f"Could not inspect near_vector parameters: {str(e)}")

# Map from string data types to Weaviate DataType enum
DATA_TYPE_MAP = {
    "text": DataType.TEXT,
    "string": DataType.TEXT,
    "int": DataType.INT,
    "integer": DataType.INT,
    "number": DataType.NUMBER,
    "float": DataType.NUMBER,
    "double": DataType.NUMBER,
    "date": DataType.DATE,
    "object": DataType.OBJECT,
    "geo": DataType.GEO_COORDINATES,
    "geo_coordinates": DataType.GEO_COORDINATES,
}

# Add boolean type if available
if hasattr(DataType, 'BOOLEAN'):
    DATA_TYPE_MAP["boolean"] = DATA_TYPE_MAP["bool"] = DataType.BOOLEAN
elif hasattr(DataType, 'BOOL'):
    DATA_TYPE_MAP["boolean"] = DATA_TYPE_MAP["bool"] = DataType.BOOL

def get_weaviate_datatype(type_str: str) -> DataType:
    """Convert string data type to Weaviate DataType enum"""
    data_type = DATA_TYPE_MAP.get(type_str.lower())
    if not data_type:
        logger.warning(f"Unknown data type: {type_str}, using TEXT as default")
        return DataType.TEXT
    return data_type


def supports_nested_properties():
    try:
        return "nested_properties" in inspect.signature(Property).parameters
    except Exception as e:
        logger.warning(f"Could not inspect Property: {e}")
        return False
    
def create_property_definition(prop_def: PropertyDefinition) -> Property:
    """Convert PropertyDefinition to Weaviate Property object"""
    data_type = get_weaviate_datatype(prop_def.data_type)
    
    # Handle nested properties if this is an object type
    nested_props = None
    if data_type == DataType.OBJECT and prop_def.nested_properties:
        nested_props = [create_property_definition(p) for p in prop_def.nested_properties]
    
    # Create property with only supported parameters
    property_params = {
        "name": prop_def.name,
        "data_type": data_type,
    }
    
    # Add optional parameters if they exist in the Property constructor
    if hasattr(Property, "__init__") and "description" in Property.__init__.__code__.co_varnames:
        property_params["description"] = prop_def.description
    
    if hasattr(Property, "__init__") and "tokenization" in Property.__init__.__code__.co_varnames:
        if prop_def.tokenization == "field":
            property_params["tokenization"] = "field"
    
    if hasattr(Property, "__init__") and "index_filterable" in Property.__init__.__code__.co_varnames:
        property_params["index_filterable"] = prop_def.index_filterable
    
    if hasattr(Property, "__init__") and "index_searchable" in Property.__init__.__code__.co_varnames:
        property_params["index_searchable"] = prop_def.index_searchable
    
    if data_type in [DataType.OBJECT, DataType.OBJECT_ARRAY] and prop_def.nested_properties and supports_nested_properties():
        property_params["nested_properties"] = [
            create_property_definition(p) for p in prop_def.nested_properties
        ]
    
    return Property(**property_params)

def create_vector_config(vector_config):
    """Create vector configuration based on the installed Weaviate client version"""
    if vector_config.vector_type == "none":
        # Handle different versions of the Weaviate client
        if 'dimensions' in none_params:
            # Newer versions support dimensions
            return Configure.NamedVectors.none(
                name=vector_config.name,
                dimensions=vector_config.dimensions
            )
        else:
            # Older versions don't support dimensions
            return Configure.NamedVectors.none(
                name=vector_config.name
            )
    elif vector_config.vector_type == "text2vec-openai":
        try:
            return Configure.NamedVectors.text2vec_openai(
                name=vector_config.name,
                model_name=vector_config.config.get("model", "ada") if vector_config.config else "ada",
                type=vector_config.config.get("type", "text") if vector_config.config else "text"
            )
        except TypeError:
            # Fallback to simpler configuration if needed
            return Configure.NamedVectors.text2vec_openai(
                name=vector_config.name
            )
    elif vector_config.vector_type == "text2vec-huggingface":
        try:
            return Configure.NamedVectors.text2vec_huggingface(
                name=vector_config.name,
                model_name=vector_config.config.get("model", "sentence-transformers/all-MiniLM-L6-v2") if vector_config.config else "sentence-transformers/all-MiniLM-L6-v2"
            )
        except TypeError:
            # Fallback to simpler configuration if needed
            return Configure.NamedVectors.text2vec_huggingface(
                name=vector_config.name
            )
    else:
        # Unknown vector type, fallback to basic configuration
        logger.warning(f"Unknown vector type: {vector_config.vector_type}, using 'none' type")
        return Configure.NamedVectors.none(name=vector_config.name)

async def weaviate_create(collection_request):
    """
    Create a new collection in Weaviate with dynamic schema.
    
    Args:
        collection_request: Collection creation request with schema definition
        
    Returns:
        JSONResponse with creation status
    """
    try:
        collection_name = collection_request.collection_name
        
        # Check if the collection already exists
        existing_collections = client.collections.list_all()
        existing_collection_names = get_collection_names(existing_collections)
        
        if collection_name in existing_collection_names:
            logger.warning(f"Collection '{collection_name}' already exists")
            return JSONResponse(
                status_code=400,
                content={"message": f"Collection '{collection_name}' already exists."}
            )

        # Configure vector configs
        vector_configs = []
        if collection_request.vectors:
            for vector_config in collection_request.vectors:
                try:
                    vector_configs.append(create_vector_config(vector_config))
                except Exception as e:
                    logger.error(f"Error creating vector config: {str(e)}")
                    # Continue with other vector configs
        
        # Use default vectors if none provided or all failed
        if not vector_configs:
            logger.info(f"No vector configurations provided or all failed, using defaults")
            try:
                vector_configs = [create_vector_config(VectorConfig(name="vector", vector_type="none"))]
            except Exception as e:
                logger.error(f"Error creating default vector config: {str(e)}")
                # Try absolute minimal configuration
                vector_configs = [Configure.NamedVectors.none(name="vector")]

        # Configure properties
        properties = []
        if collection_request.properties:
            for prop_def in collection_request.properties:
                try:
                    weaviate_prop = create_property_definition(prop_def)
                    logger.info(f"Created property: {weaviate_prop}")
                    properties.append(weaviate_prop)
                except Exception as e:
                    logger.error(f"Error creating property {prop_def.name}: {str(e)}")
                    raise
                
        # Create the collection
        logger.info(f"Creating collection: {collection_name} with {len(properties)} properties and {len(vector_configs)} vectors")
        
        # Check if description is supported
        create_params = {
            "name": collection_name,
            "vectorizer_config": vector_configs,
            "properties": properties
        }
        
        # Add description if supported
        if hasattr(client.collections.create, "__code__") and "description" in client.collections.create.__code__.co_varnames:
            create_params["description"] = collection_request.description
        
        logger.info(f"Create parameters: {json.dumps(create_params, default=str)}")
        client.collections.create(**create_params)

        # Return success message with collection list
        collections = client.collections.list_all()
        collection_names = get_collection_names(collections)
        
        logger.info(f"Collection '{collection_name}' created successfully")
        return JSONResponse(
            status_code=201,
            content={
                "message": f"Collection '{collection_name}' created successfully.",
                "collection": collection_name,
                "total_collections": len(collection_names),
                "collections": collection_names
            }
        )
    except Exception as e:
        # Handle any unexpected errors
        error_msg = f"Error creating collection: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": error_msg, "traceback": traceback.format_exc()}
        )
        
async def weaviate_insert(insert_request):
    """
    Insert data into a Weaviate collection.
    
    Args:
        insert_request: Data insertion request with dynamic properties and vectors
        
    Returns:
        JSONResponse with insertion status
    """
    try:
        collection_name = insert_request.collection_name
        
        # Get collection
        try:
            logger.debug(f"Getting collection: {collection_name}")
            collection = client.collections.get(collection_name)
        except Exception as e:
            error_msg = f"Collection not found: {collection_name}"
            logger.error(error_msg)
            return JSONResponse(
                status_code=404,
                content={"message": error_msg, "error": str(e)}
            )

        # Insert data in batch
        logger.info(f"Inserting {len(insert_request.data)} items into collection: {collection_name}")
        with collection.batch.dynamic() as batch:
            for item in insert_request.data:
                # Convert item to dictionary format Weaviate expects
                properties = item.properties
                vectors = item.vectors if item.vectors else {}
                
                # Add the object to the batch
                object_params = {
                    "properties": properties,
                }
                
                # Add vectors if provided
                if vectors:
                    object_params["vector"] = vectors
                
                # Add UUID if provided
                if item.id:
                    object_params["uuid"] = item.id
                
                logger.info(f"Inserting object with properties: {json.dumps(properties, default=str)}")
                batch.add_object(**object_params)

        # Check for failed objects
        failed_objects = collection.batch.failed_objects
        if failed_objects:
            logger.warning(f"Failed to import {len(failed_objects)} objects")
            # Convert failed objects to serializable format
            failed_details = []
            for obj in failed_objects:
                if hasattr(obj, '__dict__'):
                    # Convert the object to a dictionary
                    obj_dict = obj.__dict__
                    # Convert any non-serializable attributes to strings
                    serializable_dict = {}
                    for key, value in obj_dict.items():
                        try:
                            json.dumps(value)  # Test if value is serializable
                            serializable_dict[key] = value
                        except (TypeError, ValueError):
                            serializable_dict[key] = str(value)
                    failed_details.append(serializable_dict)
                else:
                    failed_details.append(str(obj))
            return JSONResponse(
                status_code=500,
                content={
                    "message": f"Failed to import {len(failed_objects)} objects",
                    "details": failed_details
                }
            )
        else:
            logger.info(f"Successfully imported all {len(insert_request.data)} objects")
            return JSONResponse(
                status_code=200,
                content={"message": "All objects imported successfully!"}
            )

    except Exception as e:
        error_msg = f"Unexpected error during insert operation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        # Convert error to serializable format
        error_details = str(e)
        if hasattr(e, '__dict__'):
            error_dict = e.__dict__
            # Convert any non-serializable attributes to strings
            error_details = {}
            for key, value in error_dict.items():
                try:
                    json.dumps(value)  # Test if value is serializable
                    error_details[key] = value
                except (TypeError, ValueError):
                    error_details[key] = str(value)
        return JSONResponse(
            status_code=500,
            content={"message": error_msg, "error": error_details}
        )

async def weaviate_search(search_request):
    """
    Search for vectors in Weaviate.
    
    Args:
        search_request: Search request parameters
        
    Returns:
        Dict containing search results
    """
    try:
        # Get collection
        try:
            collection = client.collections.get(search_request.collection_name)
            logger.debug(f"Found collection: {search_request.collection_name}")
        except Exception as e:
            error_msg = f"Collection not found: {search_request.collection_name}"
            logger.error(error_msg)
            return JSONResponse(
                status_code=404,
                content={"status_code": 404, "details": error_msg, "error": str(e)}
            )
        
        # Validate query vector
        if not search_request.query.vector:
            error_msg = "Search request must include a vector"
            logger.error(error_msg)
            return JSONResponse(
                status_code=400,
                content={"status_code": 400, "details": error_msg}
            )
            
        # Handle both single vector and multiple vectors
        query_vector = search_request.query.vector
        if isinstance(query_vector, list):
            # Single vector case
            near_vector = query_vector
            target_vector = TargetVectors.manual_weights(search_request.columns)
        else:
            # Multiple vectors case
            near_vector = query_vector
            target_vector = list(query_vector.keys())
            
        logger.debug(f"Using provided vector(s) with length(s): {[len(v) if isinstance(v, list) else len(v) for v in query_vector.values()] if isinstance(query_vector, dict) else len(query_vector)}")

        # Process filters if provided
        filters = None
        if search_request.filters:
            try:
                filter_property = search_request.filters.property
                filter_value = search_request.filters.valueText
                
                # Create filter based on operator
                if search_request.filters.operator == "Equal":
                    filters = Filter.by_property(filter_property).equal(filter_value)
                elif search_request.filters.operator == "Like":
                    filters = Filter.by_property(filter_property).like(filter_value)
                elif search_request.filters.operator == "GreaterThan":
                    filters = Filter.by_property(filter_property).greater_than(filter_value)
                elif search_request.filters.operator == "LessThan":
                    filters = Filter.by_property(filter_property).less_than(filter_value)
                elif search_request.filters.operator == "ContainsAny":
                    filters = Filter.by_property(filter_property).contains_any(filter_value)
                elif search_request.filters.operator == "ContainsAll":
                    filters = Filter.by_property(filter_property).contains_all(filter_value)
                else:
                    raise ValueError(f"Unsupported filter operator: {search_request.filters.operator}")
                    
                logger.debug(f"Applied filters: {search_request.filters}")
            except Exception as filter_error:
                error_msg = f"Invalid filter format: {str(filter_error)}"
                logger.error(error_msg)
                raise HTTPException(
                    status_code=400,
                    detail=error_msg
                )
        
        # Execute vector search
        try:
            logger.info(f"Executing vector search with columns: {search_request.columns}")
            
            response = collection.query.near_vector(
                near_vector=near_vector,
                limit=search_request.top_k,
                target_vector=target_vector,
                return_metadata=MetadataQuery(distance=True),
                filters=filters
            )
            
            # Process results
            results = []
            for o in response.objects:
                filtered_data = {field: o.properties.get(field) for field in search_request.output_fields}
                results.append({
                    "score": 1 - o.metadata.distance,
                    "data": filtered_data
                })
            
            logger.info(f"Returning {len(results)} search results")
            return {"results": results}
            
        except Exception as e:
            error_msg = f"Error during Weaviate search: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Special handling for vector length mismatch errors
            error_text = str(e)
            if "vector lengths don't match" in error_text:
                beautified_error = beautify_grpc_error(error_text)
                logger.debug(f"Beautified vector length mismatch error")
                return {"results": beautified_error, "error": "vector_length_mismatch"}
            else:
                raise HTTPException(
                    status_code=500,
                    detail=error_msg
                )

    except HTTPException as e:
        # Pass through HTTP exceptions
        raise e
    except Exception as e:
        # Catch all other exceptions
        error_msg = f"Unexpected error during search operation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status_code": 500, "details": error_msg, "traceback": traceback.format_exc()}
        )
