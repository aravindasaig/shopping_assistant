import os
import sys
import logging
import importlib.util
import uvicorn
import httpx
from fastapi import FastAPI, HTTPException, Body, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import atexit

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import *
from src.config import *
from src.weaviate_query import client
from src.weaviate_delete import delete_collection, delete_by_id, delete_by_filter, delete_by_ids, dry_run_delete
from src.weaviate_read import read_objects, read_object_by_id
from src.weaviate_update import update_properties, update_with_vector, replace_object

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if client is available
if client is None:
    logger.error("Weaviate client is not initialized")
    raise RuntimeError("Weaviate client is not initialized")

# Register cleanup handler
def cleanup():
    if client is not None:
        try:
            client.close()
            logger.info("Weaviate connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing Weaviate connection: {str(e)}")

atexit.register(cleanup)

# Setup file-based logging if enabled
if LOG_TO_FILE:
    try:
        # Ensure log directory exists
        log_dir = os.path.dirname(LOG_FILE_PATH)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Add rotating file handler
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            LOG_FILE_PATH,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)
        logger.info(f"File logging enabled at {LOG_FILE_PATH}")
    except Exception as e:
        logger.error(f"Failed to set up file logging: {str(e)}")

# Import database modules with error handling
def import_with_fallback(module_name, fallback_function=None):
    try:
        return importlib.import_module(f"src.{module_name}")
    except Exception as e:
        logger.warning(f"Could not import {module_name}: {str(e)}")
        if fallback_function:
            return type('DummyModule', (), {name: fallback_function for name in fallback_function.__annotations__})
        return None

# Fallback function for database operations
async def database_error_fallback(*args, **kwargs):
    """Fallback function when database connection fails"""
    return {
        "error": "database_unavailable",
        "message": f"The {VECTOR_DB_TYPE} database is not available or properly configured",
        "details": "Check your database configuration and ensure the service is running"
    }

# Import database modules
weaviate_module = import_with_fallback("weaviate_query", database_error_fallback)
milvus_module = import_with_fallback("milvus_search", database_error_fallback)

# Import functions or use fallbacks
weaviate_search = getattr(weaviate_module, "weaviate_search", database_error_fallback)
weaviate_create = getattr(weaviate_module, "weaviate_create", database_error_fallback)
weaviate_insert = getattr(weaviate_module, "weaviate_insert", database_error_fallback)
milvus_search = getattr(milvus_module, "milvus_search", database_error_fallback)

app = FastAPI(
    title="Vector Database API",
    description="A flexible API service for vector database operations",
    version="1.0.0"
)

# Database status check
def get_database_status():
    """Check if the database is available"""
    db_available = False
    db_name = VECTOR_DB_TYPE
    error_message = None
    
    if VECTOR_DB_TYPE == "weaviate":
        db_available = weaviate_module is not None and weaviate_module != database_error_fallback
    elif VECTOR_DB_TYPE == "milvus":
        db_available = milvus_module is not None and milvus_module != database_error_fallback
    
    if not db_available:
        error_message = f"The {db_name} database is not available"
    
    return {"available": db_available, "error": error_message}

# Status endpoint
@app.get("/status", summary="Check API and database status")
async def status():
    """Return the status of the API and configured database"""
    db_status = get_database_status()
    return {
        "status": "operational" if db_status["available"] else "degraded",
        "api_version": "1.0.0",
        "database": {
            "type": VECTOR_DB_TYPE,
            "status": "connected" if db_status["available"] else "disconnected",
            "error": db_status["error"]
        }
    }

# Database dependency
async def get_db():
    """Dependency to check database availability before each request"""
    db_status = get_database_status()
    if not db_status["available"]:
        raise HTTPException(
            status_code=503, 
            detail=f"Database unavailable: {db_status['error']}"
        )
    return True

# Search Endpoint
@app.post("/search", summary="Search vector database")
async def search_endpoint(search_request: SearchRequest, db: bool = Depends(get_db)):
    """
    Search for similar vectors in the database.
    
    - **collection_name**: Name of the collection to search in
    - **query**: Vector or text to search for
    - **columns**: Weight for each vector field
    - **filters**: Optional filters to apply
    - **output_fields**: Fields to return in results
    - **top_k**: Number of results to return
    """
    try:
        logger.info(f"Received search request for collection: {search_request.collection_name}")
        if VECTOR_DB_TYPE == "milvus":
            logger.debug(f"Using Milvus search for request")
            return await milvus_search(search_request)
        elif VECTOR_DB_TYPE == "weaviate":
            logger.debug(f"Using Weaviate search for request")
            return await weaviate_search(search_request)
        else:
            logger.error(f"Invalid DB type: {VECTOR_DB_TYPE}")
            raise HTTPException(status_code=400, detail="Invalid db_type. Must be 'milvus' or 'weaviate'.")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Status Error: {str(e)}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Unexpected error in search endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create", summary="Create a new collection schema")
async def create_collection(collection_request: CollectionRequest, db: bool = Depends(get_db)):
    """
    Create a new collection with custom schema.
    - **collection_name**: Name of the collection to create
    - **properties**: Optional list of property definitions
    - **vectors**: Optional list of vector configurations
    - **description**: Optional collection description
    """
    try:
        logger.info(f"Received create request for collection: {collection_request.collection_name}")
        if VECTOR_DB_TYPE != "weaviate":
            logger.warning(f"Collection creation not supported for DB type: {VECTOR_DB_TYPE}")
            raise HTTPException(status_code=501, detail="Collection creation is only supported for Weaviate.")
        elif VECTOR_DB_TYPE == "weaviate":
            logger.debug(f"Creating Weaviate collection: {collection_request.collection_name}")
            return await weaviate_create(collection_request)
        else:
            logger.error(f"Invalid DB type: {VECTOR_DB_TYPE}")
            raise HTTPException(status_code=400, detail="Invalid db_type. Must be 'milvus' or 'weaviate'.")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Status Error: {str(e)}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Unexpected error in create endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/insert", summary="Insert data into collection")
async def insert_data(insert_request: InsertRequest, db: bool = Depends(get_db)):
    """
    Insert data into a collection.
    
    - **collection_name**: Name of the collection to insert into
    - **data**: List of items to insert with properties and vectors
    """
    try:
        logger.info(f"Received insert request for collection: {insert_request.collection_name} with {len(insert_request.data)} items")
        if VECTOR_DB_TYPE != "weaviate":
            logger.warning(f"Data insertion not supported for DB type: {VECTOR_DB_TYPE}")
            raise HTTPException(status_code=501, detail="Data insertion is only supported for Weaviate.")
        elif VECTOR_DB_TYPE == "weaviate":
            logger.debug(f"Inserting data into Weaviate collection: {insert_request.collection_name}")
            return await weaviate_insert(insert_request)
        else:
            logger.error(f"Invalid DB type: {VECTOR_DB_TYPE}")
            raise HTTPException(status_code=400, detail="Invalid db_type. Must be 'milvus' or 'weaviate'.")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Status Error: {str(e)}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Unexpected error in insert endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections", summary="List all collections")
async def list_collections(db: bool = Depends(get_db)):
    """
    List all available collections in the vector database.
    """
    try:
        logger.info("Listing all collections")
        if VECTOR_DB_TYPE == "weaviate":
            collections = client.collections.list_all()
            collection_names = get_collection_names(collections)
            return {
                "collections": collection_names,
                "total": len(collection_names)
            }
        else:
            logger.warning(f"Collection listing not supported for DB type: {VECTOR_DB_TYPE}")
            raise HTTPException(status_code=501, detail="Collection listing is only supported for Weaviate.")
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collections/{collection_name}", summary="Delete a collection")
async def delete_collection_endpoint(collection_name: str, db: bool = Depends(get_db)):
    """
    Delete a collection from the vector database.
    
    - **collection_name**: Name of the collection to delete
    """
    return await delete_collection(collection_name, client)

@app.delete("/objects/{collection_name}/{object_id}", summary="Delete an object by ID")
async def delete_object_endpoint(collection_name: str, object_id: str, db: bool = Depends(get_db)):
    """
    Delete a specific object by its ID.
    
    - **collection_name**: Name of the collection
    - **object_id**: UUID of the object to delete
    """
    return await delete_by_id(collection_name, object_id, client)

@app.delete("/objects/{collection_name}", summary="Delete objects by filter")
async def delete_objects_by_filter_endpoint(
    collection_name: str,
    filter_criteria: dict = Body(...),
    db: bool = Depends(get_db)
):
    """
    Delete objects matching specific filter criteria.
    
    - **collection_name**: Name of the collection
    - **filter_criteria**: Dictionary containing filter conditions
    """
    return await delete_by_filter(collection_name, filter_criteria, client)

@app.delete("/objects/batch/{collection_name}", summary="Delete multiple objects by IDs")
async def delete_objects_by_ids_endpoint(
    collection_name: str,
    object_ids: list = Body(...),
    db: bool = Depends(get_db)
):
    """
    Delete multiple objects by their IDs.
    
    - **collection_name**: Name of the collection
    - **object_ids**: List of UUIDs to delete
    """
    return await delete_by_ids(collection_name, object_ids, client)

@app.post("/objects/dry-run/{collection_name}", summary="Dry run deletion")
async def dry_run_delete_endpoint(
    collection_name: str,
    filter_criteria: dict = Body(...),
    db: bool = Depends(get_db)
):
    """
    Perform a dry run of deletion to see which objects would be affected.
    
    - **collection_name**: Name of the collection
    - **filter_criteria**: Dictionary containing filter conditions
    """
    return await dry_run_delete(collection_name, filter_criteria, client)

# Legacy endpoints for backward compatibility
@app.post("/legacy/create", summary="Create a collection with predefined schema (legacy)")
async def create_weaviate_collection_legacy(collection_name: CollectionRequest, db: bool = Depends(get_db)):
    logger.info(f"Received legacy create request for collection: {collection_name.collection_name}")
    return await create_collection(collection_name)

@app.post("/legacy/insert", summary="Insert data using legacy format")
async def insert_legacy(legacy_request: LegacyInsertRequest, db: bool = Depends(get_db)):
    """
    Insert data into a collection using the legacy format.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database is not available")
    
    try:
        result = await weaviate_insert(legacy_request)
        return result
    except Exception as e:
        logger.error(f"Error in legacy insert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/read", summary="Read objects from a collection")
async def read_endpoint(request: ReadRequest, db: bool = Depends(get_db)):
    """
    Read objects from a collection with various options.
    
    - **collection_name**: Name of the collection to read from
    - **include_vectors**: Whether to include vectors in the response (default: false)
    - **object_id**: Optional ID to read a specific object
    - **limit**: Maximum number of objects to return
    - **offset**: Number of objects to skip
    - **filters**: Optional filter criteria with property, operator, and value
    """
    try:
        logger.info(f"Reading from collection: {request.collection_name}")
        
        if request.object_id:
            # Read specific object by ID
            return await read_object_by_id(
                collection_name=request.collection_name,
                object_id=request.object_id,
                include_vectors=request.include_vectors
            )
        else:
            # Read objects with optional pagination and filtering
            return await read_objects(
                collection_name=request.collection_name,
                include_vectors=request.include_vectors,
                limit=request.limit,
                offset=request.offset,
                filters=request.filters
            )
            
    except Exception as e:
        logger.error(f"Error reading objects: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reading objects: {str(e)}"
        )

# Server start function
def start_server(host=HOST, port=PORT, num_workers=4, loop="asyncio", reload=False):
    logger.info(f"Starting server on {host}:{port} with {num_workers} workers")
    uvicorn.run("src.main:app",  # Update the import path
                host=host,
                port=port,
                workers=num_workers,
                loop=loop,
                reload=reload)

class DeleteRequest(BaseModel):
    operation_type: str  # "collection", "object", "filter", "batch", "dry-run"
    collection_name: str
    object_id: Optional[str] = None
    object_ids: Optional[List[str]] = None
    filter_criteria: Optional[Dict[str, Any]] = None

    class Config:
        schema_extra = {
            "example": {
                "operation_type": "filter",
                "collection_name": "test_collection",
                "filter_criteria": {
                    "property": "name",
                    "operator": "Like",
                    "valueText": "test*"
                }
            }
        }

@app.post("/delete", summary="Sample endpoint for all delete operations")
async def sample_delete_endpoint(request: DeleteRequest, db: bool = Depends(get_db)):
    """
    Sample endpoint demonstrating all delete operations.
    
    Request body examples:
    
    1. Delete collection:
    ```json
    {
        "operation_type": "collection",
        "collection_name": "test_collection"
    }
    ```
    
    2. Delete object by ID:
    ```json
    {
        "operation_type": "object",
        "collection_name": "test_collection",
        "object_id": "123e4567-e89b-12d3-a456-426614174000"
    }
    ```
    
    3. Delete by filter:
    ```json
    {
        "operation_type": "filter",
        "collection_name": "test_collection",
        "filter_criteria": {
            "property": "name",
            "operator": "Like",
            "valueText": "test*"
        }
    }
    ```
    
    4. Delete multiple objects:
    ```json
    {
        "operation_type": "batch",
        "collection_name": "test_collection",
        "object_ids": [
            "123e4567-e89b-12d3-a456-426614174000",
            "987fcdeb-a123-45d6-7890-123456789012"
        ]
    }
    ```
    
    5. Dry run deletion:
    ```json
    {
        "operation_type": "dry-run",
        "collection_name": "test_collection",
        "filter_criteria": {
            "property": "name",
            "operator": "Like",
            "valueText": "test*"
        }
    }
    ```
    """
    try:
        if request.operation_type == "collection":
            return await delete_collection(request.collection_name, client)
            
        elif request.operation_type == "object":
            if not request.object_id:
                raise HTTPException(status_code=400, detail="object_id is required for object deletion")
            return await delete_by_id(request.collection_name, request.object_id, client)
            
        elif request.operation_type == "filter":
            if not request.filter_criteria:
                raise HTTPException(status_code=400, detail="filter_criteria is required for filter deletion")
            return await delete_by_filter(request.collection_name, request.filter_criteria, client)
            
        elif request.operation_type == "batch":
            if not request.object_ids:
                raise HTTPException(status_code=400, detail="object_ids is required for batch deletion")
            return await delete_by_ids(request.collection_name, request.object_ids, client)
            
        elif request.operation_type == "dry-run":
            if not request.filter_criteria:
                raise HTTPException(status_code=400, detail="filter_criteria is required for dry run")
            return await dry_run_delete(request.collection_name, request.filter_criteria, client)
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid operation_type: {request.operation_type}. Must be one of: collection, object, filter, batch, dry-run"
            )
            
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in delete operation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update/properties", summary="Update object properties")
async def update_properties_endpoint(request: UpdatePropertiesRequest):
    """
    Update specific properties of an object.
    
    - **collection_name**: Name of the collection
    - **object_id**: UUID of the object to update
    - **properties**: Dictionary of properties to update
    """
    try:
        return await update_properties(
            collection_name=request.collection_name,
            object_id=request.object_id,
            properties=request.properties
        )
    except Exception as e:
        logger.error(f"Error in update properties endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update/with-vector", summary="Update object with vector")
async def update_with_vector_endpoint(request: UpdateWithVectorRequest):
    """
    Update object properties and its vector.
    
    - **collection_name**: Name of the collection
    - **object_id**: UUID of the object to update
    - **properties**: Dictionary of properties to update
    - **vector**: New vector for the object
    """
    try:
        return await update_with_vector(
            collection_name=request.collection_name,
            object_id=request.object_id,
            properties=request.properties,
            vector=request.vector
        )
    except Exception as e:
        logger.error(f"Error in update with vector endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update/replace", summary="Replace entire object")
async def replace_object_endpoint(request: ReplaceObjectRequest):
    """
    Replace an entire object with new properties and optionally a new vector.
    
    - **collection_name**: Name of the collection
    - **object_id**: UUID of the object to replace
    - **properties**: Dictionary of new properties
    - **vector**: Optional new vector for the object
    """
    try:
        return await replace_object(
            collection_name=request.collection_name,
            object_id=request.object_id,
            properties=request.properties,
            vector=request.vector
        )
    except Exception as e:
        logger.error(f"Error in replace object endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info(f"RAG API Service starting - Vector DB Type: {VECTOR_DB_TYPE}")
    logger.info(f"API documentation available at http://{HOST}:{PORT}/docs")
    start_server(host=HOST, port=PORT, num_workers=1) 