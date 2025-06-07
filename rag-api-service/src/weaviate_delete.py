from fastapi import HTTPException
from fastapi.responses import JSONResponse
from weaviate.classes.query import Filter
import logging
from typing import List, Optional, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def delete_collection(collection_name: str, client) -> JSONResponse:
    """
    Delete an entire collection from Weaviate.
    
    Args:
        collection_name: Name of the collection to delete
        client: Weaviate client instance
        
    Returns:
        JSONResponse with deletion status
    """
    try:
        logger.info(f"Deleting collection: {collection_name}")
        
        # Check if collection exists
        collections = client.collections.list_all()
        collection_names = [str(col) for col in collections]
        
        if collection_name not in collection_names:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        # Delete the collection
        client.collections.delete(collection_name)
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Collection '{collection_name}' deleted successfully",
                "collection": collection_name
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def delete_by_id(collection_name: str, object_id: str, client) -> JSONResponse:
    """
    Delete a specific object by its ID.
    
    Args:
        collection_name: Name of the collection
        object_id: UUID of the object to delete
        client: Weaviate client instance
        
    Returns:
        JSONResponse with deletion status
    """
    try:
        collection = client.collections.get(collection_name)
        collection.data.delete_by_id(object_id)
        return JSONResponse(
            status_code=200,
            content={"message": f"Object {object_id} deleted successfully"}
        )
    except Exception as e:
        logger.error(f"Error deleting object {object_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def delete_by_filter(collection_name: str, filter_criteria: Dict[str, Any], client) -> JSONResponse:
    """
    Delete objects matching specific filter criteria.
    
    Args:
        collection_name: Name of the collection
        filter_criteria: Dictionary containing filter conditions
        client: Weaviate client instance
        
    Returns:
        JSONResponse with deletion status
    """
    try:
        collection = client.collections.get(collection_name)
        
        # Create filter using the correct Weaviate Filter API
        filter_obj = Filter.by_property(filter_criteria["property"]).equal(filter_criteria["valueText"])
        
        result = collection.data.delete_many(
            where=filter_obj,
            verbose=True
        )
        print(result,"assasas")
        return JSONResponse(
            status_code=200,
            content={
                "message": "Objects deleted successfully",
                "details": {
                    "deleted": str(result)
                }
            }
        )
    except Exception as e:
        logger.error(f"Error deleting objects with filter: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def delete_by_ids(collection_name: str, object_ids: List[str], client) -> JSONResponse:
    """
    Delete multiple objects by their IDs.
    
    Args:
        collection_name: Name of the collection
        object_ids: List of UUIDs to delete
        client: Weaviate client instance
        
    Returns:
        JSONResponse with deletion status
    """
    try:
        collection = client.collections.get(collection_name)
        filter_obj = Filter.by_id().contains_any(object_ids)
        
        result = collection.data.delete_many(
            where=filter_obj,
            verbose=True
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Objects deleted successfully",
                "details": {
                    "deleted": result.results.successful,
                    "failed": result.results.failed,
                    "total_matches": result.results.matches
                }
            }
        )
    except Exception as e:
        logger.error(f"Error deleting objects by IDs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def dry_run_delete(collection_name: str, filter_criteria: Dict[str, Any], client) -> JSONResponse:
    """
    Perform a dry run of deletion to see which objects would be affected.
    
    Args:
        collection_name: Name of the collection
        filter_criteria: Dictionary containing filter conditions
        client: Weaviate client instance
        
    Returns:
        JSONResponse with dry run results
    """
    try:
        collection = client.collections.get(collection_name)
        
        # Create filter using the correct Weaviate Filter API
        filter_obj = Filter.by_property(filter_criteria["property"]).equal(filter_criteria["valueText"])
        
        result = collection.data.delete_many(
            where=filter_obj,
            dry_run=True,
            verbose=True
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Dry run completed",
                "details": {
                    "would_delete": result.results.matches,
                    "objects": [
                        {
                            "id": obj.id,
                            "status": obj.status
                        } for obj in result.results.objects
                    ]
                }
            }
        )
    except Exception as e:
        logger.error(f"Error in dry run deletion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 