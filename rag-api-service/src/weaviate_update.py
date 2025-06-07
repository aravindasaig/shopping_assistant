from fastapi import HTTPException
from typing import Dict, Any, List, Optional
import logging
from src.weaviate_query import client
from src.utils import get_collection_names

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def update_properties(
    collection_name: str,
    object_id: str,
    properties: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update specific properties of an object.
    
    Args:
        collection_name: Name of the collection
        object_id: UUID of the object to update
        properties: Dictionary of properties to update
        
    Returns:
        Dict containing the update status
    """
    try:
        # Get the collection
        collection = client.collections.get(collection_name)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found"
            )
        
        # Update the object
        collection.data.update(
            uuid=object_id,
            properties=properties
        )
        
        return {
            "status": "success",
            "message": f"Object {object_id} updated successfully",
            "collection": collection_name,
            "updated_properties": list(properties.keys())
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error updating object properties: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating object properties: {str(e)}"
        )

async def update_with_vector(
    collection_name: str,
    object_id: str,
    properties: Dict[str, Any],
    vector: List[float]
) -> Dict[str, Any]:
    """
    Update object properties and its vector.
    
    Args:
        collection_name: Name of the collection
        object_id: UUID of the object to update
        properties: Dictionary of properties to update
        vector: New vector for the object
        
    Returns:
        Dict containing the update status
    """
    try:
        # Get the collection
        collection = client.collections.get(collection_name)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found"
            )
        
        # Update the object with vector
        collection.data.update(
            uuid=object_id,
            properties=properties,
            vector=vector
        )
        
        return {
            "status": "success",
            "message": f"Object {object_id} updated successfully with new vector",
            "collection": collection_name,
            "updated_properties": list(properties.keys())
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error updating object with vector: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating object with vector: {str(e)}"
        )

async def replace_object(
    collection_name: str,
    object_id: str,
    properties: Dict[str, Any],
    vector: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Replace an entire object with new properties and optionally a new vector.
    
    Args:
        collection_name: Name of the collection
        object_id: UUID of the object to replace
        properties: Dictionary of new properties
        vector: Optional new vector for the object
        
    Returns:
        Dict containing the replace status
    """
    try:
        # Get the collection
        collection = client.collections.get(collection_name)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found"
            )
        
        # Replace the object
        if vector:
            collection.data.replace(
                uuid=object_id,
                properties=properties,
                vector=vector
            )
        else:
            collection.data.replace(
                uuid=object_id,
                properties=properties
            )
        
        return {
            "status": "success",
            "message": f"Object {object_id} replaced successfully",
            "collection": collection_name,
            "new_properties": list(properties.keys())
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error replacing object: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error replacing object: {str(e)}"
        ) 