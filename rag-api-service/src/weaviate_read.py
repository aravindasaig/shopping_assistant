from fastapi import HTTPException
from typing import List, Dict, Any, Optional, Union
import logging
from src.weaviate_query import client
from src.utils import get_collection_names, FilterCriteria
from weaviate.classes.query import Filter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def read_objects(
    collection_name: str,
    include_vectors: bool = False,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    filters: Optional[FilterCriteria] = None
) -> Dict[str, Any]:
    """
    Read objects from a collection with optional vector inclusion, pagination, and filtering.
    
    Args:
        collection_name: Name of the collection to read from
        include_vectors: Whether to include vectors in the response
        limit: Maximum number of objects to return
        offset: Number of objects to skip
        filters: Optional filter criteria
        
    Returns:
        Dict containing the objects and their properties/vectors
    """
    try:
        # Get the collection
        collection = client.collections.get(collection_name)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found"
            )
        
        # Create filter if provided
        filter_obj = None
        if filters:
            # Create filter based on operator
            if filters.operator == "Equal":
                filter_obj = Filter.by_property(filters.property).equal(filters.valueText)
            elif filters.operator == "Like":
                filter_obj = Filter.by_property(filters.property).like(filters.valueText)
            elif filters.operator == "GreaterThan":
                filter_obj = Filter.by_property(filters.property).greater_than(filters.valueText)
            elif filters.operator == "LessThan":
                filter_obj = Filter.by_property(filters.property).less_than(filters.valueText)
            elif filters.operator == "ContainsAny":
                filter_obj = Filter.by_property(filters.property).contains_any(filters.valueText)
            elif filters.operator == "ContainsAll":
                filter_obj = Filter.by_property(filters.property).contains_all(filters.valueText)
        
        # Execute query with filters
        result = collection.query.fetch_objects(
            filters=filter_obj,
            limit=limit,
            offset=offset,
            include_vector=include_vectors
        )
        
        # Process results
        objects = []
        for item in result.objects:
            # Create object data with basic properties
            object_data = {
                "id": str(item.uuid),
                "properties": item.properties
            }
            
            # Add vectors if requested and available
            if include_vectors and hasattr(item, 'vector'):
                object_data["vector"] = item.vector
            
            objects.append(object_data)
        
        # Return response
        return {
            "collection": collection_name,
            "total_objects": len(result.objects),
            "returned_objects": len(objects),
            "offset": offset if offset is not None else 0,
            "limit": limit if limit is not None else len(result.objects),
            "objects": objects
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error reading objects: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reading objects: {str(e)}"
        )

async def read_multi_tenant_objects(
    collection_name: str,
    include_vectors: bool = False
) -> Dict[str, Any]:
    """
    Read objects from a multi-tenant collection with optional vector inclusion.
    
    Args:
        collection_name: Name of the collection to read from
        include_vectors: Whether to include vectors in the response
        
    Returns:
        Dict containing the objects and their properties/vectors, grouped by tenant
    """
    try:
        # Get the collection
        collection = client.collections.get(collection_name)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found"
            )
        
        # Get tenants
        tenants = collection.tenants.get()
        if not tenants:
            raise HTTPException(
                status_code=404,
                detail=f"No tenants found in collection '{collection_name}'"
            )
        
        # Process objects for each tenant
        tenant_objects = {}
        
        for tenant_name in tenants.keys():
            tenant_collection = collection.with_tenant(tenant_name)
            
            # Build query to fetch objects
            query = tenant_collection.query.fetch_objects()
            
            # Include vectors if requested
            if include_vectors:
                query = query.with_vector()
            
            # Execute query
            result = query.do()
            
            # Process objects
            objects = []
            for item in result.objects:
                # Create object data with basic properties
                object_data = {
                    "id": str(item.uuid),
                    "properties": item.properties
                }
                
                # Add vectors if requested and available
                if include_vectors and hasattr(item, 'vector'):
                    object_data["vectors"] = item.vector
                
                objects.append(object_data)
            
            tenant_objects[tenant_name] = objects
        
        # Return response
        return {
            "collection": collection_name,
            "tenants": list(tenants.keys()),
            "objects_by_tenant": tenant_objects
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error reading multi-tenant objects: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reading multi-tenant objects: {str(e)}"
        )

async def read_object_by_id(
    collection_name: str,
    object_id: str,
    include_vectors: Union[bool, List[str]] = False
) -> Dict[str, Any]:
    """
    Read a specific object by its ID.
    
    Args:
        collection_name: Name of the collection
        object_id: UUID of the object to read
        include_vectors: Whether to include vectors in the response
        
    Returns:
        Dict containing the object's properties and optionally vectors
    """
    try:
        # Get the collection
        collection = client.collections.get(collection_name)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found"
            )
        
        # Check if object exists
        exists = collection.data.exists(object_id)
        if not exists:
            raise HTTPException(
                status_code=404,
                detail=f"Object with ID '{object_id}' not found in collection '{collection_name}'"
            )
        
        # Get the object
        data_object = collection.query.fetch_object_by_id(
            uuid=object_id,
            include_vector=include_vectors
        )
        
        if not data_object:
            raise HTTPException(
                status_code=404,
                detail=f"Object with ID '{object_id}' not found in collection '{collection_name}'"
            )
        
        # Create result with basic properties
        result = {
            "id": str(data_object.uuid),
            "properties": data_object.properties
        }
        
        # Add vectors if requested and available
        if include_vectors and hasattr(data_object, 'vector'):
            if isinstance(include_vectors, list):
                # Include only specified named vectors
                result["vectors"] = {
                    name: data_object.vector[name]
                    for name in include_vectors
                    if name in data_object.vector
                }
            else:
                # Include all vectors
                result["vectors"] = data_object.vector
        
        return result
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error reading object: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reading object: {str(e)}"
        ) 