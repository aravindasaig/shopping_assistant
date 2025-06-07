import os
import json
import base64
import asyncio
import tempfile
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from embedding_generator import EmbeddingGenerator
from fastapi import UploadFile, File

import sys
sys.path.append('../')
from logger_config import logger

# Initialize FastAPI app
app = FastAPI(title="Embedding Generation Service")

# Initialize the embedding generator
embedding_generator = EmbeddingGenerator()

class ContentItem(BaseModel):
    chunk_id: str
    chunk_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding_content_texts: str = None
    generated_context: str = None
    image_base64: str = None
    image_format: str = None

class EmbeddingRequest(BaseModel):
    content: List[ContentItem]
    output_path: str
    job_id: str

class TextEmbeddingRequest(BaseModel):
    text: str

class ImageEmbeddingRequest(BaseModel):
    image_path: str

def validate_uploaded_file(file: UploadFile) -> None:
    """
    Validate uploaded file with proper None checks.
    
    Args:
        file: FastAPI UploadFile object
        
    Raises:
        HTTPException: If file validation fails
    """
    # Check if filename exists and is valid
    if not file.filename:
        logger.error("No filename provided")
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if file.filename == "":
        logger.error("Empty filename")
        raise HTTPException(status_code=400, detail="Empty filename")
    
    # Check file extension
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        logger.error(f"Unsupported file extension: {file_extension}")
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Check content type if available (with None check)
    if file.content_type is not None:  # Fix for the startswith error
        if not file.content_type.startswith("image/"):
            logger.error(f"Invalid content type: {file.content_type}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid content type: {file.content_type}. Expected image/*"
            )
    else:
        logger.warning("No content type provided, relying on file extension validation")
    
    logger.info(f"File validation passed for: {file.filename}")

@app.post("/generate-embeddings")
async def generate_embeddings(request: Request):
    """
    API endpoint to receive content and generate embeddings.
    
    The endpoint receives content items (text or images), processes them,
    generates embeddings, and returns the embeddings along with job_id and output_path.
    """
    try:
        # Parse the request body
        request_data = await request.json()
        
        # Log request information
        job_id = request_data.get("job_id", "unknown")
        content_items = request_data.get("content", [])
        output_path = request_data.get("output_path", "")

        logger.info(f"Received embedding generation request for job {job_id} with {len(content_items)} items")
        if content_items:
            logger.info(f"Content items keys: {content_items[0].keys()}")

        # Process the content items
        processed_content = await process_content_items(content_items)
        
        # Generate embeddings
        embedding_results = await embedding_generator.generate_embeddings(processed_content, output_path)
        
        # Add job_id to the response
        embedding_results["job_id"] = job_id
        
        # Clean up temporary image files
        await cleanup_temp_files(processed_content)
        
        return embedding_results
        
    except Exception as e:
        logger.error(f"Error processing embedding request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing embedding request: {str(e)}")

async def process_content_items(content_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process content items, converting base64 images to temporary files.
    
    Args:
        content_items: List of content items from the request
        
    Returns:
        List of processed content items ready for embedding generation
    """
    processed_items = []
    
    for item in content_items:
        try:
            processed_item = item.copy()
            
            # Process based on chunk type
            if item.get("chunk_type") == "image" and item.get("image_base64"):
                # Convert base64 to image file
                temp_image_path = await base64_to_image(
                    item["image_base64"], 
                    item.get("image_format", "png")
                )
                
                # Update the item with the temporary image path
                processed_item["chunk"] = temp_image_path
                
                # Remove the base64 data to save memory
                if "image_base64" in processed_item:
                    del processed_item["image_base64"]
                
                logger.info(f"Processed image chunk {item.get('chunk_id')}, saved to temporary file")
                
            elif item.get("chunk_type") == "text":
                # For text chunks, ensure the content is in the expected format
                if "embedding_content_texts" in item:
                    processed_item["chunk"] = item["embedding_content_texts"]
                
                logger.info(f"Processed text chunk {item.get('chunk_id')}")
            
            processed_items.append(processed_item)
            
        except Exception as e:
            logger.error(f"Error processing content item {item.get('chunk_id')}: {str(e)}")
            # Continue with other items even if one fails
    
    logger.info(f"Processed {len(processed_items)} content items")
    return processed_items

async def base64_to_image(base64_string: str, image_format: str) -> str:
    """
    Convert a base64 string to an image file.
    
    Args:
        base64_string: Base64 encoded image data
        image_format: Format of the image (jpg, png, etc.)
        
    Returns:
        Path to the temporary image file
    """
    try:
        # Decode the base64 string
        image_data = base64.b64decode(base64_string)
        
        # Create a temporary file with the appropriate extension
        suffix = f".{image_format}" if image_format else ".png"
        
        # Create a temporary file that won't be automatically deleted
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_file_path = temp_file.name
        
        # Write the image data to the file
        with open(temp_file_path, "wb") as f:
            f.write(image_data)
        
        logger.info(f"Converted base64 image to temporary file: {temp_file_path}")
        return temp_file_path
        
    except Exception as e:
        logger.error(f"Error converting base64 to image: {str(e)}")
        raise

# Startup event to load the embedding model
@app.on_event("startup")
async def startup_event():
    logger.info("Starting embedding service and loading model...")
    await embedding_generator.load_model()
    logger.info("Embedding service started and model loaded")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "embedding_generation"}

@app.post("/embed-image")
async def embed_image(file: UploadFile = File(...)):
    """
    Generate embedding for an uploaded image file.
    """
    temp_file_path = None
    try:
        logger.info("=== Image Embedding Request ===")
        logger.info(f"Filename: {file.filename}")
        logger.info(f"Content type: {file.content_type}")
        logger.info(f"Size: {file.size if hasattr(file, 'size') else 'unknown'}")
        
        # Validate the uploaded file
        validate_uploaded_file(file)
        
        # Read content and check size
        content = await file.read()
        
        if not content:
            raise HTTPException(status_code=400, detail="Empty file content")
        
        if len(content) > 10 * 1024 * 1024:  # 10 MB limit
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")
        
        logger.info(f"File size: {len(content)} bytes")
        
        # Create temporary file
        suffix = os.path.splitext(file.filename)[1] or ".png"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file_path = temp_file.name
        
        # Write content to temporary file
        with open(temp_file_path, "wb") as f:
            f.write(content)

        logger.info(f"Received image file {file.filename}, saved to {temp_file_path}")

        # Generate the embedding
        image_embedding = await embedding_generator.generate_image_embedding(temp_file_path)
        
        if image_embedding is None:
            raise HTTPException(status_code=500, detail="Failed to generate embedding")
        
        # Convert numpy array to list if needed
        if hasattr(image_embedding, 'tolist'):
            embedding_list = image_embedding.tolist()
        else:
            embedding_list = image_embedding

        logger.info(f"Successfully generated embedding with {len(embedding_list)} dimensions")

        return {
            "embedding": embedding_list,
            "dimensions": len(embedding_list),
            "filename": file.filename
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error generating image embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate image embedding: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file_path}: {e}")

@app.post("/generate-text-embedding")
async def generate_text_embedding(request: TextEmbeddingRequest):
    """
    Generate an embedding for a single text string.
    
    This endpoint allows users to get embeddings for arbitrary text
    without going through the full document processing pipeline.
    """
    try:
        logger.info(f"=== Text Embedding Request ===")
        logger.info(f"Text length: {len(request.text)}")
        
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")
        
        # Use the already initialized embedding_generator
        result = await embedding_generator.generate_embedding_for_text(request.text)
        
        logger.info(f"Successfully generated text embedding")
        
        return {
            "embedding": result["embedding"],
            "dimensions": len(result["embedding"]),
            "text_length": len(request.text)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating text embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def cleanup_temp_files(processed_content):
    """
    Clean up temporary files created during processing.
    """
    for item in processed_content:
        if (item.get("chunk_type") == "image" and 
            item.get("chunk") and 
            isinstance(item.get("chunk"), str) and
            item.get("chunk").startswith(tempfile.gettempdir())):
            try:
                os.remove(item["chunk"])
                logger.info(f"Removed temporary file: {item['chunk']}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {item['chunk']}: {e}")

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.error(f"ValueError: {str(exc)}")
    return HTTPException(status_code=400, detail=str(exc))

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    logger.error(f"RuntimeError: {str(exc)}")
    return HTTPException(status_code=500, detail=str(exc))

# Additional debugging endpoint
@app.get("/debug/info")
async def debug_info():
    """Debug endpoint to check service status"""
    return {
        "model_loaded": embedding_generator.model is not None,
        "temp_dir": tempfile.gettempdir(),
        "python_version": sys.version,
        "endpoints": [
            "/health",
            "/embed-image", 
            "/generate-text-embedding",
            "/generate-embeddings"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI embedding service...")
    uvicorn.run(app, host="0.0.0.0", port=6006, log_level="info")