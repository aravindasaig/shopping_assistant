import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image
import tempfile

import sys
sys.path.append('../')
from logger_config import logger
# from config import TEXT_CHUNK_WEIGHT, TEXT_CONTEXT_WEIGHT, IMAGE_CONTENT_WEIGHT, IMAGE_DESCRIPTION_WEIGHT

class EmbeddingGenerator:
    """
    A class to generate and manage embeddings for text and image content.
    """
    
    def __init__(self, model_name="llamaindex/vdr-2b-multi-v1", device="cuda:0"):
        """
        Initialize the EmbeddingGenerator.
        
        Args:
            model_name: Name of the embedding model to use
            device: Device to run the model on (cuda:0, cpu, etc.)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        
    async def load_model(self):
        """
        Load the embedding model.
        """
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from llama_index.core import Document
            from langchain_community.vectorstores import FAISS
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = HuggingFaceEmbedding(
                model_name=self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading vdr embedding model: {e}")
            raise RuntimeError(f"Failed to load vdr embedding model: {e}")
    
    async def generate_image_embedding(self, image_path: str):
        if not self.model:
            raise RuntimeError("Model not loaded")
        return self.model.get_image_embedding(image_path)
    
    async def generate_text_embeddings(self, enriched_text_chunks):
        """
        Generate embeddings for text chunks.
        
        Args:
            enriched_text_chunks: List of enriched text chunks
            
        Returns:
            List of dictionaries with text chunks and their separate embeddings
        """
        if self.model is None:
            await self.load_model()
            
        logger.info(f"Generating embeddings for {len(enriched_text_chunks)} text chunks")
        logger.info("Storing separate embeddings for chunk content and context")
        
        text_embeddings = []
        
        for i, chunk in enumerate(enriched_text_chunks):
            try:
                # Extract text content and context
                chunk_text = chunk.get('embedding_content_texts', '')
                context = chunk.get('generated_context', 'self-contained')
                heading = chunk.get('metadata', {}).get('heading', '')
                
                # Skip if no content
                if not chunk_text:
                    logger.warning(f"Skipping chunk {i+1} - no content")
                    logger.info(f" Skipping Chunk {i+1} : {chunk}")
                    continue
                
                # Get embeddings for the chunk text
                chunk_embedding = np.array(self.model.get_query_embedding(chunk_text))
                
                # Create a copy of the chunk and add the embeddings
                enriched_chunk = chunk.copy()
                enriched_chunk['chunk_embedding'] = chunk_embedding.tolist()
                
                # Only get context embedding if it's not "self-contained"
                if context and context != "self-contained":
                    context_embedding = np.array(self.model.get_query_embedding(context))
                    enriched_chunk['context_embedding'] = context_embedding.tolist()
                else:
                    enriched_chunk['context_embedding'] = None
                
                text_embeddings.append(enriched_chunk)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(enriched_text_chunks)} text chunks")
                
            except Exception as e:
                logger.error(f"Error generating embedding for text chunk {i+1}: {e}")
        
        logger.info(f"Completed generating embeddings for {len(text_embeddings)} text chunks")
        return text_embeddings
    
    async def generate_image_embeddings(self, enriched_image_chunks):
        """
        Generate embeddings for image chunks.
        
        Args:
            enriched_image_chunks: List of enriched image chunks
            
        Returns:
            List of dictionaries with image chunks and their separate embeddings
        """
        if self.model is None:
            await self.load_model()
            
        logger.info(f"Generating embeddings for {len(enriched_image_chunks)} image chunks")
        logger.info("Storing separate embeddings for image content and description")
        
        image_embeddings = []
        
        for i, chunk in enumerate(enriched_image_chunks):
            try:
                # Extract image path and metadata
                image_path = chunk.get('chunk', '')
                metadata = chunk.get('metadata', {})
                detailed_analysis = metadata.get('detailed_analysis', {})
                
                # Extract text description from detailed analysis
                description = ""
                if isinstance(detailed_analysis, dict):
                    description = detailed_analysis.get('detailed_description', '')
                    if not description and 'key_elements' in detailed_analysis:
                        description = ', '.join(detailed_analysis.get('key_elements', []))
                    if not description and 'text_content' in detailed_analysis:
                        description = detailed_analysis.get('text_content', '')
                
                # Skip if no image path
                if not image_path or not os.path.exists(image_path):
                    logger.warning(f"Skipping image chunk {i+1} - invalid image path: {image_path}")
                    continue
                
                # Load the image
                image = Image.open(image_path)
                
                # Create a temporary file to save the image (required by the model)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_image_path = temp_file.name
                    image.save(temp_image_path)
                
                # Get image embedding
                image_embedding = np.array(self.model.get_image_embedding(temp_image_path))
                
                # Create a copy of the chunk and add the image embedding
                enriched_chunk = chunk.copy()
                enriched_chunk['chunk_embedding'] = image_embedding.tolist()
                
                # Get text embedding for the description if available
                if description:
                    text_embedding = np.array(self.model.get_query_embedding(description))
                    enriched_chunk['context_embedding'] = text_embedding.tolist()
                else:
                    enriched_chunk['context_embedding'] = None
                
                image_embeddings.append(enriched_chunk)
                
                # Clean up temporary file
                os.unlink(temp_image_path)
                
                if (i + 1) % 5 == 0:
                    logger.info(f"Processed {i+1}/{len(enriched_image_chunks)} image chunks")
                
            except Exception as e:
                logger.error(f"Error generating embedding for image chunk {i+1}: {e}")
        
        logger.info(f"Completed generating embeddings for {len(image_embeddings)} image chunks")
        return image_embeddings
    
    def _standardize_embedding_keys(self, embeddings, embedding_type=None):
        """
        Standardize the keys in the embeddings to ensure consistency.
        
        Args:
            embeddings: List of embedding dictionaries
            embedding_type: Optional type to enforce (text or image)
            
        Returns:
            List of standardized embedding dictionaries
        """
        standardized_embeddings = []
        
        for item in embeddings:
            standardized_item = item.copy()
            
            # Rename 'chunk' to 'content' for consistency
            if 'chunk' in standardized_item:
                standardized_item['content'] = standardized_item.pop('chunk')
            
            # For image embeddings, rename detailed_analysis to context
            if item.get('chunk_type') == 'image' and 'metadata' in standardized_item:
                metadata = standardized_item['metadata']
                if isinstance(metadata, dict) and 'detailed_analysis' in metadata:
                    metadata['context'] = metadata.pop('detailed_analysis')
            
            # Ensure chunk_type is set
            if embedding_type == 'text' and 'chunk_type' not in standardized_item:
                standardized_item['chunk_type'] = 'text'
            elif embedding_type == 'image' and 'chunk_type' not in standardized_item:
                standardized_item['chunk_type'] = 'image'
            
            standardized_embeddings.append(standardized_item)
        
        return standardized_embeddings

    async def generate_embeddings(self, enriched_content, output_path):
        """
        Generate embeddings for all content (text and images).
        
        Args:
            enriched_content: List of enriched content chunks
            output_path: Path to save the embeddings
            
        Returns:
            Dictionary with embeddings (but doesn't save them to disk)
        """
        if self.model is None:
            await self.load_model()
        
        # Separate text and image chunks
        text_chunks = [chunk for chunk in enriched_content if chunk.get('chunk_type', '') == 'text']
        image_chunks = [chunk for chunk in enriched_content if chunk.get('chunk_type', '') == 'image']
        
        # Log the actual counts to verify
        logger.info(f"Processing {len(text_chunks)} text chunks and {len(image_chunks)} image chunks")
        logger.info(f"Text chunk IDs: {[chunk.get('chunk_id') for chunk in text_chunks]}")
        
        # Generate embeddings
        text_embeddings = await self.generate_text_embeddings(text_chunks)
        image_embeddings = await self.generate_image_embeddings(image_chunks)
        
        # Standardize keys in embeddings
        logger.info("Standardizing embedding keys for consistency")
        text_embeddings = self._standardize_embedding_keys(text_embeddings, 'text')
        image_embeddings = self._standardize_embedding_keys(image_embeddings, 'image')
        
        # Combine embeddings for the full dataset
        all_embeddings = text_embeddings + image_embeddings
        all_embeddings = self._standardize_embedding_keys(all_embeddings)
        
        # Sort by chunk_id
        all_embeddings.sort(key=lambda x: float(x.get('chunk_id', 0)))
        text_embeddings.sort(key=lambda x: float(x.get('chunk_id', 0)))
        image_embeddings.sort(key=lambda x: float(x.get('chunk_id', 0)))
        
        # Return the embeddings without saving them
        logger.info(f"Generated embeddings for {len(all_embeddings)} content items")
        logger.info(f"All embeddings keys: {all_embeddings[0].keys()}")
        # Return just the embeddings and output_path
        return {
            "all_embeddings": all_embeddings,
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
            "output_path": output_path
        }

    async def generate_embedding_for_text(self, text: str) -> Dict[str, Any]:
        """
        Generate an embedding for a single text string.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            Dictionary containing the embedding vector
        """
        if self.model is None:
            await self.load_model()
            
        logger.info(f"Generating embedding for text of length {len(text)}")
        
        try:
            # Generate the embedding
            embedding = np.array(self.model.get_query_embedding(text))
            
            # Return as a dictionary
            result = {
                "text": text[:100] + "..." if len(text) > 100 else text,  # Truncate for logging
                "embedding": embedding.tolist()
            }
            
            logger.info(f"Successfully generated embedding with {embedding.shape[0]} dimensions")
            return result
            
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}") 