#!/usr/bin/env python3
"""
vLLM Client
===========

Client functions to generate embeddings by calling the deployed vLLM server.
These functions run locally and make HTTP requests to the remote server.
"""

import os
import json
import requests
import logging
import asyncio
import aiohttp
import time
import numpy as np
from typing import List, Dict, Any, Union, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vllm_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get server URL from environment variables
VLLM_SERVER = os.environ.get("MODAL_VLLM_SERVER", "https://yourusername--vllm-server-vllm-server.modal.run")

# ======================================================================
# EMBEDDING GENERATION FUNCTIONS
# ======================================================================
def generate_embedding(text: str) -> Tuple[List[float], Dict[str, Any]]:
    """
    Generate embedding for a single text using the deployed vLLM server.
    
    Args:
        text (str): The text to generate embedding for
        
    Returns:
        tuple: (embedding_vector, metadata)
            - embedding_vector (list): The embedding vector
            - metadata (dict): Additional information about the embedding
    """
    if not text:
        logger.warning("Empty text provided for embedding generation")
        return [], {"error": "Empty text provided"}
    
    logger.info(f"Generating embedding for text: {text[:50]}...")
    
    # Make request to vLLM server
    try:
        start_time = time.time()
        response = requests.get(VLLM_SERVER, params={"text": text}, timeout=120)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            metadata = {
                "model": result.get("model", "unknown"),
                "dimension": result.get("dimension", 0),
                "elapsed_seconds": elapsed,
                "status": "success"
            }
            
            if "embeddings" in result and len(result["embeddings"]) > 0:
                embedding = result["embeddings"][0]
                logger.info(f"Successfully generated embedding with dimension {len(embedding)} in {elapsed:.2f} seconds")
                return embedding, metadata
            else:
                logger.warning(f"No embeddings found in response: {result}")
                if "error" in result:
                    logger.error(f"Error from server: {result['error']}")
                    metadata["error"] = result["error"]
                    if "traceback" in result:
                        logger.error(f"Traceback: {result['traceback']}")
                        metadata["traceback"] = result["traceback"]
                return [], metadata
        else:
            logger.error(f"Error generating embedding (status {response.status_code}): {response.text}")
            return [], {"error": f"HTTP error {response.status_code}", "response": response.text, "status": "error"}
    except Exception as e:
        logger.error(f"Exception during embedding generation: {e}")
        return [], {"error": str(e), "status": "error"}

def generate_embeddings_batch(texts: List[str], batch_size: int = 10, delay_between_batches: float = 0.1) -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    Generate embeddings for a list of texts using the deployed vLLM server.
    
    Args:
        texts (list): List of texts to generate embeddings for
        batch_size (int): Number of texts to process in each batch
        delay_between_batches (float): Seconds to wait between sending batches
        
    Returns:
        tuple: (embeddings_list, metadata)
            - embeddings_list (list): List of embedding vectors
            - metadata (dict): Additional information about the embeddings
    """
    logger.info(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}, delay {delay_between_batches}s...")
    
    if not texts:
        logger.warning("No texts provided for embedding generation")
        return [], {"error": "No texts provided", "status": "error"}
    
    all_embeddings = []
    metadata = {
        "total_texts": len(texts),
        "batch_size": batch_size,
        "success_count": 0,
        "error_count": 0,
        "batches": []
    }
    
    total_start_time = time.time()
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_idx = i // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info(f"Processing batch {batch_idx}/{total_batches}")
        
        batch_metadata = {
            "batch_index": batch_idx,
            "batch_size": len(batch),
            "start_index": i,
            "end_index": min(i + batch_size, len(texts))
        }
        
        try:
            # Make request to vLLM server
            batch_start_time = time.time()
            response = requests.get(VLLM_SERVER, params={"texts": json.dumps(batch)}, timeout=120)
            batch_elapsed = time.time() - batch_start_time
            
            batch_metadata["elapsed_seconds"] = batch_elapsed
            
            if response.status_code == 200:
                result = response.json()
                batch_metadata["model"] = result.get("model", "unknown")
                batch_metadata["dimension"] = result.get("dimension", 0)
                
                if "embeddings" in result:
                    batch_embeddings = result["embeddings"]
                    all_embeddings.extend(batch_embeddings)
                    batch_metadata["success_count"] = len(batch_embeddings)
                    batch_metadata["error_count"] = len(batch) - len(batch_embeddings)
                    batch_metadata["status"] = "success"
                    metadata["success_count"] += len(batch_embeddings)
                    
                    logger.info(f"Successfully processed batch with {len(batch_embeddings)} embeddings in {batch_elapsed:.2f} seconds")
                else:
                    logger.warning(f"No embeddings found in response: {result}")
                    batch_metadata["status"] = "error"
                    batch_metadata["error"] = "No embeddings in response"
                    if "error" in result:
                        logger.error(f"Error from server: {result['error']}")
                        batch_metadata["error"] = result["error"]
                        if "traceback" in result:
                            logger.error(f"Traceback: {result['traceback']}")
                            batch_metadata["traceback"] = result["traceback"]
                    
                    # Add empty embeddings for this batch
                    batch_metadata["error_count"] = len(batch)
                    metadata["error_count"] += len(batch)
                    all_embeddings.extend([[] for _ in range(len(batch))])
            else:
                logger.error(f"Error generating embeddings (status {response.status_code}): {response.text}")
                
                batch_metadata["status"] = "error"
                batch_metadata["error"] = f"HTTP error {response.status_code}"
                batch_metadata["response"] = response.text
                batch_metadata["error_count"] = len(batch)
                metadata["error_count"] += len(batch)
                
                # Add empty embeddings for this batch
                all_embeddings.extend([[] for _ in range(len(batch))])
        except Exception as e:
            logger.error(f"Exception during batch embedding generation: {e}")
            
            batch_metadata["status"] = "error"
            batch_metadata["error"] = str(e)
            batch_metadata["error_count"] = len(batch)
            metadata["error_count"] += len(batch)
            
            # Add empty embeddings for this batch
            all_embeddings.extend([[] for _ in range(len(batch))])
        
        metadata["batches"].append(batch_metadata)
        
        # --- Add delay before next batch ---
        if i + batch_size < len(texts) and delay_between_batches > 0:
            logger.debug(f"Waiting {delay_between_batches} seconds before next batch...")
            time.sleep(delay_between_batches)
        # --- End added delay ---
    
    total_elapsed = time.time() - total_start_time
    metadata["total_elapsed_seconds"] = total_elapsed
    
    logger.info(f"Generated {len(all_embeddings)} embeddings in {total_elapsed:.2f} seconds")
    logger.info(f"Success rate: {metadata['success_count']}/{metadata['total_texts']} texts ({metadata['success_count']/metadata['total_texts']*100:.1f}%)")
    
    # Add overall status
    if metadata["error_count"] == 0:
        metadata["status"] = "success"
    elif metadata["success_count"] == 0:
        metadata["status"] = "error"
    else:
        metadata["status"] = "partial_success"
    
    return all_embeddings, metadata

# ======================================================================
# ASYNC EMBEDDING GENERATION FUNCTIONS
# ======================================================================
async def generate_embedding_async(text: str) -> Tuple[List[float], Dict[str, Any]]:
    """
    Generate embedding for a single text using the deployed vLLM server (async version).
    
    Args:
        text (str): The text to generate embedding for
        
    Returns:
        tuple: (embedding_vector, metadata)
            - embedding_vector (list): The embedding vector
            - metadata (dict): Additional information about the embedding
    """
    if not text:
        logger.warning("Empty text provided for embedding generation")
        return [], {"error": "Empty text provided"}
    
    logger.info(f"Generating embedding for text: {text[:50]}...")
    
    # Make request to vLLM server
    try:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.get(VLLM_SERVER, params={"text": text}, timeout=120) as response:
                elapsed = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    metadata = {
                        "model": result.get("model", "unknown"),
                        "dimension": result.get("dimension", 0),
                        "elapsed_seconds": elapsed,
                        "status": "success"
                    }
                    
                    if "embeddings" in result and len(result["embeddings"]) > 0:
                        embedding = result["embeddings"][0]
                        logger.info(f"Successfully generated embedding with dimension {len(embedding)} in {elapsed:.2f} seconds")
                        return embedding, metadata
                    else:
                        logger.warning(f"No embeddings found in response: {result}")
                        if "error" in result:
                            logger.error(f"Error from server: {result['error']}")
                            metadata["error"] = result["error"]
                            if "traceback" in result:
                                logger.error(f"Traceback: {result['traceback']}")
                                metadata["traceback"] = result["traceback"]
                        return [], metadata
                else:
                    error_text = await response.text()
                    logger.error(f"Error generating embedding (status {response.status}): {error_text}")
                    return [], {"error": f"HTTP error {response.status}", "response": error_text, "status": "error"}
    except Exception as e:
        logger.error(f"Exception during embedding generation: {e}")
        return [], {"error": str(e), "status": "error"}

async def generate_embeddings_batch_async(texts: List[str], batch_size: int = 10, delay_between_batches: float = 0.0) -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    Generate embeddings for a list of texts using the deployed vLLM server (async version).
    
    Args:
        texts (list): List of texts to generate embeddings for
        batch_size (int): Number of texts to process in each batch
        delay_between_batches (float): Seconds to wait between queuing batches (approximate)
        
    Returns:
        tuple: (embeddings_list, metadata)
            - embeddings_list (list): List of embedding vectors
            - metadata (dict): Additional information about the embeddings
    """
    logger.info(f"Generating embeddings for {len(texts)} texts async...")
    
    if not texts:
        logger.warning("No texts provided for embedding generation")
        return [], {"error": "No texts provided", "status": "error"}
    
    # Use asyncio to send requests in parallel
    async def fetch_embeddings_parallel():
        all_embeddings = []
        metadata = {
            "total_texts": len(texts),
            "batch_size": batch_size,
            "success_count": 0,
            "error_count": 0,
            "batches": []
        }
        
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests to avoid overwhelming the server
        total_start_time = time.time()
        
        async def process_batch(batch_idx, start_idx, batch):
            batch_metadata = {
                "batch_index": batch_idx,
                "batch_size": len(batch),
                "start_index": start_idx,
                "end_index": start_idx + len(batch)
            }
            
            async with semaphore:
                # --- Optional delay before processing batch ---
                # Note: This delays *before* processing starts, affecting concurrency.
                # A delay *after* might be more typical for rate limiting,
                # but would require restructuring the task creation loop.
                if delay_between_batches > 0:
                     logger.debug(f"Delaying batch {batch_idx} start by {delay_between_batches}s...")
                     await asyncio.sleep(delay_between_batches)
                # --- End optional delay ---
                try:
                    batch_start_time = time.time()
                    async with aiohttp.ClientSession() as session:
                        async with session.get(VLLM_SERVER, params={"texts": json.dumps(batch)}, timeout=120) as response:
                            batch_elapsed = time.time() - batch_start_time
                            batch_metadata["elapsed_seconds"] = batch_elapsed
                            
                            if response.status == 200:
                                result = await response.json()
                                batch_metadata["model"] = result.get("model", "unknown")
                                batch_metadata["dimension"] = result.get("dimension", 0)
                                
                                if "embeddings" in result:
                                    batch_embeddings = result["embeddings"]
                                    batch_metadata["success_count"] = len(batch_embeddings)
                                    batch_metadata["error_count"] = len(batch) - len(batch_embeddings)
                                    batch_metadata["status"] = "success"
                                    
                                    logger.info(f"Successfully processed batch {batch_idx} with {len(batch_embeddings)} embeddings in {batch_elapsed:.2f} seconds")
                                    return batch_embeddings, batch_metadata
                                else:
                                    logger.warning(f"No embeddings found in response for batch {batch_idx}: {result}")
                                    batch_metadata["status"] = "error"
                                    batch_metadata["error"] = "No embeddings in response"
                                    if "error" in result:
                                        logger.error(f"Error from server: {result['error']}")
                                        batch_metadata["error"] = result["error"]
                                        if "traceback" in result:
                                            logger.error(f"Traceback: {result['traceback']}")
                                            batch_metadata["traceback"] = result["traceback"]
                                    
                                    # Return empty embeddings for this batch
                                    batch_metadata["error_count"] = len(batch)
                                    return [[] for _ in range(len(batch))], batch_metadata
                            else:
                                error_text = await response.text()
                                logger.error(f"Error generating embeddings for batch {batch_idx} (status {response.status}): {error_text}")
                                
                                batch_metadata["status"] = "error"
                                batch_metadata["error"] = f"HTTP error {response.status}"
                                batch_metadata["response"] = error_text
                                batch_metadata["error_count"] = len(batch)
                                
                                # Return empty embeddings for this batch
                                return [[] for _ in range(len(batch))], batch_metadata
                except Exception as e:
                    logger.error(f"Exception during embedding generation for batch {batch_idx}: {e}")
                    
                    batch_metadata["status"] = "error"
                    batch_metadata["error"] = str(e)
                    batch_metadata["error_count"] = len(batch)
                    
                    # Return empty embeddings for this batch
                    return [[] for _ in range(len(batch))], batch_metadata
        
        # Process in batches
        tasks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_idx = i // batch_size + 1
            logger.info(f"Queuing batch {batch_idx}/{(len(texts) + batch_size - 1)//batch_size}")
            tasks.append(process_batch(batch_idx, i, batch))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Process results
        all_batch_results = []
        for batch_embeddings, batch_metadata in results:
            all_batch_results.append((batch_metadata["start_index"], batch_embeddings, batch_metadata))
            metadata["batches"].append(batch_metadata)
            metadata["success_count"] += batch_metadata["success_count"]
            metadata["error_count"] += batch_metadata["error_count"]
        
        # Sort batches by start index to maintain original order
        all_batch_results.sort(key=lambda x: x[0])
        
        # Flatten embeddings
        for _, batch_embeddings, _ in all_batch_results:
            all_embeddings.extend(batch_embeddings)
        
        total_elapsed = time.time() - total_start_time
        metadata["total_elapsed_seconds"] = total_elapsed
        
        logger.info(f"Generated {len(all_embeddings)} embeddings in {total_elapsed:.2f} seconds")
        logger.info(f"Success rate: {metadata['success_count']}/{metadata['total_texts']} texts ({metadata['success_count']/metadata['total_texts']*100:.1f}%)")
        
        # Add overall status
        if metadata["error_count"] == 0:
            metadata["status"] = "success"
        elif metadata["success_count"] == 0:
            metadata["status"] = "error"
        else:
            metadata["status"] = "partial_success"
        
        return all_embeddings, metadata
    
    return await fetch_embeddings_parallel()

# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================
def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1 (list): First embedding
        embedding2 (list): Second embedding
        
    Returns:
        float: Cosine similarity (between -1 and 1)
    """
    if not embedding1 or not embedding2:
        return 0.0
    
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)

def calculate_similarities(query_embedding: List[float], candidate_embeddings: List[List[float]]) -> List[float]:
    """
    Calculate similarities between a query embedding and a list of candidate embeddings.
    
    Args:
        query_embedding (list): Query embedding
        candidate_embeddings (list): List of candidate embeddings
        
    Returns:
        list: List of similarity scores
    """
    if not query_embedding or not candidate_embeddings:
        return []
    
    vec1 = np.array(query_embedding)
    norm1 = np.linalg.norm(vec1)
    
    if norm1 == 0:
        return [0.0] * len(candidate_embeddings)
    
    similarities = []
    for embedding in candidate_embeddings:
        if not embedding:
            similarities.append(0.0)
            continue
            
        vec2 = np.array(embedding)
        norm2 = np.linalg.norm(vec2)
        
        if norm2 == 0:
            similarities.append(0.0)
        else:
            similarities.append(np.dot(vec1, vec2) / (norm1 * norm2))
    
    return similarities

# ======================================================================
# MAIN FUNCTION (FOR TESTING)
# ======================================================================
def main():
    """Test the embedding functions."""
    # Test single embedding
    text = "This is a test sentence for embedding generation."
    embedding, metadata = generate_embedding(text)
    
    if embedding:
        print(f"Generated embedding with dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        print(f"Model: {metadata.get('model', 'unknown')}")
        print(f"Time taken: {metadata.get('elapsed_seconds', 0):.2f} seconds")
    else:
        print(f"Failed to generate embedding: {metadata.get('error', 'Unknown error')}")
    
    # Test batch embeddings
    texts = [
        "This is the first test sentence.",
        "This is the second test sentence.",
        "This is the third test sentence."
    ]
    embeddings, metadata = generate_embeddings_batch(texts)
    
    print(f"\nGenerated {len(embeddings)} embeddings")
    if embeddings and len(embeddings) > 0 and len(embeddings[0]) > 0:
        print(f"First embedding dimension: {len(embeddings[0])}")
        print(f"First 5 values of first embedding: {embeddings[0][:5]}")
        print(f"Success rate: {metadata['success_count']}/{metadata['total_texts']}")
        print(f"Total time: {metadata['total_elapsed_seconds']:.2f} seconds")
        
        # Calculate similarity between first two embeddings
        if len(embeddings) >= 2 and all(len(e) > 0 for e in embeddings[:2]):
            similarity = calculate_similarity(embeddings[0], embeddings[1])
            print(f"Similarity between first two embeddings: {similarity:.5f}")
    else:
        print(f"Failed to generate batch embeddings: {metadata.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 