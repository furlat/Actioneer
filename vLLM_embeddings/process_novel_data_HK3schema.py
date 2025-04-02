#!/usr/bin/env python3
"""
Novel Data Processing Pipeline with New Schema (Parquet Version)
=============================================

This script processes novel data to extract chunks and actions, generate embeddings,
and store the data in a structured format according to a new schema.

The schema includes:
1. Chunk-level embeddings with full JSON objects
2. Action-level embeddings with multiple types of embeddings per action

This version exclusively uses Parquet files for both input and output.

== Updates ==
- Now uses the vllm_client.py library for embedding generation instead of making direct HTTP requests
- This allows the script to run locally while using the centralized vLLM server for embeddings
- The script handles both the metadata and embeddings returned by the client
- Supports the BGE model hosted on the vLLM server
- Simplified to process Parquet files only
- Uses real chunks from Parquet files
- Uses asynchronous embedding generation for maximum throughput
"""

# ======================================================================
# GLOBAL CONFIGURATION
# ======================================================================
# Embedding generation parameters
EMBEDDING_BATCH_SIZE = 20           # Number of texts to process in each batch
EMBEDDING_DELAY = 0.0               # No delay needed for async processing
EMBEDDING_CONCURRENCY = 5           # Number of concurrent requests (semaphore limit)

# Processing limits
DEFAULT_LIMIT = 1000                 # Default limit for number of rows to process

# ======================================================================
# IMPORTS
# ======================================================================
import json
import os
import polars as pl
import numpy as np
import logging
import asyncio
from typing import Dict, List, Any, Optional

# Import dotenv for environment variables
from dotenv import load_dotenv

# Import the vLLM client functions
from vllm_client import (
    generate_embedding, 
    generate_embeddings_batch,
    generate_embeddings_batch_async
)

# Load environment variables from .env file
load_dotenv()

# Check and log which vLLM server endpoint will be used
vllm_server = os.environ.get("MODAL_VLLM_SERVER", "https://yourusername--vllm-server-vllm-server.modal.run")
print(f"Using vLLM server endpoint: {vllm_server}")

# ======================================================================
# LOGGING SETUP
# ======================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing_parquet.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======================================================================
# DATA LOADING FUNCTIONS
# ======================================================================
def load_actions_data(file_path: str, limit: int = None) -> Dict[str, Any]:
    """
    Load the actions data from Parquet file.
    
    Args:
        file_path (str): Path to the Parquet file containing actions data
        limit (int, optional): Limit the number of rows to load
        
    Returns:
        dict: The loaded actions data
    """
    logger.info(f"Loading actions data from {file_path}{' with limit ' + str(limit) if limit else ''}...")
    try:
        # Load from Parquet
        if limit:
            df = pl.scan_parquet(file_path).head(limit).collect()
        else:
            df = pl.read_parquet(file_path)
        
        # Convert DataFrame to a dictionary structure
        data = {
            "book_id": df.get_column("book_id")[0] if "book_id" in df.columns else os.path.basename(file_path).split('.')[0],
            "title": df.get_column("title")[0] if "title" in df.columns else "Unknown",
            "all_actions": df.to_dicts()
        }
        logger.info(f"Actions data loaded successfully. Number of actions: {len(data.get('all_actions', []))}")
        
        return data
    except Exception as e:
        logger.error(f"Error loading actions data: {e}")
        raise

def load_chunks_data(file_path: str, limit: int = None) -> List[Dict[str, Any]]:
    """
    Load the chunks data from Parquet file.
    
    Args:
        file_path (str): Path to the Parquet file containing chunks data
        limit (int, optional): Limit the number of rows to load
        
    Returns:
        list: The loaded chunks data
    """
    logger.info(f"Loading chunks data from {file_path}{' with limit ' + str(limit) if limit else ''}...")
    try:
        # Load from Parquet
        if limit:
            df = pl.scan_parquet(file_path).head(limit).collect()
        else:
            df = pl.read_parquet(file_path)
        
        # Convert DataFrame to a list of dictionaries
        chunks = df.to_dicts()
        logger.info(f"Chunks data loaded successfully. Number of chunks: {len(chunks)}")
        
        return chunks
    except Exception as e:
        logger.error(f"Error loading chunks data: {e}")
        # If chunks file doesn't exist or has issues, return an empty list
        logger.warning("Returning empty chunks list")
        return []

# ======================================================================
# DATA PREPARATION FUNCTIONS
# ======================================================================
def prepare_chunk_level_data(chunks: List[Dict[str, Any]], book_id: str, source_filename: str) -> List[Dict[str, Any]]:
    """
    Prepare chunk-level data for embedding.
    
    Args:
        chunks (list): List of chunks
        book_id (str): ID of the book
        source_filename (str): Name of the source file
        
    Returns:
        list: List of chunk data records
    """
    logger.info("Preparing chunk-level data for embedding...")
    
    if not chunks:
        logger.warning("No chunks provided to prepare")
        return []
    
    chunk_data = []
    
    for chunk in chunks:
        # Extract the text from the chunk using the correct field name
        # The schema inspection showed the text content is in the "chunk" field
        chunk_text = chunk.get("chunk", "")
        
        # Create a record for each chunk with its text and full chunk as JSON
        record = {
            "book_id": book_id,
            "source_filename": source_filename,
            "chunk_id": chunk.get("chunk_id"),
            "chunk_text": chunk_text,
            "full_json": json.dumps(chunk)  # Store the complete chunk object
        }
        
        chunk_data.append(record)
    
    logger.info(f"Prepared {len(chunk_data)} chunk-level records")
    return chunk_data

def prepare_action_level_data(actions: List[Dict[str, Any]], book_id: str, source_filename: str) -> List[Dict[str, Any]]:
    """
    Prepare action-level data for embedding.
    
    Args:
        actions (list): List of actions
        book_id (str): ID of the book
        source_filename (str): Name of the source file
        
    Returns:
        list: List of action data records
    """
    logger.info("Preparing action-level data for embedding...")
    
    action_data = []
    
    for i, action in enumerate(actions):
        # Extract source, action, target components
        source = action.get("source", "")
        action_verb = action.get("action", "")
        target = action.get("target", "")
        
        # Extract quotes
        action_quote = action.get("text_describing_the_action", "")
        consequence_quote = action.get("text_describing_the_consequence", "")
        
        # Create action tuple text
        action_tuple = f"{source} {action_verb} {target}".strip()
        
        # Create action tuple with quote
        action_tuple_with_quote = f"{action_tuple}: {action_quote}".strip()
        
        # Create record
        record = {
            "book_id": book_id,
            "source_filename": source_filename,
            "chunk_id": action.get("chunk_id"),
            "temporal_order": action.get("temporal_order_id", i),  # Use index as fallback
            "source": source,
            "action": action_verb,
            "target": target,
            "action_quote": action_quote,
            "consequence_quote": consequence_quote,
            "action_tuple": action_tuple,
            "action_tuple_with_quote": action_tuple_with_quote
        }
        
        action_data.append(record)
    
    logger.info(f"Prepared {len(action_data)} action-level records")
    return action_data

# ======================================================================
# DATAFRAME CREATION FUNCTIONS
# ======================================================================
async def create_chunk_level_frame(chunk_data: List[Dict[str, Any]]) -> Optional[pl.DataFrame]:
    """
    Create the chunk-level frame with embeddings.
    
    Args:
        chunk_data (list): List of chunk data records
        
    Returns:
        pl.DataFrame: Polars DataFrame with chunk-level data and embeddings
    """
    logger.info("Creating chunk-level frame...")
    
    if not chunk_data:
        logger.warning("No chunk data to process")
        return None
    
    # Extract chunk texts for embedding
    chunk_texts = [record["chunk_text"] for record in chunk_data]
    
    # Generate embeddings for chunks using our client library (async version)
    logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
    chunk_embeddings, metadata = await generate_embeddings_batch_async(
        chunk_texts, 
        batch_size=EMBEDDING_BATCH_SIZE,
        delay_between_batches=EMBEDDING_DELAY
    )
    
    # Log metadata about the embedding generation
    logger.info(f"Embedding generation status: {metadata.get('status', 'unknown')}")
    logger.info(f"Success rate: {metadata.get('success_count', 0)}/{metadata.get('total_texts', 0)}")
    logger.info(f"Model used: {metadata.get('model', 'unknown')}")
    logger.info(f"Dimension: {metadata.get('dimension', 0)}")
    
    # Create DataFrame
    try:
        # Create a new list with the embeddings as numpy arrays
        processed_records = []
        for i, record in enumerate(chunk_data):
            processed_record = {k: v for k, v in record.items()}
            
            # Store embedding as numpy array
            if i < len(chunk_embeddings) and chunk_embeddings[i]:
                processed_record["chunk_embedding"] = np.array(chunk_embeddings[i], dtype=np.float32)
            else:
                processed_record["chunk_embedding"] = None
                
            processed_records.append(processed_record)
        
        # Convert to Polars DataFrame
        chunk_df = pl.DataFrame(processed_records)
        logger.info("Chunk-level frame created successfully")
        return chunk_df
    
    except Exception as e:
        logger.error(f"Error creating chunk-level frame: {e}")
        return None

async def create_action_level_frame(action_data: List[Dict[str, Any]]) -> Optional[pl.DataFrame]:
    """
    Create the action-level frame with embeddings.
    
    Args:
        action_data (list): List of action data records
        
    Returns:
        pl.DataFrame: Polars DataFrame with action-level data and embeddings
    """
    logger.info("Creating action-level frame...")
    
    # Extract texts for embedding
    action_quotes = [record["action_quote"] for record in action_data]
    consequence_quotes = [record["consequence_quote"] for record in action_data]
    action_tuples = [record["action_tuple"] for record in action_data]
    action_tuples_with_quotes = [record["action_tuple_with_quote"] for record in action_data]
    
    # Generate embeddings using our client library (async version)
    logger.info(f"Generating embeddings for {len(action_quotes)} action quotes...")
    action_quote_embeddings, action_quote_metadata = await generate_embeddings_batch_async(
        action_quotes,
        batch_size=EMBEDDING_BATCH_SIZE,
        delay_between_batches=EMBEDDING_DELAY
    )
    logger.info(f"Action quote embedding generation status: {action_quote_metadata.get('status', 'unknown')}")
    
    logger.info(f"Generating embeddings for {len(consequence_quotes)} consequence quotes...")
    consequence_quote_embeddings, consequence_quote_metadata = await generate_embeddings_batch_async(
        consequence_quotes,
        batch_size=EMBEDDING_BATCH_SIZE,
        delay_between_batches=EMBEDDING_DELAY
    )
    logger.info(f"Consequence quote embedding generation status: {consequence_quote_metadata.get('status', 'unknown')}")
    
    logger.info(f"Generating embeddings for {len(action_tuples)} action tuples...")
    action_tuple_embeddings, action_tuple_metadata = await generate_embeddings_batch_async(
        action_tuples,
        batch_size=EMBEDDING_BATCH_SIZE,
        delay_between_batches=EMBEDDING_DELAY
    )
    logger.info(f"Action tuple embedding generation status: {action_tuple_metadata.get('status', 'unknown')}")
    
    logger.info(f"Generating embeddings for {len(action_tuples_with_quotes)} action tuples with quotes...")
    action_tuple_with_quote_embeddings, action_tuple_with_quote_metadata = await generate_embeddings_batch_async(
        action_tuples_with_quotes,
        batch_size=EMBEDDING_BATCH_SIZE,
        delay_between_batches=EMBEDDING_DELAY
    )
    logger.info(f"Action tuple with quote embedding generation status: {action_tuple_with_quote_metadata.get('status', 'unknown')}")
    
    # Create DataFrame
    try:
        # Create a new list without the embedding fields
        processed_records = []
        for i, record in enumerate(action_data):
            processed_record = {k: v for k, v in record.items()}
            
            # Store embeddings as numpy arrays to fix potential JSON serialization issues
            if i < len(action_quote_embeddings) and action_quote_embeddings[i]:
                processed_record["action_quote_embedding"] = np.array(action_quote_embeddings[i], dtype=np.float32)
            else:
                processed_record["action_quote_embedding"] = None
                
            if i < len(consequence_quote_embeddings) and consequence_quote_embeddings[i]:
                processed_record["consequence_quote_embedding"] = np.array(consequence_quote_embeddings[i], dtype=np.float32)
            else:
                processed_record["consequence_quote_embedding"] = None
                
            if i < len(action_tuple_embeddings) and action_tuple_embeddings[i]:
                processed_record["action_tuple_embedding"] = np.array(action_tuple_embeddings[i], dtype=np.float32)
            else:
                processed_record["action_tuple_embedding"] = None
                
            if i < len(action_tuple_with_quote_embeddings) and action_tuple_with_quote_embeddings[i]:
                processed_record["action_tuple_with_quote_embedding"] = np.array(action_tuple_with_quote_embeddings[i], dtype=np.float32)
            else:
                processed_record["action_tuple_with_quote_embedding"] = None
                
            processed_records.append(processed_record)
        
        # Convert to Polars DataFrame
        action_df = pl.DataFrame(processed_records)
        logger.info("Action-level frame created successfully")
        return action_df
    
    except Exception as e:
        logger.error(f"Error creating action-level frame: {e}")
        return None

# ======================================================================
# MAIN FUNCTION
# ======================================================================
async def async_main():
    """
    Asynchronous main function that runs the entire processing pipeline.
    """
    try:
        logger.info("Starting processing pipeline with new schema using centralized vLLM server (Parquet version)...")
        
        # Create output directories
        output_dir = "vllm_embeddings_parquet"
        os.makedirs(output_dir, exist_ok=True)
        
        # Use the Gutenberg Parquet files with limit
        actions_file = "data/gutenberg_en_novels_all_processed_actions_with_matches_and_metadata.parquet"
        chunks_file = "data/gutenberg_en_novels_all_processed_chunks_with_metadata.parquet"
        limit = DEFAULT_LIMIT  # Use global configuration
        
        # Load the actions data with limit
        actions_data = load_actions_data(actions_file, limit=limit)
        
        # Load the chunks data with same limit
        chunks = load_chunks_data(chunks_file, limit=limit)
        
        # Get book ID and source filename
        book_id = actions_data.get("book_id", os.path.basename(actions_file).split('.')[0])
        source_filename = os.path.basename(actions_file)
        
        # Extract actions
        actions = actions_data["all_actions"]
        
        # Prepare data according to schema
        chunk_data = prepare_chunk_level_data(chunks, book_id, source_filename)
        action_data = prepare_action_level_data(actions, book_id, source_filename)
        
        # Create frames with embeddings (async)
        chunk_frame = await create_chunk_level_frame(chunk_data)
        action_frame = await create_action_level_frame(action_data)
        
        # Save frames as Parquet
        if chunk_frame is not None:
            # Convert object columns to serializable format
            if "chunk_embedding" in chunk_frame.columns:
                chunk_frame = chunk_frame.with_columns([
                    pl.col("chunk_embedding").map_elements(
                        lambda x: x.tolist() if isinstance(x, np.ndarray) else None, 
                        return_dtype=pl.List(pl.Float32)
                    ).alias("chunk_embedding")
                ])
                
            # Convert full_json to string if needed
            if "full_json" in chunk_frame.columns:
                chunk_frame = chunk_frame.with_columns([
                    pl.col("full_json").cast(pl.Utf8).alias("full_json")
                ])
                
            # Save as Parquet
            output_file = f'{output_dir}/chunk_level_frame_gutenberg_sample.parquet'
            chunk_frame.write_parquet(output_file)
            logger.info(f"Chunk-level frame saved as Parquet to {output_file}")
        
        if action_frame is not None:
            # Convert all embedding columns to lists
            embedding_cols = [col for col in action_frame.columns if col.endswith("_embedding")]
            for col in embedding_cols:
                action_frame = action_frame.with_columns([
                    pl.col(col).map_elements(
                        lambda x: x.tolist() if isinstance(x, np.ndarray) else None,
                        return_dtype=pl.List(pl.Float32)
                    ).alias(col)
                ])
                
            # Save as Parquet
            output_file = f'{output_dir}/action_level_frame_gutenberg_sample.parquet'
            action_frame.write_parquet(output_file)
            logger.info(f"Action-level frame saved as Parquet to {output_file}")
        
        logger.info(f"Processing completed successfully! Results saved to {output_dir}/")
    
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        raise

# ======================================================================
# SCRIPT ENTRY POINT
# ======================================================================
def main():
    """
    Main function wrapper to run the async pipeline with asyncio.
    """
    asyncio.run(async_main())

if __name__ == "__main__":
    main() 