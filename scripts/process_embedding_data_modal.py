#!/usr/bin/env python3
"""
Novel Data Processing Pipeline with New Schema
=============================================

This script processes novel data to extract chunks and actions, generate embeddings,
and store the data in a structured format according to a new schema.

The schema includes:
1. Chunk-level embeddings with full JSON objects
2. Action-level embeddings with multiple types of embeddings per action
"""

# ======================================================================
# IMPORTS
# ======================================================================
import json
import os
import re
import requests
import polars as pl
import numpy as np
import logging
import asyncio
import aiohttp
### from collections import defaultdict

# Import nest_asyncio for handling nested event loops
import nest_asyncio

# ======================================================================
# LOGGING SETUP
# ======================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing_newschema.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======================================================================
# DATA LOADING FUNCTIONS
# ======================================================================
def load_novel_data(file_path):
    """
    Load the novel data from JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing novel data
        
    Returns:
        dict: The loaded novel data
    """
    logger.info(f"Loading data from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Data loaded successfully. Title: {data.get('title', 'Unknown')}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# ======================================================================
# DATA EXTRACTION FUNCTIONS
# ======================================================================
def extract_chunks(data):
    """
    Extract chunks from the novel data.
    
    Tries multiple approaches to accommodate different data formats.
    
    Args:
        data (dict): The novel data
        
    Returns:
        list: A list of chunks with text and analysis
    """
    logger.info("Extracting chunks from novel data...")
    
    # Try different approaches to extract chunks based on common data formats
    
    # Approach 1: Direct chunks list
    if "chunks" in data and isinstance(data["chunks"], list) and data["chunks"]:
        chunks = data["chunks"]
        logger.info(f"Found {len(chunks)} chunks in 'chunks' list")
        return chunks
    
    # Approach 2: Document structure with sections
    elif "document" in data and isinstance(data["document"], list) and data["document"]:
        chunks = []
        for i, section in enumerate(data["document"]):
            if isinstance(section, dict) and "text" in section:
                chunks.append({
                    "chunk_id": i,
                    "text": section["text"],
                    "analysis": section.get("analysis", {})
                })
        if chunks:
            logger.info(f"Constructed {len(chunks)} chunks from document structure")
            return chunks
    
    # Approach 3: Extract from text_analysis if available
    elif "text_analysis" in data and isinstance(data["text_analysis"], dict):
        chunks = []
        for chunk_id, chunk_data in data["text_analysis"].items():
            if isinstance(chunk_data, dict):
                chunk = {
                    "chunk_id": chunk_id,
                    "text": chunk_data.get("text", ""),
                    "analysis": {k: v for k, v in chunk_data.items() if k != "text"}
                }
                chunks.append(chunk)
        if chunks:
            logger.info(f"Constructed {len(chunks)} chunks from text_analysis")
            return chunks
    
    # Approach 4: Handle case where we have paragraphs or sections
    elif "paragraphs" in data and isinstance(data["paragraphs"], list) and data["paragraphs"]:
        chunks = []
        for i, para in enumerate(data["paragraphs"]):
            if isinstance(para, str):
                chunks.append({
                    "chunk_id": i,
                    "text": para,
                    "analysis": {}
                })
            elif isinstance(para, dict) and "text" in para:
                chunks.append({
                    "chunk_id": i,
                    "text": para["text"],
                    "analysis": {k: v for k, v in para.items() if k != "text"}
                })
        if chunks:
            logger.info(f"Constructed {len(chunks)} chunks from paragraphs")
            return chunks
    
    # Approach 5: Create a single chunk from the entire book if we have a 'text' field
    elif "text" in data and isinstance(data["text"], str):
        chunks = [{
            "chunk_id": 0,
            "text": data["text"],
            "analysis": {}
        }]
        logger.info("Created a single chunk from the book's full text")
        return chunks
    
    # Last resort: Create synthetic chunks from actions if available
    elif "all_actions" in data and isinstance(data["all_actions"], list) and data["all_actions"]:
        actions = data["all_actions"]
        # Group actions by chunk_id
        chunk_map = {}
        for action in actions:
            chunk_id = action.get("chunk_id", 0)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": "",
                    "analysis": {"actions": []}
                }
            chunk_map[chunk_id]["analysis"]["actions"].append(action)
            # If the action has text, add it to the chunk text
            if "text_describing_the_action" in action and action["text_describing_the_action"]:
                if chunk_map[chunk_id]["text"]:
                    chunk_map[chunk_id]["text"] += " " + action["text_describing_the_action"]
                else:
                    chunk_map[chunk_id]["text"] = action["text_describing_the_action"]
        
        chunks = list(chunk_map.values())
        logger.info(f"Constructed {len(chunks)} synthetic chunks from actions")
        return chunks
    
    # If we get here, we couldn't extract chunks
    logger.error("Could not extract chunks from the data structure")
    
    # As a final fallback, create an empty chunk to prevent processing errors
    logger.warning("Creating a minimal empty chunk as fallback")
    return [{
        "chunk_id": 0,
        "text": "Empty chunk created as fallback",
        "analysis": {}
    }]

def extract_actions(data, chunks=None):
    """
    Extract all actions from the novel data.
    
    Args:
        data (dict): The novel data
        chunks (list, optional): List of chunks to extract actions from
        
    Returns:
        list: A list of actions
    """
    logger.info("Extracting actions from novel data...")
    
    all_actions = []
    
    # Try direct access to all_actions list first
    if "all_actions" in data and isinstance(data["all_actions"], list):
        all_actions = data["all_actions"]
        logger.info(f"Found {len(all_actions)} actions in all_actions list")
    
    # If no actions found or we also have chunks, extract from chunks
    if (not all_actions or chunks) and chunks:
        logger.info("Extracting actions from chunks...")
        chunk_actions = []
        
        for chunk in chunks:
            if "analysis" in chunk and "actions" in chunk["analysis"]:
                actions = chunk["analysis"]["actions"]
                # Add chunk_id to each action if not already present
                for action in actions:
                    if "chunk_id" not in action:
                        action["chunk_id"] = chunk.get("chunk_id")
                chunk_actions.extend(actions)
        
        if chunk_actions:
            # If we already had all_actions, merge with chunk_actions
            if all_actions:
                # Create a set of existing action texts for deduplication
                existing_action_texts = {a.get("text_describing_the_action", "") for a in all_actions}
                # Add only new actions
                for action in chunk_actions:
                    if action.get("text_describing_the_action", "") not in existing_action_texts:
                        all_actions.append(action)
                logger.info(f"Added {len(all_actions) - len(existing_action_texts)} unique actions from chunks")
            else:
                all_actions = chunk_actions
                logger.info(f"Extracted {len(all_actions)} actions from chunks")
    
    if not all_actions:
        logger.error("No actions found in the data")
    
    return all_actions

# ======================================================================
# EMBEDDING GENERATION FUNCTIONS
# ======================================================================
def generate_embedding(text, model="Alibaba-NLP/gte-Qwen2-7B-instruct"):
    """
    Generate embedding for a single text using Modal embedding server.
    
    Args:
        text (str): The text to generate embedding for
        model (str): The model to use for generating embeddings
        
    Returns:
        list: The embedding vector
    """
    if not text:
        logger.warning("Empty text provided for embedding generation")
        return []
    
    # Modal server endpoint - you need to change this to the new endpoint
    url = "https://YOUR_MODAL_APP_NAME--infinity-serve-serve.modal.run/embeddings"
    
    # Request payload
    payload = {
        "model": model,
        "input": [text]
    }
    
    # Define async function to fetch embedding
    async def fetch_embedding_async():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=60) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['data'][0]['embedding']
                    else:
                        logger.error(f"Error generating embedding (status {response.status}): {await response.text()}")
                        return []
        except Exception as e:
            logger.error(f"Exception during embedding generation: {e}")
            return []
    
    # Run the async function using asyncio.run 
    try:
        if asyncio._get_running_loop() is not None:
            # If we're already in an event loop (like in Jupyter), use nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(fetch_embedding_async())
        else:
            # Otherwise use asyncio.run which handles the loop creation/cleanup properly
            return asyncio.run(fetch_embedding_async())
    except Exception as e:
        logger.error(f"Error in asyncio handling: {e}")
        # Fallback to old approach if needed, works but gives error on start as there is no loop before we start, no issues though
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(fetch_embedding_async())

def generate_embeddings_batch(texts, model="Alibaba-NLP/gte-Qwen2-7B-instruct", batch_size=32):
    """
    Generate embeddings for a list of texts using the Modal embedding server with parallel requests.
    
    Args:
        texts (list): List of texts to generate embeddings for
        model (str): The model to use for generating embeddings
        batch_size (int): Number of texts to process in each batch
        
    Returns:
        list: List of embedding vectors
    """
    logger.info(f"Generating embeddings for {len(texts)} texts using model {model}...")
    
    if not texts:
        logger.warning("No texts provided for embedding generation")
        return []
    
    # Modal server endpoint - updated to new endpoint
    url = "https://YOUR_MODAL_APP_NAME--infinity-serve-serve.modal.run/embeddings"
    
    # Use asyncio to send requests in parallel
    async def fetch_embeddings_parallel():
        all_embeddings = []
        semaphore = asyncio.Semaphore(100)  # Limit to 100 concurrent requests
        
        async def process_batch(batch):
            async with semaphore:
                payload = {
                    "model": model,
                    "input": batch
                }
                
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, json=payload, timeout=60) as response:
                            if response.status == 200:
                                result = await response.json()
                                batch_embeddings = [item['embedding'] for item in result['data']]
                                logger.info(f"Batch processed successfully")
                                return batch_embeddings
                            else:
                                logger.error(f"Error generating embeddings (status {response.status}): {await response.text()}")
                                # Return empty embeddings for this batch
                                return [[] for _ in range(len(batch))]
                except Exception as e:
                    logger.error(f"Exception during embedding generation: {e}")
                    # Return empty embeddings for this batch
                    return [[] for _ in range(len(batch))]
        
        # Process in batches
        tasks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            logger.info(f"Queuing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            tasks.append(process_batch(batch))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        for batch_result in results:
            all_embeddings.extend(batch_result)
        
        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    # Run the async function using asyncio.run instead of manual loop handling
    try:
        if asyncio._get_running_loop() is not None:
            # If we're already in an event loop (like in Jupyter), use nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            all_embeddings = loop.run_until_complete(fetch_embeddings_parallel())
        else:
            # Otherwise use asyncio.run which handles the loop creation/cleanup properly
            all_embeddings = asyncio.run(fetch_embeddings_parallel())
    except Exception as e:
        logger.error(f"Error in asyncio handling: {e}")
        # Fallback to old approach if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        all_embeddings = loop.run_until_complete(fetch_embeddings_parallel())
    
    return all_embeddings

# ======================================================================
# DATA PREPARATION FUNCTIONS
# ======================================================================
def prepare_chunk_level_data(chunks, book_id, source_filename):
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
    
    chunk_data = []
    
    for chunk in chunks:
        # Create a record for each chunk with its text and full JSON object
        record = {
            "book_id": book_id,
            "source_filename": source_filename,
            "chunk_id": chunk.get("chunk_id"),
            "chunk_text": chunk.get("text", ""),
            "full_json": json.dumps(chunk)  # Store the complete JSON object
        }
        
        chunk_data.append(record)
    
    logger.info(f"Prepared {len(chunk_data)} chunk-level records")
    return chunk_data

def prepare_action_level_data(actions, book_id, source_filename):
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
def create_chunk_level_frame(chunk_data):
    """
    Create the chunk-level frame with embeddings.
    
    Args:
        chunk_data (list): List of chunk data records
        
    Returns:
        pl.DataFrame: Polars DataFrame with chunk-level data and embeddings
    """
    logger.info("Creating chunk-level frame...")
    
    # Extract chunk texts for embedding
    chunk_texts = [record["chunk_text"] for record in chunk_data]
    
    # Generate embeddings for chunks
    logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
    chunk_embeddings = generate_embeddings_batch(chunk_texts)
    
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

def create_action_level_frame(action_data):
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
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(action_quotes)} action quotes...")
    action_quote_embeddings = generate_embeddings_batch(action_quotes)
    
    logger.info(f"Generating embeddings for {len(consequence_quotes)} consequence quotes...")
    consequence_quote_embeddings = generate_embeddings_batch(consequence_quotes)
    
    logger.info(f"Generating embeddings for {len(action_tuples)} action tuples...")
    action_tuple_embeddings = generate_embeddings_batch(action_tuples)
    
    logger.info(f"Generating embeddings for {len(action_tuples_with_quotes)} action tuples with quotes...")
    action_tuple_with_quote_embeddings = generate_embeddings_batch(action_tuples_with_quotes)
    
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
def main():
    """
    Main function that runs the entire processing pipeline.
    """
    try:
        logger.info("Starting processing pipeline with new schema...")
        
        # Create output directories
        os.makedirs("newschema_embeddings", exist_ok=True)
        
        # Load novel data
        input_file = "data/INPUT_NOVEL_FILE.json" # Replace with your input file
        novel_data = load_novel_data(input_file)
        
        # Get book ID and source filename
        book_id = novel_data.get("book_id", "5001")  # Default to 5001 if not available
        source_filename = os.path.basename(input_file)
        
        # Extract chunks
        chunks = extract_chunks(novel_data)
        
        # Extract actions
        actions = extract_actions(novel_data, chunks)
        
        # Prepare data according to tom-furl schema
        chunk_data = prepare_chunk_level_data(chunks, book_id, source_filename)
        action_data = prepare_action_level_data(actions, book_id, source_filename)
        
        # Create frames with embeddings
        chunk_frame = create_chunk_level_frame(chunk_data)
        action_frame = create_action_level_frame(action_data)
        
        # Save frames - convert Object columns to serializable formats
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
                
            chunk_frame.write_parquet('newschema_embeddings/chunk_level_frame.parquet')
            logger.info("Chunk-level frame saved successfully")
        
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
                
            action_frame.write_parquet('newschema_embeddings/action_level_frame.parquet')
            logger.info("Action-level frame saved successfully")
        
        # Save raw data as JSON for reference
        with open('newschema_embeddings/chunks_data.json', 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)
        
        with open('newschema_embeddings/actions_data.json', 'w', encoding='utf-8') as f:
            json.dump(actions, f, indent=2)
        
        logger.info("Processing completed successfully!")
    
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        raise

# ======================================================================
# SCRIPT ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    main() 
