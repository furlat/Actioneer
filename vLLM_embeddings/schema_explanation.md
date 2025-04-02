# Data Embedding Schema

## Overview
The schema describes a dual-frame embedding approach for novels, with:
1. **Chunk-level embedding**: Embedding of chunks with complete content and metadata
2. **Action-level embedding**: Processing the actions with multiple embedding types per action

## Data Sources
The pipeline processes parquet files from the Gutenberg novels dataset:
- `gutenberg_en_novels_all_processed_chunks_with_metadata.parquet`: Contains all chunks
- `gutenberg_en_novels_all_processed_actions_with_matches_and_metadata.parquet`: Contains all actions

## Embedding Details

### For Each Novel
We generate two output frames:

#### Frame 1: Chunk Level
- Contains rows of chunks with original metadata
- Adds Qwen 2.5 1.5B embeddings of the chunk text
- Preserves all original columns from the source parquet file

#### Frame 2: Action Level
- Contains rows of actions with original metadata
- Each row represents an action with multiple embedding types
- We embed:
  - Action quote
  - Consequence quote
  - Source-action-target tuple with associated quotes
  
### For Each Action Row
We generate:
- Embedding of action quote
- Embedding of consequence quote (when available)
- Embedding of action tuple (source, action, target)
- Embedding of action tuple concatenated with action quote

### Important Fields
- Book ID
- Source information
- Chunk ID
- Temporal order of the action (position in the list for each action)

## Implementation Note
The source files are parquet files in the data directory:
```
data/gutenberg_en_novels_all_processed_chunks_with_metadata.parquet
data/gutenberg_en_novels_all_processed_actions_with_matches_and_metadata.parquet
```

The embedding process uses the Qwen 2.5 1.5B model hosted on Modal.
Output creates folder called qwen_embeddings, with the following files:
- chunks_with_qwen_embeddings.parquet
- actions_with_qwen_embeddings.parquet 