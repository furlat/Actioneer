# vLLM Embedding Service

This project provides a scalable, efficient embedding service using vLLM on Modal. It's designed to generate high-quality text embeddings for semantic search and other NLP applications.

## Overview

The service uses the `Alibaba-NLP/gte-Qwen2-1.5B-instruct` model to generate 1536-dimensional embeddings. It's deployed as a serverless application on Modal, making it cost-effective and easy to scale.

Key features:
- Generates embeddings with 1536 dimensions
- Uses persistent caching for faster cold starts
- Configurable idle timeout and scale-down settings
- Supports single texts or batches of texts
- Simple REST API interface
- Detailed embedding metadata and error handling

## Production Files

- `deploy_vllm_server.py` - Main Modal application for the embedding service
- `vllm_client.py` - Client library for interacting with the service
- `process_novel_data_HK3schema.py` - Pipeline for processing novel data with embeddings
- `deploy.sh` - Helper script for deployment
- `.env` - Configuration file (create from `.env.example`)

## Deployment

### Prerequisites

1. Modal account and CLI tool installed
2. Python 3.6+ with required packages

### Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a `.env` file based on the example:
   ```
   cp .env.example .env
   ```

3. Edit the `.env` file as needed (the defaults work well for most cases).

### Deploy to Modal

Use the deploy script:
```
./deploy.sh
```

Or deploy manually:
```
python -m modal deploy deploy_vllm_server.py
```

After deployment, update your `.env` file with the URL provided by Modal.

## Usage

### Using the Client Library

The `vllm_client.py` provides convenient methods for generating embeddings:

```python
from vllm_client import generate_embedding, generate_embeddings_batch

# Single text embedding
text = "This is a sample text for embedding."
embedding, metadata = generate_embedding(text)

# Process a batch of texts
texts = ["First text", "Second text", "Third text"]
embeddings, metadata = generate_embeddings_batch(texts)
```

### Direct API Access

You can also access the API directly:

```python
import requests
import json

url = "https://your-username--vllm-server-vllm-server.modal.run"

# Single text
response = requests.get(url, params={"text": "Sample text"})

# Multiple texts
texts = ["Text 1", "Text 2", "Text 3"]
response = requests.get(url, params={"texts": json.dumps(texts)})

result = response.json()
embeddings = result["embeddings"]
```

## Output Format

The embeddings are returned as lists of floats with 1536 dimensions. When stored in parquet files, they use the following schema:

- Stored as float32 values
- Each embedding is a vector of 1536 dimensions
- Column name is typically 'chunk_embedding' or 'embedding'

Example:
```
Embedding dimensions: 1536
Data type: float32
First few values: [0.02621431 -0.04592579 0.02045203 0.01110464 -0.00691456]
```

## Performance Optimizations

Several optimizations have been implemented:

1. **Persistent Volumes**: Model weights are cached in Modal volumes to dramatically reduce cold start times:
   - `huggingface-cache` - Caches model downloads
   - `vllm-cache` - Caches vLLM-specific optimizations

2. **Resource Management**:
   - `min_containers=0` - Allows complete scale-down when idle to minimize costs
   - `scaledown_window=5 * MINUTES` - Keeps server alive for 5 minutes after last request
   - `timeout=600` - Allows long-running requests up to 10 minutes

## Troubleshooting

If you encounter issues:

1. Check the Modal dashboard for logs and errors
2. Verify your `.env` file has the correct URL
3. Ensure you have sufficient GPU quota on Modal

## Maintenance

For maintenance:

1. Update the model by changing the default in `deploy_vllm_server.py`
2. Adjust timeout and scaling parameters in the app function decorator
3. Monitor usage in the Modal dashboard 
