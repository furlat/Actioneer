#!/usr/bin/env python3
"""
Unified vLLM Server Deployment
===========================

This script deploys a single vLLM server to Modal that handles embedding operations.
The server is designed to be accessed remotely by clients and only provides embedding functionality.
"""

import argparse
import modal
from typing import List, Dict, Any, Optional

# Define MINUTES constant (60 seconds) since it's not exported in this Modal version
MINUTES = 60

# Print Modal version
print(f"Modal version: {modal.__version__}")

# Create an app (renamed from stub for newer Modal versions)
app = modal.App("vllm-server")

# Create volumes for caching
hf_cache_vol = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True
)
vllm_cache_vol = modal.Volume.from_name(
    "vllm-cache", create_if_missing=True
)

# Define the vLLM image
vllm_image = modal.Image.debian_slim().pip_install(
    "vllm>=0.3.0",  # Specify a recent version
    "torch>=2.0.0",
    "transformers>=4.36.0",
    "fastapi[standard]",  # Required for web endpoints
    "numpy", 
    "einops",  # Often needed by newer models
    "accelerate",  # Helps with model loading
).run_commands(
    # Install additional system dependencies
    "apt-get update && apt-get install -y build-essential",
    # Set environment variables for better GPU utilization
    "echo 'export CUDA_VISIBLE_DEVICES=0' >> /etc/profile",
    "echo 'export NCCL_ASYNC_ERROR_HANDLING=1' >> /etc/profile",
    "echo 'export VLLM_USE_V1=0' >> /etc/profile",
    # Avoid PyTorch distributed process group warnings
    "echo 'export TORCH_DISTRIBUTED_DEBUG=DETAIL' >> /etc/profile",
)

# Define the GPU requirements
gpu = "A100-40GB"

# Define the vLLM server
@app.function(
    image=vllm_image,
    gpu=gpu,
    timeout=600,
    scaledown_window=5 * MINUTES,  # Keep server alive for 5 minutes
    min_containers=0,  # Allow complete scale down when idle for cost savings
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    }
)
@modal.fastapi_endpoint(method="GET")
def vllm_server(text: str = None, texts: str = None, model: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"):
    """
    Main vLLM server endpoint that handles all embedding requests.
    Can be called directly with GET requests.
    
    Args:
        text (str, optional): Single text to embed
        texts (str, optional): JSON string containing array of texts to embed
        model (str, optional): Model to use for embeddings
        
    Returns:
        dict: Dictionary containing embeddings or error information
    """
    import json
    import os
    import torch
    import numpy as np
    import atexit
    from transformers import AutoTokenizer, AutoModel
    
    # Set environment variables to avoid warnings
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    
    # Register cleanup function for distributed processes
    def cleanup_distributed():
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    
    # Register the cleanup function to run at exit
    atexit.register(cleanup_distributed)
    
    try:
        # Handle single text
        if text:
            input_texts = [text]
        # Handle multiple texts
        elif texts:
            # Try to parse as JSON
            try:
                input_texts = json.loads(texts)
                if not isinstance(input_texts, list):
                    # If not a list, treat the original string as a single text item
                    input_texts = [texts]
            except json.JSONDecodeError:
                # If JSON parsing fails, treat the original string as a single text item
                input_texts = [texts]
        else:
            return {
                "error": "No text provided. Please include either 'text' or 'texts' parameter.",
                "status_code": 400
            }
        
        # --- Add logging for text count ---
        print(f"INFO: Processing {len(input_texts)} texts...")
        # --- End added logging ---
        
        # Using transformers directly for embedding support
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_instance = AutoModel.from_pretrained(model, trust_remote_code=True)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_instance = model_instance.to(device)
        
        # Generate embeddings
        embeddings = []
        
        # Process each text and get its embedding
        for t in input_texts:
            # Tokenize and prepare input
            encoded_input = tokenizer(t, padding=True, truncation=True, return_tensors='pt')
            # Move inputs to the same device as the model
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            # Get model embeddings
            with torch.no_grad():
                model_output = model_instance(**encoded_input)
                # For models like BGE, sentence embedding is in the last hidden state
                if hasattr(model_output, 'last_hidden_state'):
                    # Mean pooling to get sentence embedding
                    tokens_embeddings = model_output.last_hidden_state
                    attention_mask = encoded_input['attention_mask']
                    mask = attention_mask.unsqueeze(-1).expand(tokens_embeddings.size()).float()
                    sum_embeddings = torch.sum(tokens_embeddings * mask, 1)
                    sum_mask = torch.sum(mask, 1)
                    embedding = sum_embeddings / sum_mask
                # For models that output pooler_output 
                elif hasattr(model_output, 'pooler_output'):
                    embedding = model_output.pooler_output
                else:
                    raise ValueError(f"Unsupported model output format for {model}")
                
                # Convert to list and normalize
                embedding_np = embedding.squeeze().cpu().numpy()
                embedding_normalized = embedding_np / np.linalg.norm(embedding_np)
                embeddings.append(embedding_normalized.tolist())
        
        return {
            "embeddings": embeddings,
            "dimension": len(embeddings[0]) if embeddings else 0,
            "model": model,
            "status_code": 200
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        
        # Ensure we clean up distributed process groups even on error
        cleanup_distributed()
        
        return {
            "error": f"Failed to generate embeddings: {str(e)}",
            "traceback": error_details,
            "status_code": 500
        }
    finally:
        # Make sure we clean up in all cases
        cleanup_distributed()

# Main function for CLI usage
def main():
    parser = argparse.ArgumentParser(description="Deploy vLLM server to Modal")
    parser.add_argument("--model", type=str, default="Alibaba-NLP/gte-Qwen2-1.5B-instruct", 
                        help="Model to deploy")
    parser.add_argument("--deploy", action="store_true",
                        help="Deploy the server")
    parser.add_argument("--gpu", type=str, default="A100-40GB", choices=["T4", "A10G", "A100-40GB"],
                        help="GPU type to use (T4, A10G, or A100-40GB)")
    parser.add_argument("--gpu-count", type=int, default=1,
                        help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    # Update GPU configuration based on args
    global gpu
    if args.gpu == "T4":
        gpu = "T4"
    elif args.gpu == "A10G":
        gpu = "A10G"
    elif args.gpu == "A100-40GB":
        gpu = "A100-40GB"
    
    if args.deploy:
        # For deployment, simply print instructions to use Modal CLI
        print(f"To deploy embedding server with the {args.model} model, run:")
        print(f"python -m modal deploy {__file__}")
        print("\nAfter deployment, update your .env file with the server URL:")
        print(f"MODAL_VLLM_SERVER=https://yourusername--vllm-server-vllm-server.modal.run")
        print("\nYou can find the exact URL in the Modal dashboard.")
    else:
        # Print usage instructions
        print("Please specify --deploy to deploy the server")
        parser.print_help()

if __name__ == "__main__":
    main() 