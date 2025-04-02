#!/bin/bash
# Deploy script for vLLM on Modal

# Load .env file if it exists
if [ -f .env ]; then
    echo "Loading environment from .env file"
    set -a
    source .env
    set +a
else
    echo "No .env file found, using defaults"
fi

# Default values
MODEL=${MODAL_MODEL:-"Alibaba-NLP/gte-Qwen2-1.5B-instruct"}
GPU_TYPE=${MODAL_GPU_TYPE:-"A100"}  # Default to A100 since vLLM needs it
GPU_COUNT=${MODAL_GPU_COUNT:-1}

# Get command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --gpu)
            GPU_TYPE="$2"
            shift 2
            ;;
        --gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./deploy.sh [options]"
            echo ""
            echo "Options:"
            echo "  --model      Model to deploy (default: $MODEL)"
            echo "  --gpu        GPU type to use (T4, A10G, A100) (default: $GPU_TYPE)"
            echo "  --gpu-count  Number of GPUs to use (default: $GPU_COUNT)"
            echo "  --help       Display this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Deploying vLLM server with model: $MODEL"
echo "GPU configuration: $GPU_COUNT x $GPU_TYPE"

# Deploy the server
python -m modal deploy deploy_vllm_server.py

# Prompt the user to update the .env file with the endpoint URL
echo ""
echo "After deployment, update your .env file with the server URL:"
echo "MODAL_VLLM_SERVER=https://yourusername--vllm-server-vllm-server.modal.run"
echo "You can find the exact URL in the Modal dashboard." 