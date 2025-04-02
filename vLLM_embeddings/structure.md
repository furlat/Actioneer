# Repository Structure

This document outlines the structure of the vLLM Embedding Service repository.

## Core Files

- `deploy_vllm_server.py` - Main deployment script for the vLLM server on Modal
- `vllm_client.py` - Client library for making embedding requests to the deployed server
- `process_novel_data_HK3schema.py` - Pipeline for processing novel data with embeddings

## Configuration Files

- `requirements.txt` - Python package dependencies
- `.env.example` - Example environment variables template (copy to `.env` for local use)

## Deployment Files

- `deploy.sh` - Deployment script for the vLLM server
- `modal_reference.md` - Reference for Modal configuration options

## Documentation

- `HK3_TEAM_README.md` - Main project documentation
- `structure.md` - This file, describing repository organization

## Data Directories

- `data/` - Input data directory
  - Contains Gutenberg novel parquet files used for embedding generation
  
- `vllm_embeddings_parquet/` - Output directory for embedding results
  - `chunk_level_frame_gutenberg_sample.parquet` - Embedded chunks
  - `action_level_frame_gutenberg_sample.parquet` - Embedded actions

## Generated Files (not in version control)

- `*.log` - Log files generated during processing
- `.env` - Local environment variables file (create from `.env.example`) 