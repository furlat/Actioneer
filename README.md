Below is a cleaned-up, concise README that provides the same information in a clearer format:

---

# Actioneer
**Mapping Narrative Books to Action Sequences**

## Overview
**Actioneer** is a project focused on extracting action sequences from narrative books, leveraging external inference services to process and interpret text from large corpora (e.g., Project Gutenberg).

## Installation

1. **Clone the Actioneer Repository**  
   ```bash
   git clone https://github.com/furlat/Actioneer
   cd Actioneer
   ```
2. **Set Up a Virtual Environment**  
   Create and activate a virtual environment at this level using your preferred method (e.g., `virtualenv`, `conda`, or `venv`).

3. **Clone the MultiInference Repository**  
   ```bash
   git clone https://github.com/marketagents-ai/MultiInference
   cd MultiInference
   ```
4. **Install MultiInference**  
   ```bash
   pip install -e .
   ```

5. **Copy and Configure Environment File**  
   ```bash
   cp env.example .env
   ```
   Open the newly created `.env` file and add the appropriate API endpoints, keys, and other required credentials.

6. **Return to the Actioneer Folder**  
   From within the `Actioneer` directory, verify that you can import the `minference` package in Python:
   ```python
   import minference
   ```
   (If the import works without error, the installation is successful.)

## Usage

1. **Download Project Gutenberg Data**  
   Run the Jupyter notebook to download data:
   ```bash
   jupyter notebook scripts/gutenberg_actions.ipynb
   ```

2. **Set Up the Modal Server**  
   If the Modal server is not running, you can deploy the server:
   ```bash
   modal deploy vllm_inference.py
   ```

3. **Parse a Book with Qwen**  
   Once the server is deployed, you can parse a specific book:
   ```bash
   jupyter notebook scripts/gutenberg_processing.ipynb
   ```
   This notebook will demonstrate how to use Qwen to process and analyze the text for action mapping.

---

**Happy hacking!** For questions, bug reports, or feature requests, feel free to open an issue or submit a pull request.