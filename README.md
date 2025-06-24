# REPLUG LSR for Code: A Refactored Implementation

This repository contains a modified implementation of **REPLUG: Retrieval-Augmented Black-Box Language Models**. The primary goal of this refactoring was to create a stable, working version of the LSR (LM-Supervised Retrieval) fine-tuning process, as the original implementation appears to be incomplete.

The key modifications include:

* **Local LLM Integration:** The main Black Box LLM is built using the **vLLM server**, allowing for high-performance, on-device inference instead of relying on the costly OpenAI API.
* **Robust Driver Scripts:** New driver scripts were created to handle argument parsing and execute the training and evaluation pipelines correctly.
* **Code-Focused Baseline:** The entire pipeline has been adapted and debugged to work with code-based datasets, making it suitable for creating RAG baselines for code generation tasks.

Disclaimer: Further testing still needs to be performed to see if the modifications to get LSR to work yield similar results. Use with caution. Feel free to fix / improve things as you see fit.

## Prerequisites

* A CUDA-enabled GPU is required.
* Python 3.10
* Conda for dependency management.
* [vLLM](https://github.com/vllm-project/vllm) for serving the local LLM.

## Setup & Training Workflow

### 1. Initial Setup

First, install all the required Python packages using Conda and create a virtual environment.

```bash
conda env create -f environment.yml
```

If Conda doesn't work, Poetry can also be used to download the dependencies from .lock and .toml file, however some of the packages might be outdated.

### 2. Data Preparation

The LSR process requires two separate, non-overlapping datasets to prevent the model from learning "trivial retrieval" (i.e., just finding the exact source of its query).

* **Retrieval Corpus (`--passages`):** The large knowledge base the retriever will search through.
* **Training Query Corpus (`--data`):** A smaller, separate set of documents to generate training examples from.

You will need to prepare these two `.csv` files. Each file should contain at least two columns: `id` and `text`.

### 3. Generate Embeddings

Create a searchable vector index from your **retrieval corpus**. This step uses the base retriever model (e.g., MiniLM) to convert your code documents into embeddings.

* **Input:** Your retrieval corpus CSV file (e.g., `github_corpus.csv`).
* **Output:** A directory containing the embeddings (e.g., `code_embeddings/`).

```bash
# This command uses the smaller and faster MiniLM model
python generate_passage_embeddings.py \
    --model_name_or_path "sentence-transformers/all-MiniLM-L6-v2" \
    --passages "github_corpus.csv" \
    --output_dir "code_embeddings" \
    --projection_size 384 \
    --shard_id 0 \
    --num_shards 1
```

**Note:** The `--projection_size` must match the dimension of your chosen retriever model (384 for `all-MiniLM-L6-v2`).

### 4. Run the vLLM Server

In a **separate terminal**, start the vLLM server to host your supervisor LLM. This server will act as the "teacher" for the retriever.

```bash
# This command serves the TinyLlama model with an OpenAI-compatible API
python -m vllm.entrypoints.openai.api_server --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### 5. Run REPLUG LSR Fine-Tuning

With the data prepared and the vLLM server running, you can now start the main training process. This script will train the retriever model.

* **Input:** Your two separate CSV files (`github_corpus.csv` and `training_queries.csv`), the generated embedding, and the name of the model being served by vLLM.
* **Output:** Your new, fine-tuned retriever model saved to a directory.

```bash
python run_replug_lsr.py \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --passages "github_corpus.csv" \
    --passages_embeddings "code_embeddings/passages_00" \
    --data "training_queries.csv" \
    --output_dir "output_finetuned_retriever" \
    --re_model_name_or_path "sentence-transformers/all-MiniLM-L6-v2" \
    --projection_size 384 \
    --learning_rate 2e-5 \
    --per_gpu_batch_size 4 \
    --context_len 128 \
    --pred_len 128 \
    --retrieved_max_length 128 \
    --n_docs 10 \
    --save_or_load_index
```

## Evaluation

### Quick Qualitative Test

To quickly see if the fine-tuned model behaves differently from the original, you can use the `test_retriever.py` script. This script will show you a side-by-side comparison of the retrieval results for a sample query.

```bash
python test_retriever.py
```

### Formal Benchmarking

The `test_retriever.py` script is only for a quick qualitative check. For a rigorous, quantitative evaluation of your new retriever's performance on coding tasks, it is highly recommended to use a formal benchmark like **CodeRagBench**.
