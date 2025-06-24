# REPLUG LSR: 
This includes a modified implementation of **REPLUG: Retrieval-Augmented Black-Box Language Models** to try to get LSR Finetuning to work and finetune retrievers since the original implementation seems to be incomplete. The main Black Box LLM is built using vLLM to speed up the inference process by performing on-device inference instead of relying on cost expensive OpenAI server. The code was refactored to work well with code generation related tasks.

Further tests need to be conducted to see if this modified script yields similar results. Work in progress... This is NOT a final, consumer ready script. It is working as far as I can tell but the code and parameters might have to be updated based on specific use cases and for your system.

## LSR finetuning:

### Download Test Retriever Dataset and reformat it
```
python3 dataset_downloader_formatter.py
```

### Generating Embeddings
```
python generate_passage_embeddings.py \
    --model_name_or_path "sentence-transformers/all-MiniLM-L6-v2" \
    --passages "python-github-code.csv" \
    --output_dir "embeddings" \
    --shard_id 0 \
    --num_shards 1
```

### Running vLLM Server
```
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype auto --api-key vllm
```

### Running REPLUG LSR 
```
python run_replug_lsr.py \
    --model_config_path "local_vllm_config.json" \
    --passages "python-github-code.csv" \
    --passages_embeddings "embeddings/passages_00" \
    --data "python-github-code-data.csv" \
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

### Basic Test for newly created retriever

```
python3 test_retriever.py
```



