# REPLUG: 
This includes a modified implementation of **REPLUG: Retrieval-Augmented Black-Box Language Models** to try to get LSR Finetuning to work on retrievers. The main Black Box LLM is built using vLLM to speed up the process and perform on-device inference instead of relying on cost expensive OpenAI server.

Further tests need to be conducted to see if this modified script yields similar results. Work in progress...

## LSR finetuning:

### Download Dataset and reformat it
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
    --passages "psgs_w100_50k.tsv" \
    --passages_embeddings "embeddings_50k_miniLM" \
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



