# REPLUG: 
This includes a modified implementation of **REPLUG: Retrieval-Augmented Black-Box Language Models** to try to get LSR Finetuning to work on retrievers. The main Black Box LLM is built using vLLM to speed up the process and perform on-device inference instead of relying on cost expensive OpenAI server.

Further tests need to be conducted to see if this modified script yields similar results. Work in progress...

## LSR finetuning:

### Download Dataset
```
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
```

### Downsizing it to 50k passages
```
python downsample_corpus.py --input_file "psgs_w100.tsv" --output_file "psgs_w100_50k.tsv" --num_passages 50000
```

### Generating Embeddings
```
python generate_passage_embeddings.py \
    --model_name_or_path "sentence-transformers/all-MiniLM-L6-v2" \
    --passages "psgs_w100_50k.tsv" \
    --output_dir "embeddings_50k_miniLM" \
    --shard_id 0 \
    --n_shards 1
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



