# REPLUG: 
This includes a modified implementation of **REPLUG: Retrieval-Augmented Black-Box Language Models** to try to get LSR Finetuning to work on retrievers. The main Black Box LLM is built using vLLM to speed up the process and perform on-device inference instead of relying on cost expensive OpenAI server.

## LSR finetuning:
```
python LSR_finetune/replug_lsr.py       
       --model_config_path $MODEL_CONFIG  \
       --passages   $ENCODE_PATH  # the path to the raw corpus from step 1 \
       --passages_embeddings  $EMB_PATH # the path to encoded corpus from step 1 \
       --re_model_name_or_path $RETRIEVER  \
       --data   wikitext-2-v1    # dataset you want to use. Change the dataloading in line82/92 in save_logprob_data.py \
       --retrieved_max_length 128      \ # max length of each retrieved documents.
       --context_len 128     \ # Prior context used as the retrieval query
       --pred_len 768        \ # length of the next sentence following the prior context. This next sentence will be used to compute the log probability
       --output_path  outputs/ppl.data  \
       --ensemble $ENSEMBLE_DOCS    \ 
       --n_docs $ENSEMBLE_DOCS    \
       --save_or_load_index
```



