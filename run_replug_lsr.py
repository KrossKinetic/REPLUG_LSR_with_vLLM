import argparse
import torch
import utils
import csv
from tqdm import tqdm
from pathlib import Path
from replug_lsr import GPT3LM

def parse_args():
    """
    Argument parser for LSR Fine-tuning. Most of these are copied directly from logprobs script and might not have been used.
    """
    parser = argparse.ArgumentParser()
    # Model and Data
    parser.add_argument('--model', required=True, help="HuggingFace model for the supervisor LM.")
    parser.add_argument('--output_dir', required=True, help="Directory to save the fine-tuned retriever model.")
    
    # Training
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate for the optimizer.")
    parser.add_argument('--per_gpu_batch_size', type=int, default=16, help="Effective batch size for training.")
    
    # Retrieval
    parser.add_argument('--passages', type=str, required=True, help='Path to passages file (.tsv or .jsonl). This will also be used as the training data.')
    parser.add_argument('--passages_embeddings', type=str, required=True, help='Glob path to encoded passages')
    parser.add_argument('--re_model_name_or_path', type=str, default="facebook/contriever", help="Path to the retriever model to be fine-tuned.")
    parser.add_argument('--n_docs', type=int, default=10, help="Number of documents to retrieve per query.")
    parser.add_argument("--save_or_load_index", action='store_true', help='If enabled, save index and load index if it exists')
    parser.add_argument('--cache_dict', type=str, default="cache", help="Path to a cache file for retrieval results.")
    parser.add_argument('--data', type=str, required=True, help="Path to the corpus for generating training queries (.tsv, .jsonl, etc).")
    parser.add_argument('--do_retrieval', type=int, default=1) # Always have to perform retrieval for LSR
    
    parser.add_argument('--normalize_text', action='store_true', help="Normalize text before processing.")
    parser.add_argument('--question_maxlength', type=int, default=128, help="Maximum number of tokens in a query.")
    parser.add_argument('--passage_maxlength', type=int, default=128, help="Maximum number of tokens in a passage for the retriever's tokenizer.")
    parser.add_argument('--chunk_size', type=int, default=100, help="Chunk size for processing passages (if applicable).")
    parser.add_argument('--no_title', action='store_true', help="Do not use titles when processing passages.")

    # Sequence Lengths
    parser.add_argument('--retrieved_max_length', type=int, default=128, help="Max length of each retrieved document to be passed to the LM.") # USED
    parser.add_argument('--context_len', type=int, default=128, help="Prior context used as the retrieval query.")
    parser.add_argument('--pred_len', type=int, default=128, help="Length of the next sentence for computing log probability.")

    # Faiss Indexing
    parser.add_argument('--use-faiss-gpu', action="store_true", help='If enabled, use faiss GPU for retrieval inference')
    parser.add_argument('--projection_size', type=int, default=768, help="Dimension of the retriever model embeddings.")
    parser.add_argument("--n_subquantizers", type=int, default=0, help='Number of subquantizers for vector quantization.')
    parser.add_argument("--n_bits", type=int, default=8, help='Number of bits per subquantizer')
    parser.add_argument('--indexing_batch_size', type=int, default=1000000, help="Batch size of the number of passages indexed")

    # LSR Specific
    parser.add_argument('--temperature_gold', type=float, default=0.1) # USED
    parser.add_argument('--temperature_score', type=float, default=0.1) # USED
 
    return parser.parse_args()


def load_text_from_local_corpus(file_path):
    print(f"Loading training data from local file: {file_path}")
    text_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f) # Using default delimiter
        header = next(reader) # Skip header
        text_column_index = header.index('text') 
        for row in reader:
            if len(row) > text_column_index:
                text_data.append(row[text_column_index])
    print(f"Loaded {len(text_data)} documents for training.")
    return text_data


def main():
    args = parse_args()
    
    # --- Model Initialization ---
    config = {"engine": args.model}

    model = GPT3LM(
        **config, 
        context_len=args.context_len,
        max_seq_len=args.context_len + args.pred_len,
        batch_size=args.per_gpu_batch_size, 
        args=args
    )
    
    # --- Retriever and Optimizer Setup ---
    model.initialize_retriever(args)
    
    model.retriever.model.train()

    optimizer = torch.optim.Adam(model.retriever.model.parameters(), lr=args.learning_rate)
    model.optimizer = optimizer
    
    # --- Data Loading and Training Loop ---
    text_data = load_text_from_local_corpus(args.data)

    print(f"Starting LSR fine-tuning for {len(text_data)} documents...")

    for doc in tqdm(text_data, desc="Fine-tuning Retriever"):
        if not doc.strip():
            continue   
        model.forward_training(doc)

    # --- Save the Fine-tuned Retriever ---
    print("Training complete. Saving the fine-tuned retriever model...")
    save_path = Path(args.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model.retriever.model.save_pretrained(save_path)
    model.retriever.tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
