import argparse
import copy
from retriever import Retriever  # Assuming the corrected retriever.py is in the same directory

def setup_retriever(args):
    """Initializes a Retriever instance with the given arguments."""
    print("-" * 50)
    print(f"Loading retriever from: {args.re_model_name_or_path}")
    retriever = Retriever(args)
    print("Retriever loaded successfully.")
    print("-" * 50)
    return retriever

def main():
    # We will reuse the arguments from our main driver script
    # We can hardcode them here for this simple comparison test
    
    # --- Base Arguments (Shared by both retrievers) ---
    base_args = argparse.Namespace(
        passages="psgs_w100.tsv",
        passages_embeddings="embeddings_50k_miniLM", 
        save_or_load_index=True,
        n_docs=5, # Let's just look at the top 5 results
        projection_size=384, # Correct dimension for MiniLM
        n_subquantizers=0,
        n_bits=8,
        use_faiss_gpu=False,
        cache_dict=None,
        normalize_text=False,
        question_maxlength=128,
        passage_maxlength=128,
        chunk_size=100,
        no_title=False,
        per_gpu_batch_size=4,
        # This last one is needed by the retriever init but not used in this script
        num_gpus=-1 
    )

    # --- Setup for the ORIGINAL Retriever ---
    args_original = copy.copy(base_args)
    args_original.re_model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
    
    # --- Setup for YOUR FINE-TUNED Retriever ---
    args_finetuned = copy.copy(base_args)
    args_finetuned.re_model_name_or_path = "./output_finetuned_retriever" # Path to your new model

    # Initialize both retrievers
    original_retriever = setup_retriever(args_original)
    finetuned_retriever = setup_retriever(args_finetuned)

    # --- The Test Query ---
    query = "The game is a tactical role @-@ playing game where players take control"
    
    print(f"\n\n===== COMPARING RETRIEVAL RESULTS FOR QUERY =====")
    print(f"QUERY: '{query}'\n")

    # --- Perform Retrieval with BOTH models ---
    original_docs, original_scores = original_retriever.retrieve_passage([query])[0]
    finetuned_docs, finetuned_scores = finetuned_retriever.retrieve_passage([query])[0]

    # --- Print Results Side-by-Side ---
    print("--- ORIGINAL MODEL RESULTS ---")
    # FIX: Loop over docs and scores together using zip
    for i, (doc, score) in enumerate(zip(original_docs, original_scores)):
        print(f"{i+1}. (Score: {score:.4f}) {doc['text'][:150]}...")
    
    print("\n--- YOUR FINE-TUNED MODEL RESULTS ---")
    # FIX: Loop over docs and scores together using zip
    for i, (doc, score) in enumerate(zip(finetuned_docs, finetuned_scores)):
        print(f"{i+1}. (Score: {score:.4f}) {doc['text'][:150]}...")
        
    print("\n===== COMPARISON COMPLETE =====")


if __name__ == "__main__":
    main()

