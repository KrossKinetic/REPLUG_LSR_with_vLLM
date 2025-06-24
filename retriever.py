import copy
import pickle
import numpy as np
import glob
from logging import getLogger
import time
import os
import torch
import index_utils.index
import index_utils.contriever
import index_utils.dragon
import index_utils.data_contriever
import index_utils.normalize_text


logger = getLogger()

class Retriever():
    def __init__(self, args):
        self.args = args
        if 'dragon' in args.re_model_name_or_path.lower():
            self.model, self.tokenizer = index_utils.dragon.load_retriever(
                args.re_model_name_or_path
            )
        else:
            self.model, self.tokenizer = index_utils.contriever.load_retriever(
                args.re_model_name_or_path
            )
        self.model.cuda()
        self.model.eval()

        self.index = index_utils.index.Indexer(
            args.projection_size, args.n_subquantizers, args.n_bits)

        input_paths = glob.glob(args.passages_embeddings)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, 'index.faiss')
        if args.save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f'Indexing passages from files {input_paths}')
            start_time_indexing = time.time()
            self.index_encoded_data(
                self.index, input_paths, args.indexing_batch_size)
            print(f'Indexing time: {time.time() - start_time_indexing:.1f} s.')
            if args.save_or_load_index:
                self.index.serialize(embeddings_dir)

        passages = index_utils.data_contriever.load_passages(args.passages)
        self.passage_id_map = {x['id']: x for x in passages}

    def index_encoded_data(self, index, embedding_files, indexing_batch_size):
        allids = []
        allembeddings = np.array([])
        for i, file_path in enumerate(embedding_files):
            print(f'Loading file {file_path}')
            with open(file_path, 'rb') as fin:
                ids, embeddings = pickle.load(fin)
            allembeddings = np.vstack(
                (allembeddings, embeddings)) if allembeddings.size else embeddings
            allids.extend(ids)
            while allembeddings.shape[0] > indexing_batch_size:
                allembeddings, allids = self.add_embeddings(
                    index, allembeddings, allids, indexing_batch_size)
        while allembeddings.shape[0] > 0:
            allembeddings, allids = self.add_embeddings(
                index, allembeddings, allids, indexing_batch_size)
        print('Data indexing completed.')

    def add_embeddings(self, index, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_toadd = ids[:end_idx]
        embeddings_toadd = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        index.index_data(ids_toadd, embeddings_toadd)
        return embeddings, ids

    def embed_queries(self, queries):
        # --- FIX: REMOVED torch.no_grad() to allow for backpropagation ---
        embeddings, batch_question = [], []
        for k, q in enumerate(queries):
            if self.args.normalize_text:
                q = index_utils.normalize_text.normalize(q)
            batch_question.append(q)
            if len(batch_question) == self.args.per_gpu_batch_size or k == len(queries) - 1:
                encoded_batch = self.tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=self.args.question_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                output = self.model(**encoded_batch)
                embeddings.append(output.cpu())
                batch_question = []
        embeddings = torch.cat(embeddings, dim=0)
        # --- FIX: Return a torch tensor, not a numpy array ---
        return embeddings

    def retrieve_passage(self, queries):
        questions_embedding = self.embed_queries(queries)
        questions_embedding_np = questions_embedding.detach().cpu().numpy() # Use detached numpy version for FAISS search
        
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(
            questions_embedding_np, self.args.n_docs)
        
        print(f"Retrieval completed in {time.time() - start_time_retrieval}s")
        
        num_queries = len(top_ids_and_scores)
        assert(num_queries == len(queries))
        
        top_docs_and_scores = []
        for i in range(num_queries):
            docs = [] 
            for doc_id in top_ids_and_scores[i][0]:
                if doc_id in self.passage_id_map:
                    doc = copy.deepcopy(self.passage_id_map[doc_id])
                    docs.append(doc)
            scores = top_ids_and_scores[i][1]
            top_docs_and_scores.append((docs, scores))
            
        return top_docs_and_scores
