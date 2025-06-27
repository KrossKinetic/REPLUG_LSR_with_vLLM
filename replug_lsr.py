import numpy as np
from tqdm import tqdm
from retriever import Retriever
from typing import Optional
from openai import OpenAI
import torch
import transformers
import utils

class LM:
    @classmethod
    def create_from_config(cls, path):
        raise NotImplementedError

    def initialize_retriever(self, args):
        self.args = args # Removed if-else statement because do_retrieval is always true
        self.retriever = Retriever(args)

class GPT3LM(LM):

    def __init__(self, engine, context_len=1024, max_seq_len=2048, verbose=False, batch_size=16, optimizer=None, args=None):
        
        # Added new
        self.client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="vllm"
        )
        
        self.engine = engine
        self.context_len = context_len
        self.max_seq_len = max_seq_len
        self.wb = utils.WaitBlocker()
        self.verbose = verbose
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.args = args

        print(f"Loading tokenizer for model: {self.engine}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.engine) # Load tokenizer for the HuggingFace Modle you are using for vLLM
        
        if self.tokenizer.eos_token_id is not None:
             self.end_of_text_token_id = self.tokenizer.eos_token_id
        else:
             self.end_of_text_token_id = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]

    def forward_training(self, text):
        input_ids = self.tokenizer.encode_plus(text)["input_ids"]
        rolling_token_windows = utils.get_rolling_token_windows(
            token_list=input_ids,
            prefix_token=self.end_of_text_token_id,
            max_seq_len=self.max_seq_len,
            context_len=self.context_len,
        )

        batch_loss = []
        batch_index = 0
        for input_tokens, pred_tokens in tqdm(rolling_token_windows, desc="Processing windows"):
            retriever_loss = self.forward_training_single(input_tokens, pred_tokens)
            if retriever_loss is None:
                continue
            batch_loss.append(retriever_loss)
            batch_index += 1
            if batch_index >= self.batch_size:
                total_loss = torch.stack(batch_loss).mean()
                total_loss.backward()
                
                self.optimizer.step()
                self.optimizer.zero_grad()

                batch_loss = []
                batch_index = 0
        
        if batch_loss:
            total_loss = torch.stack(batch_loss).mean()
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def forward_training_single(self, input_tokens, pred_tokens):
        if len(input_tokens) == len(pred_tokens): return None
        query_id = input_tokens[:-len(pred_tokens)]
        if not query_id: return None
            
        query = self.tokenizer.decode(query_id)

        retrieved_list = self.retriever.retrieve_passage([query])
        
        # Debug Message
        if not retrieved_list or not retrieved_list[0][0]:
            tqdm.write(f"Query: '{query[:80].strip().replace(chr(10), ' ')}...' | Docs found: 0")
            return None
            
        docs, _ = retrieved_list[0]

        # Debug Message
        tqdm.write(f"Query: '{query[:80].strip().replace(chr(10), ' ')}...' | Docs found: {len(docs)}")

        plain_docs = [doc["text"] for doc in docs]

        # --- FIX: Re-calculate the retriever score to keep it attached to the graph ---
        # 1. Embed the query (this will now have requires_grad=True)
        query_embedding = self.retriever.embed_queries([query])
        # 2. Embed the retrieved passages (this also has requires_grad=True)
        passages_embedding = self.retriever.embed_queries(plain_docs)
        # 3. Calculate the dot product. This `retriever_score` is now "live".
        retriever_score = torch.einsum("bd,bd->b", query_embedding.repeat(passages_embedding.size(0), 1), passages_embedding).unsqueeze(0)
        
        all_gold_score = []
        for i in range(len(docs)):
            doc_str = plain_docs[i]
            doc_encodings = self.retriever.tokenizer.encode(doc_str, truncation=True, max_length=self.args.retrieved_max_length)
            input_tokens_tmp = doc_encodings + input_tokens
            
            block_output = self.get_token_logprobs(input_tokens=input_tokens_tmp, pred_tokens=pred_tokens)
            if block_output is None: continue
            
            gold_score = block_output["logprobs"].sum()
            all_gold_score.append(gold_score)
            
        if not all_gold_score: return None

        all_gold_score = torch.FloatTensor(all_gold_score).unsqueeze(0)
        retriever_loss = self.kldivloss(retriever_score, all_gold_score)
        return retriever_loss

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score / self.args.temperature_gold, dim=-1)
        score = torch.nn.functional.log_softmax(score / self.args.temperature_score, dim=-1)
        return torch.nn.KLDivLoss(reduction='batchmean')(score, gold_score)

    def get_token_logprobs(self, input_tokens, pred_tokens):
        pred_start = len(input_tokens) - len(pred_tokens) + 1
        token_ids = input_tokens + [pred_tokens[-1]]
        
        try:
            with self.wb.check_valid():

                # Changed to use the client
                response = self.client.completions.create(
                    model=self.engine,
                    prompt=token_ids,
                    max_tokens=0,
                    temperature=0.0,
                    logprobs=1, 
                    echo=True,
                )
            
            logprobs_of_pred_tokens = np.array(response.choices[0].logprobs.token_logprobs[pred_start:])
            positions = np.arange(pred_start - 1, pred_start - 1 + len(pred_tokens))

            return {"logprobs": logprobs_of_pred_tokens, "positions": positions}
        except Exception as e:
            tqdm.write(f"API call failed: {e}")
            return None

    @classmethod
    def create_from_config(cls, config):
        return cls(**config)
