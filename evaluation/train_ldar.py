import tiktoken
from eval_utils import (
    load_data,
    create_msgs,
    truncate_input
)
import json
import os

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from search.simpleHybridSearcher import SimpleHybridSearcher
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm

from optim_utils import OneShotTransformer
from random import shuffle
from optim_utils import process_example as evaluate_example
from torch.distributions import Beta
from collections import defaultdict
import wandb
import numpy as np
import gc
from distutils.util import strtobool
import random
from collections import deque

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--query_type', type=str, help='the type of the query')
parser.add_argument('--context_length', type=str, help='the length of the context')
parser.add_argument('--eval_model', type=str, help='model')
parser.add_argument('--save_path', type=str, help='save path for the predictions', default='reproduce_prediction')

parser.add_argument('--device', type=int, help='device to load our lightweight adaptive retriever', default=3)
parser.add_argument('--run_name', type=str, default='LDAR')
parser.add_argument('--debug', type=strtobool, default=False)
parser.add_argument('--val_while_training', type=strtobool, default=True)
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--full_eval_only', type=strtobool, default=False)

MODEL_PATH = {
    "llama-3.1-8b": "../models/Meta-Llama-3___1-8B-Instruct",
    "llama-3.2-3b": "../models/Llama-3___2-3B-Instruct",
    "qwen-2.5-7b": "../models/Qwen-2.5-7B-Instruct",
    "qwen-3-4b": "../models/Qwen-3-4B-Instruct",
    "mistral-nemo-12b": "../models/Mistral-Nemo-Instruct-2407",
}

args = parser.parse_args()
eval_model = args.eval_model
query_type = args.query_type
context_length = args.context_length
model_path = MODEL_PATH[eval_model]
device = torch.device(args.device)
seed = args.seed
full_eval_only = args.full_eval_only

random.seed(seed)                        # Python built-in RNG
np.random.seed(seed)                     # NumPy RNG
torch.manual_seed(seed)                  # CPU RNG
torch.cuda.manual_seed(seed)             # Current GPU
torch.cuda.manual_seed_all(seed)         # All GPUs (multi-GPU)

# ----------------------------------------------------------------------
# Simple State-Value Network (baseline)
# ----------------------------------------------------------------------
from rtdl_num_embeddings import PeriodicEmbeddings
class BaselineNet(nn.Module):
    def __init__(self, input_dim=1, model_dim=256):
        super().__init__()
        self.pos_sim = PeriodicEmbeddings(input_dim, model_dim, lite=False)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 1),
        )
        
    def forward(self, sim):
        B, N, sim_dim = sim.size()
        emb = self.pos_sim(sim.reshape(-1, sim_dim)).reshape(B, N, -1)  # [B, N, D]
        h = emb.mean(dim=1)  # [B, D] simple mean pooling over documents
        v = self.mlp(h).squeeze(-1)  # [B,]
        return v

class ReplayBufferFIFO:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
        
    def push(self, states_cpu, q_low_val, q_delta_val, reward_val, beh_logp_val, qkey,
             doc_usage_ratio, num_chunk, num_token):
        self.buf.append({
            "states": states_cpu,            # CPU tensor [1, N, 1]
            "q_low": float(q_low_val),       # scalar float
            "q_delta": float(q_delta_val),   # scalar float
            "reward": float(reward_val),     # scalar float
            "beh_logp": float(beh_logp_val), # log b(a|s)
            "qkey": qkey,                    # string
            "dur": float(doc_usage_ratio),
            "num_chunk": int(num_chunk),
            "num_token": int(num_token),
        })
                 
    def sample(self, n):
        n = min(n, len(self.buf))
        return random.sample(self.buf, n)
        
    def uniform_fraction(self):
        # uniform transitions have beh_logp==0 (since Uniform pdf=1)
        if len(self.buf) == 0:
            return 0.0
        return sum(1 for x in self.buf if abs(x["beh_logp"]) < 1e-12) / len(self.buf)
        
    def __len__(self):
        return len(self.buf)

# Calculate Reward
def calc_reward(sampled_index_list, sorted_indices, nodes, eg):
    eg['context'] = ''
    num_chunk = 0
    for node_index in sampled_index_list:
        og_node_index = sorted_indices[node_index]
        eg['context'] += f'chunk {num_chunk}: {nodes[og_node_index].text}.' + '\n\n'
        num_chunk += 1
    llm_reward = float(tf_reward(g4_tokenizer, eg, c_type, 'rag', pipeline))
    return llm_reward, num_chunk, eg['context']

def init_model(model_path):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        device_map="auto",
    )
    return pipeline

def call_model(pipeline, messages):
    outputs = pipeline(messages, max_new_tokens=1024)
    response = outputs[0]["generated_text"][-1]['content']
    return response

def get_eg_doc(eg, c_type):
    file_name = eg['file']
    eg_file_path = f"../datasets/{context_length}/{c_type}/{file_name}"
    with open(eg_file_path, 'r', encoding='utf-8') as f:
        eg_txt = f.read()
    return [Document(text=eg_txt)]

def tf_reward(tokenizer, eg, data_name, model_name, pipeline, not_eval=False):
    # Create msgs
    msgs, prompt = create_msgs(
        tokenizer, eg, data_name=data_name, model_name=model_name
    )
    # Make prediction
    try:
        response = call_model(pipeline, msgs)
        if not_eval:
            return response, eg
    except Exception as e:
        print("ERROR:", e)
        if not_eval:
            return False, eg

    sample = {'prediction': response,
            'question': eg['question'],
            'ground_truth': eg['answer'],
            'query_type': eg['level']
            }
    
    score, sample = evaluate_example(sample, sample['query_type'])
    return score

def eval_single_valscore(eval_eg, embed_model, tokenizer, rag_pipeline, pipeline, adaptive_retriever):
    val_llm_reward_list = []
    val_doc_usage_list = []
    val_num_chunk_list = []
    val_num_token_list = []
    ours_eval_eg = eval_eg.copy()
    for eg in tqdm(ours_eval_eg):
        file_name = eg['file']

        if eg['type'] in ['novelette', 'novelette_fake', 'novel_fake', 'novel']:
            c_type = 'book'
        else:
            c_type = eg['type']

        eg_file_path = f"../datasets/{eg['length']}/{c_type}/{file_name}"
        if os.path.exists(eg_file_path):
            eg, doc_usage_ratio, num_chunk, num_token = process_example(eg, embed_model, tokenizer, rag_pipeline, pipeline, adaptive_retriever, c_type)
            llm_reward = float(tf_reward(g4_tokenizer, eg, c_type, 'rag', pipeline))
            val_llm_reward_list.append(llm_reward)
            val_doc_usage_list.append(doc_usage_ratio)
            val_num_chunk_list.append(num_chunk)
            val_num_token_list.append(num_token)
        
        gc.collect()
        torch.cuda.empty_cache()
    print("val_score:", np.mean(val_llm_reward_list))
    print("val_dur:", np.mean(val_doc_usage_list))
    print("val_num_chunk:", np.mean(val_num_chunk_list))
    print("val_num_token:", np.mean(val_num_token_list))
    return np.mean(val_llm_reward_list), np.mean(val_doc_usage_list), np.mean(val_num_chunk_list), np.mean(val_num_token_list)

def LC_eval_single_valscore(eval_eg, tokenizer, rag_pipeline, pipeline):
    LC_val_llm_reward_list = []
    LC_val_doc_usage_list = []
    LC_val_num_chunk_list = []
    LC_val_num_token_list = []
    LC_eval_eg = eval_eg.copy()
    for eg in tqdm(LC_eval_eg):
        file_name = eg['file']

        if eg['type'] in ['novelette', 'novelette_fake', 'novel_fake', 'novel']:
            c_type = 'book'
        else:
            c_type = eg['type']

        eg_file_path = f"../datasets/{eg['length']}/{c_type}/{file_name}"
        if os.path.exists(eg_file_path):
            with open(eg_file_path, 'r', encoding='utf-8') as f:
                eg_txt = f.read()
            eg['context'] = eg_txt

            llm_reward = float(tf_reward(tokenizer, eg, c_type, 'full', pipeline))
            num_token = compute_num_token(tokenizer, eg['context'])

            doc = get_eg_doc(eg, c_type)
            nodes = rag_pipeline.run(documents=doc, show_progress=False)  # document is splitted into passages
            num_chunk = len(nodes)
            doc_usage_ratio = num_chunk / len(nodes)

            LC_val_llm_reward_list.append(llm_reward)
            LC_val_doc_usage_list.append(doc_usage_ratio)
            LC_val_num_chunk_list.append(num_chunk)
            LC_val_num_token_list.append(num_token)
        
        gc.collect()
        torch.cuda.empty_cache()
    print("LC_val_score:", np.mean(LC_val_llm_reward_list))
    print("LC_val_dur:", np.mean(LC_val_doc_usage_list))
    print("LC_val_num_chunk:", np.mean(LC_val_num_chunk_list))
    print("LC_val_num_token:", np.mean(LC_val_num_token_list))
    return np.mean(LC_val_llm_reward_list), np.mean(LC_val_doc_usage_list), np.mean(LC_val_num_chunk_list), np.mean(LC_val_num_token_list)

def RAG_eval_single_valscore(eval_eg, tokenizer, rag_pipeline, pipeline):
    RAG_val_llm_reward_list = []
    RAG_val_doc_usage_list = []
    RAG_val_num_chunk_list = []
    RAG_val_num_token_list = []
    RAG_eval_eg = eval_eg.copy()
    for eg in tqdm(RAG_eval_eg):
        file_name = eg['file']

        if eg['type'] in ['novelette', 'novelette_fake', 'novel_fake', 'novel']:
            c_type = 'book'
        else:
            c_type = eg['type']

        eg_file_path = f"../datasets/{eg['length']}/{c_type}/{file_name}"
        if os.path.exists(eg_file_path):
            with open(eg_file_path, 'r', encoding='utf-8') as f:
                eg_txt = f.read()

            eg_doc = Document(text=eg_txt)
            documents = [eg_doc]
            nodes = rag_pipeline.run(documents=documents, show_progress=False)
            config = {
                "class_name": "SimpleHybridSearcher",
                "class_file": "simpleHybridSearcher",
                "remove_if_exists": False,
                "thread_num": 1,
                "rerank_size": 5,
                "vector_ratio": 0.5,
                "embed_model_name": "../embedding_models/gte-enzh-emb-large-v1.5",
                "rerank_model": "../embedding_models/gte-rerank-large-v1.5"
            }
            searcher = SimpleHybridSearcher(config, nodes)
            search_nodes = searcher.process(eg['question'])
            eg["context"] = ''
            for i, node in enumerate(search_nodes):
                node_txt = node.text
                eg["context"] += f'chunk {i}: {node_txt}.' + '\n\n'
            llm_reward = float(tf_reward(g4_tokenizer, eg, c_type, 'rag', pipeline))
            num_token = compute_num_token(tokenizer, eg['context'])

            RAG_val_llm_reward_list.append(llm_reward)
            RAG_val_doc_usage_list.append(len(search_nodes)/len(nodes))
            RAG_val_num_chunk_list.append(len(search_nodes))
            RAG_val_num_token_list.append(num_token)
        
        gc.collect()
        torch.cuda.empty_cache()
    print("RAG_val_score:", np.mean(RAG_val_llm_reward_list))
    print("RAG_val_dur:", np.mean(RAG_val_doc_usage_list))
    print("RAG_val_num_chunk:", np.mean(RAG_val_num_chunk_list))
    print("RAG_val_num_token:", np.mean(RAG_val_num_token_list))
    return np.mean(RAG_val_llm_reward_list), np.mean(RAG_val_doc_usage_list), np.mean(RAG_val_num_chunk_list), np.mean(RAG_val_num_token_list)

def process_example(eg, embed_model, tokenizer, rag_pipeline, pipeline, adaptive_retriever, c_type):
    # Get query embedding
    query_embed = embed_model.get_text_embedding(eg['question'])
    query_embed = torch.tensor(query_embed) # [embed_size,]

    # Get document embedding
    doc = get_eg_doc(eg, c_type)
    nodes = rag_pipeline.run(documents=doc, show_progress=False)  # document is splitted into passages
    doc_embed = torch.tensor([node.embedding for node in nodes]) # [num_passages, embed_size]

    # Get cosine similarity
    q2d_cos_sim = torch.nn.functional.cosine_similarity(query_embed, doc_embed)
    sorted_q2d_cos_sim, sorted_indices = torch.sort(q2d_cos_sim)
    states = sorted_q2d_cos_sim.reshape(1, -1, 1).to(device)
    
    adaptive_retriever.eval()
    with torch.no_grad():
        num_all_chunk = len(nodes)
        
        # adaptive_retriever action
        q_low_alpha, q_low_beta, q_delta_alpha, q_delta_beta = adaptive_retriever(states)

        q_low = q_low_alpha / (q_low_alpha + q_low_beta)
        q_delta = q_delta_alpha / (q_delta_alpha + q_delta_beta)

        q_high = torch.clamp(q_low + q_delta, max=1.0)

        low_kth = round((len(nodes) - 1) * q_low.item()) + 1
        high_kth = round((len(nodes) - 1) * q_high.item()) + 1
        sampled_index_list = []
        for range_kth in reversed(range(low_kth-1, high_kth)):
            sampled_index_list.append(range_kth)

        eg['context'] = ''
        num_chunk = 0
        for node_index in sampled_index_list:
            og_node_index = sorted_indices[node_index]
            eg['context'] += f'chunk {num_chunk}: {nodes[og_node_index].text}.' + '\n\n'
            num_chunk += 1

        num_token = compute_num_token(g4_tokenizer, eg['context'])

    del nodes
    torch.cuda.empty_cache()

    return eg, num_chunk/num_all_chunk, num_chunk, num_token

def make_qkey(eg, c_type):
    # choose what uniquely identifies a "query"
    # e.g., (question, type, level), or hash it if long
    return f"{eg['question']}||{c_type}||{eg.get('level','')}"

def compute_num_token(tokenizer, input):
    tokens = tokenizer.encode(input)
    tokens = truncate_input(tokens, 128_000 - 1000, manner="middle")
    return len(tokens)

if __name__ == "__main__":    
    print("========loading the model=======")
    pipeline = init_model(model_path)
    print("=======loading done!=======")

    train_dataset_path = f'../datasets/traintest/{context_length}_{query_type}_train.jsonl'
    test_dataset_path = f'../datasets/traintest/{context_length}_{query_type}_test.jsonl'
    if os.path.exists(train_dataset_path) and os.path.exists(test_dataset_path):
        with open(train_dataset_path, "r", encoding="utf-8") as f:
            total_examples = json.load(f)
        with open(test_dataset_path, "r", encoding="utf-8") as f:
            total_examples_test = json.load(f)
    else:
        total_examples = []
        total_examples_test = []
        for c_type in ['financial', 'book', 'paper']:
            c_examples = []
            data_path = f'../datasets/query/{context_length}_{c_type}_{query_type}.jsonl'
            c_examples.extend(load_data(data_path))

            # shuffle per c_type (seed already set above)
            shuffle(c_examples)

            n_train = int(len(c_examples) * 0.8)
            total_examples.extend(c_examples[:n_train])
            total_examples_test.extend(c_examples[n_train:])

        # (optional) shuffle the combined splits
        shuffle(total_examples)
        shuffle(total_examples_test)

        os.makedirs(os.path.dirname(train_dataset_path), exist_ok=True)
        with open(train_dataset_path, "w", encoding="utf-8") as f:
            json.dump(total_examples, f, ensure_ascii=False, indent=2)

        with open(test_dataset_path, "w", encoding="utf-8") as f:
            json.dump(total_examples_test, f, ensure_ascii=False, indent=2)

    g4_tokenizer = tiktoken.encoding_for_model("gpt-4")

    ckpt_save_base_path = f"./{args.save_path}/ckpt/{eval_model}/{args.run_name}"
    if not os.path.exists(ckpt_save_base_path) or args.debug or full_eval_only:
        print('Creating Adaptive Retriever')
        adaptive_retriever = OneShotTransformer().to(device)
        baseline_net = BaselineNet().to(device)
        adaptive_retriever_optimizer = torch.optim.Adam(list(adaptive_retriever.parameters()) + list(baseline_net.parameters()), lr=3e-4)

        if args.checkpoint == '':
            print('Not resume. Training from scratch.')
            start_epoch = 0
        else:
            print('Resume. Call the pretrained adaptive_retriever...')
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            
            adaptive_retriever.load_state_dict(checkpoint['adaptive_retriever_state_dict'])
            baseline_net.load_state_dict(checkpoint['baseline_state_dict'])
            adaptive_retriever_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            print('load checkpoint from %s' % args.checkpoint)
            ckpt_save_base_path = os.path.dirname(args.checkpoint)
            start_epoch = checkpoint['epoch'] + 1

        # Set the retriever
        transformations = []
        splitter = SentenceSplitter(
            include_metadata=True, include_prev_next_rel=True,
            chunk_size=600,
            chunk_overlap=100,
            separator=' ',       
            paragraph_separator='\n\n\n', secondary_chunking_regex='[^,.;。？！]+[,.;。？！]?')
        transformations.append(splitter)
        embed_model = HuggingFaceEmbedding(model_name="../embedding_models/gte-enzh-emb-large-v1.5")
        transformations.append(embed_model)
        rag_pipeline = IngestionPipeline(
            transformations=transformations
        )

        # Initialize wandb
        wandb.init(project="LDAR", 
                   entity="your_wandb_account_name",
                   name=args.run_name)

        if full_eval_only:
            if query_type == 'hallu':   # Use all for hallu
                total_examples_test = total_examples_test + total_examples

            # LDAR
            llm_eval_reward, val_doc_usage, val_num_chunk, val_num_token = eval_single_valscore(total_examples_test, embed_model, g4_tokenizer, rag_pipeline, pipeline, adaptive_retriever)
            avg_log = {
                # eval & usage stats
                "test/reward_mean": float(llm_eval_reward),
                "test/doc_usage_ratio_mean": float(val_doc_usage),
                "test/num_chunk": float(val_num_chunk),
                "test/num_token": float(val_num_token),
            }
            wandb.log(avg_log, step=0)

            # LC
            LC_llm_eval_reward, LC_val_doc_usage, LC_val_num_chunk, LC_val_num_token = LC_eval_single_valscore(total_examples_test, g4_tokenizer, rag_pipeline, pipeline)
            LC_avg_log = {
                # eval & usage stats
                "LC/reward_mean": float(LC_llm_eval_reward),
                "LC/doc_usage_ratio_mean": float(LC_val_doc_usage),
                "LC/num_chunk": float(LC_val_num_chunk),
                "LC/num_token": float(LC_val_num_token),
            }
            wandb.log(LC_avg_log, step=0)

            # RAG
            RAG_llm_eval_reward, RAG_val_doc_usage, RAG_val_num_chunk, RAG_val_num_token = RAG_eval_single_valscore(total_examples_test, g4_tokenizer, rag_pipeline, pipeline)
            RAG_avg_log = {
                # eval & usage stats
                "RAG/reward_mean": float(RAG_llm_eval_reward),
                "RAG/doc_usage_ratio_mean": float(RAG_val_doc_usage),
                "RAG/num_chunk": float(RAG_val_num_chunk),
                "RAG/num_token": float(RAG_val_num_token),
            }
            wandb.log(RAG_avg_log, step=0)
            exit()

        # ================== LDAR with total_examples =======
        max_epoch = 30
        batch_size = 128 if not args.debug else 4
        global_update_step = 0
        global_logging_step = 0

        ####### entropy scheduling and Replay Buffer #######
        entropy_coef_start = 5e-1
        entropy_coef_end   = 1e-4
        num_steps_total = max_epoch * len(total_examples)
        global_env_step = 0

        warmup_env_steps = int(0.1 * num_steps_total)
        replay_capacity  = int(0.2 * num_steps_total)  # FIFO size (bigger => slower replacement)
        replay_buffer = ReplayBufferFIFO(replay_capacity)

        baseline_coef = 0.5  # weight for baseline_loss relative to adaptive_retriever_loss
        ###################################################

        for epoch in tqdm(range(start_epoch, max_epoch)):
            shuffle(total_examples)
            running_rewards = []
            running_dur = []
            running_num_token = []
            running_num_chunk = []
            running_entropy = []
            for ith_eg, eg in tqdm(enumerate(total_examples), total=len(total_examples)):
                global_env_step += 1
                
                # entropy schedule
                frac = (global_env_step - 1) / (num_steps_total - 1)
                entropy_coef = entropy_coef_start + frac * (entropy_coef_end - entropy_coef_start)
                
                if eg['type'] in ['novelette', 'novelette_fake', 'novel_fake', 'novel']:
                    c_type = 'book'
                else:
                    c_type = eg['type']

                # Get query embedding
                query_embed = embed_model.get_text_embedding(eg['question'])
                query_embed = torch.tensor(query_embed) # [embed_size,]

                # Get document embedding
                doc = get_eg_doc(eg, c_type)
                nodes = rag_pipeline.run(documents=doc, show_progress=False)  # document is splitted into passages
                doc_embed = torch.tensor([node.embedding for node in nodes]) # [num_passages, embed_size]

                # Get cosine similarity
                q2d_cos_sim = torch.nn.functional.cosine_similarity(query_embed, doc_embed)
                sorted_q2d_cos_sim, sorted_indices = torch.sort(q2d_cos_sim)
                states = sorted_q2d_cos_sim.reshape(1, -1, 1).to(device)    # [B, N, dim] = [1, N, 1]
                
                # ==========================Collect Episode===========================
                num_all_chunk = len(nodes)
                
                # adaptive_retriever action
                q_low_alpha, q_low_beta, q_delta_alpha, q_delta_beta = adaptive_retriever(states)

                q_low_dist = Beta(q_low_alpha, q_low_beta)
                q_delta_dist = Beta(q_delta_alpha, q_delta_beta)

                if global_env_step <= warmup_env_steps:
                    # (1) uniform action
                    q_low = torch.rand(1, device=device)   # Uniform(0, 1)
                    q_delta = torch.rand(1, device=device)
                    beh_logp = 0.0                         # log Uniform pdf = log(1)=0
                else:
                    # (2) adaptive_retriever action
                    q_low = q_low_dist.sample()
                    q_delta = q_delta_dist.sample()
                    beh_logp = float((q_low_dist.log_prob(q_low) + q_delta_dist.log_prob(q_delta)).item())
                q_high = torch.clamp(q_low + q_delta, max=1.0)

                low_kth  = round((num_all_chunk - 1) * q_low.item()) + 1
                high_kth = round((num_all_chunk - 1) * q_high.item()) + 1
                
                sampled_index_list = []
                for range_kth in reversed(range(low_kth - 1, high_kth)):
                    sampled_index_list.append(range_kth)

                llm_reward, num_chunk, eg_context = calc_reward(sampled_index_list, sorted_indices, nodes, eg)
                num_token = compute_num_token(g4_tokenizer, eg_context)
                doc_usage_ratio = num_chunk / num_all_chunk
                
                running_rewards.append(llm_reward)
                running_dur.append(doc_usage_ratio)
                running_num_token.append(num_token)
                running_num_chunk.append(num_chunk)
                running_entropy.append(curr_ent)

                # -------------------- push transition to FIFO buffer --------------------
                qkey = make_qkey(eg, c_type)
                replay_buffer.push(
                    states_cpu=states.detach().cpu(),
                    q_low_val=q_low.item(),
                    q_delta_val=q_delta.item(),
                    reward_val=llm_reward,
                    beh_logp_val=beh_logp,
                    qkey=qkey,
                    doc_usage_ratio=doc_usage_ratio,
                    num_chunk=num_chunk,
                    num_token=num_token
                )
                del nodes
                gc.collect()
                torch.cuda.empty_cache()
                
                # ==========================Update adaptive_retriever + Baseline===========================
                if len(replay_buffer) >= batch_size:
                    batch = replay_buffer.sample(batch_size)

                    logp_list, ent_list, w_list = [], [], []
                    rewards_list, baseline_preds_list = [], []
                    dur_list, nchunk_list, ntok_list = [], [], []

                    adaptive_retriever.train()
                    baesline_net.train()

                    for tr in batch:
                        st = tr["states"].to(device)  # (1,N,1)
                        a_val = torch.tensor([tr["q_low"]], device=device)
                        a_val2 = torch.tensor([tr["q_delta"]], device=device)
                        r_val = torch.tensor([tr["reward"]], device=device)
                        beh_lp = torch.tensor([tr["beh_logp"]], device=device)
                        
                        # current adaptive_retriever logp/entropy on stored action
                        alpha, beta, alpha2, beta2 = adaptive_retriever(st)
                        dist = Beta(alpha, beta)
                        dist2 = Beta(alpha2, beta2)
                        
                        logp_cur = dist.log_prob(a_val) + dist2.log_prob(a_val2)
                        ent_cur  = dist.entropy() + dist2.entropy()
                        
                        # importance weight: w = pi_cur / b
                        w = (logp_cur - beh_lp).exp().detach()
                        
                        # baseline prediction
                        baseline_pred = baseline(st)  # shape (1,)
                        baseline_preds_list.append(baseline_pred.squeeze())
                        
                        logp_list.append(logp_cur.squeeze())
                        ent_list.append(ent_cur.squeeze())
                        w_list.append(w.squeeze())
                        rewards_list.append(r_val.squeeze())
                        dur_list.append(tr["dur"])
                        nchunk_list.append(tr["num_chunk"])
                        ntok_list.append(tr["num_token"])

                    logp_sums = torch.stack(logp_list)      # [B]
                    ent_avgs  = torch.stack(ent_list)       # [B]
                    w_s       = torch.stack(w_list)         # [B]
                    rewards   = torch.stack(rewards_list)   # [B]
                    baseline_preds = torch.stack(baseline_preds_list) # [B]

                    # Advantage using state-value baseline: A = r - V(s)
                    advantage = rewards - baseline_preds.detach()
                    
                    # retriever loss (with importance weights)
                    adaptive_retriever_loss = -(w_s * logp_sums * advantage).mean()
                    
                    # Baseline loss (MSE between V(s) and reward)
                    baseline_loss = F.mse_loss(baseline_preds, rewards)
                    
                    # Entropy for exploration regularization
                    entropy = ent_avgs.mean()
                    
                    # Total loss: retriever + baseline - entropy term
                    loss = adaptive_retriever_loss + baseline_coef * baseline_loss - entropy_coef * entropy

                    adaptive_retriever_optimizer.zero_grad()
                    loss.backward()
                    adaptive_retriever_optimizer.step()

                    avg_log = {
                        # training stats
                        "train/reward_mean": float(np.mean(running_rewards)),
                        "train/adaptive_retriever_loss": float(adaptive_retriever_loss.item()),
                        "train/baseline_loss": float(baseline_loss.item()),
                        "train/epoch": int(epoch),
                        
                        "train/doc_usage_ratio_mean": float(np.mean(running_dur)),
                        "train/num_chunk": float(np.mean(running_num_chunk)),
                        "train/num_token": float(np.mean(running_num_token)),

                        "train/entropy_coef": float(entropy_coef),
                        "train/entropy": float(torch.stack(running_entropy).mean().item()),

                        "train/replay_size": int(len(replay_buffer)),
                        "train/uniform_frac": float(replay_buffer.uniform_fraction()),
                    }

                    if global_update_step % 32 == 0:
                        global_logging_step += 1
                        print(f"[step {global_update_step}] avg_log:", avg_log)
                        wandb.log(avg_log, step=global_logging_step)

                        running_rewards.clear()
                        running_dur.clear()
                        running_num_chunk.clear()
                        running_num_token.clear()
                        running_entropy.clear()

                    gc.collect()
                    torch.cuda.empty_cache()

                    global_update_step += 1

                    if args.debug:
                        print('Evaluating for debugging...')
                        test_eg = random.sample(total_examples_test, 1)
                        llm_eval_reward, val_doc_usage, val_num_chunk, val_num_token = eval_single_valscore(test_eg, embed_model, g4_tokenizer, rag_pipeline, pipeline, adaptive_retriever)
                        LC_llm_eval_reward, LC_val_doc_usage, LC_val_num_chunk, LC_val_num_token = LC_eval_single_valscore(test_eg, g4_tokenizer, rag_pipeline, pipeline)
                        RAG_llm_eval_reward, RAG_val_doc_usage, RAG_val_num_chunk, RAG_val_num_token = RAG_eval_single_valscore(test_eg, g4_tokenizer, rag_pipeline, pipeline)
                        print("Finished Debugging!")
                        exit()

            if args.val_while_training:
                shuffle(total_examples_test)

                # Ours
                llm_eval_reward, val_doc_usage, val_num_chunk, val_num_token = eval_single_valscore(total_examples_test, embed_model, g4_tokenizer, rag_pipeline, pipeline, adaptive_retriever)
                avg_log = {
                    # eval & usage stats
                    "test/reward_mean": float(llm_eval_reward),
                    "test/doc_usage_ratio_mean": float(val_doc_usage),
                    "test/num_chunk": float(val_num_chunk),
                    "test/num_token": float(val_num_token),
                }
                print(f"[step {global_update_step}] avg_log:", avg_log)
                wandb.log(avg_log, step=global_logging_step)

                if epoch == 0:  # just eval once
                    # LC
                    LC_llm_eval_reward, LC_val_doc_usage, LC_val_num_chunk, LC_val_num_token = LC_eval_single_valscore(total_examples_test, g4_tokenizer, rag_pipeline, pipeline)
                    LC_avg_log = {
                        # eval & usage stats
                        "LC/reward_mean": float(LC_llm_eval_reward),
                        "LC/doc_usage_ratio_mean": float(LC_val_doc_usage),
                        "LC/num_chunk": float(LC_val_num_chunk),
                        "LC/num_token": float(LC_val_num_token),
                    }
                    wandb.log(LC_avg_log, step=global_logging_step)

                    # RAG
                    RAG_llm_eval_reward, RAG_val_doc_usage, RAG_val_num_chunk, RAG_val_num_token = RAG_eval_single_valscore(total_examples_test, g4_tokenizer, rag_pipeline, pipeline)
                    RAG_avg_log = {
                        # eval & usage stats
                        "RAG/reward_mean": float(RAG_llm_eval_reward),
                        "RAG/doc_usage_ratio_mean": float(RAG_val_doc_usage),
                        "RAG/num_chunk": float(RAG_val_num_chunk),
                        "RAG/num_token": float(RAG_val_num_token),
                    }
                    wandb.log(RAG_avg_log, step=global_logging_step)

            # Save adaptive_retriever per epoch
            ckpt_save_path = f"{ckpt_save_base_path}/adaptive_retriever_epoch_{epoch}.pt"
            os.makedirs(os.path.dirname(ckpt_save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'adaptive_retriever_state_dict': adaptive_retriever.state_dict(),
                'baseline_state_dict': baseline_net.state_dict(),
                'optimizer_state_dict': adaptive_retriever_optimizer.state_dict()
            }, ckpt_save_path)
            print(f"[INFO] adaptive_retriever saved at {ckpt_save_path}")
