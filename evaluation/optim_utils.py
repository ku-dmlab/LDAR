import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI

from rtdl_num_embeddings import PeriodicEmbeddings

class OneShotTransformer(nn.Module):
    def __init__(self, model_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.model_dim = model_dim

        self.pos_sim = PeriodicEmbeddings(1, model_dim, lite=False)

        enc_layer = nn.TransformerEncoderLayer(model_dim, num_heads, 4*model_dim, norm_first=True, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=nn.LayerNorm(model_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.raw_mlp = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim), nn.GELU(),
            nn.Linear(model_dim, model_dim), nn.GELU()
        )

        self.q_low_delta_alpha_beta = nn.Linear(model_dim, 4)

    def forward(self, sim):
        B, N, sim_dim = sim.size()  # N: number of documents

        sim_embeddings = self.pos_sim(sim.reshape(-1, sim_dim)).reshape(B, N, self.model_dim)  # [B, N, model_dim]

        num_emb = 1 # the number of stacked embeddings
        stacked_inputs = torch.stack(
            (
            sim_embeddings, 
            # you can add any embedding here.
            ), dim=1
        ).permute(0, 2, 1, 3).reshape(B, num_emb*N, self.model_dim)

        cls = self.cls_token.expand(B, 1, self.model_dim)  # [B, 1, model_dim]
        x = torch.cat([cls, stacked_inputs], dim=1)        # [B, 1+num_emb, model_dim]

        out = self.encoder(x)              # [B, 1,+num_emb, model_dim]
        cls_h = out[:, 0]                  # [B, model_dim] – global summary

        raw_token_feats = self.raw_mlp(sim)  # [B, num_emb, model_dim]
        raw_h = raw_token_feats.mean(dim=1)  # [B, model_dim]

        combined = cls_h + raw_h            # [B, model_dim]
        g = self.head(combined)             # [B, model_dim]

        q_low_delta_alpha_beta_logits = self.q_low_delta_alpha_beta(g)
        q_low_alpha_logits = q_low_delta_alpha_beta_logits[:, 0]
        q_low_beta_logits = q_low_delta_alpha_beta_logits[:, 1]
        q_delta_alpha_logits = q_low_delta_alpha_beta_logits[:, 2]
        q_delta_beta_logits = q_low_delta_alpha_beta_logits[:, 3]

        # Apply softplus to ensure the outputs are strictly positive.
        q_low_alpha = F.softplus(q_low_alpha_logits) + 1e-5  # Ensure alpha > 0
        q_low_beta = F.softplus(q_low_beta_logits) + 1e-5  # Ensure beta > 0
        q_delta_alpha = F.softplus(q_delta_alpha_logits) + 1e-5  # Ensure alpha > 0
        q_delta_beta = F.softplus(q_delta_beta_logits) + 1e-5  # Ensure beta > 0

        return q_low_alpha.squeeze(), q_low_beta.squeeze(), q_delta_alpha.squeeze(), q_delta_beta.squeeze()


def call_gpt(model, messages, retry_num=5, retry_interval=5):
    client = OpenAI(
        api_key="<your api key>",
        organization='',
    )
    for _ in range(retry_num):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            text = completion.choices[0].message.content
            if isinstance(text, str) and text:
                return text
        except:
            time.sleep(retry_interval)
            print(f'retry after {retry_interval} seconds')
    return False


def get_score_one_llm(pred, label, query, query_type, model='qwen-max-allinone') -> float:
    if query_type == 'hallu':
        prompt = f'''You need to help me determine whether an AI model is hallucinating. I will provide you with a question, a ground truth answer, and a prediction from the model. If the prediction from the model and the ground truth answer are basically consistent, and it is determined that this question is not mentioned in the text, then the model is deemed not to be hallucinating and the answer is considered correct. If it is correct, you should only output True; if it is incorrect, only output False.

[Query] {query}

[Groundtruth Answer] {label}

[AI Assistant's Answer] {pred}

Now, start your judgement:'''
    else:
        prompt = f'''I will provide you with a question and its groundtruth answer, as well as an answer from an AI assistant. You need to judge whether the AI assistant's answer is correct based on the groundtruth answer. If it is correct, you should only output True; if it is incorrect, only output False.

[Query] {query}

[Groundtruth Answer] {label}

[AI Assistant's Answer] {pred}

Now, start your judgment:'''
    msg = [
        {
            "role": "system",
            "content": "You are a discriminator that judges whether the predictions to questions are correct.",
        },
        {"role": "user", "content": prompt},
    ]
    for _ in range(20):
        try:
            # response = call_qwen(model=model, messages=msg)
            response = call_gpt('gpt-4o', msg)
            if not response:
                return False
            else:
                if 'true' in response.lower():
                    return 1.0
                else:
                    return 0.0
        except:
            time.sleep(5)

def process_example(sample, query_type):
    pred = sample['prediction']
    query = sample['question']
    label = sample['ground_truth']
    score = get_score_one_llm(pred=pred, label=label, query=query, query_type=query_type)
    return score, sample
