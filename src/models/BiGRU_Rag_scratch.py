"""
src/models/BiGRU_RAG.py
 
From-scratch Bidirectional GRU + Attention Pooling model for the Smart MCQ
Solver, trained on RAG-augmented input (prompt + retrieved Wikipedia context + option), instead of a pretrained transformer.
 
Pipeline:
    prompt --> FAISS search (MiniLM retriever) --> top-k Wikipedia passages
           --> rag_context (cached per row)
           --> encode_pair(prompt, rag_context, option) --> token ids + mask
           --> OptionScorer (BiGRU + attention pooling) --> one score/option
           --> top-3 predicted letters (for MAP@3) """

import re
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

OPTIONS = ['A', 'B', 'C', 'D', 'E']
 
MAX_LEN = 175
CONTEXT_MAX_TOKENS = 50
MIN_FREQ = 2

# Word dropout: randomly replaces ~10% of input tokens with <unk> during training.
# Prevents the model from memorizing/shortcut-learning on a few fixed keywords —
# forces it to rely on the full context instead of any single token.
WORD_DROPOUT_PROB = 0.10

# ---------------------------------------------------------------------------
# RAG context retrieval (FAISS + MiniLM), cached once per split
# ---------------------------------------------------------------------------
""" Encode all prompts in df in batches, 
    FAISS-search once per batch,
    return a list[str] of concatenated top-k Wikipedia passages 
    (one string per row)."""

def build_rag_context(
        df, retriever, wiki_index, wiki_index_df, k:int=3, batch_size:int=64, passage_chars: int = 400
) -> list[str]:
    prompts = df['prompt'].astype(str).tolist()   # list of all prompts given
    contexts = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="retrieving context"):
        batch = prompts[i: i+batch_size]
        query_vecs = retriever.encode(
            batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype('float32')

        _, indices = wiki_index.search(query_vecs, k)
        for row_indices in indices:     # indices is a list of indices list(named as row_indicies)
            passages = []
            for idx in row_indices:
                if 0 <= idx < len(wiki_index_df):
                    row = wiki_index_df.iloc[idx]
                    passages.append(f"{row['title']}: {row['text'][:passage_chars]}")     #  3 passages

            contexts.append(" ".join(passages))

    return contexts


# --------------------------------------------------------------------
# Tokenizer + vocab (built from scratch, no pretrained tokenizer)
# ------------------------------------------------------------------
def simple_tokenize(text) -> list[str]:
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)   # everything except a-z0-9 and whitespace replaced with a single space.
    return text.split()   # return a list of words of the given text


def build_vocab(train_split, min_freq:int = MIN_FREQ) -> dict:
    counter = Counter()
    for _, row in train_split.iterrows():
        for col in ['prompt', 'rag_context'] + OPTIONS:
            counter.update(simple_tokenize(row[col]))

    vocab = {'<pad>': 0, '<unk>': 1, '<sep>': 2}   # vocab with special tokens
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


"""Encode prompt + <sep> + rag_context + <sep> + option into a fixed-length
    id sequence + attention mask."""
def encode_pair(
        prompt, context, option, vocab, max_len:int=MAX_LEN, word_dropout:bool=False
):
    prompt_tokens = simple_tokenize(prompt)
    context_tokens = simple_tokenize(context)[:CONTEXT_MAX_TOKENS]
    option_tokens = simple_tokenize(option)

    tokens = prompt_tokens + ['<sep>'] + context_tokens + ['<sep>'] + option_tokens
    tokens = tokens[:max_len]

    ids = [vocab.get(t, 1) for t in tokens]   # get ids for vocab, if token not in vocab, give it 1 - unknown token <unk>
    if word_dropout:   # true while training for regularisation, reduce shortcut learning/memoization
        import random
        ids = [1 if (i > 2 and random.random() < WORD_DROPOUT_PROB) else i for i in ids]   
        # i > 2 (apply dropout only on regular token not on special ones) and based on condition, replace any real word token with <unk> if random.random < word drop prob for the token
    
    # pad if the ids len is less than MAX_LEN, if more than then pad is itself 0
    mask = [1] * len(ids)
    pad = max_len - len(ids)
    ids += [0] * pad
    mask += [0] * pad
    return ids, mask


# ---------------------------------------------------------------------------
# Dataset — one row -> 5 (prompt+context+option) sequences, one per option
# ---------------------------------------------------------------------------
class MCQDataset(Dataset):
    """augment=True: shuffles option order + applies word dropout (train only).
    augment=False: keeps A..E order, no dropout (val/test)."""

    def __init__(self, df, vocab, max_len:int = MAX_LEN, augment:bool=False):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        has_label = 'answer' in row and row['answer'] == row['answer']  # NaN-safe (NaN == NaN is False)
        context = row['rag_context'] if row['rag_context'] == row['rag_context'] else ""

        if self.augment:
            perm = np.random.permutation(5)   # 0-4 number in shuffled order
            option_order = [OPTIONS[p] for p in perm]    # shuffle options - new options order
            if has_label:
                orig_pos = OPTIONS.index(row['answer'])
                label = int(np.where(perm == orig_pos)[0][0])   # np.where return a tuple which has an array at index 0
            else:
                label = -1
        else:
            option_order = OPTIONS    # if augment is false (for val/test set)
            label = OPTIONS.index(row['answer']) if has_label else -1

        all_ids, all_mask = [], []
        for opt in option_order:
            ids, mask = encode_pair(row['prompt'], context, row[opt], self.vocab,
                                     self.max_len, word_dropout=self.augment)
            all_ids.append(ids)
            all_mask.append(mask)
        return (
            torch.tensor(all_ids, dtype=torch.long),
            torch.tensor(all_mask, dtype=torch.float),
            label,
        )

# ---------------------------------------------------------------------------
# Model — Bidirectional GRU + additive attention pooling, shared across options
# ---------------------------------------------------------------------------
class OptionScorer(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=64, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, ids, mask):
        emb = self.embed_drop(self.embedding(ids))
        out, _ = self.gru(emb)
        attn_scores = self.attn(out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        pooled = (out * attn_weights).sum(dim=1)
        return pooled
 
    def forward(self, ids, mask):
        B, O, L = ids.shape
        ids = ids.view(B * O, L)
        mask = mask.view(B * O, L)
        pooled = self.encode(ids, mask)
        scores = self.fc(pooled).squeeze(-1)
        return scores.view(B, O)