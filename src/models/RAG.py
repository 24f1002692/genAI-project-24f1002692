"""
src/models/RAG_pretrained.py

RAG (Retrieval-Augmented Generation) predictor for the Smart MCQ Solver.
Pipeline:
    question --> FAISS search over Wikipedia passages (MiniLM retriever)
             --> top-k real passages retrieved
             --> prompt = context + question + options
             --> Flan-T5 reads context, ranks options by logits
             --> top-3 predicted letters (for MAP@3)

Usage from a notebook:
    import sys, os
    sys.path.append(os.path.abspath("../../"))
    from src.models.RAG_pretrained import RAGPredictor

    rag = RAGPredictor(wiki_index, wiki_index_df)
    top3 = rag.predict_top3(row)   # row = a DataFrame row with 'prompt', 'A'..'E'
"""

import torch
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

OPTIONS = ['A', 'B', 'C', 'D', 'E']

class RAG_predictor:
  def __init__(
    self, wiki_index, wiki_index_df,
    retriever_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    reader_name: str = "google/flan-t5-base",
    device: str = None
  ):
    self.wiki_index = wiki_index
    self.wiki_index_df = wiki_index_df
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading retriever: {retriever_name} ...")
    self.retriever = SentenceTransformer(retriever_name)

    print(f"Loading reader: {reader_name} ...")
    self.tokenizer = T5Tokenizer.from_pretrained(reader_name)
    self.model = T5ForConditionalGeneration.from_pretrained(reader_name).to(self.device)
    self.model.eval()
    print("RAGPredictor ready.")

    def retrieve(self, query: str, k: int = 3) -> list[str]:
      """Embed query, search Wikipedia FAISS index, return top-k article texts."""
      query_vec = self.retriever.encode(
          [query], convert_to_numpy=True, normalize_embeddings=True
      ).astype('float32')

      _, indices = self.wiki_index.search(query_vec, k)

      passages = []
      for idx in indices[0]:
          if 0 <= idx < len(self.wiki_index_df):
              row = self.wiki_index_df.iloc[idx]
              passages.append(f"{row['title']}: {row['text'][:300]}")
      return passages

    def build_prompt(self, row, passages: list[str]) -> str:
      context = "\n".join(f"- {p}" for p in passages)
      options = "  ".join(f"{opt}) {row[opt]}" for opt in OPTIONS)
      return (
          f"Context:\n{context}\n\n"
          f"Question: {row['prompt']}\n"
          f"{options}\n\n"
          f"Which option is correct? Answer with only the letter (A, B, C, D, or E)."
      )

    def flan_top3(self, prompt_text: str) -> str:
      """Run Flan-T5 once, rank all 5 options by logits, return top-3 letters."""
      inputs = self.tokenizer(
          prompt_text, return_tensors="pt", max_length=512, truncation=True
      ).to(self.device)

      with torch.no_grad():
        out = self.model.generate(
            **inputs,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

      logits = out.scores[0][0]
      option_token_ids = {
          opt: self.tokenizer(opt, add_special_tokens=False).input_ids[0]
          for opt in OPTIONS
      }
      option_logits = {opt: logits[tid].item() for opt, tid in option_token_ids.items()}
      ranked = sorted(option_logits, key=option_logits.get, reverse=True)
      return " ".join(ranked[:3])

    def predict_top3(self, row, k: int = 3) -> str:
      """End-to-end: retrieve passages, build prompt, get Flan-T5's top-3 letters."""
      passages = self.retrieve(row['prompt'], k=k)
      prompt_text = self.build_prompt(row, passages)
      return self.flan_top3(prompt_text)
