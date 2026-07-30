"""
app.py — Gradio deployment for the from-scratch BiGRU-RAG MCQ Solver.

Lives at the repo root. Reuses the model class + tokenizer functions from
src/models/BiGRU_Rag_scratch.py, and loads the trained checkpoint + vocab
from outputs/BiGRU_Rag/.
"""

import spaces  # must be imported before torch/CUDA-related packages

import os
import sys
import pickle
import torch
import torch.nn.functional as F
import gradio as gr

# Make src/ importable (app.py sits at repo root, so this is just the repo root itself)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.BiGRU_Rag_scratch import OPTIONS, encode_pair, OptionScorer

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "BiGRU_Rag")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_scratch_rag_model.pt")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.pkl")

device = "cuda" if torch.cuda.is_available() else "cpu"

with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

model = OptionScorer(vocab_size=len(vocab)).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

@spaces.GPU
def predict(prompt, context, opt_a, opt_b, opt_c, opt_d, opt_e):
    row = {"prompt": prompt, "A": opt_a, "B": opt_b, "C": opt_c, "D": opt_d, "E": opt_e}
    context = context or ""

    all_ids, all_mask = [], []
    for opt in OPTIONS:
        ids, mask = encode_pair(row["prompt"], context, row[opt], vocab)
        all_ids.append(ids)
        all_mask.append(mask)

    ids_t = torch.tensor([all_ids], dtype=torch.long).to(device)
    mask_t = torch.tensor([all_mask], dtype=torch.float).to(device)

    with torch.no_grad():
        scores = model(ids_t, mask_t)
        probs = F.softmax(scores, dim=1).squeeze(0).cpu().tolist()

    ranked = sorted(zip(OPTIONS, probs), key=lambda x: x[1], reverse=True)
    top3_letters = " ".join([opt for opt, _ in ranked[:3]])

    breakdown = "\n".join([f"{opt}: {prob*100:.1f}%" for opt, prob in ranked])
    return top3_letters, breakdown

with gr.Blocks(title="BiGRU-RAG MCQ Solver") as demo:
    gr.Markdown(
        "# BiGRU-RAG MCQ Solver\n"
        "A from-scratch Bidirectional GRU + Attention model trained on "
        "RAG-augmented MCQ data. Enter a question, optional context, "
        "and 5 options — the model ranks them and returns its top-3 guesses."
    )

    prompt_in = gr.Textbox(label="Question (prompt)", lines=2)
    context_in = gr.Textbox(
        label="Context (optional — paste relevant passage text here; leave blank if none)",
        lines=3,
    )

    with gr.Row():
        a_in = gr.Textbox(label="Option A")
        b_in = gr.Textbox(label="Option B")
    with gr.Row():
        c_in = gr.Textbox(label="Option C")
        d_in = gr.Textbox(label="Option D")
    with gr.Row():
        e_in = gr.Textbox(label="Option E")

    submit_btn = gr.Button("Predict", variant="primary")

    top3_out = gr.Textbox(label="Top-3 Predicted Options")
    breakdown_out = gr.Textbox(label="Full Score Breakdown", lines=5)

    submit_btn.click(
        fn=predict,
        inputs=[prompt_in, context_in, a_in, b_in, c_in, d_in, e_in],
        outputs=[top3_out, breakdown_out],
    )

if __name__ == "__main__":
    demo.launch()