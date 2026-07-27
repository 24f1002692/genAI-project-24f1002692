"""
src/inference/LoRA_MC_inference.py

Runs the LoRA multiple-choice predictor (src/models/LoRA_MC.py) over a full
dataset in batches and returns/saves top-3 predictions.
"""

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import default_data_collator
from tqdm import tqdm

from src.models.LoRA_MC import ID_TO_LABEL, build_hf_mc_dataset


def run_lora_mc_predictions(
    predictor, df: pd.DataFrame, id_col: str = "id", batch_size: int = 32
) -> pd.DataFrame:
    """
    Run the fine-tuned LoRA MC model over every row of df in batches.
    `predictor` must expose .model, .tokenizer, .device, .max_length
    (as set up by MC_LoRA_Predictor in src/models/LoRA_MC.py).
    Returns a DataFrame with columns: [id_col, 'Prediction']
    """
    hf_test = build_hf_mc_dataset(
        df, predictor.tokenizer, max_length=predictor.max_length, has_labels=False
    )
    loader = DataLoader(hf_test, batch_size=batch_size, collate_fn=default_data_collator)

    all_logits = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting on test set"):
            batch = {k: v.to(predictor.device) for k, v in batch.items()}
            outputs = predictor.model(**batch)
            all_logits.append(outputs.logits.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_probs = F.softmax(all_logits, dim=-1)
    top3_idx = torch.argsort(all_probs, dim=1, descending=True)[:, :3]

    preds = [" ".join(ID_TO_LABEL[i.item()] for i in row) for row in top3_idx]
    return pd.DataFrame({"ID": df[id_col], "Prediction": preds})


def save_submission(submission_df: pd.DataFrame, out_path: str = "submission_lora_mc.csv"):
    submission_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(submission_df)} rows)")