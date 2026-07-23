""" src/inference/rag_inference.py

Runs the RAG predictor (src/models/RAG_pretrained.py) over a full dataset
and returns/saves top-3 predictions — src/inference/inference.py."""

import pandas as pd
from tqdm import tqdm


def run_rag_predictions(rag_predictor, df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """
    Run RAGPredictor.predict_top3() over every row of df.
    Returns a DataFrame with columns: [id_col, 'rag_pred']
    """
    preds = [
        rag_predictor.predict_top3(row)
        for _, row in tqdm(df.iterrows(), total=len(df))
    ]
    return pd.DataFrame({
        "ID": df[id_col],
        "Prediction": preds,
    })


def save_submission(submission_df: pd.DataFrame, out_path: str = "submission_m3.csv"):
    submission_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(submission_df)} rows)")