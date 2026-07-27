"""
src/dataset/data_loader.py

Single, shared place to load every dataset used across this project
(EDA.ipynb and all notebooks/Milestones/Milestone-*.ipynb):

  - Smart MCQ Solver competition data: train.csv, test.csv, sample_submission.csv
  - RAG resources used in Milestone 3: pre-built Wikipedia FAISS index +
    the parquet file with the actual Wikipedia passage text """

import os
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LOCAL_DATA_DEFAULT = os.path.join(_REPO_ROOT, "data")


def _competition_dir():
    default = (
        _LOCAL_DATA_DEFAULT
        if os.path.isdir(_LOCAL_DATA_DEFAULT)
        else "/kaggle/input/competitions/smart-mcq-solver-challenge"
    )
    return os.environ.get("MCQ_COMPETITION_DIR", default)


def _wiki_faiss_index_dir():
    return os.environ.get(
        "WIKI_FAISS_INDEX_DIR",
        "/kaggle/input/datasets/jjinho/wikipedia-2023-07-faiss-index",
    )


def _wiki_text_dir():
    return os.environ.get(
        "WIKI_TEXT_DIR",
        "/kaggle/input/datasets/jjinho/wikipedia-20230701",
    )


# ---------------------------------------------------------------------------
# Competition data (used by EDA.ipynb and every Milestone notebook)
# ---------------------------------------------------------------------------
def load_competition_data(drop_id_from_train: bool = True):
    """Load train, test and sample_submission for the Smart MCQ Solver competition.

    Looks in (in order): $MCQ_COMPETITION_DIR, then <repo_root>/data if it
    exists locally, then the Kaggle default path.

    Returns
    -------
    (train_dataset, test_dataset, sample_submission) : tuple[pd.DataFrame, ...]
    """
    competition_dir = _competition_dir()

    for fname in ("sample_submission.csv", "train.csv", "test.csv"):
        fpath = os.path.join(competition_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Could not find '{fname}' in '{competition_dir}'.\n"
                f"Set the MCQ_COMPETITION_DIR environment variable to the folder "
                f"containing train.csv / test.csv / sample_submission.csv, "
                f"or place them in '{_LOCAL_DATA_DEFAULT}'."
            )

    sample_submission = pd.read_csv(os.path.join(competition_dir, "sample_submission.csv"))
    train_dataset = pd.read_csv(os.path.join(competition_dir, "train.csv"))
    test_dataset = pd.read_csv(os.path.join(competition_dir, "test.csv"))

    if drop_id_from_train and "id" in train_dataset.columns:
        train_dataset = train_dataset.drop(columns=["id"])

    return train_dataset, test_dataset, sample_submission

# ---------------------------------------------------------------------------
# RAG resources (used by Milestone 3)
# ---------------------------------------------------------------------------
def load_wiki_faiss_index():
    """Load the pre-built Wikipedia FAISS index (6.4M article embeddings)."""
    import faiss  # imported lazily — only needed for Milestone 3

    index_path = os.path.join(_wiki_faiss_index_dir(), "wikipedia_202307.index")
    return faiss.read_index(index_path)

def load_wiki_passages() -> pd.DataFrame:
    """Load the Wikipedia passage text that maps 1:1 to the FAISS index rows."""
    index_path = os.path.join(_wiki_text_dir(), "wiki_2023_index.parquet")
    return pd.read_parquet(index_path)

def load_rag_resources():
    """Load everything Milestone 3 (RAG) needs in one call.

    Returns
    -------
    (wiki_index, wiki_index_df) : tuple[faiss.Index, pd.DataFrame]
    """
    wiki_index = load_wiki_faiss_index()
    wiki_index_df = load_wiki_passages()
    return wiki_index, wiki_index_df

def load_retriever(retriever_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load the MiniLM sentence-embedding retriever used to query the
    Wikipedia FAISS index (used by Milestone 3 RAG and the from-scratch
    BiGRU-RAG model)."""
    from sentence_transformers import SentenceTransformer  # imported lazily

    return SentenceTransformer(retriever_name)