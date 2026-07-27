"""
src/models/LoRA_MC.py

LoRA-fine-tuned Multiple-Choice model for the Smart MCQ Solver ("model of choice").
Pipeline:
    question + each option --> AutoModelForMultipleChoice input (5 sequences/question)
                            --> bert-base-uncased + LoRA adapter (r=8, alpha=16, target=[query,value])
                            --> fine-tuned with HF Trainer
                            --> top-3 predicted letters (for MAP@3) """

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForMultipleChoice, AutoTokenizer

OPTIONS = ['A', 'B', 'C', 'D', 'E']
LABEL_TO_ID = {opt: i for i, opt in enumerate(OPTIONS)}
ID_TO_LABEL = {i: opt for opt, i in LABEL_TO_ID.items()}

DEFAULT_MODEL_NAME = "bert-base-uncased"


def format_option(prompt, option_text) -> str:
    """Format a single (prompt, option) pair as one MC input sequence."""
    return f"{prompt} [SEP] {option_text}"


def format_row(row) -> list[str]:
    """Format all 5 options of a row into 5 input sequences."""
    return [format_option(row['prompt'], row[opt]) for opt in OPTIONS]


def build_hf_mc_dataset(df, tokenizer, max_length: int = 128, has_labels: bool = True) -> HFDataset:
    """Build a HF Dataset of tokenized multiple-choice examples.
    Set has_labels=False for test rows with no 'answer' column."""
    records = []
    for _, row in df.iterrows():
        formatted = format_row(row)
        encoded = tokenizer(
            formatted, padding="max_length", truncation=True, max_length=max_length,
        )
        record = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if has_labels:
            record["labels"] = int(row["label"])
        records.append(record)
    return HFDataset.from_list(records)


def build_lora_mc_model(
    model_name: str = DEFAULT_MODEL_NAME,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules=("query", "value"),
):
    """Load AutoModelForMultipleChoice and wrap it with a LoRA adapter."""
    base_model = AutoModelForMultipleChoice.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model


def mc_compute_metrics(eval_pred):
    """accuracy + MAP@3, used as compute_metrics for the HF Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = (preds == labels).mean()

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    top3 = np.argsort(-probs, axis=1)[:, :3]
    map3_scores = []
    for rank_row, true_label in zip(top3, labels):
        rank_row = list(rank_row)
        map3_scores.append(1.0 / (rank_row.index(true_label) + 1) if true_label in rank_row else 0.0)
    return {"accuracy": accuracy, "map3": float(np.mean(map3_scores))}


class MC_LoRA_Predictor:
    """Loads a fine-tuned LoRA multiple-choice checkpoint for inference."""

    def __init__(
        self,
        lora_checkpoint_dir: str,
        base_model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = 128,
        device: str = None,
    ):
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForMultipleChoice.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, lora_checkpoint_dir)
        self.model = model.merge_and_unload()  # merge LoRA weights for faster inference
        self.model.to(self.device)
        self.model.eval()

    def predict_top3(self, row) -> str:
        """Tokenize one row's 5 options, run the model, return top-3 letters."""
        formatted = format_row(row)
        encoded = self.tokenizer(
            formatted, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        inputs = {k: v.unsqueeze(0).to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=-1).squeeze(0)
        top3_idx = torch.argsort(probs, descending=True)[:3]
        return " ".join(ID_TO_LABEL[i.item()] for i in top3_idx)