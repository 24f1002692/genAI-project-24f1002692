""" src/inference/BiGRU_RAG_inference.py
 
Trains the from-scratch BiGRU-RAG model (src/models/BiGRU_RAG.py), evaluates
it with MAP@3, and runs inference on test.csv """

import pandas as pd
import torch
import torch.nn as nn
 
from src.models.BiGRU_Rag_scratch import OPTIONS
 
 
# ---------------------------------------------------------------------------
# Metric — MAP@3 (Mean Average Precision at 3), the competition metric
# ---------------------------------------------------------------------------
def apk(actual, predicted, k: int = 3) -> float:
    predicted = predicted[:k]
    if actual not in predicted:
        return 0.0
    return 1.0 / (predicted.index(actual) + 1)
 
 
def mapk(df: pd.DataFrame, pred_col: str) -> float:
    return df.apply(lambda row: apk(row['answer'], row[pred_col].split()), axis=1).mean()

# ---------------------------------------------------------------------------
# Evaluation loop — used for both train_eval_loader and val_loader
# ---------------------------------------------------------------------------
def evaluate(model, loader, df: pd.DataFrame, criterion, device):
    model.eval()
    all_top3 = []
    total_loss = 0.0
    with torch.no_grad():
        for ids, mask, labels in loader:
            ids, mask = ids.to(device), mask.to(device)
            labels_t = torch.tensor(labels).to(device)
            scores = model(ids, mask)
            loss = criterion(scores, labels_t)
            total_loss += loss.item()
            top3 = torch.topk(scores, k=3, dim=1).indices.cpu().tolist()
            all_top3.extend(top3)
    avg_loss = total_loss / len(loader)
    preds = [[OPTIONS[i] for i in row] for row in all_top3]
    actuals = df['answer'].tolist()
    map3 = sum(apk(a, p) for a, p in zip(actuals, preds)) / len(actuals)
    accuracy = sum(1 for a, p in zip(actuals, preds) if a == p[0]) / len(actuals)
    return avg_loss, map3, accuracy

# ---------------------------------------------------------------------------
# Training loop — early stopping on val MAP@3, checkpoints best model
# ---------------------------------------------------------------------------
def train_bigru_rag(
    model,
    train_loader, train_eval_loader, val_loader,
    train_split: pd.DataFrame, val_split: pd.DataFrame,
    device, epochs: int = 25,
    lr: float = 1e-3, weight_decay: float = 3e-4,
    label_smoothing: float = 0.1, patience: int = 5,
    checkpoint_path: str = "best_scratch_rag_model.pt",
    use_wandb: bool = True,
    wandb_project: str = "24f1002692-t22026",
    wandb_run_name: str = "Scratch-BiGRU-RAG",
):
    """Trains `model` in-place, saves the best checkpoint to `checkpoint_path`,
    and returns (history, best_map3)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
 
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "embed_dim": model.embedding.embedding_dim,
                "hidden_dim": model.gru.hidden_size,
                "lr": lr,
                "epochs": epochs,
            },
        )
 
    best_map3 = 0.0
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
 
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        for ids, mask, labels in train_loader:
            ids, mask, labels = ids.to(device), mask.to(device), torch.tensor(labels).to(device)
            optimizer.zero_grad()
            scores = model(ids, mask)
            loss = criterion(scores, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_train_loss += loss.item()
 
        train_loss_avg = total_train_loss / len(train_loader)
        _, train_map3, train_acc = evaluate(model, train_eval_loader, train_split, criterion, device)
        val_loss, val_map3, val_acc = evaluate(model, val_loader, val_split, criterion, device)
 
        print(f"Epoch {epoch:2d} | train_loss {train_loss_avg:.4f} | train MAP@3 {train_map3:.4f} "
              f"| val_loss {val_loss:.4f} | val MAP@3 {val_map3:.4f}")
 
        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
 
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
 
        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss_avg,
                "train_map3": train_map3,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_map3": val_map3,
                "val_acc": val_acc,
                "lr": current_lr,
            })
 
        if val_map3 > best_map3:
            best_map3 = val_map3
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
            if use_wandb:
                import wandb
                wandb.save(checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping. Best val MAP@3: {best_map3:.4f}")
                break
 
    print(f"\nTraining done. Best val MAP@3: {best_map3:.4f}")
    if use_wandb:
        import wandb
        wandb.finish()
 
    return history, best_map3

# ---------------------------------------------------------------------------
# Inference — top-3 predictions on any loader (typically test_loader)
# ---------------------------------------------------------------------------
def predict_top3(model, loader, device) -> list[list[str]]:
    model.eval()
    all_top3 = []
    with torch.no_grad():
        for ids, mask, _ in loader:
            ids, mask = ids.to(device), mask.to(device)
            scores = model(ids, mask)
            top3 = torch.topk(scores, k=3, dim=1).indices.cpu().tolist()
            all_top3.extend(top3)
    return [[OPTIONS[i] for i in row] for row in all_top3]

def save_submission(test_df: pd.DataFrame, preds: list[list[str]],
                     id_col: str = "id", out_path: str = "submission_scratch_rag.csv") -> pd.DataFrame:
    submission = pd.DataFrame({
        "ID": test_df[id_col],
        "Prediction": [" ".join(p) for p in preds],
    })
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(submission)} rows)")
    return submission