"""
MIMIC-IV-ED MLP Branch: Training Script
------------------------------------------
Trains a feedforward MLP on the vitals-based features prepared by
prepare_mlp_features.py, with:
  - Class-weighted loss (handles the ~2.2% positive-class imbalance)
  - Early stopping on validation AUC
  - Threshold selection via Youden's J statistic on the validation set
    (rather than a naive 0.5 cutoff, which is a poor choice under this
    much class imbalance)
  - Final evaluation reported as Sensitivity, Specificity, and their
    average "Score", matching the ICBHI-convention reporting style used
    for the CNN branch, so both branches are comparably reported.

Run via the accompanying Slurm batch script (train_mlp.sbatch) on CHUI's
gpu1 partition -- do not run multi-hour training in an interactive srun
session, since it will be killed if your terminal disconnects.

USAGE:
  python3 train_mlp.py mimic_features_train.csv mimic_features_val.csv mimic_features_test.csv
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

NON_FEATURE_COLS = {"subject_id", "stay_id", "label", "split"}
EPOCHS = 100
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
PATIENCE = 10  # early stopping patience, in epochs


class VitalsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class VitalsMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_split(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    return X, y, feature_cols


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(yb.numpy())
    return np.array(all_probs), np.array(all_labels)


def find_best_threshold(probs, labels):
    """Youden's J statistic: maximizes (sensitivity + specificity - 1).
    A BALANCED operating point -- treats missing a case and a false alarm
    as equally costly."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def find_sensitivity_targeted_threshold(probs, labels, target_sensitivity=0.90):
    """Finds the highest threshold that still achieves at least
    target_sensitivity on the given (validation) set -- appropriate when
    missing a real case is considered costlier than a false alarm, per
    this project's stated sensitivity-first design priority."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    valid = np.where(tpr >= target_sensitivity)[0]
    if len(valid) == 0:
        print(f"WARNING: no threshold achieves sensitivity >= {target_sensitivity}; using lowest available threshold.")
        return thresholds[-1]
    # Among thresholds meeting the sensitivity target, pick the one with
    # the LOWEST false positive rate (best specificity available at that
    # sensitivity level)
    best_idx = valid[np.argmin(fpr[valid])]
    return thresholds[best_idx]


def report_metrics(probs, labels, threshold, label_str):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    score = (sensitivity + specificity) / 2.0
    auc = roc_auc_score(labels, probs)

    print(f"\n=== {label_str} Results (threshold={threshold:.4f}) ===")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Score (avg of Se/Sp): {score:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    return {"sensitivity": sensitivity, "specificity": specificity, "score": score, "auc": auc}


def main(train_path, val_path, test_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train, y_train, feature_cols = load_split(train_path)
    X_val, y_val, _ = load_split(val_path)
    X_test, y_test, _ = load_split(test_path)
    print(f"Features used: {feature_cols}")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Standardize -- fit ONLY on train, apply same transform to val/test (no leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_loader = DataLoader(VitalsDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(VitalsDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(VitalsDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = VitalsMLP(n_features=X_train.shape[1]).to(device)

    # Class-weighted loss: pos_weight = n_negative / n_positive
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    print(f"Class imbalance: {n_neg:.0f} negative, {n_pos:.0f} positive, pos_weight={pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_auc = -1
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_probs, val_labels = evaluate(model, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_probs)

        print(f"Epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.4f}  val_AUC={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (no val_AUC improvement for {PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)
    print(f"\nBest validation AUC: {best_val_auc:.4f}")

    # Select operating threshold from validation set (NOT test set -- avoids
    # picking a threshold that flatters the test result)
    val_probs, val_labels = evaluate(model, val_loader, device)
    best_threshold = find_best_threshold(val_probs, val_labels)
    sensitivity_threshold = find_sensitivity_targeted_threshold(val_probs, val_labels, target_sensitivity=0.90)
    print(f"Selected BALANCED threshold (Youden's J on validation): {best_threshold:.4f}")
    print(f"Selected SENSITIVITY-PRIORITIZED threshold (>=90% sens on validation): {sensitivity_threshold:.4f}")

    report_metrics(val_probs, val_labels, best_threshold, "Validation (balanced)")
    report_metrics(val_probs, val_labels, sensitivity_threshold, "Validation (sensitivity-prioritized)")

    test_probs, test_labels = evaluate(model, test_loader, device)
    report_metrics(test_probs, test_labels, best_threshold, "TEST -- balanced operating point (final)")
    report_metrics(test_probs, test_labels, sensitivity_threshold, "TEST -- sensitivity-prioritized operating point (final)")

    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "feature_cols": feature_cols,
        "threshold_balanced": best_threshold,
        "threshold_sensitivity_prioritized": sensitivity_threshold,
    }, "mlp_pneumonia_model.pt")
    print("\nSaved model + scaler + BOTH thresholds to mlp_pneumonia_model.pt")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 train_mlp.py <train.csv> <val.csv> <test.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
