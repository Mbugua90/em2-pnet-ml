"""
ICBHI CNN Branch: Leave-One-Patient-Out Cross-Validation Training
----------------------------------------------------------------------
Trains a shallow 1D-CNN (matching the architecture described in Section 3.6
of the methodology: "shallow and efficient... designed to extract relevant
patterns from the temporal acoustic data") over MFCC sequences, using
Leave-One-Patient-Out Cross-Validation across all patients in
icbhi_cnn_features.npz.

REPORTING METHODOLOGY (important -- read before interpreting results):
  LOPO-CV produces one held-out prediction per patient, not a natural
  per-fold Sensitivity/Specificity. Following standard practice for this
  dataset (see literature review), all held-out (outer-fold) window-level
  predictions are POOLED across all folds into one combined confusion
  matrix for final reporting -- this matches how prior published ICBHI
  LOPO-CV results (e.g. Petmezas et al.) are reported.

  Threshold selection: each fold reserves a small random subset of its
  TRAINING patients as an inner validation set (never the held-out test
  patient). Inner-validation predictions are pooled across all folds to
  select the operating threshold (Youden's J and a sensitivity-prioritized
  threshold, matching the MLP branch's reporting convention), which is
  then applied to the pooled OUTER (true held-out) predictions for the
  final reported result -- avoiding any threshold selection on test data.

Class-weighted loss is used given the severe window-level imbalance
(COPD >> Pneumonia at the window level, ~23.6:1 in the full dataset).

USAGE:
  python3 train_cnn_lopo.py icbhi_cnn_features.npz --debug   (pneumonia-holdout folds only, ~6 folds)
  python3 train_cnn_lopo.py icbhi_cnn_features.npz            (full run, all folds)
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 8
INNER_VAL_FRACTION = 0.15  # fraction of TRAINING patients held out per fold as inner validation
RANDOM_SEED = 42


class MFCCDataset(Dataset):
    def __init__(self, X, y):
        # X: (n, n_frames, n_mfcc) -> CNN wants (n, n_mfcc, n_frames) as channels-first
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ShallowCNN1D(nn.Module):
    """Shallow, efficient 1D-CNN per Section 3.6's stated architecture goal."""
    def __init__(self, n_mfcc=40):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mfcc, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(self.conv(x)).squeeze(-1)


def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            p = torch.sigmoid(model(xb)).cpu().numpy()
            probs.extend(p)
            labels.extend(yb.numpy())
    return np.array(probs), np.array(labels)


def find_best_threshold(probs, labels):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    return thresholds[np.argmax(tpr - fpr)]


def find_sensitivity_targeted_threshold(probs, labels, target=0.90):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    valid = np.where(tpr >= target)[0]
    if len(valid) == 0:
        return thresholds[-1]
    return thresholds[valid[np.argmin(fpr[valid])]]


def report_metrics(probs, labels, threshold, label_str):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    score = (sens + spec) / 2.0
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    print(f"\n=== {label_str} (threshold={threshold:.4f}) ===")
    print(f"  Sensitivity: {sens:.4f}  Specificity: {spec:.4f}  Score: {score:.4f}  AUC: {auc:.4f}")
    print(f"  Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    return {"sensitivity": sens, "specificity": spec, "score": score, "auc": auc}


def train_one_fold(X_train, y_train, X_innerval, y_innerval, device):
    train_loader = DataLoader(MFCCDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    innerval_loader = DataLoader(MFCCDataset(X_innerval, y_innerval), batch_size=BATCH_SIZE, shuffle=False)

    model = ShallowCNN1D(n_mfcc=X_train.shape[2]).to(device)
    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_auc, best_state, no_improve = -1, None, 0
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_probs, val_labels = evaluate(model, innerval_loader, device)
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            val_auc = 0.5  # inner validation fold had only one class present

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


def main(npz_path, debug_mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = np.load(npz_path, allow_pickle=True)
    X, y_raw, patient_ids = data["X"], data["y"], data["patient_ids"]
    y = (y_raw == "Pneumonia").astype(np.float32)  # binary: Pneumonia=1, COPD=0

    print(f"Loaded {X.shape[0]} windows, {len(np.unique(patient_ids))} patients")

    logo = LeaveOneGroupOut()
    all_folds = list(logo.split(X, y, groups=patient_ids))

    if debug_mode:
        # Debug subset: only folds where the held-out patient is Pneumonia
        fold_indices = [i for i, (_, test_idx) in enumerate(all_folds) if y[test_idx][0] == 1]
        print(f"DEBUG MODE: running only {len(fold_indices)} pneumonia-holdout folds")
    else:
        fold_indices = list(range(len(all_folds)))
        print(f"FULL RUN: running all {len(fold_indices)} folds")

    pooled_outer_probs, pooled_outer_labels = [], []
    pooled_inner_probs, pooled_inner_labels = [], []
    rng = np.random.RandomState(RANDOM_SEED)

    for fold_i in fold_indices:
        train_idx, test_idx = all_folds[fold_i]
        held_out_patient = patient_ids[test_idx][0]
        held_out_label = "Pneumonia" if y[test_idx][0] == 1 else "COPD"
        print(f"\n--- Fold {fold_i}: held-out patient {held_out_patient} ({held_out_label}), "
              f"{len(test_idx)} test windows ---")

        # Carve inner validation from TRAINING patients only (never touches test patient)
        train_patients = np.unique(patient_ids[train_idx])
        n_inner_val = max(1, int(len(train_patients) * INNER_VAL_FRACTION))
        inner_val_patients = set(rng.choice(train_patients, size=n_inner_val, replace=False))

        inner_val_mask = np.isin(patient_ids[train_idx], list(inner_val_patients))
        actual_train_idx = train_idx[~inner_val_mask]
        inner_val_idx = train_idx[inner_val_mask]

        model = train_one_fold(X[actual_train_idx], y[actual_train_idx],
                                X[inner_val_idx], y[inner_val_idx], device)

        # Pool inner-validation predictions (for threshold selection later)
        iv_loader = DataLoader(MFCCDataset(X[inner_val_idx], y[inner_val_idx]), batch_size=BATCH_SIZE, shuffle=False)
        iv_probs, iv_labels = evaluate(model, iv_loader, device)
        pooled_inner_probs.extend(iv_probs)
        pooled_inner_labels.extend(iv_labels)

        # Pool outer (true held-out) predictions -- the actual result
        test_loader = DataLoader(MFCCDataset(X[test_idx], y[test_idx]), batch_size=BATCH_SIZE, shuffle=False)
        test_probs, test_labels = evaluate(model, test_loader, device)
        pooled_outer_probs.extend(test_probs)
        pooled_outer_labels.extend(test_labels)

        fold_pred = "Pneumonia" if np.mean(test_probs) >= 0.5 else "COPD"
        print(f"  Fold mean predicted probability: {np.mean(test_probs):.4f} (naive call: {fold_pred})")

    pooled_outer_probs = np.array(pooled_outer_probs)
    pooled_outer_labels = np.array(pooled_outer_labels)
    pooled_inner_probs = np.array(pooled_inner_probs)
    pooled_inner_labels = np.array(pooled_inner_labels)

    print(f"\n{'='*60}\nPOOLED RESULTS ACROSS {len(fold_indices)} FOLD(S)\n{'='*60}")

    if len(np.unique(pooled_inner_labels)) < 2 or len(np.unique(pooled_outer_labels)) < 2:
        print("NOTE: debug subset lacks both classes in inner-val and/or outer pool -- "
              "threshold selection and AUC are not meaningful until the FULL run "
              "(all 70 folds) is executed. This debug run is for verifying the "
              "pipeline runs correctly end-to-end, not for interpreting results.")
    else:
        balanced_thresh = find_best_threshold(pooled_inner_probs, pooled_inner_labels)
        sens_thresh = find_sensitivity_targeted_threshold(pooled_inner_probs, pooled_inner_labels, target=0.90)
        print(f"Balanced threshold (from pooled inner-validation): {balanced_thresh:.4f}")
        print(f"Sensitivity-prioritized threshold (from pooled inner-validation): {sens_thresh:.4f}")

        report_metrics(pooled_outer_probs, pooled_outer_labels, balanced_thresh, "OUTER (final) -- balanced")
        report_metrics(pooled_outer_probs, pooled_outer_labels, sens_thresh, "OUTER (final) -- sensitivity-prioritized")

    np.savez_compressed(
        "cnn_lopo_results" + ("_debug" if debug_mode else "_full") + ".npz",
        outer_probs=pooled_outer_probs, outer_labels=pooled_outer_labels,
        inner_probs=pooled_inner_probs, inner_labels=pooled_inner_labels,
    )
    print(f"\nSaved pooled predictions to cnn_lopo_results{'_debug' if debug_mode else '_full'}.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path")
    parser.add_argument("--debug", action="store_true", help="Run only pneumonia-holdout folds for pipeline verification")
    args = parser.parse_args()
    main(args.npz_path, args.debug)
