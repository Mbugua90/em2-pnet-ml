"""
ICBHI CNN Branch: Keras Reimplementation, LOPO-CV Validation, Final Model + PTQ
------------------------------------------------------------------------------------
Two stages:
  1. LOPO-CV across all 70 patients (Keras version) -- validates the
     reimplementation against the original PyTorch result (AUC=0.9244,
     Score=0.8752 balanced). This is a VALIDATION exercise, not the
     deployment model -- no single fold's model is the right artifact to
     ship, since each excludes one patient's data.
  2. Final model: trained on ALL 70 patients (no held-out patient), which
     IS the deployment candidate. This is the model that gets quantized
     and exported for the ESP32-S3.

Architecture matches the original PyTorch ShallowCNN1D exactly (same layer
sizes/kernels), per Section 3.6's "shallow and efficient" design goal.

USAGE:
  python3 train_cnn_keras_ptq.py icbhi_cnn_features.npz --debug   (6 pneumonia-holdout folds only)
  python3 train_cnn_keras_ptq.py icbhi_cnn_features.npz            (full 70-fold validation + final model + PTQ)
"""

import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 8
INNER_VAL_FRACTION = 0.15
RANDOM_SEED = 42


def build_model(n_frames, n_mfcc):
    inputs = keras.Input(shape=(n_frames, n_mfcc))
    x = keras.layers.Conv1D(32, 5, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(64, 5, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(64, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


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


def train_model(X_train, y_train, X_val, y_val, n_frames, n_mfcc):
    model = build_model(n_frames, n_mfcc)
    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    class_weight = {0: 1.0, 1: n_neg / max(n_pos, 1)}
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss="binary_crossentropy", metrics=[keras.metrics.AUC(name="auc")])
    early_stop = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                patience=PATIENCE, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weight,
              callbacks=[early_stop], verbose=0)
    return model


def run_lopo_validation(X, y, patient_ids, debug_mode):
    logo = LeaveOneGroupOut()
    all_folds = list(logo.split(X, y, groups=patient_ids))
    n_frames, n_mfcc = X.shape[1], X.shape[2]

    if debug_mode:
        fold_indices = [i for i, (_, test_idx) in enumerate(all_folds) if y[test_idx][0] == 1]
        print(f"DEBUG MODE: {len(fold_indices)} pneumonia-holdout folds")
    else:
        fold_indices = list(range(len(all_folds)))
        print(f"FULL VALIDATION: {len(fold_indices)} folds")

    pooled_outer_probs, pooled_outer_labels = [], []
    pooled_inner_probs, pooled_inner_labels = [], []
    rng = np.random.RandomState(RANDOM_SEED)

    for fold_i in fold_indices:
        train_idx, test_idx = all_folds[fold_i]
        held_out = patient_ids[test_idx][0]
        held_label = "Pneumonia" if y[test_idx][0] == 1 else "COPD"

        train_patients = np.unique(patient_ids[train_idx])
        train_patient_labels = np.array([y[train_idx][patient_ids[train_idx] == p][0] for p in train_patients])
        n_inner_val = max(2, int(len(train_patients) * INNER_VAL_FRACTION))
        inner_val_patients = set()
        for cls in np.unique(train_patient_labels):
            cls_patients = train_patients[train_patient_labels == cls]
            n_from_class = max(1, int(round(n_inner_val * (cls_patients.size / len(train_patients)))))
            n_from_class = min(n_from_class, cls_patients.size)
            inner_val_patients.update(rng.choice(cls_patients, size=n_from_class, replace=False))

        inner_val_mask = np.isin(patient_ids[train_idx], list(inner_val_patients))
        actual_train_idx = train_idx[~inner_val_mask]
        inner_val_idx = train_idx[inner_val_mask]

        model = train_model(X[actual_train_idx], y[actual_train_idx],
                             X[inner_val_idx], y[inner_val_idx], n_frames, n_mfcc)

        iv_probs = model.predict(X[inner_val_idx], verbose=0).flatten()
        pooled_inner_probs.extend(iv_probs)
        pooled_inner_labels.extend(y[inner_val_idx])

        test_probs = model.predict(X[test_idx], verbose=0).flatten()
        pooled_outer_probs.extend(test_probs)
        pooled_outer_labels.extend(y[test_idx])

        print(f"Fold {fold_i}: held-out {held_out} ({held_label}), "
              f"mean pred={np.mean(test_probs):.4f}")

    return (np.array(pooled_outer_probs), np.array(pooled_outer_labels),
            np.array(pooled_inner_probs), np.array(pooled_inner_labels))


def make_c_array(tflite_bytes, var_name="cnn_pneumonia_model_tflite"):
    lines = [f"unsigned char {var_name}[] = {{"]
    for i in range(0, len(tflite_bytes), 12):
        chunk = tflite_bytes[i:i + 12]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    lines.append("};")
    lines.append(f"unsigned int {var_name}_len = {len(tflite_bytes)};")
    return "\n".join(lines)


def main(npz_path, debug_mode):
    data = np.load(npz_path, allow_pickle=True)
    X, y_raw, patient_ids = data["X"], data["y"], data["patient_ids"]
    y = (y_raw == "Pneumonia").astype(np.float32)
    n_frames, n_mfcc = X.shape[1], X.shape[2]
    print(f"Loaded {X.shape[0]} windows, {len(np.unique(patient_ids))} patients")

    # ---- STAGE 1: LOPO-CV validation ----
    outer_probs, outer_labels, inner_probs, inner_labels = run_lopo_validation(
        X, y, patient_ids, debug_mode)

    print(f"\n{'='*60}\nSTAGE 1 RESULTS: POOLED LOPO-CV VALIDATION\n{'='*60}")
    if len(np.unique(inner_labels)) < 2 or len(np.unique(outer_labels)) < 2:
        print("Debug subset -- metrics not meaningful until full run.")
    else:
        bal_thresh = find_best_threshold(inner_probs, inner_labels)
        sens_thresh = find_sensitivity_targeted_threshold(inner_probs, inner_labels, 0.90)
        report_metrics(outer_probs, outer_labels, bal_thresh,
                        "Keras LOPO-CV -- balanced (compare to PyTorch: Score=0.8752, AUC=0.9244)")
        report_metrics(outer_probs, outer_labels, sens_thresh, "Keras LOPO-CV -- sensitivity-prioritized")

    if debug_mode:
        print("\nDebug mode: skipping Stage 2 (final model + PTQ). "
              "Re-run without --debug once validation looks correct.")
        return

    # ---- STAGE 2: Final model trained on ALL patients (the deployment candidate) ----
    print(f"\n{'='*60}\nSTAGE 2: FINAL MODEL (trained on ALL {len(np.unique(patient_ids))} patients)\n{'='*60}")

    rng = np.random.RandomState(RANDOM_SEED)
    all_patients = np.unique(patient_ids)
    patient_labels = np.array([y[patient_ids == p][0] for p in all_patients])
    val_patients = set()
    for cls in np.unique(patient_labels):
        cls_patients = all_patients[patient_labels == cls]
        n_val = max(1, int(round(len(cls_patients) * 0.1)))
        val_patients.update(rng.choice(cls_patients, size=n_val, replace=False))

    val_mask = np.isin(patient_ids, list(val_patients))
    final_model = train_model(X[~val_mask], y[~val_mask], X[val_mask], y[val_mask], n_frames, n_mfcc)

    val_probs = final_model.predict(X[val_mask], verbose=0).flatten()
    final_bal_thresh = find_best_threshold(val_probs, y[val_mask])
    report_metrics(val_probs, y[val_mask], final_bal_thresh, "Final model -- internal validation check")

    final_model.save("cnn_pneumonia_keras.keras")
    print("\nSaved final Keras model: cnn_pneumonia_keras.keras")

    def representative_dataset():
        idx = rng.choice(len(X[~val_mask]), size=min(200, len(X[~val_mask])), replace=False)
        for i in idx:
            yield [X[~val_mask][i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    with open("cnn_pneumonia_quantized.tflite", "wb") as f:
        f.write(tflite_model)
    print(f"\nSaved quantized TFLite model: cnn_pneumonia_quantized.tflite ({len(tflite_model)} bytes)")

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    in_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]
    in_scale, in_zero = in_d["quantization"]
    out_scale, out_zero = out_d["quantization"]

    quant_probs = []
    for i in range(len(X[val_mask])):
        x_int8 = np.round(X[val_mask][i:i+1] / in_scale + in_zero).astype(np.int8)
        interpreter.set_tensor(in_d["index"], x_int8)
        interpreter.invoke()
        out = interpreter.get_tensor(out_d["index"])
        quant_probs.append((out.astype(np.float32) - out_zero) * out_scale)
    quant_probs = np.array(quant_probs).flatten()

    report_metrics(quant_probs, y[val_mask], final_bal_thresh, "QUANTIZED (INT8) -- accuracy check post-PTQ")

    with open("cnn_pneumonia_model.h", "w") as f:
        f.write("// Auto-generated from cnn_pneumonia_quantized.tflite\n\n")
        f.write(make_c_array(tflite_model))
    print("Saved C header: cnn_pneumonia_model.h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.npz_path, args.debug)
