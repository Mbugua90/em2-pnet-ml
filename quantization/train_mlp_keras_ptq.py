"""
MIMIC-IV-ED MLP Branch: Keras Reimplementation + Post-Training Quantization
-------------------------------------------------------------------------------
Rebuilds the validated PyTorch MLP architecture in Keras/TensorFlow (required
for TFLite Micro deployment -- PyTorch models cannot be converted directly to
TFLite Micro reliably), retrains on the same data splits, confirms comparable
performance to the original PyTorch result, then applies FULL INTEGER
Post-Training Quantization (PTQ) -- the quantization mode required for
microcontroller deployment (not just mobile int8-with-float-fallback).

Architecture matches the original: 64-32-16-1 with BatchNorm and Dropout,
trained with class-weighted loss for the ~2.2% positive-class imbalance.

Reporting uses the SAME convention as the PyTorch script (balanced and
sensitivity-prioritized thresholds via Youden's J / target sensitivity) so
results are directly comparable to the original PyTorch run (AUC=0.7939,
balanced Score=0.7237 on test).

USAGE:
  python3 train_mlp_keras_ptq.py mimic_features_train.csv mimic_features_val.csv mimic_features_test.csv
"""

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

NON_FEATURE_COLS = {"subject_id", "stay_id", "label", "split"}
EPOCHS = 100
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
PATIENCE = 10


def load_split(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    return X, y, feature_cols


def build_model(n_features):
    inputs = keras.Input(shape=(n_features,))
    x = keras.layers.Dense(64)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(16, activation="relu")(x)
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
    auc = roc_auc_score(labels, probs)
    print(f"\n=== {label_str} (threshold={threshold:.4f}) ===")
    print(f"  Sensitivity: {sens:.4f}  Specificity: {spec:.4f}  Score: {score:.4f}  AUC: {auc:.4f}")
    print(f"  Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")


def make_c_array(tflite_bytes, var_name="mlp_pneumonia_model_tflite"):
    """Convert TFLite bytes to a C header array -- portable alternative to
    the `xxd -i` shell command, in case it's unavailable on the deployment
    build environment."""
    lines = [f"unsigned char {var_name}[] = {{"]
    for i in range(0, len(tflite_bytes), 12):
        chunk = tflite_bytes[i:i + 12]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    lines.append("};")
    lines.append(f"unsigned int {var_name}_len = {len(tflite_bytes)};")
    return "\n".join(lines)


def main(train_path, val_path, test_path):
    X_train, y_train, feature_cols = load_split(train_path)
    X_val, y_val, _ = load_split(val_path)
    X_test, y_test, _ = load_split(test_path)
    print(f"Features: {feature_cols}")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    class_weight = {0: 1.0, 1: n_neg / n_pos}
    print(f"Class weights: {class_weight}")

    model = build_model(n_features=X_train_s.shape[1])
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])

    early_stop = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                patience=PATIENCE, restore_best_weights=True)

    model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weight,
              callbacks=[early_stop], verbose=2)

    val_probs = model.predict(X_val_s, verbose=0).flatten()
    test_probs = model.predict(X_test_s, verbose=0).flatten()

    balanced_threshold = find_best_threshold(val_probs, y_val)
    sens_threshold = find_sensitivity_targeted_threshold(val_probs, y_val, target=0.90)

    print(f"\nBalanced threshold: {balanced_threshold:.4f}")
    print(f"Sensitivity-prioritized threshold: {sens_threshold:.4f}")
    report_metrics(test_probs, y_test, balanced_threshold, "KERAS TEST -- balanced (compare to PyTorch: Score=0.7237, AUC=0.7939)")
    report_metrics(test_probs, y_test, sens_threshold, "KERAS TEST -- sensitivity-prioritized")

    model.save("mlp_pneumonia_keras.keras")
    print("\nSaved Keras model to mlp_pneumonia_keras.keras")

    # =========================================================================
    # POST-TRAINING QUANTIZATION -- full integer, required for TFLite Micro
    # =========================================================================
    def representative_dataset():
        # A sample of real, scaled training inputs -- calibrates the int8
        # quantization ranges. 200 samples is ample for a 5-feature model.
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train_s), size=min(200, len(X_train_s)), replace=False)
        for i in idx:
            yield [X_train_s[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open("mlp_pneumonia_quantized.tflite", "wb") as f:
        f.write(tflite_model)
    print(f"\nSaved quantized TFLite model: mlp_pneumonia_quantized.tflite ({len(tflite_model)} bytes)")

    # --- Evaluate the QUANTIZED model to confirm accuracy survived PTQ ---
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    in_scale, in_zero = input_details["quantization"]

    quant_probs = []
    for i in range(len(X_test_s)):
        x_int8 = np.round(X_test_s[i:i+1] / in_scale + in_zero).astype(np.int8)
        interpreter.set_tensor(input_details["index"], x_int8)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        out_scale, out_zero = output_details["quantization"]
        prob = (out.astype(np.float32) - out_zero) * out_scale
        quant_probs.append(prob[0][0])
    quant_probs = np.array(quant_probs)

    report_metrics(quant_probs, y_test, balanced_threshold, "QUANTIZED (INT8) TEST -- balanced (accuracy check post-PTQ)")

    # --- Export as C header for firmware embedding ---
    with open("mlp_pneumonia_model.h", "w") as f:
        f.write("// Auto-generated from mlp_pneumonia_quantized.tflite\n")
        f.write("// Include this in EM2-PNET firmware for TFLite Micro deployment\n\n")
        f.write(make_c_array(tflite_model))
    print("Saved C header: mlp_pneumonia_model.h")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 train_mlp_keras_ptq.py <train.csv> <val.csv> <test.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
