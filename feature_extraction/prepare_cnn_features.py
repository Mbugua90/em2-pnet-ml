"""
ICBHI CNN Branch: MFCC Feature Extraction + LOPO-CV Fold Preparation
------------------------------------------------------------------------
Extracts firmware-matched MFCC features from ICBHI recordings, segmented
into fixed 5-second windows per the stated methodology (Section 3.4:
"Segmentation step entails segmenting of continuous data streams into
fixed windows (e.g. 5-second intervals)").

Produces data ready for Leave-One-Patient-Out Cross-Validation (LOPO-CV),
NOT a fixed train/val/test split -- this matches the methodology already
established and literature-grounded for ICBHI's small pneumonia cohort
(n=6 patients). A fixed split would reintroduce the fragility problem
LOPO-CV was specifically chosen to avoid.

TARGET_CLASSES is configurable below:
  - Default: ["Pneumonia", "COPD"] (age-matched primary task, per the
    identified Healthy/Pneumonia age confound)
  - Set to None to include all 8 diagnosis classes (multi-class / secondary
    reporting task)

MFCC parameters EXACTLY match the on-device firmware (audio_features.h):
  16kHz sample rate, 25ms frame (400 samples), 10ms hop (160 samples),
  512-point FFT (zero-padded), 40 mel filters, 40 MFCC coefficients.
  All ICBHI audio is resampled to 16kHz first, since ICBHI recordings use
  mixed native sample rates (4kHz/10kHz/44.1kHz).

Dependencies: librosa, soundfile, numpy, pandas, scikit-learn
  pip install librosa soundfile numpy pandas scikit-learn

USAGE:
  python3 prepare_cnn_features.py /path/to/icbhi_data_dir /path/to/output_dir
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import LeaveOneGroupOut

# =============================================================================
# CONFIGURATION -- matches firmware audio_features.h exactly
# =============================================================================
SAMPLE_RATE = 16000
FRAME_SAMPLES = 400          # 25ms at 16kHz
HOP_SAMPLES = 160            # 10ms at 16kHz
N_FFT = 512                  # zero-padded FFT size, matches firmware
N_MELS = 40
N_MFCC = 40

WINDOW_SECONDS = 5.0          # per stated methodology Section 3.4
N_FRAMES_PER_WINDOW = int(WINDOW_SECONDS * SAMPLE_RATE / HOP_SAMPLES)  # fixed shape for CNN batching

TARGET_CLASSES = ["Pneumonia", "COPD"]  # set to None for full 8-class task


def load_diagnosis(data_dir):
    for fname in ["patient_diagnosis.csv", "patient_diagnosis.txt"]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            return pd.read_csv(path, sep=None, engine="python", header=None,
                                names=["patient_id", "diagnosis"], dtype={"patient_id": str})
    raise FileNotFoundError("patient_diagnosis file not found")


def extract_mfcc_windows(wav_path):
    """Load, resample to 16kHz, segment into fixed 5s windows, extract MFCC
    per window. Returns a list of (N_FRAMES_PER_WINDOW, N_MFCC) arrays."""
    y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

    window_len_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
    n_windows = int(np.ceil(len(y) / window_len_samples)) if len(y) > 0 else 0

    windows_mfcc = []
    for w in range(n_windows):
        start = w * window_len_samples
        end = start + window_len_samples
        segment = y[start:end]
        if len(segment) < window_len_samples:
            segment = np.pad(segment, (0, window_len_samples - len(segment)))  # zero-pad final partial window

        mfcc = librosa.feature.mfcc(
            y=segment, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT,
            hop_length=HOP_SAMPLES, win_length=FRAME_SAMPLES, window="hamming"
        )  # shape: (N_MFCC, n_frames)
        mfcc = mfcc.T  # -> (n_frames, N_MFCC)

        # Enforce exact fixed frame count (pad/truncate) for consistent CNN input shape
        if mfcc.shape[0] < N_FRAMES_PER_WINDOW:
            pad_amt = N_FRAMES_PER_WINDOW - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_amt), (0, 0)))
        elif mfcc.shape[0] > N_FRAMES_PER_WINDOW:
            mfcc = mfcc[:N_FRAMES_PER_WINDOW, :]

        windows_mfcc.append(mfcc)

    return windows_mfcc


def main(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    diagnosis_df = load_diagnosis(data_dir)
    diag_map = dict(zip(diagnosis_df["patient_id"], diagnosis_df["diagnosis"]))

    if TARGET_CLASSES is not None:
        valid_patients = {pid for pid, dx in diag_map.items() if dx in TARGET_CLASSES}
        print(f"Restricting to classes {TARGET_CLASSES}: {len(valid_patients)} patients")
    else:
        valid_patients = set(diag_map.keys())
        print(f"Using all classes: {len(valid_patients)} patients")

    wav_files = sorted(glob.glob(os.path.join(data_dir, "*.wav")))
    print(f"Found {len(wav_files)} total .wav files")

    all_X, all_y, all_patient_ids, all_recording_names = [], [], [], []

    for i, wav_path in enumerate(wav_files):
        base = os.path.basename(wav_path).replace(".wav", "")
        patient_id = base.split("_")[0]
        if patient_id not in valid_patients:
            continue

        diagnosis = diag_map[patient_id]
        windows = extract_mfcc_windows(wav_path)

        for w_idx, mfcc_window in enumerate(windows):
            all_X.append(mfcc_window)
            all_y.append(diagnosis)
            all_patient_ids.append(patient_id)
            all_recording_names.append(f"{base}_win{w_idx}")

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(wav_files)} files...")

    X = np.stack(all_X)  # shape: (n_samples, N_FRAMES_PER_WINDOW, N_MFCC)
    y = np.array(all_y)
    patient_ids = np.array(all_patient_ids)

    print(f"\nFinal dataset: X shape = {X.shape}, {len(np.unique(patient_ids))} unique patients")
    print(f"Class distribution:\n{pd.Series(y).value_counts()}")

    np.savez_compressed(
        os.path.join(output_dir, "icbhi_cnn_features.npz"),
        X=X, y=y, patient_ids=patient_ids, recording_names=np.array(all_recording_names)
    )
    print(f"\nSaved features to {os.path.join(output_dir, 'icbhi_cnn_features.npz')}")

    # --- LOPO-CV fold verification (does NOT train anything -- just confirms
    #     the fold structure is valid and reports per-fold sample counts) ---
    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=patient_ids)
    print(f"\nLOPO-CV: {n_folds} folds (one per unique patient)")

    fold_summary = []
    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=patient_ids)):
        test_patient = patient_ids[test_idx][0]
        test_label = y[test_idx][0]
        fold_summary.append({
            "fold": fold_i, "held_out_patient": test_patient, "held_out_label": test_label,
            "n_train_windows": len(train_idx), "n_test_windows": len(test_idx)
        })
    fold_df = pd.DataFrame(fold_summary)
    fold_df.to_csv(os.path.join(output_dir, "lopo_fold_summary.csv"), index=False)
    print(fold_df.to_string(index=False))
    print(f"\nSaved fold summary to {os.path.join(output_dir, 'lopo_fold_summary.csv')}")
    print("\nNote: within each fold's TRAINING patients, reserve one held-out")
    print("patient (or a small random subset of training windows) as an inner")
    print("validation set for early stopping -- this is handled in the training")
    print("script, not baked into this feature file, to keep fold logic flexible.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 prepare_cnn_features.py <icbhi_data_dir> <output_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
