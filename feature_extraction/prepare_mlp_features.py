"""
MIMIC-IV-ED MLP Branch: Feature Engineering + Patient-Level Train/Val/Test Split
------------------------------------------------------------------------------------
Builds the labeled (X, y) dataset for MLP training from triage.csv + diagnosis.csv,
using the same pneumonia-matching logic (dual ICD code/text match, with the
exclusion list and negation check) already validated in Phase 1 EDA.

CRITICAL: splitting is done at the SUBJECT level (subject_id), not the stay
level -- a single patient can have multiple ED visits (stay_id), and allowing
the same patient's visits to land in both train and test would leak
patient-specific signal across the split, inflating reported performance.

Run on KENET CHUI (or locally) -- this step is lightweight (pandas only, no
GPU needed). Output is a single feature-engineered CSV ready for the
subsequent MLP training script.

USAGE:
  python3 prepare_mlp_features.py diagnosis.csv triage.csv output_features.csv
"""

import sys
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ICD9_PNEUMONIA_PREFIXES = ["480", "481", "482", "483", "484", "485", "486"]
ICD10_PNEUMONIA_PREFIXES = ["J12", "J13", "J14", "J15", "J16", "J17", "J18"]
EXCLUDED_CODES = {"J852", "A3790", "B961"}
NEGATION_PATTERN = re.compile(r"without\s+pneumonia", re.IGNORECASE)

TARGET_FIELDS = ["temperature", "heartrate", "resprate", "o2sat"]


def f_to_c(f):
    return (f - 32) * 5.0 / 9.0


def code_based_match(row):
    code = str(row["icd_code"]).strip().upper()
    version = row["icd_version"]
    if version == 9:
        return any(code.startswith(p) for p in ICD9_PNEUMONIA_PREFIXES)
    elif version == 10:
        return any(code.startswith(p) for p in ICD10_PNEUMONIA_PREFIXES)
    return False


def text_based_match(row):
    return "pneumonia" in str(row["icd_title"]).lower()


def is_excluded(row):
    code = str(row["icd_code"]).strip().upper().replace(".", "")
    if code in EXCLUDED_CODES:
        return True
    if NEGATION_PATTERN.search(str(row["icd_title"])):
        return True
    return False


def build_pneumonia_labels(diagnosis):
    diagnosis = diagnosis.copy()
    diagnosis["match_code"] = diagnosis.apply(code_based_match, axis=1)
    diagnosis["match_text"] = diagnosis.apply(text_based_match, axis=1)
    diagnosis["excluded"] = diagnosis.apply(is_excluded, axis=1)
    diagnosis["is_pneumonia"] = (diagnosis["match_code"] | diagnosis["match_text"]) & (~diagnosis["excluded"])
    pneumonia_stays = set(diagnosis[diagnosis["is_pneumonia"]]["stay_id"].unique())
    return pneumonia_stays


def clean_vitals(triage):
    df = triage.copy()
    if "temperature" in df.columns:
        df["temperature_c"] = df["temperature"].apply(lambda f: f_to_c(f) if pd.notna(f) else f)
        df = df[df["temperature_c"].between(30, 43) | df["temperature_c"].isna()]
    for field, lo, hi in [("o2sat", 0, 100), ("resprate", 0, 60), ("heartrate", 0, 250)]:
        if field in df.columns:
            df = df[df[field].between(lo, hi) | df[field].isna()]
    return df


def main(diagnosis_path, triage_path, output_path):
    diagnosis = pd.read_csv(diagnosis_path)
    triage = pd.read_csv(triage_path)

    print(f"Loaded diagnosis.csv: {len(diagnosis)} rows")
    print(f"Loaded triage.csv: {len(triage)} rows")

    pneumonia_stays = build_pneumonia_labels(diagnosis)
    print(f"Pneumonia-labeled stays identified: {len(pneumonia_stays)}")

    triage_clean = clean_vitals(triage)
    triage_clean["label"] = triage_clean["stay_id"].isin(pneumonia_stays).astype(int)
    triage_clean["complete_vitals"] = triage_clean[TARGET_FIELDS].notna().all(axis=1)

    # Keep only rows with complete vitals -- matches Phase 1 EDA's usable-sample definition
    dataset = triage_clean[triage_clean["complete_vitals"]].copy()
    print(f"Rows with complete vitals (usable dataset): {len(dataset)}")
    print(f"  Positive (pneumonia): {dataset['label'].sum()}")
    print(f"  Negative (non-pneumonia): {(dataset['label']==0).sum()}")

    # Feature set: the 4 core vitals, matching what the EM2-PNET hardware
    # itself measures (temperature, SpO2, and IMU-derived respiratory
    # effort/heart-rate-adjacent signals) -- keeps offline features aligned
    # with what the deployed device can actually provide in Cycle 2.
    feature_cols = ["temperature_c", "heartrate", "resprate", "o2sat"]
    if "acuity" in dataset.columns:
        feature_cols.append("acuity")  # triage acuity score, if present -- optional extra signal

    final_cols = ["subject_id", "stay_id", "label"] + feature_cols
    final_cols = [c for c in final_cols if c in dataset.columns]
    dataset_out = dataset[final_cols].dropna(subset=feature_cols)

    print(f"\nFinal feature set: {feature_cols}")
    print(f"Final dataset size after dropping any remaining NaNs: {len(dataset_out)}")

    # --- Patient-level split (GroupShuffleSplit on subject_id) ---
    # 70% train / 15% val / 15% test, grouped so no subject_id appears in
    # more than one split -- prevents patient-level leakage.
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss1.split(dataset_out, groups=dataset_out["subject_id"]))
    train_df = dataset_out.iloc[train_idx]
    temp_df = dataset_out.iloc[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["subject_id"]))
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    # Verify no subject_id leakage across splits
    train_subj = set(train_df["subject_id"])
    val_subj = set(val_df["subject_id"])
    test_subj = set(test_df["subject_id"])
    assert len(train_subj & val_subj) == 0, "LEAKAGE: subject_id overlap between train and val!"
    assert len(train_subj & test_subj) == 0, "LEAKAGE: subject_id overlap between train and test!"
    assert len(val_subj & test_subj) == 0, "LEAKAGE: subject_id overlap between val and test!"
    print("\nPatient-level split verified: NO subject_id leakage across train/val/test.")

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n = len(split_df)
        pos = split_df["label"].sum()
        print(f"  {name}: {n} rows, {pos} positive ({100*pos/n:.2f}%), {split_df['subject_id'].nunique()} unique subjects")

    train_df.assign(split="train").to_csv(output_path.replace(".csv", "_train.csv"), index=False)
    val_df.assign(split="val").to_csv(output_path.replace(".csv", "_val.csv"), index=False)
    test_df.assign(split="test").to_csv(output_path.replace(".csv", "_test.csv"), index=False)
    print(f"\nSaved: {output_path.replace('.csv','_train.csv')}, _val.csv, _test.csv")
    print("Report only aggregate counts above externally -- files themselves stay within your approved CHUI environment.")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 prepare_mlp_features.py <diagnosis.csv> <triage.csv> <output_features.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
