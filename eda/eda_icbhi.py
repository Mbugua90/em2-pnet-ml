"""
ICBHI 2017 EDA: Diagnosis Distribution, Cycle Counts, Train/Test Split Analysis
---------------------------------------------------------------------------------
Fully public dataset -- safe to run on Google Colab or locally. Formalizes
the manual analysis already performed on the pasted diagnosis/split files
into a reusable script that also parses the actual audio annotation (.txt)
files for cycle-level counts and crackle/wheeze labels.

UPDATED: wider console/display output so summary tables print fully instead
of being column-truncated (the previous run's cycle-level and pneumonia-
detail tables were cut off mid-table).

Expected folder layout (standard ICBHI release):
  <data_dir>/
    patient_diagnosis.csv          (or .txt -- tab or comma separated, no header)
    ICBHI_challenge_train_test.txt (filename <tab> train/test)
    demographic_info.txt           (optional)
    *.wav, *.txt                   (recording + per-cycle annotation pairs)

USAGE:
  python3 eda_icbhi.py /path/to/icbhi_data_dir
"""

import sys
import os
import glob
import pandas as pd

# --- Wider console/display output: prevents pandas from truncating columns
#     or wrapping wide tables when printed to a Colab cell or terminal. ---
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)


def load_diagnosis(data_dir):
    for fname in ["patient_diagnosis.csv", "patient_diagnosis.txt"]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, sep=None, engine="python", header=None,
                              names=["patient_id", "diagnosis"], dtype={"patient_id": str})
            print(f"Loaded diagnosis file: {path} ({len(df)} patients)")
            return df
    raise FileNotFoundError("Could not find patient_diagnosis.csv/.txt in data_dir")


def load_split(data_dir):
    candidates = glob.glob(os.path.join(data_dir, "*train_test*"))
    if not candidates:
        raise FileNotFoundError("Could not find the official train/test split file in data_dir")
    path = candidates[0]
    df = pd.read_csv(path, sep=None, engine="python", header=None,
                      names=["filename", "split"])
    df["patient_id"] = df["filename"].str.split("_").str[0]
    print(f"Loaded train/test split: {path} ({len(df)} recording entries)")
    return df


def load_demographics(data_dir):
    for fname in ["demographic_info.txt", "demographic_info.csv"]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, sep=None, engine="python", header=None,
                              names=["patient_id", "age", "sex", "adult_bmi", "child_weight", "child_height"],
                              dtype={"patient_id": str})
            print(f"Loaded demographics: {path} ({len(df)} patients)")
            return df
    print("No demographics file found -- skipping demographic analysis.")
    return None


def diagnosis_by_split(diagnosis_df, split_df):
    patient_split = split_df.groupby("patient_id")["split"].first()
    merged = diagnosis_df.copy()
    merged["split"] = merged["patient_id"].map(patient_split)

    print("\n=== Patient count by diagnosis and split ===")
    table = merged.groupby(["diagnosis", "split"]).size().unstack(fill_value=0)
    table["total"] = table.sum(axis=1)
    print(table.sort_values("total", ascending=False).to_string())
    return merged


def cycle_level_counts(data_dir, diagnosis_df):
    """Parse every .txt annotation file (NOT the diagnosis/split files) to
    count actual breathing cycles and crackle/wheeze labels per patient."""
    txt_files = [f for f in glob.glob(os.path.join(data_dir, "*.txt"))
                 if not any(skip in os.path.basename(f).lower()
                            for skip in ["diagnosis", "train_test", "demographic", "filename"])]

    print(f"\nFound {len(txt_files)} per-recording annotation files")

    rows = []
    for path in txt_files:
        base = os.path.basename(path).replace(".txt", "")
        patient_id = base.split("_")[0]
        try:
            ann = pd.read_csv(path, sep="\t", header=None,
                               names=["start", "end", "crackles", "wheezes"])
        except Exception as e:
            print(f"  [!] Failed to parse {path}: {e}")
            continue
        rows.append({
            "patient_id": patient_id,
            "recording": base,
            "n_cycles": len(ann),
            "n_crackle_cycles": int(ann["crackles"].sum()),
            "n_wheeze_cycles": int(ann["wheezes"].sum()),
        })

    cycles_df = pd.DataFrame(rows)
    merged = cycles_df.merge(diagnosis_df, on="patient_id", how="left")

    print("\n=== Cycle-level counts by diagnosis ===")
    summary = merged.groupby("diagnosis").agg(
        n_recordings=("recording", "count"),
        total_cycles=("n_cycles", "sum"),
        total_crackle_cycles=("n_crackle_cycles", "sum"),
        total_wheeze_cycles=("n_wheeze_cycles", "sum"),
    ).sort_values("total_cycles", ascending=False)
    print(summary.to_string())

    print("\n=== Pneumonia-specific cycle detail ===")
    pneumonia_recordings = merged[merged["diagnosis"] == "Pneumonia"]
    print(pneumonia_recordings[["patient_id", "recording", "n_cycles",
                                  "n_crackle_cycles", "n_wheeze_cycles"]].to_string(index=False))
    return merged


def demographic_summary(demo_df, diagnosis_df):
    if demo_df is None:
        return
    merged = diagnosis_df.merge(demo_df, on="patient_id", how="left")
    merged["age"] = pd.to_numeric(merged["age"], errors="coerce")

    print("\n=== Age summary by diagnosis ===")
    print(merged.groupby("diagnosis")["age"].agg(["count", "mean", "min", "max"]).to_string())

    print("\n=== Pneumonia patients: age detail ===")
    pneumonia = merged[merged["diagnosis"] == "Pneumonia"]
    print(pneumonia[["patient_id", "age", "sex"]].to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 eda_icbhi.py <path_to_icbhi_data_dir>")
        sys.exit(1)

    data_dir = sys.argv[1]

    diagnosis_df = load_diagnosis(data_dir)
    split_df = load_split(data_dir)
    demo_df = load_demographics(data_dir)

    merged_diag_split = diagnosis_by_split(diagnosis_df, split_df)
    cycle_merged = cycle_level_counts(data_dir, diagnosis_df)
    demographic_summary(demo_df, diagnosis_df)

    print("\n=== EDA complete. Use these outputs to finalize the LOPO-CV fold design. ===")
