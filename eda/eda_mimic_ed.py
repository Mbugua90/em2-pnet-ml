"""
MIMIC-IV-ED EDA: Vitals Completeness and Quality Check
--------------------------------------------------------
Run this LOCALLY (or on infrastructure confirmed compliant with your signed
PhysioNet DUA) against your own downloaded triage.csv and vitalsign.csv
files. Do NOT paste raw row-level output from this script back into any
chat/AI tool -- only report aggregate summary statistics (counts,
percentages, ranges), consistent with your DUA obligations.

Assumes standard MIMIC-IV-ED column names:
  triage.csv:   subject_id, stay_id, temperature, heartrate, resprate, o2sat, ...
  vitalsign.csv: subject_id, stay_id, charttime, temperature, heartrate, resprate, o2sat, ...
Adjust column names below if your actual headers differ.

USAGE:
  python3 eda_mimic_ed.py /path/to/triage.csv /path/to/vitalsign.csv
"""

import sys
import pandas as pd


def f_to_c(f):
    return (f - 32) * 5.0 / 9.0


def profile_table(df, name, fields):
    print(f"\n=== {name} ===")
    print(f"Total rows: {len(df)}")
    if "stay_id" in df.columns:
        print(f"Unique stay_id: {df['stay_id'].nunique()}")

    for field in fields:
        if field not in df.columns:
            print(f"  [!] Column '{field}' not found -- check actual header names.")
            continue
        n_missing = df[field].isna().sum()
        pct_missing = 100 * n_missing / len(df)
        print(f"  {field}: {n_missing} missing ({pct_missing:.1f}%)")

    # Joint completeness: rows where ALL target fields are present
    present_fields = [f for f in fields if f in df.columns]
    joint_complete = df[present_fields].notna().all(axis=1).sum()
    print(f"  Rows with ALL of {present_fields} present: {joint_complete} "
          f"({100*joint_complete/len(df):.1f}%)")


def sanity_check_ranges(df, name):
    print(f"\n=== {name}: Value Range Sanity Check ===")

    if "temperature" in df.columns:
        temp_c = df["temperature"].dropna().apply(f_to_c)
        implausible = temp_c[(temp_c < 30) | (temp_c > 43)]
        print(f"  Temperature (converted to C): range "
              f"{temp_c.min():.1f}-{temp_c.max():.1f} C, "
              f"n={len(temp_c)}, implausible (<30C or >43C): {len(implausible)}")

    if "o2sat" in df.columns:
        spo2 = df["o2sat"].dropna()
        implausible = spo2[(spo2 <= 0) | (spo2 > 100)]
        print(f"  O2 Saturation: range {spo2.min():.1f}-{spo2.max():.1f}%, "
              f"n={len(spo2)}, implausible (<=0 or >100): {len(implausible)}")

    if "resprate" in df.columns:
        rr = df["resprate"].dropna()
        implausible = rr[(rr <= 0) | (rr > 60)]
        print(f"  Respiratory Rate: range {rr.min():.0f}-{rr.max():.0f} /min, "
              f"n={len(rr)}, implausible (<=0 or >60): {len(implausible)}")

    if "heartrate" in df.columns:
        hr = df["heartrate"].dropna()
        implausible = hr[(hr <= 0) | (hr > 250)]
        print(f"  Heart Rate: range {hr.min():.0f}-{hr.max():.0f} bpm, "
              f"n={len(hr)}, implausible (<=0 or >250): {len(implausible)}")


def abnormal_range_prevalence(df, name):
    print(f"\n=== {name}: Clinically Abnormal Range Prevalence ===")
    if "temperature" in df.columns:
        temp_c = df["temperature"].dropna().apply(f_to_c)
        febrile = (temp_c > 38.0).sum()
        print(f"  Febrile (>38.0C): {febrile} ({100*febrile/len(temp_c):.1f}% of non-missing)")
    if "o2sat" in df.columns:
        spo2 = df["o2sat"].dropna()
        hypoxic = (spo2 < 92).sum()
        print(f"  Hypoxic (<92%): {hypoxic} ({100*hypoxic/len(spo2):.1f}% of non-missing)")
    if "resprate" in df.columns:
        rr = df["resprate"].dropna()
        tachypneic = (rr > 20).sum()  # adult threshold; note age-specific thresholds needed for pediatric analysis
        print(f"  Tachypneic (>20/min, adult threshold): {tachypneic} "
              f"({100*tachypneic/len(rr):.1f}% of non-missing)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 eda_mimic_ed.py <triage.csv> <vitalsign.csv>")
        sys.exit(1)

    triage_path, vitalsign_path = sys.argv[1], sys.argv[2]

    triage = pd.read_csv(triage_path)
    vitalsign = pd.read_csv(vitalsign_path)

    target_fields = ["temperature", "heartrate", "resprate", "o2sat"]

    profile_table(triage, "triage.csv (single intake snapshot)", target_fields)
    profile_table(vitalsign, "vitalsign.csv (longitudinal, multiple per stay)", target_fields)

    sanity_check_ranges(triage, "triage.csv")
    sanity_check_ranges(vitalsign, "vitalsign.csv")

    abnormal_range_prevalence(triage, "triage.csv")
    abnormal_range_prevalence(vitalsign, "vitalsign.csv")

    print("\n=== Summary for methodology decision ===")
    print("Compare joint-completeness and abnormal-range prevalence above between")
    print("triage.csv and vitalsign.csv to decide your primary MLP feature source.")
    print("Report ONLY these aggregate numbers in any external discussion -- do not")
    print("share row-level MIMIC data outside your approved secure environment.")
