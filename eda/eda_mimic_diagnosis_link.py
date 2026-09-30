"""
MIMIC-IV-ED Diagnosis-to-Vitals Linkage: Pneumonia Case Identification
------------------------------------------------------------------------
Joins diagnosis.csv (ICD codes) to triage.csv (vitals) via stay_id, to
identify pneumonia-labeled ED stays and quantify how many have complete,
usable vitals for MLP training.

UPDATED based on manual review of the code/text cross-check disagreement
list from the previous run:
  - EXCLUDED_CODES: 3 specific codes manually confirmed as false positives
    (J852, A3790: explicit "...WITHOUT pneumonia" in title; B961: identifies
    a causative organism for a DIFFERENT condition coded elsewhere, not a
    pneumonia diagnosis itself) -- these are now hard-excluded regardless of
    what either matching method says.
  - General negation check: any icd_title containing "without pneumonia"
    (case-insensitive) is now excluded from the text-based match, not just
    the 2 specific codes found so far -- this generalizes the fix to catch
    similar negated titles that may exist elsewhere in the dataset but
    weren't in the small disagreement sample manually reviewed.

Run LOCALLY only (per your MIMIC-IV-ED DUA decision). Report only the
printed aggregate counts externally -- no row-level data.

USAGE:
  python3 eda_mimic_diagnosis_link.py diagnosis.csv triage.csv
"""

import sys
import re
import pandas as pd

# ICD-9 pneumonia codes: 480-486 (and common 4th-digit subcodes, matched by prefix)
ICD9_PNEUMONIA_PREFIXES = ["480", "481", "482", "483", "484", "485", "486"]

# ICD-10 pneumonia codes: J12-J18 (matched by prefix, covers subcodes e.g. J189)
ICD10_PNEUMONIA_PREFIXES = ["J12", "J13", "J14", "J15", "J16", "J17", "J18"]

# --- Manual exclusion list: codes confirmed via review to be false positives
#     despite containing "pneumonia" in their title or matching a code prefix. ---
EXCLUDED_CODES = {
    "J852",   # "Abscess of lung WITHOUT pneumonia"
    "A3790",  # "Whooping cough... WITHOUT pneumonia"
    "B961",   # identifies causative organism for a DIFFERENT condition, not a pneumonia dx itself
}

# --- General negation pattern: catches any other "...without pneumonia"
#     titles not already caught by the specific exclusion list above. ---
NEGATION_PATTERN = re.compile(r"without\s+pneumonia", re.IGNORECASE)


def f_to_c(f):
    return (f - 32) * 5.0 / 9.0


def is_excluded(row):
    code = str(row["icd_code"]).strip().upper().replace(".", "")
    if code in EXCLUDED_CODES:
        return True
    if NEGATION_PATTERN.search(str(row["icd_title"])):
        return True
    return False


def code_based_match(row):
    code = str(row["icd_code"]).strip().upper()
    version = row["icd_version"]
    if version == 9:
        return any(code.startswith(p) for p in ICD9_PNEUMONIA_PREFIXES)
    elif version == 10:
        return any(code.startswith(p) for p in ICD10_PNEUMONIA_PREFIXES)
    return False


def text_based_match(row):
    title = str(row["icd_title"]).lower()
    return "pneumonia" in title


def main(diagnosis_path, triage_path):
    diagnosis = pd.read_csv(diagnosis_path)
    triage = pd.read_csv(triage_path)

    print(f"Loaded diagnosis.csv: {len(diagnosis)} rows, "
          f"{diagnosis['stay_id'].nunique()} unique stays")
    print(f"Loaded triage.csv: {len(triage)} rows, "
          f"{triage['stay_id'].nunique()} unique stays")

    diagnosis["match_code"] = diagnosis.apply(code_based_match, axis=1)
    diagnosis["match_text"] = diagnosis.apply(text_based_match, axis=1)
    diagnosis["excluded"] = diagnosis.apply(is_excluded, axis=1)

    agree = diagnosis[diagnosis["match_code"] == diagnosis["match_text"]]
    disagree = diagnosis[diagnosis["match_code"] != diagnosis["match_text"]]

    print(f"\n=== Code-match vs. Text-match Cross-Check ===")
    print(f"Rows where both methods agree: {len(agree)} ({100*len(agree)/len(diagnosis):.2f}%)")
    print(f"Rows where methods DISAGREE: {len(disagree)}")

    remaining_disagree = disagree[~disagree["excluded"]]
    print(f"  -> of these, {len(disagree) - len(remaining_disagree)} resolved by exclusion "
          f"list / negation check")
    print(f"  -> {len(remaining_disagree)} still unresolved -- review these manually:")
    if len(remaining_disagree) > 0:
        review = remaining_disagree[["icd_code", "icd_version", "icd_title", "match_code", "match_text"]].drop_duplicates()
        print(review.to_string(index=False))
    else:
        print("  (none remaining -- all disagreements resolved by exclusion/negation rules)")

    # Final pneumonia flag: union of both methods, MINUS excluded codes/negated titles
    diagnosis["is_pneumonia"] = (diagnosis["match_code"] | diagnosis["match_text"]) & (~diagnosis["excluded"])

    n_excluded_positive = ((diagnosis["match_code"] | diagnosis["match_text"]) & diagnosis["excluded"]).sum()
    print(f"\nRows excluded by manual/negation rules that would otherwise have matched: {n_excluded_positive}")

    pneumonia_diag = diagnosis[diagnosis["is_pneumonia"]]
    pneumonia_stays_any = pneumonia_diag["stay_id"].unique()
    pneumonia_stays_primary = diagnosis[(diagnosis["is_pneumonia"]) & (diagnosis["seq_num"] == 1)]["stay_id"].unique()

    print(f"\n=== Pneumonia-Labeled Stay Counts (after exclusion fix) ===")
    print(f"Total unique stays (all triage visits): {triage['stay_id'].nunique()}")
    print(f"Stays with pneumonia ANYWHERE in diagnosis list: {len(pneumonia_stays_any)}")
    print(f"Stays with pneumonia as PRIMARY diagnosis (seq_num=1): {len(pneumonia_stays_primary)}")

    icd_version_split = pneumonia_diag["icd_version"].value_counts()
    print(f"\nICD version breakdown among pneumonia-matched rows:\n{icd_version_split}")

    target_fields = ["temperature", "heartrate", "resprate", "o2sat"]
    triage_clean = triage.copy()
    if "temperature" in triage_clean.columns:
        triage_clean["temperature_c"] = triage_clean["temperature"].apply(
            lambda f: f_to_c(f) if pd.notna(f) else f)
        plausible = triage_clean["temperature_c"].between(30, 43) | triage_clean["temperature_c"].isna()
        triage_clean = triage_clean[plausible]
    if "o2sat" in triage_clean.columns:
        triage_clean = triage_clean[(triage_clean["o2sat"].between(0, 100)) | triage_clean["o2sat"].isna()]
    if "resprate" in triage_clean.columns:
        triage_clean = triage_clean[(triage_clean["resprate"].between(0, 60)) | triage_clean["resprate"].isna()]
    if "heartrate" in triage_clean.columns:
        triage_clean = triage_clean[(triage_clean["heartrate"].between(0, 250)) | triage_clean["heartrate"].isna()]

    triage_clean["complete_vitals"] = triage_clean[target_fields].notna().all(axis=1)

    pneumonia_triage = triage_clean[triage_clean["stay_id"].isin(pneumonia_stays_any)]
    nonpneumonia_triage = triage_clean[~triage_clean["stay_id"].isin(pneumonia_stays_any)]

    print(f"\n=== Vitals Completeness: Pneumonia vs. Non-Pneumonia Stays ===")
    print(f"Pneumonia stays with triage record: {len(pneumonia_triage)}")
    print(f"  -> with COMPLETE vitals (all 4 fields, plausible range): "
          f"{pneumonia_triage['complete_vitals'].sum()} "
          f"({100*pneumonia_triage['complete_vitals'].mean():.1f}%)")
    print(f"Non-pneumonia stays with triage record: {len(nonpneumonia_triage)}")
    print(f"  -> with COMPLETE vitals (all 4 fields, plausible range): "
          f"{nonpneumonia_triage['complete_vitals'].sum()} "
          f"({100*nonpneumonia_triage['complete_vitals'].mean():.1f}%)")

    print(f"\n=== Clinical Validity Check: Abnormal Vitals Rate, Pneumonia vs. Baseline ===")
    pn_complete = pneumonia_triage[pneumonia_triage["complete_vitals"]]
    if len(pn_complete) > 0:
        febrile = (pn_complete["temperature_c"] > 38.0).mean() * 100
        hypoxic = (pn_complete["o2sat"] < 92).mean() * 100
        tachypneic = (pn_complete["resprate"] > 20).mean() * 100
        print(f"Among pneumonia-labeled stays with complete vitals (n={len(pn_complete)}):")
        print(f"  Febrile (>38C): {febrile:.1f}%   [whole-population baseline: 2.1%]")
        print(f"  Hypoxic (<92%): {hypoxic:.1f}%   [whole-population baseline: 0.8%]")
        print(f"  Tachypneic (>20/min): {tachypneic:.1f}%   [whole-population baseline: 4.6%]")
        print("If these rates are notably HIGHER than baseline, this supports label/vitals validity.")

    print(f"\n=== Summary for methodology decision ===")
    print("Report only the aggregate counts/percentages above externally.")
    print("This run applies the manual exclusion list + negation check derived from")
    print("reviewing the previous run's code/text disagreement list.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 eda_mimic_diagnosis_link.py <diagnosis.csv> <triage.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
