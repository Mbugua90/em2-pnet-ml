# EM2-PNET: Machine Learning Pipeline

Code accompanying the EM2-PNET project (Development of an Edge-AI Based
Multimodal Framework for Pneumonia Detection in Resource-Constrained
Settings), Dedan Kimathi University of Technology. Ethics approval:
DeKUT/ISREC/03422/368.

This repository contains the **code only**. No patient data, extracted
features, or trained model weights are included or distributed here, in
compliance with the PhysioNet Credentialed Health Data Use Agreement
governing MIMIC-IV / MIMIC-IV-ED, and with the ICBHI 2017 dataset's terms
of use.

## Data Access (obtain independently -- not provided here)

- **ICBHI 2017 Respiratory Sound Database** (public): https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
- **MIMIC-IV v3.1** and **MIMIC-IV-ED v2.2** (credentialed access required):
  https://physionet.org/ -- requires completing CITI human-subjects/HIPAA
  training and executing a signed Data Use Agreement.

Users of this code must obtain their own access to these datasets under
their respective terms; no data from either source is redistributed here.

## Repository Structure

```
eda/
  eda_icbhi.py                    # ICBHI diagnosis/cycle/split exploratory analysis
  eda_mimic_ed.py                 # MIMIC-IV-ED vitals completeness/quality profiling
  eda_mimic_diagnosis_link.py     # MIMIC-IV-ED pneumonia diagnosis-to-vitals linkage

feature_extraction/
  prepare_cnn_features.py         # ICBHI MFCC extraction + LOPO-CV fold prep (firmware-matched parameters)
  prepare_mlp_features.py         # MIMIC-IV-ED feature engineering + patient-level train/val/test split

training/
  train_cnn_lopo.py               # CNN branch (PyTorch): LOPO-CV training (ICBHI, Pneumonia-vs-COPD)
  train_mlp.py                    # MLP branch (PyTorch): vitals-based pneumonia classifier (MIMIC-IV-ED)
  train_cnn_debug.sbatch          # Slurm job: CNN debug subset (pneumonia-holdout folds only)
  train_cnn_full.sbatch           # Slurm job: CNN full 70-fold run
  train_mlp.sbatch                # Slurm job: MLP training

quantization/
  train_mlp_keras_ptq.py          # Objective 3: MLP Keras reimplementation + full-integer PTQ -> .tflite/.h
  train_cnn_keras_ptq.py          # Objective 3: CNN Keras reimplementation, LOPO-CV validation, final all-data model + PTQ -> .tflite/.h
  train_mlp_keras.sbatch          # Slurm job: MLP quantization pipeline (CPU)
  train_cnn_keras.sbatch          # Slurm job: CNN quantization pipeline
```

## Methodology Notes

- **CNN branch** uses Leave-One-Patient-Out Cross-Validation (LOPO-CV)
  rather than a fixed train/test split, since ICBHI's official split
  places all pneumonia-labeled patients in the training partition,
  making it unusable for pneumonia-specific evaluation. See the
  accompanying literature review for methodological precedent.
- **MLP branch** uses a patient-level (subject_id) train/val/test split
  to prevent leakage across a patient's multiple ED visits.
- MFCC feature extraction parameters (16kHz, 25ms frame / 10ms hop, 40
  coefficients) are matched exactly to the on-device firmware
  implementation to preserve train/deploy feature compatibility.
- Both branches report Sensitivity, Specificity, and their average
  ("Score"), plus AUC-ROC, at both a balanced (Youden's J) and a
  sensitivity-prioritized operating threshold, consistent with this
  project's stated sensitivity-first design priority for a screening
  application.

## Compute Environment

Developed and run on KENET CHUI (Slurm-managed HPC, NVIDIA L40S GPUs) and
Google Colaboratory. MIMIC-IV-ED processing was restricted to KENET CHUI
throughout, per institutional data-handling practice for credentialed
health data.

## Citation

If you use this code, please cite the accompanying thesis/publication
[citation to be added upon publication], and the original data sources:

- Rocha, B.M. et al. "A respiratory sound database for the development
  of automated classification," ICBHI 2017.
- Johnson, A. et al. "MIMIC-IV" (version 3.1), PhysioNet.
- Johnson, A. et al. "MIMIC-IV-ED" (version 2.2), PhysioNet.

## License

MIT License (see LICENSE file) -- applies to the code in this repository
only. Dataset access and use remain governed by each dataset's own terms.
