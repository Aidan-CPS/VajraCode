# VajraCode
VajraCode is a research codebase for modeling depression as a dynamic process using longitudinal data. It compares theory-agnostic machine learning with theory-guided approaches inspired by psychology and yoga, emphasizing early detection, interpretability, and rigorous empirical evaluation.



## Data Overview

This project uses the WARN-D dataset, a longitudinal, multi-modal dataset designed to support early prediction of depression while minimizing data leakage. The data are already de-identified and split into training (70%) and test (30%) sets.

The dataset is organized into three stages, each corresponding to a different timescale:
- **Stage 1 (Baseline):** One row per participant containing static questionnaire data (e.g. demographics, mental health history, personality, coping styles). These variables describe who the person is at baseline.
- **Stage 2 (Daily Dynamics)**: Many rows per participant (≈350), each corresponding to an EMA (Ecological Momentary Assessment) survey moment. These rows include short self-reports (mood, stress, fatigue, cognition, context) and wearable-derived features aggregated over the 120 minutes prior to each EMA prompt. This stage captures fine-grained day-to-day fluctuations.
- **Stage 3 (Outcomes):** Eight rows per participant, each corresponding to a quarterly or yearly follow-up. These rows define the outcome timeline. For training data, a separate file provides a binary depression onset label (0/1) at each Stage-3 time point.

The core modeling task is to aggregate Stage-2 data up to each Stage-3 time point, combine those aggregates with Stage-1 baseline features, and predict whether a depression onset occurs at that time point. Test data include no outcome labels and are truncated to prevent future leakage.

For full details on file structure, columns, and modeling constraints, see README_data.md.
