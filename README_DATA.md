# README_data.md  
## WARN-D Dataset Overview (for VajraCode)

This document explains the structure and meaning of the **WARN-D dataset** used in the VajraCode project. It is intended for new collaborators, students, and contributors who want to understand what the data contain and how the different files relate to each other *before* writing any code.

The dataset is designed to support **early prediction of depression** using longitudinal data. It separates baseline information, intensive daily measurements, and long-term outcomes in a way that makes temporal modeling and leakage control explicit.

---

## 1. High-level structure

The dataset is provided as a single zip file and, once unpacked, is organized by **stage** and by **train/test split**.

You will find the following files:

- `stage1_train_x.csv`
- `stage1_test_x.csv`
- `stage2_train_x.csv`
- `stage2_test_x.csv`
- `stage3_train_x.csv`
- `stage3_test_x.csv`
- `stage3_train_y.csv`

Each file is a CSV table. The suffix `_x` denotes **predictors (features)**, and `_y` denotes **outcomes (labels)**.

---

## 2. The three stages (conceptual meaning)

The dataset is divided into **three stages**, each corresponding to a different timescale and type of information.

### Stage 1 — Baseline (static information)

**Files:**  
- `stage1_train_x.csv`  
- `stage1_test_x.csv`

**Structure:**  
- One row per participant  
- Many columns (questionnaire variables)

**Content:**  
Stage 1 contains baseline questionnaire data collected at the start of the study. These variables are mostly stable over time and include things like demographics, mental health history, personality traits, coping styles, and baseline wellbeing.

**How to think about it:**  
Stage 1 describes *who the person is* at baseline. These variables do not change over time in the dataset and can be treated as static covariates.

---

### Stage 2 — Intensive daily measurements (EMA + wearables)

**Files:**  
- `stage2_train_x.csv`  
- `stage2_test_x.csv`

**Structure:**  
- Many rows per participant (typically ~350)  
- Each row corresponds to one **EMA survey moment**

**What is EMA?**  
EMA stands for **Ecological Momentary Assessment**. It refers to short surveys delivered repeatedly during daily life via a smartphone app. The goal is to measure experiences *as they occur*, rather than relying on memory.

**Content:**  
Each row in Stage 2 includes:
- EMA self-reports (e.g. mood, stress, fatigue, cognition, context)
- Wearable-derived features from a smartwatch

**Wearable data details:**  
Wearable data were collected continuously, but for this challenge they are **pre-aggregated** into features computed over the **120 minutes before each EMA survey**. These include averages, standard deviations, minima, and maxima of:
- heart rate
- stress scores
- activity-related measures

**How to think about it:**  
Stage 2 captures **short-term dynamics**: how mood, energy, body, and context fluctuate across days and weeks. This is the highest-resolution data in the dataset.

---

### Stage 3 — Long-term follow-up (outcomes timeline)

**Files:**  
- `stage3_train_x.csv`  
- `stage3_test_x.csv`  
- `stage3_train_y.csv`

**Structure:**  
- 8 rows per participant  
- Each row corresponds to a **quarterly (or yearly) follow-up**

**Stage 3 predictors (`*_x.csv`):**  
These files contain longer-term questionnaire measures such as depression-related scales, anxiety, wellbeing, functioning, and meaning-related variables.

**Stage 3 outcomes (`stage3_train_y.csv`):**  
This file contains the **binary depression onset outcome**:
- `0` = no onset at that time point  
- `1` = depression onset at that time point  

There is one outcome row for each row in `stage3_train_x.csv`.

**Important note:**  
The test set does *not* include outcome labels. This is standard for prediction challenges.

---

## 3. Train vs test split and leakage control

The dataset is already split into **training (70%)** and **test (30%)** participants.

To prevent data leakage:
- Raw PHQ-9 scores are *not* provided as predictors.
- Only **binned severity categories** (low / moderate / high) appear in predictor files.
- Test data include only information *up to* the last possible onset.
- The final measurement point for each test participant is removed, since it cannot be used to predict a future onset.

You should **never mix train and test data**, and you should never use information from *after* a given outcome time point to predict that outcome.

---

## 4. How the stages fit together (very important)

For each participant:

1. Stage 1 provides **one baseline row**
2. Stage 2 provides **many daily rows** (EMA + wearables)
3. Stage 3 provides **a small number of outcome time points**

The core modeling task is to:
- **aggregate Stage 2 data up to each Stage 3 time point**
- combine those aggregates with Stage 1 features
- predict whether a depression onset occurs at that Stage 3 time point

This aggregation step is *not provided* in the dataset and must be implemented by the user.

---

## 5. Free-text data

Most variables in WARN-D are structured (ratings, scales, categories). However, some questionnaires include **open-ended or semi-open responses**, where participants describe experiences, stressors, or changes in their lives.

These free-text fields are limited but valuable. They reflect experience **before it is filtered through predefined categories** and may be useful for exploratory or supplementary analyses.

---

## 6. Summary

In short, the WARN-D dataset is:
- longitudinal
- multi-modal
- explicitly designed to support early detection
- structured to minimize leakage

It separates:
- who a person is (Stage 1),
- how they fluctuate day-to-day (Stage 2),
- and when meaningful outcomes occur (Stage 3).

Understanding this structure is essential before building any models.

---

## 7. What to do next

If you are new to the dataset, the recommended next steps are:
1. Inspect the column names and row counts in each file.
2. Identify participant IDs and time variables.
3. Understand how Stage 2 rows map onto Stage 3 time points.
4. Only then begin preprocessing or modeling.

If anything in this README is unclear, ask questions *before* writing code.