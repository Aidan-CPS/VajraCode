# VajraCode
VajraCode is a research codebase for modeling depression as a dynamic process using longitudinal data. It compares theory-agnostic machine learning with theory-guided approaches inspired by psychology and yoga, emphasizing early detection, interpretability, and rigorous empirical evaluation.



## Getting Started

This section explains how to set up the VajraCode project locally and run the first baseline analysis. It is written for new collaborators and assumes no prior familiarity with the codebase.

⸻

1. Clone the repository

Clone the VajraCode repository to your local machine:

git clone https://github.com/your-username/VajraCode.git
cd VajraCode

(Replace the URL with the actual repository URL.)

⸻

2. Set up a Python environment

We recommend using a virtual environment.

Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate

(On Windows, use .venv\\Scripts\\activate.)

Install dependencies:

pip install -r requirements.txt

⸻

3. Download and place the data

The WARN-D data should live next to the VajraCode repository, not inside it.

Your directory structure should look like this:

WARN-D/
├── VajraCode/
└── WARND_data/
└── data/
├── stage1_train_x.csv
├── stage1_test_x.csv
├── stage2_train_x.csv
├── stage2_test_x.csv
├── stage3_train_x.csv
├── stage3_test_x.csv
└── stage3_train_y.csv

Do not commit any data files to GitHub.

For a detailed explanation of the dataset, see README_data.md.

⸻

4. Verify data access

Before running any models, verify that the data can be loaded correctly.

The first script you should run is a data inspection script (for example):

python scripts/load_data.py

This script should:
	•	load each stage of the data,
	•	print basic information (row counts, column names),
	•	confirm that participant IDs and time indices are present.

⸻

5. Run the first baseline model (MML v1)

The first modeling goal is to run a simple, theory-agnostic baseline.

This baseline:
	•	aggregates Stage 2 data (EMA + wearables) up to each Stage 3 time point,
	•	combines these aggregates with Stage 1 baseline features,
	•	uses a simple linear or logistic regression model,
	•	evaluates performance using AUC-ROC.

You can run the baseline model using:

python scripts/run_mml_baseline.py

This establishes a reference point against which all theory-guided models (WML, YML) will be compared.

⸻

6. What to do next

After the MML baseline is running successfully:
	1.	Inspect predictions and evaluation metrics.
	2.	Review data aggregation logic.
	3.	Begin implementing theory-guided constructs (starting with bodily tamas).

See the main README for the project roadmap and modeling philosophy.










## Data Overview

This project uses the WARN-D dataset, a longitudinal, multi-modal dataset designed to support early prediction of depression while minimizing data leakage. The data are already de-identified and split into training (70%) and test (30%) sets.

The dataset is organized into three stages, each corresponding to a different timescale:
- **Stage 1 (Baseline):** One row per participant containing static questionnaire data (e.g. demographics, mental health history, personality, coping styles). These variables describe who the person is at baseline.
- **Stage 2 (Daily Dynamics)**: Many rows per participant (≈350), each corresponding to an EMA (Ecological Momentary Assessment) survey moment. These rows include short self-reports (mood, stress, fatigue, cognition, context) and wearable-derived features aggregated over the 120 minutes prior to each EMA prompt. This stage captures fine-grained day-to-day fluctuations.
- **Stage 3 (Outcomes):** Eight rows per participant, each corresponding to a quarterly or yearly follow-up. These rows define the outcome timeline. For training data, a separate file provides a binary depression onset label (0/1) at each Stage-3 time point.

The core modeling task is to aggregate Stage-2 data up to each Stage-3 time point, combine those aggregates with Stage-1 baseline features, and predict whether a depression onset occurs at that time point. Test data include no outcome labels and are truncated to prevent future leakage.

For full details on file structure, columns, and modeling constraints, see README_data.md.


## Repository Structure

This section explains how the VajraCode repository is organized and how the WARN-D data should live alongside the codebase. The goal is to make it immediately clear where code goes, where data live, and how contributors should work without accidentally committing large or sensitive files.

⸻

### High-level layout

The recommended setup places the data directory next to the GitHub repository, not inside it:

WARN-D/
├── VajraCode/          # GitHub repository (code only)
└── WARND_data/         # Local data directory (not tracked by git)

This separation ensures:
	•	no accidental data commits,
	•	clear boundaries between code and data,
	•	consistent setup across collaborators.

⸻

### VajraCode repository

VajraCode/
├── README.md
├── README_data.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── scripts/
│   ├── load_data.py
│   ├── aggregate_stage2.py
│   └── utils.py
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_missingness.ipynb
│   └── 03_mml_baseline.ipynb
├── models/
│   ├── mml_linear.py
│   └── yml_tamas.py
├── evaluation/
│   └── metrics.py
└── outputs/
    ├── figures/
    └── predictions/

### Key directories
- README.md: High-level project overview, goals, workflow, and links to detailed documentation.
- README_data.md: Detailed explanation of the WARN-D dataset structure and semantics.
- requirements.txt: Python dependencies required to run the code.
- data/README.md: Explains where the data live locally and how to access them. No actual data files are stored here.
- scripts/: Reusable Python scripts for data loading, aggregation, and preprocessing.
- notebooks/: Exploratory and explanatory notebooks. Each notebook should correspond to a clear task or stage.
- models/: Model definitions (e.g. MML baseline, YML tamas models).
- evaluation/: Shared evaluation and scoring utilities.
- outputs/: Generated figures, logs, and prediction files (not committed by default).

⸻

## Data directory (local only)

WARND_data/
└── data/
    ├── stage1_train_x.csv
    ├── stage1_test_x.csv
    ├── stage2_train_x.csv
    ├── stage2_test_x.csv
    ├── stage3_train_x.csv
    ├── stage3_test_x.csv
    ├── stage3_train_y.csv
    └── sample/

	•	This directory is not tracked by git.
	•	Each collaborator should download the data separately and place it here.
	•	Code in VajraCode should assume this relative path:

../WARND_data/data/


⸻

## Design principles
	•	Code and data are strictly separated.
	•	All paths are explicit and relative.
	•	Nothing in WARND_data/ is committed to GitHub.
	•	Anyone cloning the repo can recreate the setup by:
	1.	cloning VajraCode,
	2.	downloading WARN-D data,
	3.	placing it in WARND_data/data/.

⸻

For details on the dataset itself, see README_data.md.
For modeling workflow and baselines, see README.md.