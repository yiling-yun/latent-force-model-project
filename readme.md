# Latent Force Representation Model for Social Perception

This project presents a parametric model of social perception based on the force dynamics of attraction and repulsion between two agents. The model is designed to capture the kind of social interactions famously described by Heider and Simmel (1944), where simple geometric shapes moving together evoke rich social narratives (e.g., chasing, fighting, helping).

This repository includes both the modeling code and human behavioral data from labeling and similarity judgment experiments.

## Project Structure
```
project/
├── utils/                        # shared config (video order, dataset EDA, trajectory figures)
├── human/
│   ├── behavioralExpCode/        # experiment code in HTML and JavaScript
│   │   ├── exp1/                 # labeling experiment of 1156 animations
│   │   ├── exp2/                 # labeling experiment on force-generated animations
│   │   └── exp3/                 # odd-one-out similarity judgment task
│   └── behavioralExpDataAndAnalysis/  # human data and analysis
│       ├── exp1/
│       ├── exp2/
│       └── exp3/
└── models/
    ├── force/                    # latent force model (MATLAB)
    ├── histo_static/             # control model with histogram input and MLP (Python)
    └── lstm/                     # LSTM model (Python)
```

## Requirements

- **MATLAB** with the **Parallel Computing Toolbox** — for force-feature estimation (Step 1).
- **Python 3.12** — for the LSTM model (Step 2) and analysis (Step 3). A virtual environment is provided at `.venv/`; dependencies are listed in `models/lstm/requirements.txt`.

## Pipeline

The full pipeline runs in three steps: estimate force features (MATLAB) → train the LSTM and export model dissimilarity matrices (Python) → correlate model and human judgments (notebook).

### Step 1 — Estimate force features  (`models/force/`, MATLAB)

Run **`main_force_model_est_par.m`** is the initial estimator. It produces `estpart_forcemodel.v3*.mat`, which uses the Parallel Computing Toolbox to fit the attraction/repulsion force parameters from the input trajectories. Set `input` at the top of the file for the target video set:

- `input = 'all'`  → `rst/all_1133/estpart_forcemodel.v3_all.mat`  (1133 videos; consumed by the LSTM in Step 2)
- `input = 'stim'` → `rst/exp2/estpart_forcemodel.v3.mat`  (54 selected videos; consumed by the force dissimilarity matrix below)

The resulting force estimates can use as an optional warm-start seed. It is for the final V2 estimate.

Run **`main_force_model_est_par_v2.m`** — the V2 estimator, which uses the Parallel Computing Toolbox to fit the attraction/repulsion force parameters from the input trajectories. Set `input` at the top of the file for the target video set:

- `input = 'all'`  → `rst/all_1133/estpart_forcemodel.v2_all.mat`  (1133 videos; consumed by the LSTM in Step 2)
- `input = 'stim'` → `rst/exp2/estpart_forcemodel.v2_improved.mat`  (54 selected videos; consumed by the force dissimilarity matrix below)

The script loads a saved estimate if one already exists; set `force_recompute = true` to refit from scratch.

Then build the model dissimilarity matrices that Step 3 reads:

- **`get_force_dmat.m`** → `distMat/force_hist_dist.csv`  (force-feature model)
- **`get_kinematic_feat_dmat.m`** → `distMat/kinematic_feat_hist_dist.csv`  (kinematic-feature baseline)

### Step 2 — Train the LSTM  (`models/lstm/`, Python)

```bash
cd models/lstm
../../.venv/bin/python main_cls_v2.py
```

`main_cls_v2.py` is a self-contained single-file pipeline. It reads the force estimate
(`force_mat = ../force/rst/all_1133/estpart_forcemodel.v2_all.mat`) together with the trajectory data under `data/`, trains the LSTM for each input mode (`core`, `allForce`, `coreAndForce`), and writes the model dissimilarity matrices to:

- `dist_mat/lstm_distr_core.csv`
- `dist_mat/lstm_distr_force.csv`
- `dist_mat/lstm_distr_coreAndForce.csv`

All paths are anchored to the script's location, so it can be launched from any working directory.

### Step 3 — Correlate with human judgments  (`human/behavioralExpDataAndAnalysis/exp2/analysis.ipynb`)

The notebook builds the **human** dissimilarity matrix from the odd-one-out behavioral data (`585subj_cleaned_*.csv` → `full_dmat_585subj_hcordered.csv`), imports the **model** dissimilarity matrices produced in Steps 1–2:

- `models/force/distMat/force_hist_dist.csv`
- `models/force/distMat/kinematic_feat_hist_dist.csv`
- `models/lstm/dist_mat/lstm_distr_{core,force,coreAndForce}.csv`

and reports the model–human similarity correlations and odd-one-out accuracy.

---

