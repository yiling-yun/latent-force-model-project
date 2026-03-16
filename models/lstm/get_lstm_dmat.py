import os
import numpy as np
import pandas as pd
import glob
import re
from scipy.spatial.distance import cdist
# from scipy.cluster.hierarchy import linkage, dendrogram
# from scipy.spatial.distance import pdist
# import matplotlib.pyplot as plt
# import scipy.stats as stats
# import seaborn as sns

emb_files = ["./results/lstm_*charades_participant*_soft_ce_loss_charades_feat.npy"]
model_names = ["lstm_core", "lstm_force", "lstm_coreAndForce"]
output_dir = './dist_mat'
n_run = 1

def load_embeddings(patterns):
    emb = {}

    for p in patterns:
        for f in glob.glob(p):
            fname = os.path.basename(f)
            full_key = os.path.splitext(fname)[0]    # full filename w/out extension

            parts = full_key.split("_")

            # Model prefix is first two parts: e.g., lstm, noOri
            if "_run" in full_key:
                cut = full_key.index("_charades")
                prefix_raw = full_key[:cut]          # keep only part before _hd
                m = re.search(r"^lstm_(core|force).*?_run(\d+)", prefix_raw)
                if m:
                    kind = m.group(1)          # "core" or "force"
                    run  = m.group(2)   # 39
                    prefix  = f"lstm_{kind}_run{run}"   # zero-pad to 2 digits: run00..run49
                else:
                    prefix = prefix_raw  # fallback
            else:
                prefix = "_".join(parts[:2])

            if "diffStim" in full_key:
                # exp6 version → add suffix
                key = prefix + "_exp6"
            else:
                # normal version
                key = prefix

            emb[key] = np.load(f, allow_pickle=True).item()

    return emb

def build_models_by_run(emb, base_keys=("lstm_core", "lstm_force"), n_runs=50):
    models = {}
    for base in base_keys:
        models[base] = {}
        if n_runs > 1:
            for r in range(n_runs):
                main_key = f"{base}_run{r}"
                exp6_key = f"{base}_run{r}_exp6"

                main = emb.get(main_key, {})
                exp6 = emb.get(exp6_key, {})

                # merged: exp6 overwrites main if same video id appears in both
                merged = (main | exp6) if (main and exp6) else (main or exp6)

                models[base][r] = merged
        else:
            main_key = f"{base}"
            exp6_key = f"{base}_exp6"

            main = emb.get(main_key, {})
            exp6 = emb.get(exp6_key, {})

            # merged: exp6 overwrites main if same video id appears in both
            merged = (main | exp6) if (main and exp6) else (main or exp6)

            models[base][0] = merged
    return models


def write_dist_csv(emb, model_name, ordered_videos):
    ordered_ids = [v.split("_")[0] for v in ordered_videos]
    # Extract embeddings
    try:
        matrix = np.stack([emb[int(id_)]["feat"] for id_ in ordered_ids])
    except Exception as e:
        matrix = np.stack([emb[int(id_)] for id_ in ordered_ids])

    # Compute cosine distance
    dist = cdist(matrix, matrix, metric="cosine")
    df = pd.DataFrame(dist, index=ordered_videos, columns=ordered_videos)

    # Write csv
    output_filename = f'lstm_distr_{model_name.replace("lstm_", "")}.csv'
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, output_filename)

    # Save the DataFrame to the file
    df.to_csv(full_path, header=False, index=False)

    print(f"✔ wrote {model_name} → {output_filename}")


if __name__ == '__main__':

    emb = load_embeddings(emb_files)
    models = build_models_by_run(emb, base_keys=("lstm_core", "lstm_force", "lstm_coreAndForce"), n_runs=1)
    
    with open("./../../utils/video_order_from_hc_human_dmat.txt") as f:
        orderedVideosAfterHC = f.read().splitlines()

    for name in model_names:
        for run in range(n_run):
            data = models[name][run]
            write_dist_csv(data, name, orderedVideosAfterHC)
