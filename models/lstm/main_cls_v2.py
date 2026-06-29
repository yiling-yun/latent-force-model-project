"""
main_cls_v2.py  (self-contained)

Single-file LSTM pipeline (no dependency on main_cls_v1). It merges the old
3-step pipeline (get_pickle_json.ipynb + data_normalization.py + main_cls.py)
into one in-memory run, with TWO conveniences:

  1) Force-based frame cleaning (clean_zero_data, which drops timesteps where ALL
     force params are 0) is applied ONLY for input_mode == "allForce".
       - input_mode == "allForce"               -> clean_zero_data applied
       - input_mode in {"core", "coreAndForce"} -> clean_zero_data SKIPPED
     The x5 temporal downsampling still applies to all modes; only the force-zero
     frame removal is conditional.

  2) MULTIPLE input modes can be trained in ONE invocation via `input_modes`.
     Each distinct dataframe (clean for allForce / no-clean for the others) is
     built once and reused; each mode re-seeds before training so its result is
     identical to running that mode standalone. A summary table prints at the end.

Run from models/lstm/:  python main_cls_v2.py
"""
import os
import ast
import warnings
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
# Silence the harmless "DataFrame is highly fragmented" hints during data prep.
warnings.simplefilter("ignore", PerformanceWarning)
import math
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from collections import defaultdict, deque
from tqdm import tqdm
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Sampler
import matplotlib.pyplot as plt


# Anchor all relative paths to this script's directory so the script can be run
# from any working directory (inputs under data/, outputs under results/, etc.).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Inlined dependencies — these were previously imported from utils.py, model.py,
# and data.py. They are copied here verbatim (only the subset this script uses)
# so main_cls_v2.py is fully stand-alone and those files can be removed.
# =============================================================================

# ----------------------------- from utils.py ---------------------------------
def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seed for reproducibility across random, numpy, and torch.
    """
    # --- Python and system-level ---
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # --- NumPy ---
    np.random.seed(seed)

    # --- PyTorch (CPU + CUDA) ---
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # all GPUs

    # --- Torch determinism settings ---
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # This may give better performance, but less reproducibility
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # --- Optional: control global hashing / hash randomization ---
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass  # TensorFlow not installed, ignore

    print(f"✅ Seeds set to {seed} (deterministic={deterministic})")


def soft_ce_loss(logits, y_probs, eps=1e-8):
    # y_probs: [B, 30], each row sums to 1
    log_q = F.log_softmax(logits, dim=-1)           # stable log-softmax
    y_safe = torch.clamp(y_probs, min=eps)          # avoid log(0) in constant term
    return -(y_safe * log_q).sum(dim=-1).mean()     # H(y, q) = -E_y[log q]


class EarlyStopping:
    def __init__(self, patience=50, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0


def plot_smoothed_loss(e, n_epochs, train_losses, val_losses, is_acc=False, filename="results.png"):
    epochs = np.arange(1, e + 1)

    # Plot smoothed training and valing losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train')
    plt.plot(epochs, val_losses, label='Validation')
    plt.xlim([0, n_epochs])
    plt.xlabel('Epochs')
    if is_acc:
        plt.ylabel('Accuracy')
    else:
        plt.ylabel('Loss')
    plt.title('Smoothed Training and Validation')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(filename)
    plt.close()


# ----------------------------- from model.py ---------------------------------
class SocialGNN(nn.Module):
    def __init__(self, input_size, output_size, spatial_dim=12, hidden_dim=6, n_layers=1, drop_prob=0.0, apply_sigmoid=True):
        super(SocialGNN, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.apply_sigmoid = apply_sigmoid

        self.Gspatial = nn.Linear(input_size, spatial_dim)
        self.Gtemporal = nn.LSTM(spatial_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.drop = nn.Dropout(p=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, lengths=None):
        batch_size, time_step, input_size = x.shape
        x = self.Gspatial(x.reshape(-1, input_size))   # spatial embedding
        x = x.view(batch_size, time_step, -1)
        hidden = self.init_hidden(batch_size)

        if lengths is not None:
            # use pack_padded_sequence for variable-length handling
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.Gtemporal(packed, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            # take last valid timestep per sequence
            idx = (lengths - 1).clamp(min=0)
            feat = lstm_out[torch.arange(batch_size), idx]
        else:
            lstm_out, _ = self.Gtemporal(x, hidden)
            feat = lstm_out[:, -1]  # last timestep

        out = self.drop(feat)
        out = self.fc(out)
        if self.apply_sigmoid:
            out = torch.sigmoid(out)
        return out, feat

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        device = next(self.parameters()).device
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=6, n_layers=1, drop_prob=0.0, apply_sigmoid=True):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.apply_sigmoid = apply_sigmoid

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers,
                            dropout=drop_prob if n_layers > 1 else 0,
                            batch_first=True)

        self.drop = nn.Dropout(p=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)

        self._init_weights()

    def forward(self, x, lengths=None):
        if lengths is not None:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

            # output contains the packed hidden states for all steps
            # hidden contains the FINAL hidden state for each sequence in the batch
            output, (hidden, cell) = self.lstm(x)

            feat = hidden[-1]
        else:
            _, (hidden, cell) = self.lstm(x)
            feat = hidden[-1]

        out = self.drop(feat)
        out = self.fc(out)
        if self.apply_sigmoid:
            out = torch.sigmoid(out)
        return out, feat

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights
                nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights (Orthogonal is often preferred for LSTM)
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Initialize biases to zero
                nn.init.constant_(param.data, 0)
                # Forget gate bias trick: set to 1 to help long-term dependencies
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)
            elif 'fc.weight' in name:
                # Output layer
                nn.init.kaiming_uniform_(param.data, nonlinearity='relu')


# ----------------------------- from data.py ----------------------------------
def read_and_concat(filename):
    if isinstance(filename, pd.DataFrame):
        df = filename
    elif filename.endswith('.csv'):
        df = pd.read_csv(filename)
    elif filename.endswith('.xlsx'):
        df = pd.read_excel(filename, engine="openpyxl")
    elif filename.endswith('.pkl'):
        df = pd.read_pickle(filename)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    results = []
    for i, row in df.iterrows():
        id_name = 'id' if 'id' in row else 'ID'
        results.append(f"{row[id_name]}_{row['label']}")
    return results


def process_aggregated_labels(filename):
    df = pd.read_csv(filename, sep="\t")
    df.rename(columns={df.columns[0]: "id_label"}, inplace=True)
    categories = df.columns[1:].tolist()

    result = {}
    for idx, row in df.iterrows():
        video_name = row["id_label"]
        values = row.drop("id_label").to_list()
        result[video_name] = np.array(values)
    return result, categories


class CharadeV2(Dataset):
    def __init__(self, traj_filename, video_names, label_distribution_file=None, input_mode="", norm=True,
                 use_relative=False, reverse=False, input_format = "excel", force_norm_suffix="_log_norm"):  # YY
        self.traj_filename = traj_filename
        self.fps = 50
        self.norm = norm
        self.use_relative = use_relative
        self.reverse = reverse  # YY
        self.input_format = input_format
        self.input_mode = input_mode
        self.force_norm_suffix = force_norm_suffix  # which force transform's _norm columns to read (default log)
        self.id_label_mapping, self.categories = None, None

        if label_distribution_file is not None:
            self.id_label_mapping, self.categories = process_aggregated_labels(label_distribution_file)

        if isinstance(video_names, str):
            video_names = read_and_concat(video_names)
        all_video_names = read_and_concat(traj_filename)

        if isinstance(traj_filename, pd.DataFrame):
            data = traj_filename.reset_index(drop=True)  # accept an in-memory df (positional indexing below needs a RangeIndex)
        elif input_format == "excel":
            data = pd.read_excel(traj_filename, engine="openpyxl")
        elif input_format == "pkl":
            data = pd.read_pickle(traj_filename)
        usable = data['usable']

        self.data = []
        for i in range(len(usable)):
            use = usable[i]
            if use not in [0, 1] or all_video_names[i] not in video_names:
                continue

            if self.id_label_mapping is not None and all_video_names[i] not in self.id_label_mapping:
                continue

            if input_mode in ["traj", "core", "visual", "noOri", "distNoOri", "noOriNoDS", "distVel", "locOnly", "velOnly", "distOnly", "accOnly", "scrambled", "random"]:
                d = self.get_traj(data, i)
            elif input_mode == "allForce" or input_mode == "interactiveOnly" or input_mode == "interSelfA":
                d = self.get_force(data, i)
            elif input_mode in["trajAndForce", "coreAndForce", "trajLocationOnlyAndForce", "trajNoOriAndForce"]:
                d_traj = self.get_traj(data, i)
                d_force = self.get_force(data, i)
                assert d_traj.shape[0] == d_force.shape[0], f"{int(data['id'][i])} traj length {d_traj.shape[0]} and force length {d_force.shape[0]} don't match"
                d = np.concatenate([d_traj, d_force], axis=1)

            id = int(data['id'][i])
            videoLabel = data['label'][i]
            label = data['label'][i] if self.id_label_mapping is None else self.id_label_mapping[all_video_names[i]]
            cur_data = {'data': d, 'id': id, 'videoLabel': videoLabel, 'label': label}
            self.data.append(cur_data)

        print(f"Processed {self.__len__()} data")
        if self.norm:
            print("Using normalized data")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id = self.data[idx]['id']
        data = self.data[idx]['data']
        videoLabel = self.data[idx]['videoLabel']
        label = self.data[idx]['label']

        if self.use_relative:
            split = int(data.shape[1] / 2)
            data = self.calculate_relative_data(data)
            data = data[:, split:]

        return torch.FloatTensor(data), id, videoLabel, label

    def get_traj(self, data, i):
        if self.norm:
            keys = ['x1_norm', 'y1_norm', 'x2_norm', 'y2_norm', 'dist_norm', 'vx1_norm', 'vy1_norm', 'vx2_norm', 'vy2_norm', 'ax1_norm', 'ay1_norm', 'ax2_norm', 'ay2_norm', 'ori1_norm', 'ori2_norm']
            x1, y1, x2, y2, dist, vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2, ori1, ori2 = [data[k][i] for k in keys]
        else:
            keys = ['x1', 'y1', 'ori1', 'x2', 'y2', 'ori2']
            if self.input_format == "excel":
                x1, y1, ori1, x2, y2, ori2 = [self.process_traj(data[k][i]) for k in keys]
            elif self.input_format == "pkl":
                x1, y1, ori1, x2, y2, ori2 = [data[k][i] for k in keys]
            dist = [math.sqrt((x2[i] - x1[i]) ** 2 + (y2[i] - y1[i]) ** 2) for i in range(len(x1))]

            x1, y1 = np.array(x1), np.array(y1)
            x2, y2 = np.array(x2), np.array(y2)

            def get_velocity_copy_first(pos, fps):
                # Calculate difference between adjacent elements
                diffs = np.diff(pos) * fps
                # Prepend the first calculated difference to the start of the array
                return np.insert(diffs, 0, diffs[0]) if diffs.size > 0 else diffs

            # Calculate Velocities
            vx1, vy1 = get_velocity_copy_first(x1, self.fps), get_velocity_copy_first(y1, self.fps)
            vx2, vy2 = get_velocity_copy_first(x2, self.fps), get_velocity_copy_first(y2, self.fps)

            # Calculate Accelerations (using the same logic on the velocities)
            ax1, ay1 = get_velocity_copy_first(vx1, self.fps), get_velocity_copy_first(vy1, self.fps)
            ax2, ay2 = get_velocity_copy_first(vx2, self.fps), get_velocity_copy_first(vy2, self.fps)

        if self.reverse:
            for lst in [x1, y1, ori1, vx1, vy1, x2, y2, ori2, vx2, vy2]:
                lst.reverse()

        if self.input_mode == "traj" or self.input_mode == "trajAndForce":
            d = np.array([x1, y1, ori1, vx1, vy1, x2, y2, ori2, vx2, vy2])
        elif "core" in self.input_mode:
            d = np.array([vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2, dist])
        elif self.input_mode == "visual":
            d = np.array([x1, y1, x2, y2, vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2])
        elif self.input_mode in ["noOri", "trajNoOriAndForce", "noOriNoDS"]:
            d = np.array([x1, y1, vx1, vy1, x2, y2, vx2, vy2])
        elif self.input_mode == "distNoOri":
            d = np.array([x1, y1, vx1, vy1, x2, y2, vx2, vy2, dist])
        elif self.input_mode == "distVel":
            d = np.array([vx1, vy1, vx2, vy2, dist])
        elif self.input_mode == "trajLocationOnlyAndForce" or self.input_mode == "locOnly":
            d = np.array([x1, y1, x2, y2])
        elif self.input_mode == "velOnly":
            d = np.array([vx1, vy1, vx2, vy2])
        elif self.input_mode == "distOnly":
            d = np.array([dist])
        elif self.input_mode == "accOnly":
            d = np.array([ax1, ay1, ax2, ay2])
        elif self.input_mode == "scrambled":
            d = np.array([x1, y1, vx2, vy2])
        elif self.input_mode == "random":
            d = np.array([x1, y1, x2, y2])
            # d = np.random.permutation(d) # this only shuffled the order of four lists
            d = d[:, np.random.permutation(d.shape[1])] # this shuffles values within each of the four lists
        d = np.transpose(d)

        if self.input_mode in ["core", "coreAndForce", "visual", "noOriNoDS", "locOnly", "velOnly", "distOnly", "accOnly", "scrambled", "random"]:
            return d
        else: # select rows in d so that it takes step of 5, starting from the 6th frame, end at the n-5th frame
            intv = 5
            if len(d) < 2*intv + 1:
                intv = 2
            d_selected = d[np.arange(intv, len(d) - intv,intv)]
            return d_selected

    def get_force(self, data, i):
        keys = [
            'epsilon_selfA', 'sigma_selfA', 'bcoef_selfA',
            'epsilon_inter', 'sigma_inter', 'bcoef_inter',
            'epsilon_selfB', 'sigma_selfB', 'bcoef_selfB'
        ]

        if self.input_format == "excel":
            (epsilon_selfA, sigma_selfA, bcoef_selfA,
             epsilon_inter, sigma_inter, bcoef_inter,
             epsilon_selfB, sigma_selfB, bcoef_selfB) = [self.process_traj(data[k][i]) for k in keys]
        elif self.input_format == "pkl":
            if self.norm:
                keys = [k + self.force_norm_suffix for k in keys]
            (epsilon_selfA, sigma_selfA, bcoef_selfA,
             epsilon_inter, sigma_inter, bcoef_inter,
             epsilon_selfB, sigma_selfB, bcoef_selfB) = [data[k][i] for k in keys]

        if self.input_mode == "interSelfA":
            d = np.array([epsilon_selfA, sigma_selfA, bcoef_selfA, epsilon_inter, sigma_inter, bcoef_inter])
        elif self.input_mode == "interactiveOnly":
            d = np.array([epsilon_inter, sigma_inter, bcoef_inter])
        elif self.input_mode in ["allForce", "coreAndForce", "trajAndForce", "trajLocationOnlyAndForce", "trajNoOriAndForce"]:
            d = np.array([epsilon_selfA, sigma_selfA, bcoef_selfA, epsilon_inter, sigma_inter, bcoef_inter, epsilon_selfB, sigma_selfB, bcoef_selfB])

        d = np.transpose(d)
        return d

    def calculate_relative_data(self, data):
        split = int(data.shape[1] / 2)
        first_half = data[:, :split]
        second_half = data[:, split:]
        second_half = second_half - first_half

        # change angle to the same scale
        scale = 1. # if self.norm else 2 * np.pi
        ori2 = second_half[:, 2]
        ori2[ori2 < 0] += scale
        second_half[:, 2] = ori2

        data = np.concatenate([first_half, second_half], axis=1)
        return data

    def process_traj(self, traj):
        if isinstance(traj, str):
            # Safely evaluates string representations like "[1.2, 3.4]" into lists
            return np.array(ast.literal_eval(traj), dtype=float)
        return traj


def cls_collate_fn(batch):
    # batch: list of (data, id, videoLabel, label)
    datas, ids, videoLabels, labels = zip(*batch)

    # convert to tensors
    datas = [torch.as_tensor(d).float() for d in datas]     # each [T, 9]
    lengths = torch.tensor([d.shape[0] for d in datas], dtype=torch.long)

    # pad to [B, T_max, 9]
    datas_padded = pad_sequence(datas, batch_first=True)    # pads with 0

    labels = torch.stack([torch.as_tensor(l).float() for l in labels])  # [B, C]

    return datas_padded, lengths, ids, videoLabels, labels


class BalancedBatchSampler(Sampler):
    """
    Yields batches of indices with class-coverage constraints:

    - If batch_size < num_classes:
        each item in a batch comes from a different class (no repeats in-batch).
    - If batch_size >= num_classes:
        each (non-empty) class appears at least once per batch, and the remaining
        slots are filled round-robin from classes that still have samples.

    Uses every sample exactly once per epoch (unless drop_last=True).
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.shuffle = shuffle
        self.drop_last = drop_last

        self.labels = []
        for i in range(len(dataset)):
            _, _, videoLabel, _ = dataset[i]
            self.labels.append(videoLabel)

        # Build indices per class
        class_indices = defaultdict(list)
        for idx, y in enumerate(self.labels):
            class_indices[y].append(idx)

        self.class_indices = class_indices
        self.classes = sorted(self.class_indices.keys())
        self.num_classes = len(self.classes)

        self.dataset_size = len(self.labels)

        if drop_last:
            self._num_batches = self.dataset_size // self.batch_size
        else:
            self._num_batches = (self.dataset_size + self.batch_size - 1) // self.batch_size

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        # Make per-class deques for fast pops
        pools = {}
        for c in self.classes:
            idxs = list(self.class_indices[c])
            if self.shuffle:
                np.random.shuffle(idxs)
            pools[c] = deque(idxs)

        # Active classes (still have samples)
        active = [c for c in self.classes if len(pools[c]) > 0]
        if not active:
            return

        rr_ptr = 0  # round-robin pointer over active classes

        def advance_to_next_nonempty(start_ptr):
            """Advance rr_ptr to a class that still has samples."""
            if not active:
                return 0
            p = start_ptr % len(active)
            # active list only holds non-empty pools, so this is safe
            return p

        def pop_one(c):
            """Pop one index from class c, updating active list if it becomes empty."""
            nonlocal rr_ptr
            idx = pools[c].popleft()
            if len(pools[c]) == 0:
                # remove from active and fix rr_ptr if needed
                remove_pos = active.index(c)
                active.pop(remove_pos)
                if active:
                    if remove_pos < rr_ptr:
                        rr_ptr -= 1
                    rr_ptr %= len(active)
                else:
                    rr_ptr = 0
            return idx

        while active:
            batch = []

            if self.batch_size < self.num_classes:
                # Need batch_size distinct classes. Take next distinct classes from round-robin.
                k = min(self.batch_size, len(active))
                used_classes = []

                rr_ptr = advance_to_next_nonempty(rr_ptr)
                # collect k distinct classes starting from rr_ptr
                for t in range(k):
                    c = active[(rr_ptr + t) % len(active)]
                    used_classes.append(c)

                # advance rr_ptr for next batch
                rr_ptr = (rr_ptr + k) % len(active)

                for c in used_classes:
                    batch.append(pop_one(c))

            else:
                # batch_size >= num_classes:
                # ensure each non-empty class appears at least once
                current_active_snapshot = list(active)  # stable snapshot for "at least once"
                for c in current_active_snapshot:
                    batch.append(pop_one(c))
                    if not active:  # everything exhausted exactly at boundary
                        break

                # fill remaining slots round-robin from remaining active classes
                while active and len(batch) < self.batch_size:
                    rr_ptr = advance_to_next_nonempty(rr_ptr)
                    c = active[rr_ptr]
                    batch.append(pop_one(c))
                    if active:
                        rr_ptr %= len(active)

            if self.drop_last and len(batch) < self.batch_size:
                break

            yield batch


# =============================================================================
# End inlined dependencies
# =============================================================================


INPUT_MODE_CONFIG = {
    "core": ("_core", 9),
    "coreAndForce": ("_coreAndForce", 18),
    "allForce": ("_force", 9),
}

# 11 force columns in the order they appear in the MATLAB `estpara` cell columns.
FORCE_COLUMNS_11 = [
    'epsilon_selfA', 'sigma_selfA', 'bcoef_selfA',
    'epsilon_inter', 'sigma_inter', 'bcoef_inter',
    'dev_selfAinter',
    'epsilon_selfB', 'sigma_selfB', 'bcoef_selfB',
    'dev_selfB',
]
# The 9 force parameters actually normalized / consumed downstream.
FORCE_PARAMETERS = [
    'epsilon_selfA', 'sigma_selfA', 'bcoef_selfA',
    'epsilon_inter', 'sigma_inter', 'bcoef_inter',
    'epsilon_selfB', 'sigma_selfB', 'bcoef_selfB',
]
VISUAL_COLS = ["x1", "y1", "x2", "y2"]


# ============================================================================
# Data-prep helpers (verbatim from the merged pipeline)
# ============================================================================
def _safe_first_repeat(arr: np.ndarray) -> np.ndarray:
    """Overwrite the first element with the second (mimics 'copy first' safely)."""
    n = arr.size
    if n <= 1:
        return arr
    arr[0] = arr[1]
    return arr


def check_extremes(arr, name):
    """Prints a warning if the array contains NaNs, Infs, or massive values."""
    if len(arr) == 0:
        return
    if not np.isfinite(arr).all():
        print(f"!!! [NON-FINITE] Found NaN/Inf in: {name}")
    abs_max = np.abs(arr).max()
    if abs_max > 1e150:
        idx = np.argmax(np.abs(arr))
        print(f"!!! [EXTREME] {name} has massive value: {abs_max:.2e} at index {idx}")


def downsample_visual_features(row, intv=5, padding=True):
    start = intv
    step = intv
    for col in ["x1", "y1", "x2", "y2"]:
        data = np.asarray(row[col])
        if padding:
            pad_len = 2 * intv + 1
            pad_values = np.full(pad_len, data[0])
            data = np.concatenate([pad_values, data])
        stop = len(data) - intv
        row[col] = data[start:stop:step]
    return row


def compute_row_features(row: pd.Series) -> pd.Series:
    x1 = np.asarray(row["x1"], dtype=float)
    y1 = np.asarray(row["y1"], dtype=float)
    x2 = np.asarray(row["x2"], dtype=float)
    y2 = np.asarray(row["y2"], dtype=float)

    check_extremes(x1, "x1")
    check_extremes(y1, "y1")

    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    check_extremes(dist, "dist")

    vx1 = np.zeros_like(x1); vy1 = np.zeros_like(y1)
    vx2 = np.zeros_like(x2); vy2 = np.zeros_like(y2)

    n = len(x1)
    if n == 0:
        return pd.Series({
            "dist": dist.tolist(),
            "vx1": [], "vy1": [], "vx2": [], "vy2": [],
            "ax1": [], "ay1": [], "ax2": [], "ay2": []
        })
    if n == 1:
        return pd.Series({
            "dist": dist.tolist(),
            "vx1": [0.0], "vy1": [0.0], "vx2": [0.0], "vy2": [0.0],
            "ax1": [0.0], "ay1": [0.0], "ax2": [0.0], "ay2": [0.0],
        })

    vx1[1:] = np.diff(x1); vy1[1:] = np.diff(y1)
    vx2[1:] = np.diff(x2); vy2[1:] = np.diff(y2)
    vx1 = _safe_first_repeat(vx1); vy1 = _safe_first_repeat(vy1)
    vx2 = _safe_first_repeat(vx2); vy2 = _safe_first_repeat(vy2)

    check_extremes(vx1, "vx1"); check_extremes(vx2, "vx2")
    check_extremes(vy1, "vy1"); check_extremes(vy2, "vy2")

    ax1 = np.zeros_like(vx1); ay1 = np.zeros_like(vy1)
    ax2 = np.zeros_like(vx2); ay2 = np.zeros_like(vy2)

    ax1[:-1] = np.diff(vx1); ay1[:-1] = np.diff(vy1)
    ax2[:-1] = np.diff(vx2); ay2[:-1] = np.diff(vy2)
    ax1 = _safe_first_repeat(ax1); ay1 = _safe_first_repeat(ay1)
    ax2 = _safe_first_repeat(ax2); ay2 = _safe_first_repeat(ay2)

    check_extremes(ax1, "ax1"); check_extremes(ax2, "ax2")
    check_extremes(ay1, "ay1"); check_extremes(ay2, "ay2")

    return pd.Series({
        "dist": dist.tolist(),
        "vx1": vx1.tolist(), "vy1": vy1.tolist(),
        "vx2": vx2.tolist(), "vy2": vy2.tolist(),
        "ax1": ax1.tolist(), "ay1": ay1.tolist(),
        "ax2": ax2.tolist(), "ay2": ay2.tolist(),
    })


def process_traj(traj):
    if isinstance(traj, str):
        return np.array(ast.literal_eval(traj), dtype=float)
    return traj


def normalize_list_column_to_newcol(data, col, suffix="_norm", eps=1e-12, ddof=0):
    """Column-wise z-score normalization for a Series of lists/arrays; writes f"{col}{suffix}"."""
    arrs = []
    try:
        for v in data[col]:
            try:
                v = [float(d) for d in v]
            except Exception:
                v = process_traj(v)
            arrs.append(np.array(v))
    except Exception as e:
        print("\n" + "=" * 30)
        print(f"CRITICAL ERROR on col {col} value: {v}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("=" * 30)
        raise

    lengths = np.fromiter((a.size for a in arrs), dtype=int)
    total = int(lengths.sum())
    out_col = f"{col}{suffix}"

    if total == 0:
        data[out_col] = [[] for _ in range(len(data))]
        return np.nan, np.nan

    flat = np.concatenate([a for a in arrs if a.size > 0], axis=0)
    mu = float(flat.mean())
    sigma = float(flat.astype(np.float64).std(ddof=ddof))

    if sigma < eps:
        transformed = np.zeros_like(flat)
    else:
        transformed = (flat - mu) / sigma

    split_indices = np.cumsum(lengths)[:-1]
    split_chunks = np.split(transformed, split_indices)
    data[out_col] = [chunk.tolist() for chunk in split_chunks]
    return mu, sigma


def clean_zero_data(data, check_columns, all_sequence_columns):
    """Drop timestep indices where ALL `check_columns` are 0, trimming `all_sequence_columns` in sync."""
    def clean_row(row):
        lists = [row[col] for col in check_columns]
        length = len(lists[0])
        keep_indices = [
            i for i in range(length)
            if not all(lst[i] == 0 for lst in lists)
        ]
        for col in all_sequence_columns:
            row[col] = [row[col][i] for i in keep_indices]
        return row

    return data.apply(clean_row, axis=1)


# ============================================================================
# Stage 1: build the "full" dataframe (replaces get_pickle_json.ipynb)
# ============================================================================
def build_full_dataframe(summary_xlsx, force_mat):
    df_traj = pd.read_excel(summary_xlsx, sheet_name="summary")

    # Convert stringified trajectory lists to real float lists
    for col in VISUAL_COLS:
        df_traj[col] = df_traj[col].apply(
            lambda x: [float(v) for v in ast.literal_eval(x)] if isinstance(x, str)
            else [float(v) for v in x] if isinstance(x, list)
            else x
        )

    # MATLAB force estimates: estpara is a cell array, each cell an (n_frames, 11) ndarray.
    mat = loadmat(force_mat)
    estpara_flat = mat['estpara'].ravel()
    force_para = []
    for i in range(estpara_flat.size):
        para = estpara_flat[i]
        if para.size == 0:
            continue  # skip empty cell
        para_np = np.array(para)
        force_para.append({name: para_np[:, j].tolist()
                           for j, name in enumerate(FORCE_COLUMNS_11)})
    df_force = pd.DataFrame(force_para)

    return pd.concat([df_traj, df_force], axis=1)


# ============================================================================
# Stage 2: normalize (replaces data_normalization.py) -- clean_zero conditional
# ============================================================================
def normalize_dataframe(data, apply_clean_zero=True):
    """clean_zero_data is applied only when apply_clean_zero=True. With
    apply_clean_zero=False the downsampled frames are kept in full."""
    with np.errstate(all="raise"):
        # x5 temporal downsample of the visual trajectory (all modes)
        data[VISUAL_COLS] = data[VISUAL_COLS].apply(downsample_visual_features, axis=1)

        for _, row in data.iterrows():
            assert len(row["x1"]) == len(row["epsilon_selfA"]), \
                f'Size mismatch, x1: {len(row["x1"])} != {len(row["epsilon_selfA"])}'

        # FORCE-ZERO CLEANING -- conditional (applied for allForce only)
        if apply_clean_zero:
            all_time_cols = FORCE_PARAMETERS + VISUAL_COLS
            data = clean_zero_data(data, check_columns=FORCE_PARAMETERS,
                                   all_sequence_columns=all_time_cols)

        # force-parameter transforms (unused by core, but harmless to compute)
        methods = ['original', 'square root', 'cube root', 'log', 'quantile']
        for para in FORCE_PARAMETERS:
            print(f'Processing {para}...')
            lengths = [len(v) for v in data[para]]
            all_data = np.concatenate(data[para].values)
            split_indices = np.cumsum(lengths)[:-1]
            for method in methods:
                if method == 'original':
                    transformed = all_data
                elif method == 'square root':
                    transformed = np.sqrt(all_data)
                elif method == 'cube root':
                    transformed = np.cbrt(all_data)
                elif method == 'log':
                    min_val = np.min(all_data)
                    if min_val < 0:
                        transformed = np.log(all_data - min_val + 1)
                    else:
                        transformed = np.log(all_data + 1)
                elif method == 'quantile':
                    qt = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
                    transformed = qt.fit_transform(all_data.reshape(-1, 1)).flatten()
                else:
                    raise ValueError(f'Method {method} not supported')

                method_slug = method.replace(" ", "_")
                data[f"{para}_{method_slug}"] = [arr.tolist() for arr in np.split(transformed, split_indices)]

                mean_val = np.mean(transformed)
                std_val = np.std(transformed)
                if std_val > 0:
                    standardized = (transformed - mean_val) / std_val
                else:
                    standardized = transformed - mean_val
                data[f"{para}_{method_slug}_norm"] = [arr.tolist() for arr in np.split(standardized, split_indices)]

        # visual / kinematic features + their z-score normalization
        new_cols = data.apply(compute_row_features, axis=1)
        data = pd.concat([data, new_cols], axis=1)
        cols_to_norm = ["x1", "y1", "x2", "y2", "dist", "vx1", "vy1", "vx2", "vy2",
                        "ax1", "ay1", "ax2", "ay2", "ori1", "ori2"]
        for c in cols_to_norm:
            if c not in data.columns:
                raise KeyError(f"Missing column: {c}")
            normalize_list_column_to_newcol(data, c, suffix="_norm", eps=1e-12, ddof=0)
    return data


def prepare_normalized_dataframe(summary_xlsx, force_mat, apply_clean_zero=True):
    """Stages 1+2 in memory; clean_zero_data is conditional via apply_clean_zero."""
    full = build_full_dataframe(summary_xlsx, force_mat)
    return normalize_dataframe(full, apply_clean_zero=apply_clean_zero)


# ============================================================================
# Training / evaluation
# ============================================================================
def val(model, dataset, device, batch_size=1):
    val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    e_loss = 0
    model.eval()

    acc_list = []
    for data, id, videoLabel, label in val_dataloader:
        data, label = data.float().to(device), label.float().to(device)

        logits, anchor_hidden = model(data)
        loss = soft_ce_loss(logits, label)

        preds = torch.softmax(logits, dim=-1)
        pred_class = preds.argmax(dim=-1)
        true_class = label.argmax(dim=-1)
        acc_list.append((pred_class == true_class).float())  # keep on-GPU; concat/sync once below

        e_loss += loss.detach()  # accumulate on-GPU; avoid per-iter sync

    e_loss = float(e_loss) / float(len(val_dataloader))  # single sync per validation pass
    acc = torch.cat(acc_list).mean().item()
    return e_loss, acc


def train(model, train_dataset, val_dataset, model_save_path, n_epochs, lr, batch_size, device, filename, log_dir=os.path.join(SCRIPT_DIR, 'results')):
    os.makedirs(log_dir, exist_ok=True)

    batch_sampler = BalancedBatchSampler(train_dataset, batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=cls_collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,  # keep workers alive across epochs (avoids respawn overhead)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.4 * n_epochs), gamma=0.1)
    early_stopping = EarlyStopping(patience=int(0.5 * n_epochs), verbose=False)

    min_loss = 100
    max_acc = 0
    e_loss_list = []
    val_e_loss_list = []
    val_acc_list = []
    pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for e in pbar:
        model.train()
        e_loss = 0

        for data, lengths, id, videoLabel, label in train_dataloader:
            data, label, lengths = data.float().to(device, non_blocking=True), label.float().to(device, non_blocking=True), lengths.to(device)

            logits, anchor_hidden = model(data, lengths=lengths)
            loss = soft_ce_loss(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            e_loss += loss.detach()  # keep on-GPU; avoid per-iter CPU<->GPU sync

        e_loss = float(e_loss) / len(train_dataloader)  # single sync per epoch
        e_loss_list.append(e_loss)
        scheduler.step()

        val_e_loss, val_acc = val(model, val_dataset, device)
        val_e_loss_list.append(val_e_loss)
        val_acc_list.append(val_acc)

        pbar.set_postfix({
            "train_loss": f"{e_loss:.4f}",
            "val_loss": f"{val_e_loss:.4f}",
            "val_acc": f"{val_acc:.3f}"
        })

        if min_loss >= val_e_loss:
            min_loss = val_e_loss
            torch.save(model.state_dict(), model_save_path)

        if max_acc <= val_acc:
            max_acc = val_acc
        torch.save(model.state_dict(), model_save_path.replace('best', 'last'))

        filename_loss = f'{log_dir}/{filename}_soft_ce_loss.png'
        # Plot periodically rather than every epoch (matplotlib + savefig is pure CPU/IO overhead).
        if (e + 1) % 25 == 0 or (e + 1) == n_epochs:
            plot_smoothed_loss(e + 1, n_epochs, e_loss_list, val_e_loss_list, is_acc=False, filename=filename_loss)

        early_stopping(val_e_loss)

        if early_stopping.early_stop:
            print("Early stopping")
            plot_smoothed_loss(e + 1, n_epochs, e_loss_list, val_e_loss_list, is_acc=False, filename=filename_loss)
            break

    result_df = pd.DataFrame({"Epoch": range(e + 1),
                              "TrainLoss": e_loss_list,
                              "ValLoss": val_e_loss_list,
                              "ValAcc": val_acc_list})
    result_df.to_csv(f"{log_dir}/{filename}_loss.csv")

    return min_loss, max_acc


def inference_similarity(model, dataset, video_id, device):
    model.eval()
    out = {}
    for idx in range(len(dataset)):
        data, id, videoLabel, label = dataset[idx]
        if id in video_id:
            data = torch.FloatTensor(data).unsqueeze(0).to(device)
            logits, hidden = model(data)
            probs = torch.softmax(logits, dim=-1)
            out[id] = {
                "feat": hidden.squeeze(0).detach().cpu().numpy(),
                "probs": probs.squeeze(0).detach().cpu().numpy(),
            }
    return out


# ============================================================================
# Distance matrix (replaces get_lstm_dmat.py) + human-similarity correlation
# ============================================================================
def write_lstm_dist_csv(emb, kind, video_order_file, output_dir=os.path.join(SCRIPT_DIR, 'dist_mat')):
    """Cosine-distance matrix over LSTM embeddings, ordered by the human-clustering
    video order. Writes {output_dir}/lstm_distr_{kind}.csv (no header/index)."""
    with open(video_order_file) as f:
        ordered_videos = f.read().splitlines()
    ordered_ids = [v.split("_")[0] for v in ordered_videos]

    try:
        matrix = np.stack([emb[int(id_)]["feat"] for id_ in ordered_ids])
    except Exception:
        matrix = np.stack([emb[int(id_)] for id_ in ordered_ids])

    dist = cdist(matrix, matrix, metric="cosine")
    df = pd.DataFrame(dist, index=ordered_videos, columns=ordered_videos)

    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, f'lstm_distr_{kind}.csv')
    df.to_csv(full_path, header=False, index=False)
    print(f"✔ wrote distance matrix -> {full_path}")
    return dist, ordered_videos


def _upper(mat):
    """Flattened upper triangle (above the diagonal) of a square matrix."""
    n = mat.shape[0]
    return mat[np.triu_indices(n, k=1)]


def correlate_with_human(dist, ordered_videos, human_csv):
    """Spearman correlation between the model dissimilarity matrix and the human
    odd-one-out dissimilarity matrix, on the upper triangles (analysis.ipynb metric)."""
    human = pd.read_csv(human_csv, sep='\t', header=0, index_col=0)
    human = human.reindex(index=ordered_videos, columns=ordered_videos).values
    hu, mu = _upper(human), _upper(dist)
    mask = ~(np.isnan(hu) | np.isnan(mu))
    r, p = spearmanr(hu[mask], mu[mask])
    return float(r), float(p)


if __name__ == '__main__':
    # ===== modes to train in this single invocation =====
    # Any subset of: "core", "allForce", "coreAndForce".
    input_modes = ["core", "allForce", "coreAndForce"]  #["allForce"] #

    # ----- shared config -----
    norm = True
    use_relative = False
    one_hot = False
    reverse = False
    input_format = "pkl"
    # Force-feature transform for allForce / coreAndForce (ignored by core).
    #   "_log_norm" (default) | "_original_norm" | "_square_root_norm" | "_cube_root_norm" | "_quantile_norm"
    force_norm_suffix = "_log_norm" #"_original_norm" #
    label_filename = '' if not one_hot else '_onehot'

    n_epochs = 200
    lr = 0.003
    hidden_dim = 64
    n_layers = 2
    batch_size = 1

    # n_epochs = 100
    # lr = 0.001
    # hidden_dim = 64
    # n_layers = 3
    # batch_size = 1
    force_mat = os.path.join(SCRIPT_DIR, '../force/rst/all_1133/estpart_forcemodel.v2_all.mat')

    model_name = 'lstm'
    apply_sigmoid = False
    checkpoint_path = os.path.join(SCRIPT_DIR, 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)

    summary_xlsx = os.path.join(SCRIPT_DIR, 'data/charades_traj_summary_selected1133.xlsx')
    video_file = os.path.join(SCRIPT_DIR, 'data/charades_traj_all_without_inference.csv')
    inference_file = [os.path.join(SCRIPT_DIR, 'data/charades_participant_diffStim.xlsx'),
                      os.path.join(SCRIPT_DIR, 'data/charades_participant.xlsx')]
    label_distribution_file = os.path.join(SCRIPT_DIR, 'data/one_hot.csv' if one_hot else 'data/732subj_response_distributions.csv')
    human_csv = os.path.join(SCRIPT_DIR, '../../human/behavioralExpDataAndAnalysis/exp2/full_dmat_585subj_hcordered.csv')
    video_order_file = os.path.join(SCRIPT_DIR, '../../utils/video_order_from_hc_human_dmat.txt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU is available" if device.type == "cuda" else "GPU not available, CPU used")

    # ----- build each distinct dataframe once (cached by the clean_zero flag) -----
    # allForce -> clean_zero=True;  core / coreAndForce -> clean_zero=False.
    _df_cache = {}

    def get_traj_df(apply_clean_zero):
        if apply_clean_zero not in _df_cache:
            print(f"\nBuilding normalized dataframe (clean_zero_data={apply_clean_zero})...")
            df = prepare_normalized_dataframe(summary_xlsx, force_mat, apply_clean_zero=apply_clean_zero)
            print(f"  -> {df.shape[0]} rows, {df.shape[1]} columns")
            _df_cache[apply_clean_zero] = df
        return _df_cache[apply_clean_zero]

    def run_mode(input_mode):
        input_mode_filename, input_size = INPUT_MODE_CONFIG[input_mode]
        apply_clean_zero = (input_mode == "allForce")          # clean_zero only for allForce
        infer_input_filename = '_shuffledInfer' if input_mode == "random" else ""
        traj_df = get_traj_df(apply_clean_zero)

        print("\n" + "=" * 80)
        print(f"== input_mode={input_mode} | clean_zero_data={apply_clean_zero} | "
              f"epochs={n_epochs} layers={n_layers} hd={hidden_dim} bs={batch_size}")
        print("=" * 80)

        # Re-seed per mode so each run reproduces a standalone single-mode run.
        set_seed(1234)

        train_dataset = CharadeV2(traj_df, video_file, label_distribution_file, input_mode=input_mode,
                                  norm=norm, use_relative=use_relative, input_format=input_format,
                                  force_norm_suffix=force_norm_suffix)
        num_classes = len(train_dataset.categories)

        filename = f"{model_name}{input_mode_filename}{infer_input_filename}{label_filename}_batch{batch_size}_lr{lr}_hd{hidden_dim}_nl{n_layers}"
        model_filename = f"{model_name}{input_mode_filename}{label_filename}_batch{batch_size}_lr{lr}_hd{hidden_dim}_nl{n_layers}"
        if model_name == 'lstm':
            model = LSTM(input_size=input_size, output_size=num_classes, hidden_dim=hidden_dim, n_layers=n_layers,
                         drop_prob=0.5, apply_sigmoid=apply_sigmoid)
        else:
            model = SocialGNN(input_size=10, output_size=num_classes, spatial_dim=32, hidden_dim=hidden_dim,
                              n_layers=n_layers, drop_prob=0.5, apply_sigmoid=apply_sigmoid)
        model = model.to(device)
        model_save_path = os.path.join(checkpoint_path, f'{model_filename}_best.pt')

        train(model, train_dataset, train_dataset, model_save_path, n_epochs, lr, batch_size, device, filename)

        print('-' * 60)
        print('Inferencing...')
        model.load_state_dict(torch.load(model_save_path.replace('best', 'last')))
        model.eval()

        _, train_acc = val(model, train_dataset, device)
        test_datasets = [
            CharadeV2(traj_df, infer_f, label_distribution_file,
                      input_mode=input_mode, norm=norm, use_relative=use_relative,
                      input_format=input_format, force_norm_suffix=force_norm_suffix)
            for infer_f in inference_file
        ]
        test_dataset = ConcatDataset(test_datasets)
        _, test_acc = val(model, test_dataset, device)
        print(f"Training accuracy (model top-1 vs human top-1): {train_acc:.4f}  ({len(train_dataset)} videos)")
        print(f"Test accuracy     (model top-1 vs human top-1): {test_acc:.4f}  ({len(test_dataset)} videos)")

        feat_by_part = {}  # 'main' = charades_participant, 'exp6' = charades_participant_diffStim
        for infer_f in inference_file:
            feat_save_path = os.path.join(SCRIPT_DIR, 'results', f"{filename}_{os.path.basename(infer_f).split('.')[0]}_soft_ce_loss_charades_feat.npy")
            infer_dataset = CharadeV2(traj_df, infer_f, input_mode=input_mode, norm=norm,
                                      use_relative=use_relative, input_format=input_format,
                                      force_norm_suffix=force_norm_suffix)
            infer_video_id = list(pd.read_excel(infer_f, engine="openpyxl")['ID'])
            feat_dict = inference_similarity(model, infer_dataset, infer_video_id, device)
            np.save(feat_save_path, feat_dict)
            print(f"Saved {feat_save_path}")
            part = 'exp6' if 'diffStim' in os.path.basename(infer_f) else 'main'
            feat_by_part[part] = feat_dict

        # distance matrix + human correlation
        merged_emb = {**feat_by_part.get('main', {}), **feat_by_part.get('exp6', {})}
        kind = input_mode_filename.lstrip('_')  # "core" / "force" / "coreAndForce"
        dist, ordered_videos = write_lstm_dist_csv(merged_emb, kind, video_order_file)
        r, p = correlate_with_human(dist, ordered_videos, human_csv)
        print(f"Model-human similarity correlation (Spearman, lstm_{kind} vs human): r={r:.4f}  p={p:.2e}")

        return {"mode": input_mode, "kind": kind, "clean_zero": apply_clean_zero,
                "train_acc": train_acc, "test_acc": test_acc, "spearman": r, "p": p}

    results = [run_mode(m) for m in input_modes]

    # ----- summary across all requested modes -----
    print("\n" + "=" * 78)
    print(f"{'mode':14s} {'clean_zero':>10s} {'train_acc':>10s} {'test_acc':>9s} {'human_r':>9s}")
    print("-" * 78)
    for res in results:
        print(f"{res['mode']:14s} {str(res['clean_zero']):>10s} {res['train_acc']:>10.4f} "
              f"{res['test_acc']:>9.4f} {res['spearman']:>9.4f}")
    print("=" * 78)
