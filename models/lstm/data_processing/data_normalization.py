import pandas as pd
import numpy as np
import os
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer

np.seterr(all='raise')

def _safe_first_repeat(arr: np.ndarray) -> np.ndarray:
    """Repeat the first value (or return as is if length 0/1) to mimic 'copy first' behavior safely."""
    n = arr.size
    if n <= 1:
        return arr

    # Overwrite the first element with the second element
    arr[0] = arr[1]
    return arr

def _safe_first_zero(arr: np.ndarray) -> np.ndarray:
    """Repeat the first value (or return as is if length 0/1) to mimic 'copy first' behavior safely."""
    n = arr.size
    if n <= 1:
        return arr

    # Overwrite the first element with the second element
    arr[0] = 0
    return arr

def check_extremes(arr, name):
    """Prints a warning if the array contains NaNs, Infs, or massive values."""
    if len(arr) == 0:
        return

    # Check for NaN or Inf
    if not np.isfinite(arr).all():
        print(f"!!! [NON-FINITE] Found NaN/Inf in: {name}")

    # Check for extreme values that will break .std()
    # 1e150 is a safe 'danger zone' for float64 squaring
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

    # dist[t] = sqrt((x2[t]-x1[t])^2 + (y2[t]-y1[t])^2)
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    check_extremes(dist, "dist")

    # velocities: v[t] = pos[t+1] - pos[t], last repeats previous
    vx1 = np.zeros_like(x1); vy1 = np.zeros_like(y1)
    vx2 = np.zeros_like(x2); vy2 = np.zeros_like(y2)

    n = len(x1)
    if n == 0:
        # keep as empty lists
        return pd.Series({
            "dist": dist.tolist(),
            "vx1": [], "vy1": [], "vx2": [], "vy2": [],
            "ax1": [], "ay1": [], "ax2": [], "ay2": []
        })
    if n == 1:
        # no "next" frame; choose 0s
        return pd.Series({
            "dist": dist.tolist(),
            "vx1": [0.0], "vy1": [0.0], "vx2": [0.0], "vy2": [0.0],
            "ax1": [0.0], "ay1": [0.0], "ax2": [0.0], "ay2": [0.0],
        })

    vx1[1:] = np.diff(x1); vy1[1:] = np.diff(y1)
    vx2[1:] = np.diff(x2); vy2[1:] = np.diff(y2)
    vx1 = _safe_first_repeat(vx1); vy1 = _safe_first_repeat(vy1)
    vx2 = _safe_first_repeat(vx2); vy2 = _safe_first_repeat(vy2)

    check_extremes(vx1, "vx1")
    check_extremes(vx2, "vx2")
    check_extremes(vy1, "vy1")
    check_extremes(vy2, "vy2")

    # accelerations: a[t] = v[t+1] - v[t], last repeats previous
    ax1 = np.zeros_like(vx1); ay1 = np.zeros_like(vy1)
    ax2 = np.zeros_like(vx2); ay2 = np.zeros_like(vy2)

    ax1[:-1] = np.diff(vx1); ay1[:-1] = np.diff(vy1)
    ax2[:-1] = np.diff(vx2); ay2[:-1] = np.diff(vy2)
    ax1 = _safe_first_repeat(ax1); ay1 = _safe_first_repeat(ay1)
    ax2 = _safe_first_repeat(ax2); ay2 = _safe_first_repeat(ay2)

    check_extremes(ax1, "ax1")
    check_extremes(ax2, "ax2")
    check_extremes(ay1, "ay1")
    check_extremes(ay2, "ay2")

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


def normalize_list_column_to_newcol(
    data: pd.DataFrame,
    col: str,
    suffix: str = "_norm",
    eps: float = 1e-12,
    ddof: int = 0,
):
    """
    Column-wise normalization for a Series of lists/arrays.
    1) concat all elements in the column
    2) (x - mean) / std
    3) split back to per-row original lengths via np.split
    4) write to new column: f"{col}{suffix}"
    Returns: (mean, std)
    """
    # Convert each row's list/array to a 1D float array
    arrs = []
    try:
        for v in data[col]:
            try:
                v = [float(d) for d in v]
            except Exception as e:
                v = process_traj(v)
            arrs.append(np.array(v))
    except Exception as e:
        print("\n" + "=" * 30)
        print(f"CRITICAL ERROR on col {col }value: {v}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("=" * 30)
        exit(-1)

    lengths = np.fromiter((a.size for a in arrs), dtype=int)
    total = int(lengths.sum())

    out_col = f"{col}{suffix}"

    # Edge case: whole column empty (or all empty lists)
    if total == 0:
        data[out_col] = [[] for _ in range(len(data))]
        return np.nan, np.nan

    # Concatenate all non-empty
    flat = np.concatenate([a for a in arrs if a.size > 0], axis=0)

    mu = float(flat.mean())
    sigma = float(flat.astype(np.float64).std(ddof=ddof))

    # Normalize with safety for near-zero std
    if sigma < eps:
        transformed = np.zeros_like(flat)
    else:
        transformed = (flat - mu) / sigma

    # Split indices for np.split: cumulative sum of lengths excluding last
    # Example: lengths [3,2,0,4] -> split_indices [3,5,5]
    split_indices = np.cumsum(lengths)[:-1]

    # Split back into per-row chunks
    split_chunks = np.split(transformed, split_indices)

    # Re-wrap into Python lists for storage in df
    data[out_col] = [chunk.tolist() for chunk in split_chunks]
    return mu, sigma


def clean_zero_dataforceonly(data, column_names):
    """
    Remove list elements at positions where ALL specified columns
    have 0 at that same index (row-wise).
    """

    def clean_row(row):
        # Get lists for relevant columns
        lists = [row[col] for col in column_names]

        # Length assumption: all lists in the row have equal length
        length = len(lists[0])

        # Keep indices where NOT all values are zero
        keep_indices = [
            i for i in range(length)
            if not all(lst[i] == 0 for lst in lists)
        ]

        # Rebuild lists using kept indices
        for col in column_names:
            row[col] = [row[col][i] for i in keep_indices]

        return row

    return data.apply(clean_row, axis=1)

def clean_zero_data(data, check_columns, all_sequence_columns):
    """
    Remove list elements at positions where ALL `check_columns`
    have 0 at that same index, but apply the deletion to `all_sequence_columns`
    so everything stays synchronized!
    """

    def clean_row(row):
        # 1. Get lists for the columns we are CHECKING for zeros
        lists = [row[col] for col in check_columns]
        length = len(lists[0])

        # 2. Find which indices to keep
        keep_indices = [
            i for i in range(length)
            if not all(lst[i] == 0 for lst in lists)
        ]

        # 3. Rebuild lists for ALL sequence columns using kept indices
        for col in all_sequence_columns:
            row[col] = [row[col][i] for i in keep_indices]

        return row

    return data.apply(clean_row, axis=1)

def plot_hist(data, title=""):
    if isinstance(data, pd.core.series.Series):
        data = np.concatenate(data.values)
    plt.hist(data, bins=30)  # you can adjust number of bins
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)

    plt.show()


if __name__ == '__main__':
    pkl_name = '../data/charade_traj_force_summary_full.pkl'
    data = pd.read_pickle(pkl_name)

    save_root = "../data/vis"
    os.makedirs(save_root, exist_ok=True)

    force_parameters = ['epsilon_selfA', 'sigma_selfA', 'bcoef_selfA',
                        'epsilon_inter', 'sigma_inter', 'bcoef_inter',
                        'epsilon_selfB', 'sigma_selfB', 'bcoef_selfB']

    # Downsample visual features so that they are the same time interval as force parameters
    visual_cols = ["x1", "y1", "x2", "y2"]
    data[visual_cols] = data[visual_cols].apply(downsample_visual_features, axis=1)
 
    for i, row in data.iterrows():
        assert len(row["x1"]) == len(
        row["epsilon_selfA"]), f'Size mismatch, x1: {len(row["x1"])} != {len(row["epsilon_selfA"])}"'

    # clean no movement frames
    # data = clean_zero_dataforceonly(data, force_parameters)

    # Define all the columns that hold your timestep lists
    all_time_cols = force_parameters + visual_cols
    
    # Check 'force_parameters' for zeros, but trim 'all_time_cols'
    data = clean_zero_data(data, check_columns=force_parameters, all_sequence_columns=all_time_cols)

    # Using 'original' instead of '' for clearer file/column naming
    methods = ['original', 'square root', 'cube root', 'log', 'quantile']

    for para in force_parameters:
        print(f'Processing {para}...')

        # 1. Prepare global data and tracking lengths
        lengths = [len(v) for v in data[para]]
        all_data = np.concatenate(data[para].values)
        split_indices = np.cumsum(lengths)[:-1]

        # plot_hist(all_data, f"before clip - {para}")

        non_zeros = all_data[all_data != 0]

        mu_global = np.mean(non_zeros)
        std_global = np.std(non_zeros)

        # Calculate bounds
        # lower_bound = mu_global - 20 * std_global  
        # upper_bound = mu_global + 20 * std_global  

        # Apply the cap: values outside [lower, upper] become the bound values
        # all_data = np.clip(all_data, lower_bound, upper_bound)
        # plot_hist(all_data, f"after clip - {para}")
     

        for method in methods:
            # 2. Apply non-linear transformation
            if method == 'original':
                transformed = all_data
            elif method == 'square root':
                transformed = np.sqrt(all_data)
            elif method == 'cube root':
                transformed = np.cbrt(all_data)
            elif method == 'log':
                min_val = np.min(all_data)
                if min_val < 0:
                    transformed = np.log(all_data - min_val + 1)  #1
                else:
                    transformed = np.log(all_data + 1)   #1
            elif method == 'quantile':
                qt = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
                transformed = qt.fit_transform(all_data.reshape(-1, 1)).flatten()
            else:
                raise ValueError(f'Method {method} not supported')

            # 3. Save the VERSION WITHOUT Standardize (Z-score)
            method_slug = method.replace(" ", "_")
            raw_transform_col = f"{para}_{method_slug}"
            data[raw_transform_col] = [arr.tolist() for arr in np.split(transformed, split_indices)]

            # 4. Standardize (Z-score) logic
            mean_val = np.mean(transformed)
            std_val = np.std(transformed)

            if std_val > 0:
                standardized = (transformed - mean_val) / std_val
            else:
                standardized = transformed - mean_val

            # 5. Save the VERSION WITH Standardize
            norm_col = f"{para}_{method_slug}_norm"
            data[norm_col] = [arr.tolist() for arr in np.split(standardized, split_indices)]

    # Normalize visual feature as well
    new_cols = data.apply(compute_row_features, axis=1)
    data = pd.concat([data, new_cols], axis=1)

    cols_to_norm = ["x1", "y1", "x2", "y2", "dist", "vx1", "vy1", "vx2", "vy2", "ax1", "ay1", "ax2", "ay2", "ori1", "ori2"]
    for c in cols_to_norm:
        if c not in data.columns:
            raise KeyError(f"Missing column: {c}")
        normalize_list_column_to_newcol(data, c, suffix="_norm", eps=1e-12, ddof=0)

    # Final Save
    new_pkl_name = '../data/charade_traj_force_summary_normalized.pkl'
    data.to_pickle(new_pkl_name)
    print(f"Success! Dataframe now has {len(data.columns)} columns.")
    print(f"Example columns: {para}_cube_root AND {para}_cube_root_norm")

    print(list(data.keys()))