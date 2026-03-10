import re
import os
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The random seed to set.
        deterministic (bool): If True, enforce deterministic behavior in torch operations.
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
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
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


def angle_difference(a1, a2):
    a = a1 - a2
    a = (a + 180) % 360 - 180
    return a


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l


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
