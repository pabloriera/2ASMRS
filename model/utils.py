# from kiwisolver import Variable
from pytorch_lightning.callbacks import Callback
from copy import copy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

eps = 1e-10


class MetricTracker(Callback):

    def __init__(self):
        self.collection = []
        self.epoch = 0

    def on_validation_epoch_end(self, trainer, module):
        metrics = {}
        for k, v in trainer.logged_metrics.items():
            metrics[k] = copy(v.cpu().detach().numpy())
        self.collection.append(metrics)


class MyEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        if trainer.current_epoch < 10:
            pass
        else:
            self._run_early_stopping_check(trainer)

    def on_train_end(self, trainer, pl_module):
        # instead, do it at the end of training loop
        self._run_early_stopping_check(trainer)


import numpy as np


def train_val_split_no_overlap(X, d, r, random_seed=None):
    """
    Split matrix X into training and validation sets by selecting n continuous sections of d rows as validation,
    ensuring no overlap between sections.

    Parameters:
    - X: np.ndarray, the dataset to split
    - d: int, the number of continuous rows in each section
    - r: float, the validation ratio
    - random_seed: int, seed for reproducibility

    Returns:
    - X_train: np.ndarray, the training data
    - X_val: np.ndarray, the validation data
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    total_rows = X.shape[0]
    val_size = int(total_rows * r)  # total number of rows for validation
    n = val_size // d  # number of sections to select

    if n == 0 or d * n > total_rows:
        raise ValueError(
            "Validation size too small or sections too large for the given dataset size."
        )

    # List of possible start indices for sections (ensuring no overlap)
    possible_start_indices = np.arange(0, total_rows - d + 1)

    selected_indices = []
    for _ in range(n):
        # Select a random start index from the remaining possible indices
        selected_idx = np.random.choice(possible_start_indices)
        selected_indices.append(selected_idx)

        # Remove indices that would cause overlap with the selected section
        remove_indices = np.arange(selected_idx, selected_idx + d)
        possible_start_indices = np.setdiff1d(possible_start_indices, remove_indices)

    # Create mask for validation rows
    val_mask = np.zeros(total_rows, dtype=bool)
    for start_idx in selected_indices:
        val_mask[start_idx : start_idx + d] = True

    # Split the data
    X_val = X[val_mask]
    X_train = X[~val_mask]

    return X_train, X_val
def train_val_split_no_overlap(X, y, d, r, random_seed=None):
    """
    Split matrix X and target y into training and validation sets by selecting n continuous sections of d rows
    as validation, ensuring no overlap between sections.

    Parameters:
    - X: np.ndarray, the dataset to split (features)
    - y: np.ndarray, the target labels corresponding to X
    - d: int, the number of continuous rows in each section
    - r: float, the validation ratio
    - random_seed: int, seed for reproducibility
    
    Returns:
    - X_train: np.ndarray, the training data (features)
    - X_val: np.ndarray, the validation data (features)
    - y_train: np.ndarray, the training data (targets)
    - y_val: np.ndarray, the validation data (targets)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    total_rows = X.shape[0]
    val_size = int(total_rows * r)  # total number of rows for validation
    n = val_size // d  # number of sections to select
    
    if n == 0 or d * n > total_rows:
        raise ValueError("Validation size too small or sections too large for the given dataset size.")
    
    # List of possible start indices for sections (ensuring no overlap)
    possible_start_indices = np.arange(0, total_rows - d + 1)
    
    selected_indices = []
    for _ in range(n):
        # Select a random start index from the remaining possible indices
        selected_idx = np.random.choice(possible_start_indices)
        selected_indices.append(selected_idx)
        
        # Remove indices that would cause overlap with the selected section
        remove_indices = np.arange(selected_idx, selected_idx + d)
        possible_start_indices = np.setdiff1d(possible_start_indices, remove_indices)
    
    # Create mask for validation rows
    val_mask = np.zeros(total_rows, dtype=bool)
    for start_idx in selected_indices:
        val_mask[start_idx:start_idx + d] = True

    # Split the data
    X_val = X[val_mask]
    X_train = X[~val_mask]
    y_val = y[val_mask]
    y_train = y[~val_mask]

    return X_train, X_val, y_train, y_val