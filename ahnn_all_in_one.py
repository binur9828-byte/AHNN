# ahnn_all_in_one.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Constants
FEATURE_NAMES = [f"feature_{i}" for i in range(10)]
F_FEATURES = len(FEATURE_NAMES)
T_YEARS = 5
N_COMPANIES = 62
ENSEMBLE_MEMBERS = 10
RANDOM_SEED = 42
OUT_DIR = "ahnn_plots"
SPECTRAL_BOUND_RHO = 1.0
FAIRNESS_LAMBDA = 0.1
WEIGHT_DECAY = 1e-4
DROPOUT_P = 0.1
EPOCHS = 100
LEARNING_RATE = 0.01

def standardize_tensor(X_train, X_test):
    """Standardize tensors along last axis."""
    # Combine for fitting
    X_all = np.concatenate([X_train, X_test], axis=0)
    mean = X_all.mean(axis=(0, -1), keepdims=True)
    std = X_all.std(axis=(0, -1), keepdims=True) + 1e-8
    X_train_s = (X_train - mean) / std
    X_test_s = (X_test - mean) / std
    return X_train_s, X_test_s, (mean, std)

def generate_synthetic_raw():
    """Generate synthetic raw data."""
    np.random.seed(RANDOM_SEED)
    n_samples = N_COMPANIES * T_YEARS
    data = {}
    for f in FEATURE_NAMES:
        data[f] = np.random.randn(n_samples)
    data['target'] = np.random.randint(0, 2, n_samples)
    data['group'] = np.random.randint(0, 2, n_samples)
    data['company'] = np.repeat(range(N_COMPANIES), T_YEARS)
    data['year'] = np.tile(range(T_YEARS), N_COMPANIES)
    return pd.DataFrame(data)

class AHNNDataset:
    def __init__(self, X_train, y_train, X_test, y_test, groups_train, groups_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.groups_train = groups_train
        self.groups_test = groups_test
        
        # For evaluation, combine train and test
        self.X = np.concatenate([X_train, X_test], axis=0)
        self.y = np.concatenate([y_train, y_test], axis=0)
        self.group = np.concatenate([groups_train, groups_test], axis=0)
        
        # Create kfold splits
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        self.kfold_splits = list(skf.split(self.X, self.y))

    def summary(self):
        return f"""============ AHNNDataset Summary ============
Train: {self.X_train.shape[0]} samples, {self.X_train.shape[1]} features
Test:  {self.X_test.shape[0]} samples, {self.X_test.shape[1]} features
Features: {FEATURE_NAMES}
T_years: {T_YEARS}, F_features: {F_FEATURES}, N_companies: {N_COMPANIES}
============================================"""

def build_dataset_from_raw(raw_df):
    """Build dataset from raw dataframe."""
    # Reshape to [N, T, F]
    X = []
    y = []
    groups = []
    for company in range(N_COMPANIES):
        company_data = raw_df[raw_df['company'] == company]
        if len(company_data) == T_YEARS:
            x = company_data[FEATURE_NAMES].values  # [T, F]
            X.append(x)
            y.append(company_data['target'].iloc[-1])  # last year target
            groups.append(company_data['group'].iloc[0])  # group

    X = np.array(X)  # [N, T, F]
    y = np.array(y)
    groups = np.array(groups)

    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    return AHNNDataset(X_train, y_train, X_test, y_test, groups_train, groups_test)