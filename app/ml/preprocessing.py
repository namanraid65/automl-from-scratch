import numpy as np
import pandas as pd

def impute_missing_values(df: pd.DataFrame, numeric_strategy='mean', categorical_strategy='mode') -> pd.DataFrame:
    """
    Imputes missing values in a Pandas DataFrame.
    """
    df_imputed = df.copy()
    for col in df_imputed.columns:
        if df_imputed[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_imputed[col]):
                if numeric_strategy == 'mean':
                    fill_value = df_imputed[col].mean()
                elif numeric_strategy == 'median':
                    fill_value = df_imputed[col].median()
                elif numeric_strategy == 'mode':
                    fill_value = df_imputed[col].mode()[0] if not df_imputed[col].mode().empty else 0
                else:
                    fill_value = 0
                df_imputed[col] = df_imputed[col].fillna(fill_value)
            else:
                if categorical_strategy == 'mode':
                    fill_value = df_imputed[col].mode()[0] if not df_imputed[col].mode().empty else "Missing"
                else:
                    fill_value = "Missing"
                df_imputed[col] = df_imputed[col].fillna(fill_value)
    return df_imputed

class LabelEncoderScratch:
    def __init__(self):
        self.classes_ = {}

    def fit(self, y):
        unique_y = sorted(list(set(y)))
        self.classes_ = {val: idx for idx, val in enumerate(unique_y)}
        return self

    def transform(self, y):
        return np.array([self.classes_.get(val, -1) for val in y])

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        inv_classes = {idx: val for val, idx in self.classes_.items()}
        return np.array([inv_classes.get(idx, "Unknown") for idx in y])

class OneHotEncoderScratch:
    def __init__(self):
        self.categories_ = {}
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, columns):
        self.columns = columns
        self.categories_ = {}
        self.feature_names_ = []
        for col in columns:
            unique_cats = sorted(list(X[col].unique()))
            self.categories_[col] = unique_cats
            for cat in unique_cats:
                self.feature_names_.append(f"{col}_{cat}")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        encoded_cols = []
        for col in self.columns:
            cats = self.categories_[col]
            for cat in cats:
                encoded_cols.append((X[col] == cat).astype(float).values)
        return np.column_stack(encoded_cols)

    def fit_transform(self, X: pd.DataFrame, columns) -> np.ndarray:
        self.fit(X, columns)
        return self.transform(X)

class MinMaxScalerScratch:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        # X is np.ndarray or pd.DataFrame
        X_arr = np.asarray(X, dtype=float)
        self.min_ = np.min(X_arr, axis=0)
        self.max_ = np.max(X_arr, axis=0)
        # Avoid division by zero
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ == 0] = 1e-8
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return (X_arr - self.min_) / self.range_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class StandardScalerScratch:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X_arr, axis=0)
        self.std_ = np.std(X_arr, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1e-8
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return (X_arr - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def train_test_split_scratch(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    shuffled_indices = np.random.permutation(n_samples)
    
    test_set_size = int(n_samples * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    # Handle both numpy arrays and pandas dataframes/series
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    else:
        X_train, X_test = X[train_indices], X[test_indices]
        
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    else:
        y_train, y_test = y[train_indices], y[test_indices]
        
    return X_train, X_test, y_train, y_test
