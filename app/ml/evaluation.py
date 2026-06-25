import numpy as np

# --- Regression Metrics ---
def mean_absolute_error(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mean_squared_error(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def root_mean_squared_error(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def r2_score(y_true, y_pred):
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - (ss_res / ss_tot))


# --- Classification Metrics ---
def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

def classification_report_scratch(y_true, y_pred):
    """
    Computes precision, recall, and F1-score.
    Supports binary and multi-class (macro-averaged).
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    precisions = []
    recalls = []
    f1s = []
    
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1_score": float(np.mean(f1s)),
        "per_class": {
            str(c): {
                "precision": float(precisions[i]),
                "recall": float(recalls[i]),
                "f1_score": float(f1s[i])
            } for i, c in enumerate(classes)
        }
    }

def confusion_matrix_scratch(y_true, y_pred):
    """
    Computes confusion matrix. Returns matrix as list of lists and labels list.
    """
    classes = sorted(list(np.unique(np.concatenate([y_true, y_pred]))))
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    for t, p in zip(y_true, y_pred):
        t_idx = class_to_idx[t]
        p_idx = class_to_idx[p]
        matrix[t_idx, p_idx] += 1
        
    return {
        "matrix": matrix.tolist(),
        "labels": [str(c) for c in classes]
    }


# --- Clustering Metrics ---
def calculate_inertia(X, centroids, labels):
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    labels = np.asarray(labels, dtype=int)
    inertia = 0.0
    for idx, c in enumerate(centroids):
        pts = X[labels == idx]
        if len(pts) > 0:
            inertia += np.sum((pts - c) ** 2)
    return float(inertia)

def cluster_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)
    dist = {str(k): int(v) for k, v in zip(unique, counts)}
    return dist
