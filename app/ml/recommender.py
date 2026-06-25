import numpy as np
import pandas as pd

def analyze_dataset(df: pd.DataFrame, target_column: str = None) -> dict:
    """
    Analyzes dataset characteristics and returns a statistical summary.
    """
    n_rows, n_cols = df.shape
    
    # Feature types
    numeric_cols = []
    categorical_cols = []
    boolean_cols = []
    text_cols = []
    
    for col in df.columns:
        if col == target_column:
            continue
        
        # Check type
        col_type = df[col].dtype
        unique_count = df[col].nunique()
        
        if pd.api.types.is_bool_dtype(df[col]):
            boolean_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # If a numeric column has very few unique values and is integer, it could be categorical/boolean
            if unique_count <= 2:
                boolean_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            # Check if text or high cardinality
            if unique_count > 100 and df[col].apply(lambda x: isinstance(x, str) and len(x.split()) > 3).mean() > 0.5:
                text_cols.append(col)
            else:
                categorical_cols.append(col)

    # Missing values
    missing_counts = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    missing_pct = float((total_missing / (n_rows * n_cols)) * 100) if n_rows > 0 else 0.0

    # Duplicates
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = float((duplicate_count / n_rows) * 100) if n_rows > 0 else 0.0

    # Class balance (only relevant if target is provided and we do classification)
    class_distribution = None
    is_imbalanced = False
    if target_column and target_column in df.columns:
        target_series = df[target_column].dropna()
        if not pd.api.types.is_numeric_dtype(target_series) or target_series.nunique() <= 10:
            counts = target_series.value_counts()
            class_distribution = {str(k): int(v) for k, v in counts.items()}
            if len(counts) > 1:
                ratio = counts.min() / counts.max()
                is_imbalanced = ratio < 0.35  # imbalance threshold

    # Variance and Outliers for numerical features
    numeric_info = {}
    total_outliers = 0
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                var = float(col_data.var())
                mean = float(col_data.mean())
                std = float(col_data.std())
                
                # Outlier detection using IQR
                q75, q25 = np.percentile(col_data, [75 ,25])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                outliers = int(((col_data < lower_bound) | (col_data > upper_bound)).sum())
                total_outliers += outliers
                
                numeric_info[col] = {
                    "variance": var,
                    "mean": mean,
                    "std": std,
                    "outliers_count": outliers
                }

    # Correlation Matrix (for numeric features)
    correlation_matrix = {}
    has_high_multicollinearity = False
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr().fillna(0)
        correlation_matrix = corr_df.to_dict()
        
        # Check for multicollinearity (excluding diagonal)
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1 = numeric_cols[i]
                col2 = numeric_cols[j]
                if abs(corr_df.loc[col1, col2]) > 0.85:
                    has_high_multicollinearity = True

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "features": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "boolean": boolean_cols,
            "text": text_cols,
            "total_count": len(numeric_cols) + len(categorical_cols) + len(boolean_cols) + len(text_cols)
        },
        "missing_values": {
            "total_missing": total_missing,
            "missing_percentage": missing_pct,
            "column_breakdown": {str(k): int(v) for k, v in missing_counts.items() if v > 0}
        },
        "duplicates": {
            "duplicate_count": duplicate_count,
            "duplicate_percentage": duplicate_pct
        },
        "class_balance": {
            "distribution": class_distribution,
            "is_imbalanced": is_imbalanced
        },
        "outliers": {
            "total_outliers": total_outliers,
            "has_outliers": total_outliers > 0
        },
        "multicollinearity": has_high_multicollinearity,
        "numeric_details": numeric_info,
        "correlation_matrix": correlation_matrix
    }

def recommend_algorithms(analysis: dict, problem_type: str) -> list:
    """
    Heuristics to score and recommend algorithms based on dataset analysis.
    Returns list of dictionaries ordered by rank:
    [
      { "rank": 1, "algorithm": "...", "confidence": 95, "reason": "..." },
      ...
    ]
    """
    n_rows = analysis["n_rows"]
    n_features = analysis["features"]["total_count"]
    n_numeric = len(analysis["features"]["numeric"])
    n_categorical = len(analysis["features"]["categorical"])
    has_outliers = analysis["outliers"]["has_outliers"]
    has_multicollinearity = analysis["multicollinearity"]
    
    recommendations = []

    if problem_type.lower() == 'regression':
        # Algorithms: Linear Regression, Multiple Linear Regression, Polynomial Regression
        
        # 1. Linear Regression (Simple)
        lr_score = 50
        lr_reasons = []
        if n_features == 1:
            lr_score += 40
            lr_reasons.append("The dataset has exactly one feature, making Simple Linear Regression highly suitable.")
        else:
            lr_score -= 20
            lr_reasons.append("Simple Linear Regression is limited because the dataset contains multiple features. Multiple Linear Regression is preferred.")
        
        if n_numeric > 0:
            lr_score += 10
        if has_multicollinearity:
            lr_score -= 10
            lr_reasons.append("High multicollinearity present, which can destabilize coefficients.")
        
        # Ensure score is within boundaries
        lr_score = max(10, min(95, lr_score))
        
        # 2. Multiple Linear Regression
        mlr_score = 75
        mlr_reasons = []
        if n_features > 1:
            mlr_score += 15
            mlr_reasons.append(f"Suitable for multivariate regression with {n_features} features.")
        else:
            mlr_score -= 10
            mlr_reasons.append("Dataset has only 1 feature; Simple Linear Regression is sufficient.")
            
        if has_multicollinearity:
            mlr_score -= 15
            mlr_reasons.append("Multicollinearity detected. Closed-form matrix inversion could be unstable, training via Gradient Descent is recommended.")
        else:
            mlr_reasons.append("No severe multicollinearity detected, allowing stable parameter estimations.")
            
        if n_categorical > 0:
            mlr_reasons.append("Categorical columns are present and will require One-Hot encoding.")
            
        mlr_score = max(10, min(98, mlr_score))
        
        # 3. Polynomial Regression
        poly_score = 60
        poly_reasons = []
        if n_features > 0 and n_features <= 10:
            poly_score += 20
            poly_reasons.append(f"Low feature dimension ({n_features}) allows polynomial expansion without combinatorial explosion.")
        elif n_features > 15:
            poly_score -= 40
            poly_reasons.append(f"High feature count ({n_features}) will lead to extremely high polynomial dimensions, causing overfitting and heavy memory usage.")
            
        if n_numeric > 0:
            poly_reasons.append("Captures potential non-linear relationships in numerical features.")
            
        if n_rows < 100:
            poly_score -= 15
            poly_reasons.append("Small dataset size may cause polynomial features to overfit easily.")
            
        poly_score = max(5, min(95, poly_score))
        
        recommendations = [
            {"algorithm": "Multiple Linear Regression", "confidence": mlr_score, "reasons": mlr_reasons},
            {"algorithm": "Polynomial Regression", "confidence": poly_score, "reasons": poly_reasons},
            {"algorithm": "Linear Regression (Simple)", "confidence": lr_score, "reasons": lr_reasons}
        ]

    elif problem_type.lower() == 'classification':
        # Algorithms: KNN, Logistic Regression, Naive Bayes
        
        # 1. K-Nearest Neighbors (KNN)
        knn_score = 70
        knn_reasons = []
        if n_rows < 5000:
            knn_score += 20
            knn_reasons.append(f"Small-to-medium dataset size ({n_rows} rows) is highly efficient for distance calculations during inference.")
        elif n_rows > 15000:
            knn_score -= 30
            knn_reasons.append(f"Large dataset size ({n_rows} rows) makes KNN prediction very slow, as it calculates distance to all training samples.")
            
        if n_numeric > 0 and n_categorical == 0:
            knn_score += 10
            knn_reasons.append("Dataset features are purely numeric. Euclidean distances are mathematically direct and scale-dependent.")
        elif n_categorical > 0:
            knn_reasons.append("Contains categorical columns; scaling and One-Hot encoding are required to maintain meaningful distance metrics.")
            
        knn_score = max(10, min(95, knn_score))
        
        # 2. Logistic Regression
        log_score = 80
        log_reasons = []
        if n_rows > 5000:
            log_score += 10
            log_reasons.append("Highly scalable and trains quickly on larger datasets using Gradient Descent.")
            
        if analysis["class_balance"]["is_imbalanced"]:
            log_score -= 10
            log_reasons.append("Classes are imbalanced. Logistic Regression might favor the majority class unless threshold is adjusted.")
        else:
            log_reasons.append("Balanced class distribution allows for stable classification boundaries.")
            
        if n_features > n_rows:
            log_score += 5
            log_reasons.append("High feature-to-sample ratio is well suited for linear boundaries.")
            
        log_score = max(10, min(97, log_score))
        
        # 3. Naive Bayes
        nb_score = 65
        nb_reasons = []
        # Check correlation matrix for independence assumption
        corr_matrix = analysis["correlation_matrix"]
        high_corr_count = 0
        if len(corr_matrix) > 0:
            for col, targets in corr_matrix.items():
                for target_col, val in targets.items():
                    if col != target_col and abs(val) > 0.5:
                        high_corr_count += 1
                        
        if high_corr_count == 0:
            nb_score += 20
            nb_reasons.append("Low correlation between features aligns well with the Naive Bayes feature independence assumption.")
        else:
            nb_score -= 15
            nb_reasons.append("Some feature correlation detected, violating the strict 'naive' independence assumption.")
            
        if n_rows < 1000:
            nb_score += 5
            nb_reasons.append("Requires very little training data to estimate probability parameters robustly.")
            
        if analysis["class_balance"]["is_imbalanced"]:
            nb_reasons.append("Naturally handles class imbalances well via prior probabilities.")
            
        nb_score = max(10, min(94, nb_score))
        
        recommendations = [
            {"algorithm": "Logistic Regression", "confidence": log_score, "reasons": log_reasons},
            {"algorithm": "K-Nearest Neighbors (KNN)", "confidence": knn_score, "reasons": knn_reasons},
            {"algorithm": "Naive Bayes", "confidence": nb_score, "reasons": nb_reasons}
        ]

    elif problem_type.lower() == 'clustering':
        # Algorithms: K-Means, Hierarchical Clustering
        
        # 1. K-Means
        kmeans_score = 85
        kmeans_reasons = []
        if n_rows > 10000:
            kmeans_score += 10
            kmeans_reasons.append("Highly scalable algorithm that converges quickly on large datasets.")
            
        if has_outliers:
            kmeans_score -= 15
            kmeans_reasons.append("Outliers detected. K-Means is sensitive to outliers as they pull centroids away from true cluster centers.")
        else:
            kmeans_reasons.append("No significant outliers detected, which helps in converging on representative cluster centers.")
            
        if n_numeric > 0 and n_categorical == 0:
            kmeans_score += 5
            kmeans_reasons.append("Numeric features are ideal for Euclidean distance centroid updates.")
            
        kmeans_score = max(10, min(98, kmeans_score))
        
        # 2. Hierarchical Clustering
        hier_score = 65
        hier_reasons = []
        if n_rows < 3000:
            hier_score += 20
            hier_reasons.append("Small dataset size is computationally feasible. It builds a rich tree of merges (dendrogram).")
        elif n_rows >= 5000:
            hier_score = 5 # Make it extremely low confidence
            hier_reasons.append("WARNING: Dataset is too large. Agglomerative Hierarchical Clustering has O(N^2) or O(N^3) complexity and will consume excessive memory and CPU time.")
            
        if has_outliers:
            hier_reasons.append("Can be more robust to outliers compared to K-Means if complete/average linkage is used.")
            
        hier_score = max(5, min(95, hier_score))
        
        # Top 3 needs 3 algorithms. Since there are only 2 clustering algorithms in our spec,
        # we can provide a dummy or a variation (e.g. Hierarchical with Average Linkage vs Single Linkage, or just output the 2 with ranks).
        # Wait, the prompt says "Instead of recommending only one algorithm, recommend the Top 3."
        # If there are only 2 algorithms for clustering, we can add a variation of K-Means (e.g. K-Means with k++ initialization vs K-Means with random initialization) as Rank 3,
        # or we can list:
        # 1. K-Means (K-means++ initialization)
        # 2. Hierarchical Clustering (Average Linkage)
        # 3. K-Means (Random initialization)
        # This fits the "Top 3" requirement perfectly and represents real algorithmic choices!
        
        kmeans_rand_score = kmeans_score - 15
        kmeans_rand_reasons = [
            "Uses random centroid initialization. Fast, but can converge to sub-optimal local minima compared to K-Means++.",
            "Suitable as a quick clustering baseline."
        ]
        
        recommendations = [
            {"algorithm": "K-Means (K-means++)", "confidence": kmeans_score, "reasons": kmeans_reasons},
            {"algorithm": "Hierarchical Clustering", "confidence": hier_score, "reasons": hier_reasons},
            {"algorithm": "K-Means (Random Init)", "confidence": kmeans_rand_score, "reasons": kmeans_rand_reasons}
        ]

    # Sort recommendations by confidence score descending
    recommendations = sorted(recommendations, key=lambda x: x["confidence"], reverse=True)
    
    # Assign ranks
    for idx, rec in enumerate(recommendations):
        rec["rank"] = idx + 1
        
    return recommendations
