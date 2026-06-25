import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Import custom ML modules
from app.ml.preprocessing import (
    impute_missing_values,
    LabelEncoderScratch,
    OneHotEncoderScratch,
    MinMaxScalerScratch,
    StandardScalerScratch,
    train_test_split_scratch
)
from app.ml.regression import (
    LinearRegressionScratch,
    MultipleLinearRegressionScratch,
    PolynomialRegressionScratch
)
from app.ml.classification import (
    KNNClassifierScratch,
    LogisticRegressionScratch,
    NaiveBayesScratch
)
from app.ml.clustering import (
    KMeansScratch,
    HierarchicalClusteringScratch
)
from app.ml.evaluation import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report_scratch,
    confusion_matrix_scratch,
    calculate_inertia,
    cluster_distribution
)
from app.ml.recommender import analyze_dataset, recommend_algorithms

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI-Based ML Algorithm Recommendation System")

# Enable CORS for cross-origin deployments (like Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create data directory if not exists
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

CSV_FILE_PATH = os.path.join(DATA_DIR, "uploaded_dataset.csv")
MODEL_STATE_PATH = os.path.join(DATA_DIR, "model_state.pkl")

# Global state in memory for active sessions
GLOBAL_STATE = {
    "model": None,
    "target_label_encoder": None,
    "feature_one_hot_encoder": None,
    "feature_scaler": None,
    "selected_features": [],
    "numeric_features": [],
    "categorical_features": [],
    "target_column": None,
    "problem_type": None,
    "algorithm_name": None
}

def save_model_state():
    with open(MODEL_STATE_PATH, "wb") as f:
        pickle.dump(GLOBAL_STATE, f)

def load_model_state():
    global GLOBAL_STATE
    if os.path.exists(MODEL_STATE_PATH):
        try:
            with open(MODEL_STATE_PATH, "rb") as f:
                GLOBAL_STATE = pickle.load(f)
        except Exception:
            pass

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        content = await file.read()
        # Save to disk for persistence
        with open(CSV_FILE_PATH, "wb") as f:
            f.write(content)
            
        # Load sample
        df = pd.read_csv(CSV_FILE_PATH)
        n_rows, n_cols = df.shape
        
        # Build preview records
        preview_rows = df.head(10).fillna("").to_dict(orient="records")
        
        columns_info = []
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            unique_count = int(df[col].nunique())
            dtype = str(df[col].dtype)
            
            # Extract unique values for non-numeric columns
            unique_values = []
            if not pd.api.types.is_numeric_dtype(df[col]) or unique_count <= 10:
                unique_values = [str(x) for x in df[col].dropna().unique().tolist()[:100]]
                
            columns_info.append({
                "name": col,
                "type": dtype,
                "null_count": null_count,
                "unique_count": unique_count,
                "unique_values": unique_values
            })
            
        return {
            "filename": file.filename,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "columns": columns_info,
            "preview": preview_rows
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {str(e)}")

class AnalyzeRequest(BaseModel):
    problem_type: str
    target_column: Optional[str] = None

@app.post("/api/analyze")
async def analyze_and_recommend(request: AnalyzeRequest):
    if not os.path.exists(CSV_FILE_PATH):
        raise HTTPException(status_code=400, detail="No dataset uploaded yet.")
        
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        
        if request.target_column and request.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found.")
            
        # Perform analysis
        analysis = analyze_dataset(df, target_column=request.target_column)
        
        # Recommendations
        recs = recommend_algorithms(analysis, request.problem_type)
        
        return {
            "analysis": analysis,
            "recommendations": recs
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

class PreprocessingConfig(BaseModel):
    impute_numeric: str = "mean"
    impute_categorical: str = "mode"
    scale_type: str = "minmax"
    test_size: float = 0.2
    random_state: int = 42

class TrainRequest(BaseModel):
    algorithm: str
    problem_type: str
    features: List[str]
    target_column: Optional[str] = None
    preprocessing: PreprocessingConfig
    hyperparams: Dict[str, Any]

@app.post("/api/train")
async def train_model(request: TrainRequest):
    global GLOBAL_STATE
    if not os.path.exists(CSV_FILE_PATH):
        raise HTTPException(status_code=400, detail="No dataset uploaded.")
        
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        
        # Verify columns
        for feat in request.features:
            if feat not in df.columns:
                raise HTTPException(status_code=400, detail=f"Feature '{feat}' not found.")
        if request.problem_type != "clustering" and request.target_column not in df.columns:
            raise HTTPException(status_code=400, detail="Target column required.")

        # Subset dataframe
        cols_to_use = request.features.copy()
        if request.problem_type != "clustering":
            cols_to_use.append(request.target_column)
            
        sub_df = df[cols_to_use].copy()
        
        # 1. Impute missing values
        sub_df = impute_missing_values(
            sub_df, 
            numeric_strategy=request.preprocessing.impute_numeric,
            categorical_strategy=request.preprocessing.impute_categorical
        )
        
        # Split target and features
        X_df = sub_df[request.features].copy()
        y_series = None
        if request.problem_type != "clustering":
            y_series = sub_df[request.target_column].copy()
            
        # Identify numeric vs categorical features
        numeric_features = []
        categorical_features = []
        for col in X_df.columns:
            if pd.api.types.is_numeric_dtype(X_df[col]) and not pd.api.types.is_bool_dtype(X_df[col]):
                numeric_features.append(col)
            else:
                categorical_features.append(col)
                
        # Split Train/Test
        if request.problem_type != "clustering":
            X_train_df, X_test_df, y_train, y_test = train_test_split_scratch(
                X_df, y_series, 
                test_size=request.preprocessing.test_size, 
                random_state=request.preprocessing.random_state
            )
        else:
            # For clustering, we split for validation if test_size > 0, else train on all
            X_train_df, X_test_df, _, _ = train_test_split_scratch(
                X_df, pd.Series(range(len(X_df))), 
                test_size=request.preprocessing.test_size, 
                random_state=request.preprocessing.random_state
            )
            y_train, y_test = None, None

        # 2. Encode target (only if classification and target is not numeric)
        target_encoder = None
        if request.problem_type == "classification":
            y_train_arr = y_train.values
            y_test_arr = y_test.values
            if not pd.api.types.is_numeric_dtype(y_train):
                target_encoder = LabelEncoderScratch()
                y_train_arr = target_encoder.fit_transform(y_train_arr)
                y_test_arr = target_encoder.transform(y_test_arr)
        elif request.problem_type == "regression":
            y_train_arr = y_train.values.astype(float)
            y_test_arr = y_test.values.astype(float)
        else:
            y_train_arr, y_test_arr = None, None

        # 3. Encode categorical features
        feature_one_hot_encoder = None
        if len(categorical_features) > 0:
            feature_one_hot_encoder = OneHotEncoderScratch()
            X_train_cat_enc = feature_one_hot_encoder.fit_transform(X_train_df, categorical_features)
            X_test_cat_enc = feature_one_hot_encoder.transform(X_test_df)
            
            # Combine numeric features and encoded features
            if len(numeric_features) > 0:
                X_train_arr = np.column_stack([X_train_df[numeric_features].values.astype(float), X_train_cat_enc])
                X_test_arr = np.column_stack([X_test_df[numeric_features].values.astype(float), X_test_cat_enc])
            else:
                X_train_arr = X_train_cat_enc
                X_test_arr = X_test_cat_enc
        else:
            X_train_arr = X_train_df[numeric_features].values.astype(float)
            X_test_arr = X_test_df[numeric_features].values.astype(float)

        # 4. Scaling (only numeric columns, but we can scale everything or just numeric parts)
        # Traditionally, we fit scaler on numeric features and scale them, keeping one-hot variables as is (0/1).
        # This is the correct, professional way!
        feature_scaler = None
        if request.preprocessing.scale_type != "none" and len(numeric_features) > 0:
            if request.preprocessing.scale_type == "minmax":
                feature_scaler = MinMaxScalerScratch()
            elif request.preprocessing.scale_type == "standard":
                feature_scaler = StandardScalerScratch()
                
            if feature_scaler:
                # Scale the numeric subset in the combined X array (which are indices 0 to len(numeric_features)-1)
                num_count = len(numeric_features)
                X_train_arr[:, :num_count] = feature_scaler.fit_transform(X_train_arr[:, :num_count])
                X_test_arr[:, :num_count] = feature_scaler.transform(X_test_arr[:, :num_count])

        # 5. Initialize & Train Model
        algo_name = request.algorithm
        model = None
        history = []
        
        # Regression models
        if algo_name == "Linear Regression (Simple)":
            # Can only use 1 feature, use first feature
            model = LinearRegressionScratch(
                learning_rate=float(request.hyperparams.get("learning_rate", 0.01)),
                epochs=int(request.hyperparams.get("epochs", 1000)),
                method=request.hyperparams.get("solver", "gradient_descent")
            )
            # Ensure it fits simple linear regression constraints
            model.fit(X_train_arr[:, :1], y_train_arr)
            history = model.history
        elif algo_name == "Multiple Linear Regression":
            model = MultipleLinearRegressionScratch(
                learning_rate=float(request.hyperparams.get("learning_rate", 0.01)),
                epochs=int(request.hyperparams.get("epochs", 1000)),
                method=request.hyperparams.get("solver", "gradient_descent")
            )
            model.fit(X_train_arr, y_train_arr)
            history = model.history
        elif algo_name == "Polynomial Regression":
            model = PolynomialRegressionScratch(
                degree=int(request.hyperparams.get("degree", 2)),
                learning_rate=float(request.hyperparams.get("learning_rate", 0.01)),
                epochs=int(request.hyperparams.get("epochs", 1000)),
                method=request.hyperparams.get("solver", "ols")
            )
            model.fit(X_train_arr, y_train_arr)
            history = model.history
            
        # Classification models
        elif algo_name == "K-Nearest Neighbors (KNN)":
            model = KNNClassifierScratch(
                k=int(request.hyperparams.get("k", 3)),
                metric=request.hyperparams.get("metric", "euclidean")
            )
            model.fit(X_train_arr, y_train_arr)
            history = model.history
        elif algo_name == "Logistic Regression":
            model = LogisticRegressionScratch(
                learning_rate=float(request.hyperparams.get("learning_rate", 0.01)),
                epochs=int(request.hyperparams.get("epochs", 1000))
            )
            model.fit(X_train_arr, y_train_arr)
            history = model.history
        elif algo_name == "Naive Bayes":
            model = NaiveBayesScratch()
            model.fit(X_train_arr, y_train_arr)
            history = model.history
            
        # Clustering models
        elif algo_name in ["K-Means (K-means++)", "K-Means (Random Init)"]:
            init_method = "kmeans++" if "k-means++" in algo_name.lower() else "random"
            model = KMeansScratch(
                k=int(request.hyperparams.get("k", 3)),
                init=init_method,
                max_iter=int(request.hyperparams.get("max_iter", 300)),
                tol=float(request.hyperparams.get("tol", 1e-4))
            )
            model.fit(X_train_arr)
            history = model.history
        elif algo_name == "Hierarchical Clustering":
            model = HierarchicalClusteringScratch(
                k=int(request.hyperparams.get("k", 3)),
                linkage=request.hyperparams.get("linkage", "average")
            )
            model.fit(X_train_arr)
            history = model.history
        else:
            raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algo_name}")

        # 6. Evaluation
        results = {}
        if request.problem_type == "regression":
            train_preds = model.predict(X_train_arr)
            test_preds = model.predict(X_test_arr)
            
            results = {
                "train": {
                    "mae": mean_absolute_error(y_train_arr, train_preds),
                    "mse": mean_squared_error(y_train_arr, train_preds),
                    "rmse": root_mean_squared_error(y_train_arr, train_preds),
                    "r2": r2_score(y_train_arr, train_preds)
                },
                "test": {
                    "mae": mean_absolute_error(y_test_arr, test_preds),
                    "mse": mean_squared_error(y_test_arr, test_preds),
                    "rmse": root_mean_squared_error(y_test_arr, test_preds),
                    "r2": r2_score(y_test_arr, test_preds)
                }
            }
        elif request.problem_type == "classification":
            train_preds = model.predict(X_train_arr)
            test_preds = model.predict(X_test_arr)
            
            train_report = classification_report_scratch(y_train_arr, train_preds)
            test_report = classification_report_scratch(y_test_arr, test_preds)
            
            # Confusion matrix for test set
            cm = confusion_matrix_scratch(y_test_arr, test_preds)
            
            # Map back string labels if label encoder was used
            if target_encoder:
                cm["labels"] = [str(x) for x in target_encoder.inverse_transform([int(float(l)) for l in cm["labels"]])]
                
            results = {
                "train": train_report,
                "test": test_report,
                "confusion_matrix": cm
            }
        elif request.problem_type == "clustering":
            # For clustering, evaluate on train set
            train_labels = model.labels
            
            if "K-Means" in algo_name:
                inertia = model.inertia_
                centroids = model.centroids.tolist()
            else: # Hierarchical
                # Calculate inertia manually using cluster centroids
                centroids_list = []
                for label_val in range(request.hyperparams.get("k", 3)):
                    pts = X_train_arr[train_labels == label_val]
                    if len(pts) > 0:
                        centroids_list.append(np.mean(pts, axis=0))
                    else:
                        centroids_list.append(np.zeros(X_train_arr.shape[1]))
                centroids = np.array(centroids_list)
                inertia = calculate_inertia(X_train_arr, centroids, train_labels)
                centroids = centroids.tolist()
                
            results = {
                "train": {
                    "inertia": inertia,
                    "distribution": cluster_distribution(train_labels)
                },
                "centroids": centroids
            }

        # Save to GLOBAL_STATE
        GLOBAL_STATE = {
            "model": model,
            "target_label_encoder": target_encoder,
            "feature_one_hot_encoder": feature_one_hot_encoder,
            "feature_scaler": feature_scaler,
            "selected_features": request.features,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "target_column": request.target_column,
            "problem_type": request.problem_type,
            "algorithm_name": algo_name
        }
        save_model_state()

        return {
            "success": True,
            "algorithm": algo_name,
            "history": history,
            "results": results
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

class PredictRequest(BaseModel):
    inputs: Dict[str, Any]

@app.post("/api/predict")
async def predict_sample(request: PredictRequest):
    global GLOBAL_STATE
    load_model_state()
    
    if not GLOBAL_STATE["model"]:
        raise HTTPException(status_code=400, detail="No model trained yet. Please train a model first.")
        
    try:
        # Construct dataframe from inputs
        inputs_dict = {k: [v] for k, v in request.inputs.items()}
        sample_df = pd.DataFrame(inputs_dict)
        
        # Preprocess features
        # 1. Fill missing (dummy imputation, though sample should have all values)
        # If some feature is missing in request, fill with standard default (0 or Empty)
        for col in GLOBAL_STATE["selected_features"]:
            if col not in sample_df.columns:
                sample_df[col] = [0.0 if col in GLOBAL_STATE["numeric_features"] else "Missing"]

        # Reorder to match training selected features
        sample_df = sample_df[GLOBAL_STATE["selected_features"]]
        
        # 2. Categorical encoding
        one_hot_enc = GLOBAL_STATE["feature_one_hot_encoder"]
        numeric_feats = GLOBAL_STATE["numeric_features"]
        categorical_feats = GLOBAL_STATE["categorical_features"]
        
        if one_hot_enc:
            sample_cat_enc = one_hot_enc.transform(sample_df)
            if len(numeric_feats) > 0:
                sample_arr = np.column_stack([sample_df[numeric_feats].values.astype(float), sample_cat_enc])
            else:
                sample_arr = sample_cat_enc
        else:
            sample_arr = sample_df[numeric_feats].values.astype(float)
            
        # 3. Scaling
        scaler = GLOBAL_STATE["feature_scaler"]
        if scaler and len(numeric_feats) > 0:
            num_count = len(numeric_feats)
            sample_arr[:, :num_count] = scaler.transform(sample_arr[:, :num_count])
            
        # 4. Predict
        model = GLOBAL_STATE["model"]
        
        # For simple linear regression, it only expects the first column
        if GLOBAL_STATE["algorithm_name"] == "Linear Regression (Simple)":
            pred = model.predict(sample_arr[:, :1])
        else:
            pred = model.predict(sample_arr)
            
        # Format prediction
        raw_pred = pred[0]
        prediction_val = raw_pred
        
        # For classification, map back class labels if encoder exists
        lbl_encoder = GLOBAL_STATE["target_label_encoder"]
        if GLOBAL_STATE["problem_type"] == "classification":
            # prediction_val is class index or label
            if lbl_encoder:
                prediction_val = str(lbl_encoder.inverse_transform([int(float(raw_pred))])[0])
            else:
                prediction_val = str(raw_pred)
        else:
            prediction_val = float(raw_pred)
            
        return {
            "prediction": prediction_val
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    global GLOBAL_STATE
    load_model_state()
    if not GLOBAL_STATE["model"]:
        raise HTTPException(status_code=400, detail="No model trained yet. Please train a model first.")
    
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        content = await file.read()
        import io
        test_df = pd.read_csv(io.BytesIO(content))
        
        # Verify columns
        missing_cols = [col for col in GLOBAL_STATE["selected_features"] if col not in test_df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Uploaded test CSV is missing required features: {', '.join(missing_cols)}")
            
        output_df = test_df.copy()
        X_df = test_df[GLOBAL_STATE["selected_features"]].copy()
        
        # 1. Fill missing values
        for col in GLOBAL_STATE["selected_features"]:
            if X_df[col].isnull().sum() > 0:
                if col in GLOBAL_STATE["numeric_features"]:
                    X_df[col] = X_df[col].fillna(0.0)
                else:
                    X_df[col] = X_df[col].fillna("Missing")
                    
        # 2. Categorical encoding
        one_hot_enc = GLOBAL_STATE["feature_one_hot_encoder"]
        numeric_feats = GLOBAL_STATE["numeric_features"]
        
        if one_hot_enc:
            sample_cat_enc = one_hot_enc.transform(X_df)
            if len(numeric_feats) > 0:
                sample_arr = np.column_stack([X_df[numeric_feats].values.astype(float), sample_cat_enc])
            else:
                sample_arr = sample_cat_enc
        else:
            sample_arr = X_df[numeric_feats].values.astype(float)
            
        # 3. Scaling
        scaler = GLOBAL_STATE["feature_scaler"]
        if scaler and len(numeric_feats) > 0:
            num_count = len(numeric_feats)
            sample_arr[:, :num_count] = scaler.transform(sample_arr[:, :num_count])
            
        # 4. Predict
        model = GLOBAL_STATE["model"]
        if GLOBAL_STATE["algorithm_name"] == "Linear Regression (Simple)":
            preds = model.predict(sample_arr[:, :1])
        else:
            preds = model.predict(sample_arr)
            
        # 5. Decode predictions
        lbl_encoder = GLOBAL_STATE["target_label_encoder"]
        pred_col = f"Predicted_{GLOBAL_STATE['target_column'] or 'Cluster'}"
        
        if GLOBAL_STATE["problem_type"] == "classification":
            if lbl_encoder:
                decoded_preds = lbl_encoder.inverse_transform([int(float(x)) for x in preds])
                output_df[pred_col] = decoded_preds
            else:
                output_df[pred_col] = [str(x) for x in preds]
        elif GLOBAL_STATE["problem_type"] == "regression":
            output_df[pred_col] = [float(x) for x in preds]
        else:
            output_df[pred_col] = [f"Cluster {int(x)}" for x in preds]
            
        # Check if actual target column is in test_df for evaluation
        batch_metrics = None
        target_col_name = GLOBAL_STATE["target_column"]
        if target_col_name and target_col_name in test_df.columns:
            try:
                actual_y = test_df[target_col_name].values
                if GLOBAL_STATE["problem_type"] == "classification":
                    pred_str = np.array(output_df[pred_col].values).astype(str)
                    actual_str = np.array(actual_y).astype(str)
                    report = classification_report_scratch(actual_str, pred_str)
                    batch_metrics = {
                        "accuracy": report["accuracy"],
                        "precision": report["precision"],
                        "recall": report["recall"],
                        "f1_score": report["f1_score"]
                    }
                elif GLOBAL_STATE["problem_type"] == "regression":
                    pred_float = np.array(output_df[pred_col].values).astype(float)
                    actual_float = np.array(actual_y).astype(float)
                    batch_metrics = {
                        "r2": r2_score(actual_float, pred_float),
                        "mae": mean_absolute_error(actual_float, pred_float),
                        "rmse": root_mean_squared_error(actual_float, pred_float),
                        "mse": mean_squared_error(actual_float, pred_float)
                    }
            except Exception:
                pass

        # Save predictions to file for download
        pred_output_path = os.path.join(DATA_DIR, "batch_predictions.csv")
        output_df.to_csv(pred_output_path, index=False)
        
        preview_rows = output_df.head(10).fillna("").to_dict(orient="records")
        return {
            "success": True,
            "preview": preview_rows,
            "columns": output_df.columns.tolist(),
            "metrics": batch_metrics
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/api/predict/download")
async def download_predictions():
    pred_output_path = os.path.join(DATA_DIR, "batch_predictions.csv")
    if not os.path.exists(pred_output_path):
        raise HTTPException(status_code=404, detail="No predictions file found. Run a batch prediction first.")
    return FileResponse(pred_output_path, media_type="text/csv", filename="batch_predictions.csv")

# Mount static folder
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
