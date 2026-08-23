# Model training script: build the pipeline, tune with GridSearchCV, log to MLflow, evaluate, save the best model.
"""
Model Training with Experimentation Tracking
---------------------------------------------
Loads the train/test splits produced by the data-prep job, builds a
preprocessing + XGBoost pipeline, tunes it with GridSearchCV, logs every
tuned parameter and evaluation metric to MLflow, and saves the best
pipeline so the workflow can commit it into the repository.
"""

import logging
import os
import joblib
import pandas as pd
import mlflow

# Keep MLflow's own SQLite-migration logging out of the way of the training
# output below -- it's informational, not something to review each run.
logging.getLogger("alembic").setLevel(logging.WARNING)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

MODEL_OUTPUT_PATH = "tourism_project/deployment/model.joblib"

NUMERIC_FEATURES = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]

CATEGORICAL_FEATURES = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

PARAM_GRID = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0],
}


def load_splits():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")
    return Xtrain, Xtest, ytrain, ytest


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def main():
    Xtrain, Xtest, ytrain, ytest = load_splits()

    pipeline = build_pipeline()

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    # A local SQLite-backed tracking store keeps everything in one file
    # (mlflow.db) that works the same across MLflow versions -- no server
    # to run, and it survives inside a single GitHub Actions job.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("tourism_wellness_package")

    with mlflow.start_run():
        grid_search.fit(Xtrain, ytrain)
        best_model = grid_search.best_estimator_

        # Log every tuned hyperparameter that was selected.
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")

        # Evaluate on the held-out test set.
        preds = best_model.predict(Xtest)
        proba = best_model.predict_proba(Xtest)[:, 1]

        metrics = {
            "test_accuracy": accuracy_score(ytest, preds),
            "test_precision": precision_score(ytest, preds),
            "test_recall": recall_score(ytest, preds),
            "test_f1": f1_score(ytest, preds),
            "test_roc_auc": roc_auc_score(ytest, proba),
            "best_cv_f1": grid_search.best_score_,
        }
        mlflow.log_metrics(metrics)

        print("=" * 60)
        print("MODEL TRAINING SUMMARY")
        print("=" * 60)
        print("Best parameters found by GridSearchCV:")
        for k, v in grid_search.best_params_.items():
            print(f"  {k}: {v}")
        print("-" * 60)
        print("Test set performance:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print("-" * 60)
        print("Classification report (test set):")
        print(classification_report(ytest, preds))
        print("Confusion matrix (test set):")
        print(confusion_matrix(ytest, preds))
        print("=" * 60)

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    print(f"Best model saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
