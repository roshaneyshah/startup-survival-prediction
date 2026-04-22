import numpy as np
import pickle
import os
import sys

sys.path.append(os.path.dirname(__file__))

from preprocess import load_and_build, get_xy, save_encoders, FEATURES

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")


def train(data_path=None):
    print("Loading and preprocessing data...")
    df, encoders = load_and_build(data_path) if data_path else load_and_build()
    X, y = get_xy(df)

    print(f"Dataset: {len(X)} companies | Survival rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 50)
    print(classification_report(y_test, y_pred, target_names=["Closed", "Survived"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("=" * 50)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved -> {MODEL_PATH}")

    save_encoders(encoders, path=os.path.join(os.path.dirname(__file__), "..", "models", "encoders.pkl"))
    print("Encoders saved -> models/encoders.pkl")

    return model, encoders


if __name__ == "__main__":
    train()
