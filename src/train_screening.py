"""
Train and evaluate the ASD screening classifier.
"""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/asd_screening_dataset.csv"
MODEL_OUT = ROOT / "models/asd_screening_model.joblib"

df = pd.read_csv(DATA)
target = "ASD_Status"
features = [c for c in df.columns if c not in [target, "Child_ID"]]
cat = ["Gender"]
num = [c for c in features if c not in cat]

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42, stratify=df[target]
)

pre = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
])

pipe = Pipeline([
    ("preprocessor", pre),
    ("model", RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    ))
])
pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
prob = pipe.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, pred, zero_division=0), 4))
print("F1:", round(f1_score(y_test, pred, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))

joblib.dump(pipe, MODEL_OUT)
print("Saved:", MODEL_OUT)
