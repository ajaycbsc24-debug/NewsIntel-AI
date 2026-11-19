
from pathlib import Path
import re
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

FAKE_PATH = BASE_DIR / "dataset" / "Fake.csv"
REAL_PATH = BASE_DIR / "dataset" / "True.csv"

MODEL_PATH = BASE_DIR / "models" / "fake_news_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Same text cleaning used during training
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load datasets
fake_df = pd.read_csv(FAKE_PATH)
real_df = pd.read_csv(REAL_PATH)

fake_df["label"] = 0
real_df["label"] = 1

data = pd.concat([fake_df, real_df], ignore_index=True)
data = data.drop_duplicates()

data["content"] = (
    data["title"].fillna("").astype(str)
    + " "
    + data["text"].fillna("").astype(str)
)

data["cleaned_content"] = data["content"].apply(clean_text)

X = data["cleaned_content"]
y = data["label"]

# Recreate the original 80/20 stratified split.
# IMPORTANT: These settings must match train_model.py.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Transform test data and predict
X_test_vectorized = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vectorized)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["FAKE", "REAL"],
    zero_division=0
)

matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\n========== MODEL EVALUATION ==========")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\nClassification Report:")
print(report)

print("Confusion Matrix:")
print(matrix)

# Save evaluation results
results = {
    "accuracy": round(accuracy * 100, 2),
    "precision": round(precision * 100, 2),
    "recall": round(recall * 100, 2),
    "f1_score": round(f1 * 100, 2),
    "confusion_matrix": matrix.tolist(),
    "classification_report": report
}

with open(OUTPUT_DIR / "evaluation_results.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nResults saved to outputs/evaluation_results.json")
