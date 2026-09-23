import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

from preprocess import preprocess_text

EMOTION_MAP = {
    0: {"name": "Sadness", "emoji": "😢", "color": "#3b82f6"},
    1: {"name": "Joy", "emoji": "😊", "color": "#eab308"},
    2: {"name": "Love", "emoji": "❤️", "color": "#ec4899"},
    3: {"name": "Anger", "emoji": "😠", "color": "#ef4444"},
    4: {"name": "Fear", "emoji": "😨", "color": "#8b5cf6"},
    5: {"name": "Surprise", "emoji": "😲", "color": "#06b6d4"}
}

def load_data(data_dir: str):
    train_path = os.path.join(data_dir, "training.csv")
    val_path = os.path.join(data_dir, "validation.csv")
    test_path = os.path.join(data_dir, "test.csv")

    print(f"Loading datasets from {data_dir}...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    print(f"Train samples: {len(df_train)}, Val samples: {len(df_val)}, Test samples: {len(df_test)}")
    return df_train, df_val, df_test

def train_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "NLP Datasets")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    df_train, df_val, df_test = load_data(data_dir)

    print("Preprocessing texts...")
    df_train['clean_text'] = df_train['text'].apply(preprocess_text)
    df_val['clean_text'] = df_val['text'].apply(preprocess_text)
    df_test['clean_text'] = df_test['text'].apply(preprocess_text)

    # Feature extraction via TF-IDF
    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=12000,
        sublinear_tf=True
    )
    
    X_train = vectorizer.fit_transform(df_train['clean_text'])
    y_train = df_train['label']

    X_val = vectorizer.transform(df_val['clean_text'])
    y_val = df_val['label']

    X_test = vectorizer.transform(df_test['clean_text'])
    y_test = df_test['label']

    # Candidate Classifiers
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=2.0, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear Support Vector Machine": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42)),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    }

    results = {}
    best_model = None
    best_model_name = ""
    best_accuracy = 0.0

    print("\n--- Model Training & Comparison ---")
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        
        val_preds = clf.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)

        test_preds = clf.predict(X_test)
        test_acc = accuracy_score(y_test, test_preds)
        
        p, r, f1, _ = precision_recall_fscore_support(y_test, test_preds, average='weighted')
        
        results[name] = {
            "val_accuracy": round(float(val_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4)
        }
        
        print(f"  {name} -> Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | F1: {f1:.4f}")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = clf
            best_model_name = name

    print(f"\nBest Performing Model: {best_model_name} with Test Accuracy: {best_accuracy:.4f}")

    # Detailed Evaluation for Best Model
    y_test_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_test_pred, target_names=[EMOTION_MAP[i]["name"] for i in range(6)], output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_test_pred).tolist()

    # Save vectorizer and best model
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    model_path = os.path.join(models_dir, "emotion_model.pkl")
    metrics_path = os.path.join(models_dir, "metrics.json")

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(best_model, model_path)

    metrics_data = {
        "best_model": best_model_name,
        "test_accuracy": round(float(best_accuracy), 4),
        "comparison": results,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "labels": [EMOTION_MAP[i]["name"] for i in range(6)]
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"Saved best model to {model_path}")
    print(f"Saved vectorizer to {vectorizer_path}")
    print(f"Saved metrics to {metrics_path}")

if __name__ == "__main__":
    train_pipeline()
