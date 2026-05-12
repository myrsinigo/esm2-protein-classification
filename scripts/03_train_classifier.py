import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    print(f"\n{name}")
    print("=" * len(name))
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def main():
    Path("results").mkdir(exist_ok=True)

    X = np.load("results/esm2_embeddings.npy")
    y = np.load("results/labels.npy")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logistic_regression = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )
    
    svm_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf"))
        ]
    )
    results = []

    results.append(
        evaluate_model(
            "Logistic Regression",
            logistic_regression,
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    results.append(
        evaluate_model(
            "Random Forest",
            random_forest,
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    results.append(
        evaluate_model(
            "SVM (RBF)",
            svm_model,
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    results_df = pd.DataFrame(results)


    # ROC curve using Logistic Regression

    y_probs = logistic_regression.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig("results/roc_curve.png", dpi=300)
    plt.close()

    results_df.to_csv("results/metrics.csv", index=False)

    print("\nSaved metrics to results/metrics.csv")
    print(results_df)


if __name__ == "__main__":
    main()