import os
import sys

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# This is to let us import from the parent directory, not sure where I learned it from; I just use it all the time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from classifier import (
    GetPerformanceMetrics,
    PredictEnsembleSVM,
    PredictGeneralSVM,
    PreprocessInputs,
    TrainEnsembleSVM,
    TrainGeneralSVM,
)


def main():
    print("Loading datasets")

    digits = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    fashion = fetch_openml("fashion-mnist", version=1, as_frame=False, parser="auto")

    combined_data_list = []

    # Get the actual sizes of the datasets
    n_digits = digits.data.shape[0]
    n_fashion = fashion.data.shape[0]

    # Label Digits as [0, true_label]
    for i in range(n_digits):
        combined_data_list.append([[0, str(digits.target[i])], digits.data[i]])

    # Label Fashion as [1, true_label]
    for i in range(n_fashion):
        combined_data_list.append([[1, str(fashion.target[i])], fashion.data[i]])

    # Just making sure they are 28x28
    X, y = PreprocessInputs(combined_data_list, n=28, m=28)
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training general SVM")
    gen_model, gen_pca = TrainGeneralSVM(X_train, y_train, n_components=50)
    gen_preds = PredictGeneralSVM(gen_model, gen_pca, X_test)
    gen_metrics = GetPerformanceMetrics(y_test, gen_preds)

    print("Training ensemble SVM")
    # We keep n_clusters = 5 because otherwise there might not be enough data per set
    ens_models, ens_kmeans, ens_pca = TrainEnsembleSVM(X_train, y_train, n_clusters=5, n_components=50)
    ens_preds = PredictEnsembleSVM(ens_models, ens_kmeans, ens_pca, X_test)
    ens_metrics = GetPerformanceMetrics(y_test, ens_preds)

    print("Results:")

    # Architecture is index 0 and the metrics are index 1
    for arch, results in [("general SVM", gen_metrics), ("ensemble SVM", ens_metrics)]:
        print(f"\n[{arch}]")
        for task_key in ["Class 0", "Class 1"]:
            # Want to separate the results by classificaion task (in the long run, this will be like MD vs LaTeX
            task_name = "Digit vs Fashion" if task_key == "Class 0" else "True Label"
            m = results[task_key]

            print(f"  {task_name}:")
            # Using a single line per task for a clean table
            print(f"    - Accuracy:  {m['Accuracy']:.4f}    F1-Score:  {m['F1-Score']:.4f}")
            print(f"    - Precision: {m['Precision']:.4f}    Recall:    {m['Recall']:.4f}")


if __name__ == "__main__":
    main()
