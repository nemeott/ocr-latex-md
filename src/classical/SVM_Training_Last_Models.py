import numpy as np
from classifier import *
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from datasets import load_dataset

def plot_comparison_sweep(param_name, param_values, acc_pix, acc_hog, title, filename):
    plt.figure()
    if param_name == 'C':
        plt.xscale('log')
    # Plot Pixels
    plt.plot(param_values, acc_pix, marker='o', linestyle='-', color='b', label='Hierarchical (Pixels)')
    # Plot HOG
    plt.plot(param_values, acc_hog, marker='s', linestyle='--', color='g', label='Hierarchical (HOG)')
    
    plt.xlabel(param_name)
    plt.ylabel('Average Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(filename)
    plt.close()
    print(f"  -> Saved graph: {filename}")

def main():
    print("Loading datasets")
    
    # Load the datasets
    # You may need to rollback your datasets version because these datasets use custom python scripts which were removed from recent versions due to security concerns
    LaTeXTrain = load_dataset("randall-lab/hasy-v2", split = "train", trust_remote_code=True)   
    LaTeXTest = load_dataset("randall-lab/hasy-v2", split = "test", trust_remote_code=True)
    textTrain = load_dataset("Royc30ne/emnist-byclass", split = "train", trust_remote_code=True)
    textTest = load_dataset("Royc30ne/emnist-byclass", split = "test", trust_remote_code=True)

    combined_data_list_train = []
    combined_data_list_test = []

    # Get the actual sizes of the datasets
    n_LaTeXTrain = len(LaTeXTrain)
    n_LaTeXTest = len(LaTeXTest)
    n_textTrain = len(textTrain)
    n_textTest = len(textTest)

    # Label LaTeX as [0, true_label]
    for row in LaTeXTrain:
        combined_data_list_train.append([[0, str(row['symbol_id'])], row['image']])
    for row in LaTeXTest:
        combined_data_list_test.append([[0, str(row['symbol_id'])], row['image']])

    # Label text as [1, true_label]
    for row in textTrain:
        combined_data_list_train.append([[1, str(row['label'])], row['image']])
    for row in textTest:
        combined_data_list_test.append([[1, str(row['label'])], row['image']])

    print(f"Preprocessing {len(combined_data_list_train)} images")
    XTrain, yTrain = PreprocessInputs(combined_data_list_train, n = 32, m = 32)
    XTest, yTest = PreprocessInputs(combined_data_list_test, n = 32, m = 32)

    # --- Parameter Sweeps ---
    print("\nStarting Parameter Sweeps")
    
    # We fix PCA and Orientations and then just sweep over C
    bPCA, bOrient = 40, 9
    Cs = [0.01, 0.1, 1, 10, 100]

    print("Sweeping C for both Hierarchical architectures")
    h_pix_c_acc = []
    h_hog_c_acc = []
    
    for c in Cs:
        # Train and eval Pixels
        dm_p, lm_p, pca_p = TrainHierarchicalSVM(XTrain, yTrain, n_components=bPCA, c=c)
        preds_p = PredictHierarchicalSVM(dm_p, lm_p, pca_p, XTest)
        m_p = GetPerformanceMetrics(yTest, preds_p)
        h_pix_c_acc.append((m_p["Class 0"]["Accuracy"] + m_p["Class 1"]["Accuracy"]) / 2)
        
        # Train and eval HOG
        dm_h, lm_h, pca_h, _ = TrainHierarchicalSVMHOG(XTrain, yTrain, n_components=bPCA, c=c, Orientations=bOrient)
        preds_h = PredictHierarchicalSVMHOG(dm_h, lm_h, pca_h, XTest, bOrient)
        m_h = GetPerformanceMetrics(yTest, preds_h)
        h_hog_c_acc.append((m_h["Class 0"]["Accuracy"] + m_h["Class 1"]["Accuracy"]) / 2)

    # Plot them together
    plot_comparison_sweep('C', Cs, h_pix_c_acc, h_hog_c_acc, 'Hierarchical Models: C Parameter Sweep', 'plots/hierarchical_comparison_C.png')
    
    # Grab the best greedy C values for the final training
    best_h_pix_c = Cs[np.argmax(h_pix_c_acc)]
    best_h_hog_c = Cs[np.argmax(h_hog_c_acc)]

    print("\nTraining Final Optimal Models")
    
    dm_p, lm_p, pca_p = TrainHierarchicalSVM(XTrain, yTrain, n_components=bPCA, c=best_h_pix_c)
    SaveHierarchicalSVM(dm_p, lm_p, pca_p, "models/opt_hier_pix")
    opt_hier_preds = PredictHierarchicalSVM(dm_p, lm_p, pca_p, XTest)
    opt_hier_metrics = GetPerformanceMetrics(yTest, opt_hier_preds)

    dm_h, lm_h, pca_h, _ = TrainHierarchicalSVMHOG(XTrain, yTrain, n_components=bPCA, c=best_h_hog_c, Orientations=bOrient)
    SaveHierarchicalSVMHOG(dm_h, lm_h, pca_h, bOrient, "models/opt_hier_hog")
    opt_hier_hog_preds = PredictHierarchicalSVMHOG(dm_h, lm_h, pca_h, XTest, bOrient)
    opt_hier_hog_metrics = GetPerformanceMetrics(yTest, opt_hier_hog_preds)

    print("Results:")
    architectures = [
        ("Optimal Hierarchical SVM (Pixels)", opt_hier_metrics),
        ("Optimal Hierarchical SVM (HOG)", opt_hier_hog_metrics)
    ]
    
    for arch, results in architectures:
        print(f"\n[{arch}]")
        for task_key in ["Class 0", "Class 1"]:
            task_name = "LaTeX Symbols" if task_key == "Class 0" else "EMNIST Text"
            m = results[task_key]

            print(f"  {task_name}:")
            print(f"    - Accuracy:  {m['Accuracy']:.4f}    F1-Score:  {m['F1-Score']:.4f}")
            print(f"    - Precision: {m['Precision']:.4f}    Recall:    {m['Recall']:.4f}")

if __name__ == "__main__":
    main()
