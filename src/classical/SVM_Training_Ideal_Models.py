import numpy as np
import pandas as pd
import os
import time
import joblib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from contextlib import redirect_stdout, redirect_stderr
from sklearn.model_selection import train_test_split
from classifier import *

from datasets import load_dataset  

def main():
    start_total = time.time()

    # Create folders for output
    for folder in ["models", "plots"]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
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
    # Iterating through rows to access 'symbol_id' and 'image' keys
    for row in LaTeXTrain:
        combined_data_list_train.append([[0, str(row['symbol_id'])], row['image']])
    for row in LaTeXTest:
        combined_data_list_test.append([[0, str(row['symbol_id'])], row['image']])

    # Label text as [1, true_label]
    # Iterating through rows to access 'label' and 'image' keys
    for row in textTrain:
        combined_data_list_train.append([[1, str(row['label'])], row['image']])
    for row in textTest:
        combined_data_list_test.append([[1, str(row['label'])], row['image']])

    # Just making sure they are all the same size
    XTrain, yTrain = PreprocessInputs(combined_data_list_train, n = 32, m = 32)
    XTest, yTest = PreprocessInputs(combined_data_list_test, n = 32, m = 32)

    # Compute HOG ahead of tmie to speed up C/PCA/K testing
    print(f"Computing Baseline HOG features (Orient={bOrient})")
    XTr_HOG = np.array(joblib.Parallel(n_jobs=-1)(joblib.delayed(hog)(img.reshape(32,32), orientations=bOrient, pixels_per_cell=(8,8), cells_per_block=(2,2)) for img in XTrain))
    XTe_HOG = np.array(joblib.Parallel(n_jobs=-1)(joblib.delayed(hog)(img.reshape(32,32), orientations=bOrient, pixels_per_cell=(8,8), cells_per_block=(2,2)) for img in XTest))

    def get_acc(preds): return GetPerformanceMetrics(yTest, preds)['Class 1']['Accuracy']

    final_results = []

    # General model no HOG
    print("\nTraining General (No HOG): PCA=40, C=0.01")
    m1, p1 = TrainGeneralSVM(XTrain, yTrain, n_components=40, c=0.01)
    acc1 = get_acc(PredictGeneralSVM(m1, p1, XTest))
    SaveGeneralSVM(m1, p1, "models/optimal_gen_svm")
    final_results.append(["Gen (No HOG)", 40, 0.01, "N/A", "N/A", acc1])

    # Ensemble model no HOG
    print("Training Ensemble (No HOG): k=4, PCA=100, C=0.01")
    m2, k2, p2 = TrainEnsembleSVM(XTrain, yTrain, n_clusters=4, n_components=100, c=0.01)
    acc2 = get_acc(PredictEnsembleSVM(m2, k2, p2, XTest))
    SaveEnsembleSVM(m2, k2, p2, "models/optimal_ens_svm")
    final_results.append(["Ens (No HOG)", 100, 0.01, 4, "N/A", acc2])

    # Compute HOG
    print("\nComputing HOG features (Orient=7) for HOG models")
    XTr_HOG7 = np.array(joblib.Parallel(n_jobs=-1)(joblib.delayed(hog)(img.reshape(32,32), orientations=7, pixels_per_cell=(8,8), cells_per_block=(2,2)) for img in XTrain))
    XTe_HOG7 = np.array(joblib.Parallel(n_jobs=-1)(joblib.delayed(hog)(img.reshape(32,32), orientations=7, pixels_per_cell=(8,8), cells_per_block=(2,2)) for img in XTest))

    # General model with HOG
    print("Training General HOG: Orient=7, PCA=60, C=0.1")
    m3, p3 = TrainGeneralSVM(XTr_HOG7, yTrain, n_components=60, c=0.1)
    acc3 = get_acc(PredictGeneralSVM(m3, p3, XTe_HOG7))
    SaveGeneralSVMHOG(m3, p3, 7, "models/optimal_gen_hog_svm")
    final_results.append(["Gen HOG", 60, 0.1, "N/A", 7, acc3])

    # Ensemble model HOG
    print("Training Ensemble HOG: Orient=7, k=2, PCA=60, C=0.1")
    m4, k4, p4 = TrainEnsembleSVM(XTr_HOG7, yTrain, n_clusters=2, n_components=60, c=0.1)
    acc4 = get_acc(PredictEnsembleSVM(m4, k4, p4, XTe_HOG7))
    SaveEnsembleSVMHOG(m4, k4, p4, 7, "models/optimal_ens_hog_svm")
    final_results.append(["Ens HOG", 60, 0.1, 2, 7, acc4])

    # Results
    print(f"{'Architecture':<15} | {'PCA':<5} | {'C':<6} | {'K':<3} | {'Orient':<6} | {'Accuracy':<8}")
    print("\n")
    for res in final_results:
        print(f"{res[0]:<15} | {res[1]:<5} | {res[2]:<6} | {res[3]:<3} | {res[4]:<6} | {res[5]:.4f}")
    print("\n")

    print(f"\nModels saved to /models. Total time: {(time.time() - start_total)/60:.2f}m.")

if __name__ == "__main__":
    main()   
