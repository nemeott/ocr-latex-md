import numpy as np
import os
import time
import joblib
import matplotlib.pyplot as plt
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

    # Values to sweep through for testing
    bC, bPCA, bK, bOrient = 1.0, 40, 5, 9
    Cs = [0.01, 0.1, 1, 10, 100]
    PCAs = [20, 40, 60, 80, 100]
    Ks = [2, 4, 6, 8, 10]
    Orients = [7, 8, 9, 10, 11, 12]

    # Compute HOG ahead of tmie to speed up C/PCA/K testing
    print(f"Computing Baseline HOG features (Orient={bOrient})")
    XTr_HOG = np.array(joblib.Parallel(n_jobs=-1)(joblib.delayed(hog)(img.reshape(32,32), orientations=bOrient, pixels_per_cell=(8,8), cells_per_block=(2,2)) for img in XTrain))
    XTe_HOG = np.array(joblib.Parallel(n_jobs=-1)(joblib.delayed(hog)(img.reshape(32,32), orientations=bOrient, pixels_per_cell=(8,8), cells_per_block=(2,2)) for img in XTest))

    def get_acc(preds): return GetPerformanceMetrics(yTest, preds)['Class 1']['Accuracy']

    # Alter C
    print("\nTesting C")
    results_c = {"C": Cs, "Gen": [], "Ens": [], "GenHOG": [], "EnsHOG": []}
    for c in Cs:
        m, pca = TrainGeneralSVM(XTrain, yTrain, n_components=bPCA, c=c)
        results_c["Gen"].append(get_acc(PredictGeneralSVM(m, pca, XTest)))
        ms, km, pca = TrainEnsembleSVM(XTrain, yTrain, n_clusters=bK, n_components=bPCA, c=c)
        results_c["Ens"].append(get_acc(PredictEnsembleSVM(ms, km, pca, XTest)))
        m, pca = TrainGeneralSVM(XTr_HOG, yTrain, n_components=bPCA, c=c)
        results_c["GenHOG"].append(get_acc(PredictGeneralSVM(m, pca, XTe_HOG)))
        ms, km, pca = TrainEnsembleSVM(XTr_HOG, yTrain, n_clusters=bK, n_components=bPCA, c=c)
        results_c["EnsHOG"].append(get_acc(PredictEnsembleSVM(ms, km, pca, XTe_HOG)))
        print(f"  > Finished C={c}")

    # Alter PCA components
    print("\nTesting num components")
    results_pca = {"PCA": PCAs, "Gen": [], "Ens": [], "GenHOG": [], "EnsHOG": []}
    for p in PCAs:
        m, pca = TrainGeneralSVM(XTrain, yTrain, n_components=p, c=bC)
        results_pca["Gen"].append(get_acc(PredictGeneralSVM(m, pca, XTest)))
        ms, km, pca = TrainEnsembleSVM(XTrain, yTrain, n_clusters=bK, n_components=p, c=bC)
        results_pca["Ens"].append(get_acc(PredictEnsembleSVM(ms, km, pca, XTest)))
        m, pca = TrainGeneralSVM(XTr_HOG, yTrain, n_components=p, c=bC)
        results_pca["GenHOG"].append(get_acc(PredictGeneralSVM(m, pca, XTe_HOG)))
        ms, km, pca = TrainEnsembleSVM(XTr_HOG, yTrain, n_clusters=bK, n_components=p, c=bC)
        results_pca["EnsHOG"].append(get_acc(PredictEnsembleSVM(ms, km, pca, XTe_HOG)))
        print(f"  > Finished PCA={p}")

    # Test cluster count
    print("\nTesting num clusters")
    results_k = {"K": Ks, "Ens": [], "EnsHOG": [], "Gen_Base": [], "GenHOG_Base": []}
    # Baselines for comparison
    m_gen, pca_gen = TrainGeneralSVM(XTrain, yTrain, n_components=bPCA, c=bC)
    g_base = get_acc(PredictGeneralSVM(m_gen, pca_gen, XTest))
    m_genh, pca_genh = TrainGeneralSVM(XTr_HOG, yTrain, n_components=bPCA, c=bC)
    gh_base = get_acc(PredictGeneralSVM(m_genh, pca_genh, XTe_HOG))

    for k in Ks:
        ms, km, pca = TrainEnsembleSVM(XTrain, yTrain, n_clusters=k, n_components=bPCA, c=bC)
        results_k["Ens"].append(get_acc(PredictEnsembleSVM(ms, km, pca, XTest)))
        ms, km, pca = TrainEnsembleSVM(XTr_HOG, yTrain, n_clusters=k, n_components=bPCA, c=bC)
        results_k["EnsHOG"].append(get_acc(PredictEnsembleSVM(ms, km, pca, XTe_HOG)))
        results_k["Gen_Base"].append(g_base)
        results_k["GenHOG_Base"].append(gh_base)
        print(f"  > Finished K={k}")

    # Alter HOG num orientations
    print("\nTesting num orientations")
    results_orient = {"Orient": Orients, "GenHOG": [], "EnsHOG": []}
    for orient in Orients:
        # Re-extract for each orientation
        XTr_temp = np.array(joblib.Parallel(n_jobs=-1)(joblib.delayed(hog)(img.reshape(32,32), orientations=orient, pixels_per_cell=(8,8), cells_per_block=(2,2)) for img in XTrain))
        XTe_temp = np.array(joblib.Parallel(n_jobs=-1)(joblib.delayed(hog)(img.reshape(32,32), orientations=orient, pixels_per_cell=(8,8), cells_per_block=(2,2)) for img in XTest))
        
        m, pca = TrainGeneralSVM(XTr_temp, yTrain, n_components=bPCA, c=bC)
        results_orient["GenHOG"].append(get_acc(PredictGeneralSVM(m, pca, XTe_temp)))
        ms, km, pca = TrainEnsembleSVM(XTr_temp, yTrain, n_clusters=bK, n_components=bPCA, c=bC)
        results_orient["EnsHOG"].append(get_acc(PredictEnsembleSVM(ms, km, pca, XTe_temp)))
        print(f"  > Finished Orient={orient}")

    # Save the models
    print("\nSaving final baseline models to /models...")
    # General (No HOG)
    m, pca = TrainGeneralSVM(XTrain, yTrain, n_components=bPCA, c=bC)
    SaveGeneralSVM(m, pca, "models/final_gen_svm")
    # Ensemble (No HOG)
    ms, km, pca = TrainEnsembleSVM(XTrain, yTrain, n_clusters=bK, n_components=bPCA, c=bC)
    SaveEnsembleSVM(ms, km, pca, "models/final_ens_svm")
    # General (HOG)
    m, pca = TrainGeneralSVM(XTr_HOG, yTrain, n_components=bPCA, c=bC)
    SaveGeneralSVMHOG(m, pca, bOrient, "models/final_gen_hog_svm")
    # Ensemble (HOG)
    ms, km, pca = TrainEnsembleSVM(XTr_HOG, yTrain, n_clusters=bK, n_components=bPCA, c=bC)
    SaveEnsembleSVMHOG(ms, km, pca, bOrient, "models/final_ens_hog_svm")

    # Plot the results
    def save_plot(data, x_key, title, filename, labels=["Gen", "Ens", "GenHOG", "EnsHOG"]):
        plt.figure(figsize=(10, 6))
        for label in labels:
            if label in data:
                plt.plot(data[x_key], data[label], marker='o', label=label)
        if x_key == 'C': plt.xscale('log') # Do log scale for C because otherwise values clump near 0
        plt.title(title)
        plt.xlabel(x_key)
        plt.ylabel("Task 1 Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/{filename}.png")
        plt.close()

    save_plot(results_c, "C", "Regularization (C) Sweep", "sweep_c")
    save_plot(results_pca, "PCA", "PCA Components Sweep", "sweep_pca")
    save_plot(results_k, "K", "Ensemble Cluster (K) Sweep", "sweep_k", labels=["Ens", "EnsHOG", "Gen_Base", "GenHOG_Base"])
    save_plot(results_orient, "Orient", "HOG Orientation Sweep", "sweep_orient", labels=["GenHOG", "EnsHOG"])

    print(f"\nTotal Time: {(time.time() - start_total)/60:.2f}m. Plots in /plots, Models in /models.")

if __name__ == "__main__":
    main()
