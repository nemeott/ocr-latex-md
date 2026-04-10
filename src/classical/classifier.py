import numpy as np
import joblib  # used for saving and loading the models; idea taken from https://medium.com/@sidakmenyadik/save-and-load-machine-learning-models-with-joblib-in-python-kneighborsclassifier-e474512e2683

from bounding_box import BoundingBox
from symbol import Symbol, SymbolType
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier # SVM with multiple classes
from sklearn.cluster import KMeans # Used for clustering
from sklearn.decomposition import PCA # Used to speed up training by removing less necessary features
from sklearn.metrics import accuracy_score, precision_recall_fscore_support # For the results values
from sklearn.preprocessing import StandardScaler # Used for normalizing values
from svm_preprocessing import *
from skimage.feature import hog

MAX_ITERS = 5000 # A constant just to keep things from getting out of hand

"""
Converts [[label1, label2], [[pixel]]] list into flattened and normalized X and y arrays.

Args:
    data_list (list): A list of samples structured as [labels, pixel_matrix].
    n (int, optional): The target height for the reshaped image, defaults to 32
    m (int, optional): The target width for the reshaped image, defaults to 32
Returns:
    np.ndarray, np.ndarray: Flattened and normalized features of shape (n_samples, n * m), array of corresponding labels of shape (n_samples, 2)
"""
def PreprocessInputs(data_list, n=32, m=32):
    X_list = []
    y_list = []
    
    for labels, features in data_list:
        label0, label1 = labels
        feature_array = np.array(features)
        reshaped = svm_reshape_image(feature_array, n, m)
        
        # Flatten the features
        X_list.append(reshaped.flatten())
        y_list.append(labels)
        
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Normalize the valeus
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

"""
Trains the baseline/general SVM model
Args:
    X_train (np.ndarray): The flattened, normalized input image, shape of (n_samples, n_features))
    y_train (np.ndarray): The target labels for each sample
    n_components (int, optional): Number for PCA, defaults to 50
    c (float, optional): C parameter for SVM
Returns:
    LinearSVC, PCA: The trained SVM and PCA
"""
def TrainGeneralSVM(X_train, y_train, n_components = 50, c = 1):
    pca = PCA(n_components=n_components).fit(X_train) # Reduce the image to the most important n_components features
    X_pca = pca.transform(X_train) # Actually change the images
    model = LinearSVC(max_iter = MAX_ITERS, dual = False, C = c) # SVM
    model_multi_class = MultiOutputClassifier(model).fit(X_pca, y_train)
    return model_multi_class, pca

"""
Saves the baseline/general SVM model
Args:
    model (LinearSVC): The SVM model to save
    pca (PCA): The PCA model to save
    filename (str, optional): File location, defaults to "general_svm"
"""
def SaveGeneralSVM(model, pca, filename="general_svm"):
    joblib.dump({'model': model, 'pca': pca}, f"{filename}.pkl")

"""
Loads the baseline/general SVM model
Args:
    filename (str, optional): File location, defaults to "general_svm"
Returns:
    LinearSVC, PCA: The saved SVM model and PCA
"""
def LoadGeneralSVM(filename="general_svm"):
    data = joblib.load(f"{filename}.pkl")
    return data['model'], data['pca']

"""
Generates predictions from the baseline/general SVM model
Args:
    model (LinearSVC): The trained SVM model
    pca (PCA): The fitted PCA transformer used during training
    X_test (np.ndarray): Flattened image of shape (n_samples, n_features)
Returns:
    np.ndarray: Array of predicted labels for each input
"""
def PredictGeneralSVM(model, pca, X_test):
    X_pca = pca.transform(X_test)
    return model.predict(X_pca)

"""
Trains the baseline/general SVM model using Histogram of Oriented Gradients
Args:
    X_train (np.ndarray): The flattened, normalized input image, shape of (n_samples, n_features))
    y_train (np.ndarray): The target labels for each sample
    n_components (int, optional): Number for PCA, defaults to 50
    c (float, optional): C parameter for SVM
Returns:
    LinearSVC, PCA, Orientations: The trained SVM and PCA and the nubmer of orientations used
"""
def TrainGeneralSVMHOG(X_train, y_train, n_components = 50, c = 1, Orientations = 10):
    side = int(np.sqrt(X_train.shape[1]))
    X_train_HOG = X_train_HOG = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(hog)(
            img.reshape(side, side), 
            orientations=Orientations, 
            pixels_per_cell=(8, 8), 
            cells_per_block=(2, 2)
        ) for img in X_train
    )
    X_train_HOG = np.array(X_train_HOG)
    model, pca = TrainGeneralSVM(X_train_HOG, y_train, n_components, c)
    return model, pca, Orientations

"""
Saves the baseline/general SVM model using Histogram of Oriented Gradients
Args:
    model (LinearSVC): The SVM model to save
    pca (PCA): The PCA model to save
    Orientations (int): The orientations number used to save
    filename (str, optional): File location, defaults to "general_svm_hog"
"""
def SaveGeneralSVMHOG(model, pca, Orientations, filename="general_svm_hog"):
    joblib.dump({'model': model, 'pca': pca, 'orientations': Orientations}, f"{filename}.pkl")

"""
Loads the baseline/general SVM model using Histogram of Oriented Gradients
Args:
    filename (str, optional): File location, defaults to "general_svm_hog"
Returns:
    LinearSVC, PCA, Orientations: The saved SVM model and PCA and the number of orientations
"""
def LoadGeneralSVMHOG(filename="general_svm_hog"):
    data = joblib.load(f"{filename}.pkl")
    return data['model'], data['pca'], data['orientations']

"""
Generates predictions from the baseline/general SVM model using Histogram of Oriented Gradients
Args:
    model (LinearSVC): The trained SVM model
    pca (PCA): The fitted PCA transformer used during training
    X_test (np.ndarray): Flattened image of shape (n_samples, n_features)
Returns:
    np.ndarray: Array of predicted labels for each input
"""
def PredictGeneralSVMHOG(model, pca, X_test, Orientations):
    side = int(np.sqrt(X_test.shape[1]))
    X_test_HOG = X_train_HOG = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(hog)(
            img.reshape(side, side), 
            orientations=Orientations, 
            pixels_per_cell=(8, 8), 
            cells_per_block=(2, 2)
        ) for img in X_test
    )
    X_test_HOG = np.array(X_test_HOG)
    X_pca = pca.transform(X_test_HOG)
    return model.predict(X_pca)

"""
Trains the baseline/general SVM model
Args:
    X_train (np.ndarray): The flattened, normalized input image, shape of (n_samples, n_features))
    y_train (np.ndarray): The target labels for each sample
    n_clusters (int, optional): Number of clusters used when clustering
    n_components (int, optional): Number for PCA, defaults to 50
    c (float, optional): C parameter for SVM
Returns:
    LinearSVC, PCA: The trained SVM and PCA
"""
def TrainEnsembleSVM(X_train, y_train, n_clusters = 5, n_components = 50, c = 1):
    # Same PCA as before because the computation was too expensive
    pca = PCA(n_components = n_components).fit(X_train)
    X_pca = pca.transform(X_train)

    # Cluster the data
    kmeans = KMeans(n_clusters = n_clusters, n_init = 10, random_state = 42).fit(X_pca)
    clusters = kmeans.predict(X_pca)

    # Train each of the models in the ensemble
    models = {}
    for i in range(n_clusters):
        mask = (clusters == i) # Mask for the images of the cluster
        if np.any(mask): # Make sure the cluster isn't empty (was getting a rare error that I think this was caused by empty clusters)
            cluster_labels = y_train[mask]
            not_all_same = all(len(np.unique(cluster_labels[:, col])) > 1 for col in range(cluster_labels.shape[1]))
            if not_all_same: # Another bug I was having where all values in the cluster were one label, so SVM was getting mad
                model = LinearSVC(max_iter = MAX_ITERS, dual = False, C = c)
                models[i] = MultiOutputClassifier(model).fit(X_pca[mask], y_train[mask])
            else:
                models[i] = cluster_labels[0] # just return a constant value
                
    return models, kmeans, pca

"""
Saves the ensemble SVM models
Args:
    models (dict): The SVM models to save
    kmeans (KMeans): The kmeans clustering model used
    pca (PCA): The PCA model to save
    filename (str, optional): Number for PCA, defaults to "ensemble_svm"
"""
def SaveEnsembleSVM(models, kmeans, pca, filename="ensemble_svm"):
    joblib.dump({'models': models, 'kmeans': kmeans, 'pca': pca}, f"{filename}.pkl")

"""
Loads the ensemble SVM models
Args:
    filename (str, optional): Number for PCA, defaults to "ensemble_svm"
Returns:
    dict, KMeans, PCA: The models, kmeans model, and pca model
"""
def LoadEnsembleSVM(filename="ensemble_svm"):
    data = joblib.load(f"{filename}.pkl")
    return data['models'], data['kmeans'], data['pca']

"""
Generates predictions from the ensemble SVM models
Args:
    models (dict): The trained SVM models
    kmeans (KMeans): The kmeans models
    pca (PCA): The fitted PCA transformer used during training
    X_test (np.ndarray): Flattened image of shape (n_samples, n_features)
Returns:
    np.ndarray: Array of predicted labels for each input
"""
def PredictEnsembleSVM(models, kmeans, pca, X_test):
    X_pca = pca.transform(X_test)
    clusters = kmeans.predict(X_pca)
    
    predictions = []
    for i, c in enumerate(clusters): # Get image index & correct cluster c
        model = models[c]
        if hasattr(model, "predict"):
            pred = models[c].predict(X_pca[i].reshape(1, -1))[0] # Get the predicted value from the correct cluster
        else:
            pred = model # This is for the case where SVM was mad and we just stored it as a constant label vector
        predictions.append(pred)
    return np.array(predictions)

"""
Trains the Ensemble SVM model using Histogram of Oriented Gradients
Args:
    X_train (np.ndarray): The flattened, normalized input image, shape of (n_samples, n_features))
    y_train (np.ndarray): The target labels for each sample
    n_clusters (int, optional): Number of clusters used when clustering
    n_components (int, optional): Number for PCA, defaults to 50
    c (float, optional): C parameter for SVM
Returns:
    LinearSVC, PCA, Orientations: The trained SVM and PCA and the nubmer of orientations used
"""
def TrainEnsembleSVMHOG(X_train, y_train, n_clusters = 5, n_components = 50, c = 1, Orientations = 10):
    side = int(np.sqrt(X_train.shape[1]))
    X_train_HOG = X_train_HOG = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(hog)(
            img.reshape(side, side), 
            orientations=Orientations, 
            pixels_per_cell=(8, 8), 
            cells_per_block=(2, 2)
        ) for img in X_train
    )
    X_train_HOG = np.array(X_train_HOG)
    model, kmeans, pca = TrainEnsembleSVM(X_train_HOG, y_train, n_clusters, n_components, c)
    return model, kmeans, pca, Orientations

"""
Saves the ensemble SVM model using Histogram of Oriented Gradients
Args:
    models (LinearSVC): The SVM model to save
    kmeans (KMeans): The kmeans clustering model used
    pca (PCA): The PCA model to save
    Orientations (int): The orientations number used to save
    filename (str, optional): Number for PCA, defaults to "ensemble_svm_hog"
"""
def SaveEnsembleSVMHOG(models, kmeans, pca, Orientations, filename="ensemble_svm_hog"):
    joblib.dump({'models': models, 'kmeans': kmeans, 'pca': pca, 'orientations': Orientations}, f"{filename}.pkl")

"""
Loads the ensemble SVM model using Histogram of Oriented Gradients
Args:
    filename (str, optional): Location of model saved defaults to "ensemble_svm_hog"
Returns:
    dict, KMeans, PCA, Orientations: The saved SVM models, KMeans model, and PCA and the number of orientations
"""
def LoadEnsembleSVMHOG(filename="ensemble_svm_hog"):
    data = joblib.load(f"{filename}.pkl")
    return data['model'], data['kmeans'], data['pca'], data['orientations']

"""
Generates predictions from the ensemble SVM model using Histogram of Oriented Gradients
Args:
    model (LinearSVC): The trained SVM model
    kmeans (KMeans): The kmeans clustering model used
    pca (PCA): The fitted PCA transformer used during training
    X_test (np.ndarray): Flattened image of shape (n_samples, n_features)
    Orientations (int): The orientations number used
Returns:
    np.ndarray: Array of predicted labels for each input
"""
def PredictEnsembleSVMHOG(model, kmeans, pca, X_test, Orientations):
    side = int(np.sqrt(X_test.shape[1]))
    X_test_HOG = X_train_HOG = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(hog)(
            img.reshape(side, side), 
            orientations=Orientations, 
            pixels_per_cell=(8, 8), 
            cells_per_block=(2, 2)
        ) for img in X_test
    )
    X_test_HOG = np.array(X_test_HOG)
    return PredictEnsembleSVM(model, kmeans, pca, X_test_HOG)

"""
Gets the perfomance metrics between y and yhat for one of the labels.

Args:
    y_true (np.ndarray): The correct labels
    y_pred (np.ndarray): the predicted lables (i.e., yhat)
Returns:
    dict: "Accuracy": the accuracy, "Precision": the precision, "Recall": the recall, and "F1-Score": the f1 score
"""
def GetSingleLabelMetrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    # LLM USE: I use an LLM to write this because I was not familiar with the sklearn function for it: precision_recall_fscore_support
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {"Accuracy": acc, "Precision": p, "Recall": r, "F1-Score": f1}


"""
Gets the perfomance metrics between y and yhat for both labels.

Args:
    y_true (np.ndarray): The correct labels
    y_pred (np.ndarray): the predicted lables (i.e., yhat)
Returns:
    dict: "Class 0": the single class metric for class 0, "Class 1": the single class metric for class 1
"""
def GetPerformanceMetrics(y_true, y_pred):
    return {"Class 0": GetSingleLabelMetrics(y_true[:, 0], y_pred[:, 0]), "Class 1": GetSingleLabelMetrics(y_true[:, 1], y_pred[:, 1])}


class SymbolClassifier:
    """Per-crop recognizer for the classical pipeline.

    Placeholder until a trained model is connected (e.g. joblib load from `TrainGeneralSVM`).
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path

    def predict(self, features: np.ndarray, box: BoundingBox) -> Symbol:
        _ = features  # unused until a real model is wired
        return Symbol("?", SymbolType.TEXT, box)
