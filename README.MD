Вот обновлённый вариант README с учётом того, что у тебя уже реально сделано в ЛР и в коде:

---

# Decision-Making Methods and Machine Learning

This repository contains student laboratory work for the course **“Decision-Making Methods and Machine Learning”**. It brings together Python implementations and experiment notebooks for classic machine learning methods and neural network architectures, with a focus on decision-making, approximation, and analysis of data.

## Contents

**Lab 1 – Gradient Descent**
Polynomial and linear regression on synthetic 1D data, implemented with gradient descent in TensorFlow. The lab covers synthetic data generation (uniform inputs + Gaussian noise), training quadratic and linear models, visualizing samples and fitted curves, and tracking MSE loss over epochs on train/test splits. 

**Lab 2 – k-Nearest Neighbors (kNN)**
kNN classification on synthetic 2D data in the rectangle [0,10] × [0,100] (custom NumPy implementation) and on a real medical dataset (Pima Indians Diabetes from OpenML) using `scikit-learn`. Experiments include:

* preprocessing with and without feature scaling,
* varying the amount of cluster overlap and class imbalance for synthetic data,
* scanning different values of *k* and distance metrics,
* evaluation via accuracy and confusion matrix, plus plots of accuracy vs *k* and visualizations of predicted clusters. 

**Lab 3 – k-Means Clustering**
Clustering the **Wine** dataset with k-means: data exploration (types, distributions, potential outliers), scaling with `StandardScaler`, and systematic selection of the number of clusters using:

* inertia / WCSS (elbow method),
* silhouette score,
* Davies–Bouldin index,
* Calinski–Harabasz score.

The lab also reports external metrics (Adjusted Rand Index, Normalized Mutual Information) relative to the true wine classes, analyzes clustering stability over multiple random initializations, and visualizes true classes vs clusters in 2D via PCA.

**Lab 4 – Multilayer Perceptron (MLP)**
Function approximation with fully connected neural networks (MLPs) using PyTorch and/or TensorFlow. According to the lab statement, the goal is to approximate two target functions and to find the smallest architecture (in terms of depth and width / parameter count) that achieves a target accuracy of **MAPE ≤ 10%**, accompanied by plots of MAPE as a function of network width and depth. 

**Lab 5 – Convolutional Neural Networks (CNNs)**
Image classification experiments on one of the **MedMNIST2D** datasets. The lab includes:

* implementation of a baseline CNN,
* exploration of regularization techniques (data augmentation, ensembles, different optimizers),
* comparison of obtained metrics (e.g., AUC, accuracy) with those reported in the *“MedMNIST Classification Decathlon”* paper,
* discussion of how regularization changes performance. 

**Lab 6 – Recurrent Neural Networks (RNNs)**
Sequence modeling with recurrent neural networks for sequential data. The lab follows the statement:

* studying the basics of RNNs,
* building and training an RNN model,
* applying it to either time series forecasting or character-level text generation,
* evaluating the quality of predictions (e.g., via error metrics for time series or perplexity / qualitative analysis for generated text). 

---

## Each lab includes

* Structured, well-commented Python code (`.py` scripts).
* Jupyter notebooks (`.ipynb`) with reproducible experiments and visualizations (where available).
* Short theoretical notes and answers to control questions from the assignments (in Russian, embedded in code comments and/or notebooks).
* Lab statements as PDF files describing tasks, theory, and control questions.

---

## Technologies

* **Language:** Python 3.10
* **Core libraries:** NumPy, pandas, scikit-learn
* **Deep learning:** TensorFlow and/or PyTorch (depending on the lab)
* **Visualization:** Matplotlib

This repository is intended as an educational resource showcasing step-by-step implementations of fundamental decision-making and machine learning methods in a student setting.
