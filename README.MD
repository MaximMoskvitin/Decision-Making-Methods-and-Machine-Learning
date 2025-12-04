# Decision-Making Methods and Machine Learning

This repository contains student laboratory work for the course **“Decision-Making Methods and Machine Learning”**. It brings together Python implementations and experiment notebooks for classic machine learning methods and neural network architectures, with a focus on decision-making, approximation, and analysis of data.

## Contents

* **Lab 1 – Gradient Descent:**
  Polynomial and linear regression on synthetic data, implemented with gradient descent; visualization of generated samples, learned curves, and loss decrease over iterations. 

* **Lab 2 – k-Nearest Neighbors (kNN):**
  kNN classification on synthetic 2D data and sklearn datasets (e.g. diabetes or iris); experiments with preprocessing, different values of *k*, class imbalance, and evaluation via accuracy and confusion matrix. 

* **Lab 3 – k-Means Clustering:**
  Clustering the Wine dataset with k-means; data exploration and scaling, choosing the number of clusters using the elbow method, silhouette score and other internal metrics, plus external metrics such as ARI and NMI. 

* **Lab 4 – Multilayer Perceptron (MLP):**
  Function approximation with fully connected neural networks (MLP) using PyTorch / TensorFlow; comparison of architectures by width and depth and analysis of MAPE to find a minimal model that reaches a target accuracy. 

* **Lab 5 – Convolutional Neural Networks (CNNs):**
  Image classification experiments on **MedMNIST2D** datasets; implementation of baseline CNNs, use of regularization techniques (data augmentation, ensembles, optimizers), and comparison of metrics with those reported in the paper *MedMNIST Classification Decathlon*. 

* **Lab 6 – Recurrent Neural Networks (RNNs):**
  Sequence modeling with RNNs for character-level text generation; preparation of text datasets, one-hot encoding, training recurrent models, and evaluation of generated sequences with appropriate metrics. 

Each lab includes:

* Structured, well-commented Python code
* Reproducible experiments and visualizations
* Short theoretical notes and answers to control questions from the assignments

## Technologies

* **Language:** Python 3.10
* **Core libraries:** NumPy, pandas, scikit-learn
* **Deep learning:** PyTorch and/or TensorFlow (depending on lab)
* **Visualization:** Matplotlib (and optionally Seaborn / other tools)

This repository is intended as an educational resource showcasing step-by-step implementations of fundamental decision-making and machine learning methods in a student setting.
