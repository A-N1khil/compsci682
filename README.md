# Repo for assignment code of COMPSCI 682 - Neural Networks: A Modern Intro

# Contents
1. [Assignment 1](#assignment-1)  
    - [Goals](#goals)
    - [Q1: k-Nearest Neighbor classifier](#q1-k-nearest-neighbor-classifier-20-points)
    - [Q2: Training a Support Vector Machine](#q2-training-a-support-vector-machine-25-points)
    - [Q3: Implement a Softmax classifier](#q3-implement-a-softmax-classifier-20-points)
    - [Q4: Two-Layer Neural Network](#q4-two-layer-neural-network-25-points)
    - [Q5: Higher Level Representations: Image Features](#q5-higher-level-representations-image-features-10-points)
    - [Q6: Cool Bonus: Do something extra!](#q6-cool-bonus-do-something-extra-10-points)

# Assignment 1
## Goals
In this assignment you will practice putting together a simple image classification pipeline, based on the k-Nearest Neighbor or the SVM/Softmax classifier. The goals of this assignment are as follows:  
- understand the basic Image Classification pipeline and the data-driven approach (train/predict stages)
- understand the train/val/test splits and the use of validation data for hyperparameter tuning.
- develop proficiency in writing efficient vectorized code with numpy
- implement and apply a k-Nearest Neighbor (kNN) classifier
- implement and apply a Multiclass Support Vector Machine (SVM) classifier
- implement and apply a Softmax classifier
- implement and apply a Two layer neural network classifier
- understand the differences and tradeoffs between these classifiers
- get a basic understanding of performance improvements from using higher-level representations than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)

## Q1: k-Nearest Neighbor classifier (20 points)
The Notebook knn.ipynb will walk you through implementing the kNN classifier.

## Q2: Training a Support Vector Machine (25 points)
The Notebook svm.ipynb will walk you through implementing the SVM classifier.

## Q3: Implement a Softmax classifier (20 points)
The Notebook softmax.ipynb will walk you through implementing the Softmax classifier.

## Q4: Two-Layer Neural Network (25 points)
The Notebook two_layer_net.ipynb will walk you through the implementation of a two-layer neural network classifier.

## Q5: Higher Level Representations: Image Features (10 points)
The Notebook features.ipynb will walk you through this exercise, in which you will examine the improvements gained by using higher-level representations as opposed to using raw pixel values.

## Q6: Cool Bonus: Do something extra! (+10 points)
Implement, investigate or analyze something extra surrounding the topics in this assignment, and using the code you developed. For example, is there some other interesting question we could have asked? Is there any insightful visualization you can plot? Or anything fun to look at? Or maybe you can experiment with a spin on the loss function? If you try out something cool we’ll give you up to 10 extra points and may feature your results in the lecture.