# CIFAR-10 Convolutional Neural Network Classifier

## Synopsis

This repository houses the implementation of a machine learning model trained on the CIFAR-10 dataset for the purpose of 2D image classification. Built using TensorFlow, our approach leverages the power of Convolutional Neural Networks (CNNs) to discern and categorize image content. Through meticulous training and optimization, the model achieves an impressive validation accuracy of approximately 93.0%. For those keen on the intricate details of our methodology, architecture, and results, we recommend diving into our comprehensive [research paper](https://harshonyou.github.io/CIFAR-10/paper/research_paper.pdf) associated with this work.

## Implementation

#### config.py
Configuration details for the model and dataset, outlining dimensions, data configurations, model specifics, hyperparameters, and externals like loss type and file saving locations.

#### data.py
Utility functions designed to streamline the process of loading and preprocessing the CIFAR-10 dataset.

#### model.py
Specifies the architecture of the neural network for image classification on the CIFAR-10 dataset.

#### test_accuracy.py
A tool to evaluate the performance of trained neural network models on the CIFAR-10 test dataset. It predicts on the test set and subsequently computes the accuracy by comparing the predictions with the true labels.

#### train.py
The main training script orchestrating the end-to-end training process of a neural network model for the CIFAR-10 dataset. It facilitates the entire pipeline from data loading, preprocessing, model construction, to training.
