# Incident-Related Image Classification Project (Project CIRI)
Final project for Data Science, module CV&amp;IC - University of Twente 2023/2024

This repository contains the code and resources for the classification of incident-related images using machine learning techniques. The project leverages pre-trained models and transfer learning to optimize performance within computational constraints.

## Project Overview
The primary goal of this project is to classify images depicting various incidents such as natural disasters and accidents. Using the Incidents1M dataset, we adapted models like ResNet50 and Wide ResNet50 through transfer learning and hyperparameter tuning to achieve optimal classification performance.

![](https://github.com/GiuliaPais/ProjectCIRI/blob/main/figures/Slide4.jpeg)

## Dataset
The dataset used for this project is a subset of the Incidents1M dataset, which includes images depicting various types of incidents. The dataset presents various issues such as incorrectly classified images and class imbalance.

## Models
We used the following models for this project:

* ResNet50: A residual network model known for its robustness in image classification tasks.
* Wide ResNet50: An adaptation of ResNet50 with increased width for enhanced performance.

Both models were adapted to our dataset by modifying the last fully connected layer to match the number of classes in our dataset.

## Hyperparameter Tuning
Hyperparameter tuning was performed using nested cross-validation with randomized search to find the optimal configurations for epochs, batch size, and learning rate.

## Evaluation
The models were evaluated using 5-fold cross-validation. Performance metrics include accuracy, precision, recall, and F1 score.

![](https://github.com/GiuliaPais/ProjectCIRI/blob/main/figures/Slide5.jpeg)
![](https://github.com/GiuliaPais/ProjectCIRI/blob/main/figures/Slide9.jpeg)

# Technologies used

* Python 3.10
* PyTorch
* Ray (RayTune)
